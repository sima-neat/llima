import logging
import math
from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout
from afe.ir.defines import get_expected_tensor_value
from sima_lmm.model.base import (
    BaseModel, EvalMode, FileGenMode, TensorTessellateParameters, GenConfiguration,
    LayerConfiguration
)
from sima_lmm.model.gemma4_vision_model import Gemma4VisionLayerModel
from sima_lmm.model.qwen_vision_model import QwenVisionLayerModel
from sima_lmm.model.model_graph import (
    ModelGraph, Node,
    load_tensor_from_source,
)
from sima_lmm.config.vlm_config import VisionArchType, VlmArchType


def _resize_siglip2_position_embeddings(
    position_embeddings: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Resize a square SigLIP2 position grid using the upstream interpolation contract."""
    if target_height <= 0 or target_width <= 0:
        raise ValueError("SigLIP2 position embedding dimensions must be positive")

    sequence_length, embedding_dim = position_embeddings.shape
    source_grid_size = math.isqrt(sequence_length)
    if source_grid_size * source_grid_size != sequence_length:
        raise ValueError(
            "SigLIP2 position embeddings must contain a square source grid; "
            f"received {sequence_length} positions"
        )
    if target_height == source_grid_size and target_width == source_grid_size:
        return position_embeddings

    def resampling_weights(source_size: int, target_size: int) -> np.ndarray:
        # align_corners=False maps output pixel centers with a half-pixel offset.
        scale = source_size / target_size
        # Widen the linear filter while downsampling to match antialias=True.
        filter_scale = max(scale, 1.0)
        source_centers = np.arange(source_size, dtype=np.float32) + 0.5
        target_centers = (np.arange(target_size, dtype=np.float32) + 0.5) * scale
        weights = np.maximum(
            0.0,
            1.0
            - np.abs(target_centers[:, None] - source_centers[None, :]) / filter_scale,
        )
        return weights / weights.sum(axis=1, keepdims=True)

    source_dtype = position_embeddings.dtype
    position_grid = position_embeddings.astype(np.float32).reshape(
        source_grid_size, source_grid_size, embedding_dim
    )
    height_weights = resampling_weights(source_grid_size, target_height)
    width_weights = resampling_weights(source_grid_size, target_width)
    resized = np.einsum(
        "hi,ijc,wj->hwc",
        height_weights,
        position_grid,
        width_weights,
        optimize=True,
    )
    return resized.reshape(target_height * target_width, embedding_dim).astype(source_dtype)


@dataclass
class VisionModel(BaseModel):
    """Vision model implementation."""

    def run_model(self, eval_mode: EvalMode, ifms: list[np.ndarray]) -> list[np.ndarray]:
        layer_ifms = ifms
        deepstack_outputs: dict[int, np.ndarray] = {}
        deepstack_indexes = self.cfg.vm_cfg.deepstack_visual_indexes

        for layer_idx in range(self.cfg.num_vision_layers):
            layer_outputs = list(
                self._get_part_model(layer_idx).run_model(eval_mode, layer_ifms)
            )
            if (
                self.cfg.model_type == VlmArchType.VLM_QWEN3_VL
                and layer_idx in deepstack_indexes
            ):
                deepstack_idx = deepstack_indexes.index(layer_idx)
                deepstack_outputs[deepstack_idx] = layer_outputs.pop()
            layer_ifms = [layer_outputs[0]]

        return [
            *layer_outputs,
            *(deepstack_outputs[idx] for idx in range(len(deepstack_indexes))),
        ]

    def gen_files(
        self,
        gen_mode: FileGenMode,
        *,
        gen_config: GenConfiguration,
        log_level: int = logging.NOTSET,
        num_processes: int = 1,
        resume: bool = False,
    ):
        """
        Generates files based on the provided file generation mode.

        Args:
            gen_mode: File generation mode.
            gen_config: Generation configuration of precision and lora for each layer.
                Layer IDs can be obtained using VlmConfig.get_layer_ids.
                The precision dict is a map from layer ID to precision for each layer to be
                processed. Unrecognized layer IDs will be ignored.  Layers that are not in the map
                will not be processed.
                The lora dict is a map from layer ID to lora graph mode for each layer.
            log_level: Logging level.
            resume: Generate the files if missing.
        """
        precision = gen_config["precision"]
        lora_mode = gen_config.get("lora", None)

        # Create a list of all models to compile
        model_list = list()
        for layer_id, curr_precision in precision.items():
            match layer_id.part:
                case "vision":
                    part_model = self._get_part_model(layer_id.part_idx)
                case _:
                    # Not a part of this model
                    continue
            curr_cfg = {"precision": curr_precision}
            if lora_mode:
                curr_cfg["lora"] = lora_mode[layer_id]
            model_list.append((part_model, curr_cfg))

        # Finished creating model_list.  Compile these models.
        self.gen_files_from_model_list(model_list, gen_mode, num_processes, log_level, resume)

    def _get_part_model(self, layer_idx: int) -> BaseModel:
        if not 0 <= layer_idx < self.cfg.num_vision_layers:
            raise ValueError(
                f"Vision layer index {layer_idx} is outside the valid range "
                f"[0, {self.cfg.num_vision_layers})"
            )
        include_embeddings = layer_idx == 0
        include_mm_proj = layer_idx == self.cfg.num_vision_layers - 1
        model_name = f"{self.model_name}_layer{layer_idx}"

        kwargs = {
            "cfg": self.cfg,
            "model_name": model_name,
            "sima_path": self.sima_path,
            "hf_model": self.hf_model,
            "layer_idx": layer_idx,
            "include_embeddings": include_embeddings,
            "include_mm_proj": include_mm_proj,
        }

        # Dispatch based on model type
        if self.cfg.model_type in (
            VlmArchType.VLM_QWEN2_5_VL,
            VlmArchType.VLM_QWEN3_VL,
            VlmArchType.VLM_QWEN3_5_VL,
        ):
            return QwenVisionLayerModel(**kwargs)
        elif self.cfg.model_type == VlmArchType.VLM_GEMMA4:
            return Gemma4VisionLayerModel(**kwargs)
        else:
            return StandardVisionLayerModel(**kwargs)


@dataclass
class StandardVisionLayerModel(BaseModel):
    """Vision model for each transformer layer with embedding or multimodal projection.

    Handles Standard architectures: CLIP, SigLIP, LFM2 (non-Qwen).
    """

    layer_idx: int
    include_embeddings: bool
    include_mm_proj: bool

    def generate_graph(
        self,
        layer_cfg: LayerConfiguration,
        quantizable: bool
    ):
        if self.include_embeddings:
            if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
                patch_dim = 3 * (self.cfg.vm_cfg.patch_size ** 2)
                input_shape = (1, 1, self.cfg.vm_cfg.seq_len, patch_dim)
            else:
                if isinstance(self.cfg.vm_cfg.image_size, list):
                    image_h, image_w = self.cfg.vm_cfg.image_size
                else:
                    image_h = image_w = self.cfg.vm_cfg.image_size
                input_shape = (1, image_h, image_w, 3)
        else:
            input_shape = (1, 1, self.cfg.vm_cfg.seq_len, self.cfg.vm_cfg.hidden_size)

        graph = ModelGraph(self, {"input": input_shape}, quantizable)
        inputs = graph.inputs
        mla_input = inputs["input"]

        # Vision tower.
        vision_output = self._build_vision_tower(
            graph, self.hf_model.vision_model_param_base_name, mla_input
        )

        # MM projection.
        if self.include_mm_proj:
            if self.cfg.model_type == VlmArchType.VLM_LLAVA:
                llava_o_shape = get_expected_tensor_value(vision_output.get_type().output).shape
                vision_output = graph.slice(
                    vision_output, start=1, stop=llava_o_shape[2], axis=2
                )
            if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
                projector_base_name = "model.multi_modal_projector"
            else:
                projector_base_name = "multi_modal_projector"
            vision_output = self._build_mm_projector(
                graph, projector_base_name, vision_output
            )

        outputs = [vision_output]
        if self.include_mm_proj and self.cfg.pipeline_cfg.quantize_embeddings:
            vision_output, vision_scale = graph.quant(vision_output)
            outputs = [vision_output, vision_scale]
        graph.save(outputs)

    def _build_vision_tower(self, graph: ModelGraph, base_name: str, input_node: Node) -> Node:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        if self.include_embeddings:
            embeddings = self._build_patch_embeddings(graph, f"{base_name}.embeddings", input_node)

            if self.cfg.vm_cfg.arch == VisionArchType.CLIP:
                # Note that the original source code has a typo in the layer norm node name.
                encoder_input = graph.layer_norm(
                    f"{base_name}.pre_layrnorm",
                    embeddings,
                    axis=-1,
                    epsilon=epsilon,
                )
            else:
                encoder_input = embeddings
        else:
            encoder_input = input_node

        encoder_output = self._build_encoder(
            graph,
            f"{base_name}.encoder.layers.{self.layer_idx}",
            encoder_input,
        )

        if not self.include_mm_proj:
            return encoder_output

        post_layer_norm = graph.layer_norm(
            f"{base_name}.post_layernorm",
            encoder_output,
            axis=-1,
            epsilon=epsilon,
        )
        return post_layer_norm

    def _build_patch_embeddings(self, graph: ModelGraph, base_name: str, input_node: Node) -> Node:
        node_name = f"{base_name}.patch_embedding"

        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            # LFM2: input is already pre-patchified (1, 1, seq_len, patch_dim) — FC projection only.
            embeddings = graph.linear(node_name, input_node)
        else:
            if isinstance(self.cfg.vm_cfg.image_size, list):
                image_h = self.cfg.vm_cfg.image_size[0]
            else:
                image_h = self.cfg.vm_cfg.image_size
            patch_embedding = graph.conv(
                node_name,
                input_node,
                stride=(self.cfg.vm_cfg.patch_size,) * 2,
            )
            # NHWC layout: split on axis H, concat on axis W → (1, 1, seq_len, hidden)
            split_and_concat = graph.split_concat(
                patch_embedding, axis=2,
                split_axis=1,
                split_block=image_h // self.cfg.vm_cfg.patch_size,
                split_repeat=1
            )
            # CLIP requires class embedding prepended; SigLIP does not.
            if self.cfg.vm_cfg.arch == VisionArchType.CLIP:
                class_embedding_weight = load_tensor_from_source(
                    f"{base_name}.class_embedding",
                    self.get_hf_param, self.check_hf_param,
                    reshape_str="c->nhwc"
                )
                class_embedding = graph.constant(class_embedding_weight)
                embeddings = graph.concat([class_embedding, split_and_concat], axis=2)
            else:
                embeddings = split_and_concat

        # Position embedding — LFM2 may need bilinear resize if resolution differs from pretraining.
        position_embedding_weight = self.get_hf_param(f"{base_name}.position_embedding.weight")
        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            if isinstance(self.cfg.vm_cfg.num_patches, list):
                target_h, target_w = self.cfg.vm_cfg.num_patches
            else:
                target_h = target_w = self.cfg.vm_cfg.num_patches
            position_embedding_weight = _resize_siglip2_position_embeddings(
                position_embedding_weight,
                target_h,
                target_w,
            )

        # Reshape "wc->nhwc"; graph.constant selects the activation dtype.
        pos_weight = position_embedding_weight.reshape(
            1, 1, position_embedding_weight.shape[0], position_embedding_weight.shape[1]
        )
        position_embedding = graph.constant(pos_weight)
        embeddings = graph.add(embeddings, position_embedding)
        return embeddings

    def _build_encoder(self, graph: ModelGraph, base_name: str, input_node: Node) -> Node:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        layer_norm1 = graph.layer_norm(
            f"{base_name}.layer_norm1",
            input_node,
            axis=-1,
            epsilon=epsilon,
        )
        self_attn = self._build_encoder_attention(graph, f"{base_name}.self_attn", layer_norm1)
        add1 = graph.add(input_node, self_attn)
        layer_norm2 = graph.layer_norm(f"{base_name}.layer_norm2", add1, axis=-1, epsilon=epsilon)
        mlp = graph.mlp(f"{base_name}.mlp", layer_norm2, self.cfg.vm_cfg.hidden_act)
        add2 = graph.add(add1, mlp)
        return add2

    def _build_encoder_attention(self, graph: ModelGraph, base_name: str, input_node: Node) -> Node:
        num_heads = self.cfg.vm_cfg.num_attention_heads
        head_dim = self.cfg.vm_cfg.hidden_size // num_heads
        projection_options, output_options = graph._head_padding_options(
            f"{base_name}.out_proj", num_heads, head_dim
        )

        query, key, value = [
            graph.split_heads(
                graph.linear(
                    f"{base_name}.{proj}", input_node,
                    scale=head_dim ** -0.5 if proj == "q_proj" else 1.0,
                    **projection_options,
                ),
                num_heads,
            )
            for proj in ("q_proj", "k_proj", "v_proj")
        ]
        context = graph.attention(query, key, value)
        return graph.linear(f"{base_name}.out_proj", graph.merge_heads(context), **output_options)

    def _build_mm_projector(self, graph: ModelGraph, base_name: str, input_node: Node) -> Node:
        match self.cfg.model_type:
            case VlmArchType.VLM_LFM2_VL:
                # NHWC: (1, 1, seq_len, hidden) → (1, num_patches_h, num_patches_w, hidden)
                if isinstance(self.cfg.vm_cfg.num_patches, list):
                    num_patches_h = self.cfg.vm_cfg.num_patches[0]
                else:
                    num_patches_h = self.cfg.vm_cfg.num_patches
                reshaped = graph.split_concat(
                    input_node, axis=1,
                    split_axis=2,
                    split_block=num_patches_h,
                    split_repeat=1
                )
                factor = self.cfg.mm_cfg.downsample_factor
                unshuffled = graph.space_to_depth(reshaped, factor)
                projector_input = unshuffled
                if self.cfg.mm_cfg.projector_use_layernorm:
                    epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
                    projector_input = graph.layer_norm(
                        f"{base_name}.layer_norm",
                        projector_input,
                        axis=-1,
                        epsilon=epsilon,
                    )
                fc1 = graph.linear(f"{base_name}.linear_1", projector_input)
                act = graph.activation(fc1, self.cfg.mm_cfg.hidden_act)
                last = graph.linear(f"{base_name}.linear_2", act)
            case VlmArchType.VLM_LLAVA:
                fc1 = graph.linear(f"{base_name}.linear_1", input_node)
                act = graph.activation(fc1, self.cfg.mm_cfg.hidden_act)
                last = graph.linear(f"{base_name}.linear_2", act)
            case VlmArchType.VLM_GEMMA3:
                # NHWC layout, split on axis W, concat on axis H
                reshape1 = graph.split_concat(
                    input_node, axis=1,
                    split_axis=2,
                    split_block=self.cfg.vm_cfg.num_patches,
                    split_repeat=1
                )
                tokens_per_side = int(self.cfg.mm_cfg.mm_tokens_per_image ** 0.5)
                kernel_shape = tuple([self.cfg.vm_cfg.num_patches // tokens_per_side] * 2)
                avgpool = graph.avgpool2d(
                    reshape1, kernel_shape=kernel_shape, strides=kernel_shape
                )
                epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
                norm = graph.rms_norm(
                    f"{base_name}.mm_soft_emb_norm", avgpool, epsilon=epsilon, weight_offset=1.0
                )
                last = graph.linear(
                    f"{base_name}.proj",
                    norm,
                    reshape_str="io->oihw",
                    src_weight_name="multi_modal_projector.mm_input_projection_weight",
                )
            case VlmArchType.VLM_PALIGEMMA:
                last = graph.linear(f"{base_name}.linear", input_node)
            case _:
                raise ValueError(
                    f"Multi-modal projection for {self.cfg.model_type} is not supported."
                )
        return last

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's inputs on the MLA.
        """
        input_tessellate_params = TensorTessellateParameters(
            tile_shape=(0, 0, 0, 0),
            enable_mla=True,
            dram_layout=TensorDRAMLayout.HWC,
            persistent_mem_name="input",
            dram_shape=None,
        )
        return {0: input_tessellate_params}

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        """
        Get the custom tessellate params for model's output on the MLA.
        """
        # Use default tessellate params.
        return {}
