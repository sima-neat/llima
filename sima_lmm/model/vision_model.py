import logging
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import zoom

from afe.apis.defines import gen2_target, TensorDRAMLayout
from afe.ir.serializer import save_awesomenet
from afe.ir.defines import Status, get_expected_tensor_value
from afe.ir.tensor_type import TensorType, ScalarType
from afe.ir.build_node import NodeOrHandle
from sima_lmm.model.base import (
    BaseModel, FileGenMode, TensorTessellateParameters, GenConfiguration, LayerConfiguration
)
from sima_lmm.model.onnx_builder import OnnxNode
from sima_lmm.model.gemma4_vision_model import Gemma4VisionLayerModel
from sima_lmm.model.qwen_vision_model import QwenVisionLayerModel
from sima_lmm.model.sima_builder import (
    SimaBuilder, build_conv, build_activation, activation_type, activation_dtype,
    build_merge_heads_and_matmul, build_matmul_and_split_heads, build_two_stage_layer_norm,
    build_space_to_depth, load_tensor_from_source
)
from sima_lmm.config.layer_id import LayerID
from sima_lmm.config.vlm_config import VisionArchType, VlmArchType


@dataclass
class VisionModel(BaseModel):
    """Vision model implementation."""

    is_single_vision_model: bool = field(default=True, kw_only=True)
    actual_num_hidden_layers: int = field(init=False)

    def __post_init__(self):
        num_hidden_layers = self.cfg.vm_cfg.num_hidden_layers
        if self.cfg.model_type == VlmArchType.VLM_LLAVA:
            num_hidden_layers -= 1
        self.actual_num_hidden_layers = num_hidden_layers

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
        if self.is_single_vision_model:
            include_embeddings = True
            include_mm_proj = True
            num_layers = self.actual_num_hidden_layers
            model_name = self.model_name
        else:
            include_embeddings = layer_idx == 0
            include_mm_proj = layer_idx == self.actual_num_hidden_layers - 1
            num_layers = 1
            model_name = f"{self.model_name}_layer{layer_idx}"
            
        kwargs = {
            "cfg": self.cfg,
            "model_name": model_name,
            "onnx_path": self.onnx_path,
            "sima_path": self.sima_path,
            "hf_model": self.hf_model,
            "layer_idx": layer_idx,
            "num_layers": num_layers,
            "include_embeddings": include_embeddings,
            "include_mm_proj": include_mm_proj,
        }

        # Dispatch based on model type
        if self.cfg.model_type in (VlmArchType.VLM_QWEN2_5_VL, VlmArchType.VLM_QWEN3_VL):
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
    num_layers: int
    include_embeddings: bool
    include_mm_proj: bool

    def gen_onnx_files(self):
        base_name = "vision_model"
        self.create_onnx_builder()
        if self.include_embeddings:
            if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
                patch_dim = 3 * (self.cfg.vm_cfg.patch_size**2)
                self._onnx_builder.create_input_node(
                    "input", (1, patch_dim, 1, self.cfg.vm_cfg.seq_len)
                )
            else:
                if isinstance(self.cfg.vm_cfg.image_size, list):
                    image_h = self.cfg.vm_cfg.image_size[0]
                    image_w = self.cfg.vm_cfg.image_size[1]
                else:
                    image_h = image_w = self.cfg.vm_cfg.image_size

                self._onnx_builder.create_input_node("input", (1, 3, image_h, image_w))
        else:
            self._onnx_builder.create_input_node(
                "input", (1, self.cfg.vm_cfg.hidden_size, 1, self.cfg.vm_cfg.seq_len)
            )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)
        if self.include_mm_proj:
            # Include the multimodal projection in the last transformer layer.
            match self.cfg.model_type:
                case VlmArchType.VLM_LLAVA | VlmArchType.VLM_PALIGEMMA:
                    self._onnx_builder.create_output_node(
                        self._onnx_builder.get_node_output_name(output_nodes[0]),
                        (1, self.cfg.lm_cfg.hidden_size, 1, self.cfg.vm_cfg.num_patches**2),
                    )
                case VlmArchType.VLM_GEMMA3:
                    tokens_per_side = int(self.cfg.mm_cfg.mm_tokens_per_image**0.5)
                    self._onnx_builder.create_output_node(
                        self._onnx_builder.get_node_output_name(output_nodes[0]),
                        (1, self.cfg.lm_cfg.hidden_size, tokens_per_side, tokens_per_side),
                    )
                case VlmArchType.VLM_LFM2_VL:
                    if isinstance(self.cfg.vm_cfg.num_patches, list):
                        num_patches_h = self.cfg.vm_cfg.num_patches[0]
                        num_patches_w = self.cfg.vm_cfg.num_patches[1]
                    else:
                        num_patches_h = num_patches_w = self.cfg.vm_cfg.num_patches
                    factor = self.cfg.mm_cfg.downsample_factor
                    self._onnx_builder.create_output_node(
                        self._onnx_builder.get_node_output_name(output_nodes[0]),
                        (
                            1,
                            self.cfg.lm_cfg.hidden_size,
                            num_patches_h // factor,
                            num_patches_w // factor,
                        ),
                    )
        else:
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[0]),
                (1, self.cfg.vm_cfg.hidden_size, 1, self.cfg.vm_cfg.seq_len),
            )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        vision_output = self._build_vision_tower(
            self.hf_model.vision_model_param_base_name, input_nodes
        )
        if not self.include_mm_proj:
            return [vision_output]

        if self.cfg.model_type == VlmArchType.VLM_LLAVA:
            mm_project_input = self._onnx_builder.build_op(
                "slice",
                [
                    vision_output,
                    np.array([0, 0, 0, 1], dtype=np.int64),
                    np.array([sys.maxsize, sys.maxsize, sys.maxsize, sys.maxsize], dtype=np.int64),
                ],
                "Slice",
            )
        else:
            mm_project_input = vision_output
        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            projector_base_name = "model.multi_modal_projector"
        else:
            projector_base_name = "multi_modal_projector"
        mm_project_output = self._build_mm_projector(projector_base_name, [mm_project_input])
        return [mm_project_output]

    def _build_vision_tower(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        if self.include_embeddings:
            embeddings = self._build_embeddings(f"{base_name}.embeddings", input_nodes)

            if self.cfg.vm_cfg.arch == VisionArchType.CLIP:
                # Note that the original source code has a typo in the layer norm node name.
                encoder_input = self._onnx_builder.build_layer_norm(
                    f"{base_name}.pre_layrnorm", embeddings, self.cfg.vm_cfg.layer_norm_eps
                )
            else:
                encoder_input = embeddings
        else:
            encoder_input = input_nodes[0]

        if self.num_layers > 1:
            for layer_idx in range(self.num_layers):
                encoder_input = self._build_encoder(
                    f"{base_name}.encoder.layers.{layer_idx}", [encoder_input]
                )
            encoder_output = encoder_input
        else:
            encoder_output = self._build_encoder(
                f"{base_name}.encoder.layers.{self.layer_idx}", [encoder_input]
            )

        if not self.include_mm_proj:
            return encoder_output

        post_layer_norm = self._onnx_builder.build_layer_norm(
            f"{base_name}.post_layernorm", encoder_output, self.cfg.vm_cfg.layer_norm_eps
        )
        return post_layer_norm

    def _build_embeddings(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:

            # Apply the linear projection, input is already in patches
            node_name = f"{base_name}.patch_embedding"
            embeddings = self._onnx_builder.build_conv(node_name, input_nodes[0], is_fc=True)
        else:
            # Original logic for CLIP and SIGLIP
            node_name = f"{base_name}.patch_embedding"
            patch_embedding = self._onnx_builder.build_conv(
                node_name, input_nodes[0], is_fc=False, strides=[self.cfg.vm_cfg.patch_size] * 2
            )
            split_and_concat = self._onnx_builder.build_split_and_concat(
                f"{base_name}.reshape",
                patch_embedding,
                self.cfg.vm_cfg.image_size // self.cfg.vm_cfg.patch_size,
                2,
                3,
            )
            embeddings = split_and_concat

        if self.cfg.vm_cfg.arch == VisionArchType.CLIP:
            node_name = f"{base_name}.concat_class_embedding"
            embeddings = self._onnx_builder.build_op(
                node_name,
                [
                    self._onnx_builder.create_initializer(
                        f"{base_name}.class_embedding", reshape_str="c->nchw"
                    ),
                    embeddings,  # Use the embeddings from above
                ],
                "Concat",
                axis=3,
            )

        # Resize positional embeddings for Siglip2 if image_size is dynamic.
        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            node_name = f"{base_name}.add_position_embedding"
            position_embedding_weight = self._onnx_builder.get_param_func(
                f"{base_name}.position_embedding.weight"
            )

            original_seq_len = position_embedding_weight.shape[0]
            original_grid_size = int(original_seq_len**0.5)
            if isinstance(self.cfg.vm_cfg.num_patches, list):
                target_grid_height = self.cfg.vm_cfg.num_patches[0]
                target_grid_width = self.cfg.vm_cfg.num_patches[1]
            else:
                target_grid_height = target_grid_width = self.cfg.vm_cfg.num_patches
            target_seq_len = target_grid_height * target_grid_width
            pos_emb_grid = position_embedding_weight.reshape(
                original_grid_size, original_grid_size, self.cfg.vm_cfg.hidden_size
            )

            height_zoom = target_grid_height / original_grid_size
            width_zoom = target_grid_width / original_grid_size
            zoom_factors = (height_zoom, width_zoom, 1.0)
            resized_pos_emb_grid = zoom(pos_emb_grid.astype(np.float32), zoom_factors, order=1)
            final_pos_emb_weight = resized_pos_emb_grid.reshape(
                target_seq_len, self.cfg.vm_cfg.hidden_size
            )

            position_embedding = self._onnx_builder.create_initializer(
                f"{base_name}.position_embedding.weight",
                value=final_pos_emb_weight.astype(position_embedding_weight.dtype),
                reshape_str="wc->nchw",
            )
        else:
            position_embedding = self._onnx_builder.create_initializer(
                f"{base_name}.position_embedding.weight", reshape_str="wc->nchw"
            )

        embeddings = self._onnx_builder.build_op(
            f"{base_name}.add_position_embedding", [embeddings, position_embedding], "Add"
        )

        return embeddings

    def _build_encoder(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        layer_norm1 = self._onnx_builder.build_layer_norm(
            f"{base_name}.layer_norm1", input_nodes[0], self.cfg.vm_cfg.layer_norm_eps
        )
        self_attn = self._build_encoder_attention(f"{base_name}.self_attn", [layer_norm1])
        add1 = self._onnx_builder.build_op(f"{base_name}.add1", [input_nodes[0], self_attn], "Add")
        layer_norm2 = self._onnx_builder.build_layer_norm(
            f"{base_name}.layer_norm2", add1, self.cfg.vm_cfg.layer_norm_eps
        )
        mlp = self._build_encoder_mlp(f"{base_name}.mlp", [layer_norm2])
        add2 = self._onnx_builder.build_op(f"{base_name}.add2", [add1, mlp], "Add")
        return add2

    def _build_encoder_attention(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        num_heads = self.cfg.vm_cfg.num_attention_heads
        head_dim = self.cfg.vm_cfg.hidden_size // num_heads

        scaled_q_projs = self._onnx_builder.build_matmul_and_split_heads(
            f"{base_name}.q_proj",
            input_nodes[0],
            num_heads,
            self.cfg.vm_cfg.seq_len,
            post_matmul_scale=head_dim**-0.5,
        )
        k_projs = self._onnx_builder.build_matmul_and_split_heads(
            f"{base_name}.k_proj", input_nodes[0], num_heads, self.cfg.vm_cfg.seq_len
        )
        v_projs = self._onnx_builder.build_matmul_and_split_heads(
            f"{base_name}.v_proj", input_nodes[0], num_heads, self.cfg.vm_cfg.seq_len
        )

        attn_outputs = list()
        for i, scaled_q_proj, k_proj, v_proj in zip(
            range(num_heads), scaled_q_projs, k_projs, v_projs
        ):
            attn_weights = self._onnx_builder.build_op(
                f"{base_name}.attn_weights.{i}",
                [scaled_q_proj, k_proj],
                "Einsum",
                equation="nchw,nchq->nqhw",
            )

            softmax = self._onnx_builder.build_op(
                f"{base_name}.softmax.{i}", [attn_weights], "Softmax", axis=1
            )

            attn_outputs.append(
                self._onnx_builder.build_op(
                    f"{base_name}.attn_output.{i}",
                    [softmax, v_proj],
                    "Einsum",
                    equation="nchw,nqhc->nqhw",
                )
            )
        return self._onnx_builder.build_merge_heads_and_matmul(
            f"{base_name}.out_proj", attn_outputs, num_heads
        )

    def _build_encoder_mlp(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        fc1 = self._onnx_builder.build_conv(f"{base_name}.fc1", input_nodes[0])
        act = self._onnx_builder.build_activation(
            f"{base_name}.act", fc1, self.cfg.vm_cfg.hidden_act
        )
        fc2 = self._onnx_builder.build_conv(f"{base_name}.fc2", act)
        return fc2

    def _build_pixel_unshuffle(self, base_name: str, input_node: OnnxNode, factor: int) -> OnnxNode:
        """
        Builds nodes for a pixel unshuffle operation (SpaceToDepth).
        Transforms a tensor of shape (N, C, H, W) to (N, C * factor**2, H // factor, W // factor).
        """
        space_to_depth = self._onnx_builder.build_op(
            f"{base_name}.space_to_depth", [input_node], "SpaceToDepth", blocksize=factor
        )
        return space_to_depth

    def _build_mm_projector(self, base_name: str, input_nodes: list[OnnxNode]) -> OnnxNode:
        # input_nodes[0] is (1, C, 1, SeqLen) [NCHW]
        match self.cfg.model_type:
            case VlmArchType.VLM_LFM2_VL:
                if isinstance(self.cfg.vm_cfg.num_patches, list):
                    num_patches_h = self.cfg.vm_cfg.num_patches[0]
                else:
                    num_patches_h = self.cfg.vm_cfg.num_patches

                reshaped_input = self._onnx_builder.build_split_and_concat(
                    f"{base_name}.reshape1", input_nodes[0], num_patches_h, 3, 2
                )
                factor = self.cfg.mm_cfg.downsample_factor
                unshuffled_nchw = self._build_pixel_unshuffle(
                    f"{base_name}.pixel_unshuffle", reshaped_input, factor
                )

                projector_input = unshuffled_nchw
                if self.cfg.mm_cfg.projector_use_layernorm:
                    projector_input = self._onnx_builder.build_layer_norm(
                        f"{base_name}.layer_norm", projector_input, self.cfg.vm_cfg.layer_norm_eps
                    )

                fc1 = self._onnx_builder.build_conv(f"{base_name}.linear_1", projector_input)
                act = self._onnx_builder.build_activation(
                    f"{base_name}.act", fc1, self.cfg.mm_cfg.hidden_act
                )
                last = self._onnx_builder.build_conv(f"{base_name}.linear_2", act)
            case VlmArchType.VLM_LLAVA:
                fc1 = self._onnx_builder.build_conv(f"{base_name}.linear_1", input_nodes[0])
                act = self._onnx_builder.build_activation(
                    f"{base_name}.act", fc1, self.cfg.mm_cfg.hidden_act
                )
                last = self._onnx_builder.build_conv(f"{base_name}.linear_2", act)
            case VlmArchType.VLM_GEMMA3:
                reshape1 = self._onnx_builder.build_split_and_concat(
                    f"{base_name}.reshape1", input_nodes[0], self.cfg.vm_cfg.num_patches, 3, 2
                )
                tokens_per_side = int(self.cfg.mm_cfg.mm_tokens_per_image**0.5)
                kernel_shape = [self.cfg.vm_cfg.num_patches // tokens_per_side] * 2
                avgpool = self._onnx_builder.build_op(
                    f"{base_name}.avgpool",
                    [reshape1],
                    "AveragePool",
                    kernel_shape=kernel_shape,
                    strides=kernel_shape,
                )
                norm = self._onnx_builder.build_rms_norm(
                    f"{base_name}.mm_soft_emb_norm", avgpool, self.cfg.vm_cfg.layer_norm_eps, 1.0
                )
                last = self._onnx_builder.build_conv(
                    f"{base_name}.proj",
                    norm,
                    reshape_str="cn->nchw",
                    src_weight_name="multi_modal_projector.mm_input_projection_weight",
                )
            case VlmArchType.VLM_PALIGEMMA:
                last = self._onnx_builder.build_conv(f"{base_name}.linear", input_nodes[0])
            case _:
                raise ValueError(
                    f"Multi-modal projection for {self.cfg.model_type} is not supported."
                )
        return last

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool
    ):
        g = self._build_sima_nodes(self.hf_model.vision_model_param_base_name, quantizable)
        save_awesomenet(g, self.model_name + (".fp32" if quantizable else ""), str(self.sima_model_sdk_path))

    def _build_sima_nodes(self, base_name: str, quantizable: bool):
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

        builder = SimaBuilder(Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target)
        model_input = builder.create_placeholder_node("input", TensorType(activation_type(quantizable), input_shape))

        # MLA subgraph inputs are the same as the model inputs, except the node names are different
        builder.begin_subnet([model_input])
        mla_input = builder.create_placeholder_node("MLA_0/input", TensorType(activation_type(quantizable), input_shape))

        # Vision tower.
        vision_output = self._build_sima_vision_tower(
            builder, self.hf_model.vision_model_param_base_name, mla_input, quantizable
        )

        # MM projection.
        if self.include_mm_proj:
            if self.cfg.model_type == VlmArchType.VLM_LLAVA:
                llava_o_shape = get_expected_tensor_value(vision_output.get_type().output).shape
                vision_output = builder.create_slice_node(
                    vision_output,
                    begin=[0, 0, 1, 0],
                    end=list(llava_o_shape),
                    stride=[1, 1, 1, 1],
                    axis=[0, 1, 2, 3]
                )
            if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
                projector_base_name = "model.multi_modal_projector"
            else:
                projector_base_name = "multi_modal_projector"
            _ = self._build_sima_mm_projector(builder, projector_base_name, vision_output, quantizable)

        mla_node = builder.finish_subnet("MLA_0")

        # Ensure that output type is float32
        if activation_type(quantizable) != ScalarType.float32:
            _ = builder.create_cast_node(mla_node, ScalarType.float32)
        net = builder.finish(self.model_name)
        return net

    def _build_sima_vision_tower(self, builder, base_name: str, input_node: NodeOrHandle, quantizable: bool) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        if self.include_embeddings:
            embeddings = self._build_sima_patch_embeddings(builder, f"{base_name}.embeddings", input_node, quantizable)

            if self.cfg.vm_cfg.arch == VisionArchType.CLIP:
                # Note that the original source code has a typo in the layer norm node name.
                encoder_input = build_two_stage_layer_norm(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.pre_layrnorm", embeddings,
                    -1, epsilon
                )
            else:
                encoder_input = embeddings
        else:
            encoder_input = input_node

        if self.num_layers > 1:
            for layer_idx in range(self.num_layers):
                encoder_input = self._build_sima_encoder(
                    builder, f"{base_name}.encoder.layers.{layer_idx}", encoder_input, quantizable
                )
            encoder_output = encoder_input
        else:
            encoder_output = self._build_sima_encoder(
                builder, f"{base_name}.encoder.layers.{self.layer_idx}", encoder_input, quantizable
            )

        if not self.include_mm_proj:
            return encoder_output

        post_layer_norm = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.post_layernorm", encoder_output,
            -1, epsilon
        )
        return post_layer_norm

    def _build_sima_patch_embeddings(self, builder, base_name: str, input_node: NodeOrHandle, quantizable: bool) -> NodeOrHandle:
        node_name = f"{base_name}.patch_embedding"

        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            # LFM2: input is already pre-patchified (1, 1, seq_len, patch_dim) — FC projection only.
            embeddings = build_conv(
                builder, self.get_hf_param, self.check_hf_param, node_name, input_node,
                is_fc=True
            )
        else:
            if isinstance(self.cfg.vm_cfg.image_size, list):
                image_h = self.cfg.vm_cfg.image_size[0]
            else:
                image_h = self.cfg.vm_cfg.image_size
            patch_embedding = build_conv(
                builder, self.get_hf_param, self.check_hf_param, node_name, input_node,
                is_fc=False, stride=(self.cfg.vm_cfg.patch_size,) * 2
            )
            # NHWC layout: split on axis H, concat on axis W → (1, 1, seq_len, hidden)
            split_and_concat = builder.create_slice_concat_node(
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
                ).astype(activation_dtype(quantizable))
                class_embedding = builder.create_constant_node(class_embedding_weight)
                embeddings = builder.create_concat_node([class_embedding, split_and_concat], axis=2)
            else:
                embeddings = split_and_concat

        # Position embedding — LFM2 may need bilinear resize if resolution differs from pretraining.
        position_embedding_weight = self.get_hf_param(f"{base_name}.position_embedding.weight")
        if self.cfg.model_type == VlmArchType.VLM_LFM2_VL:
            original_grid_size = int(position_embedding_weight.shape[0] ** 0.5)
            if isinstance(self.cfg.vm_cfg.num_patches, list):
                target_h, target_w = self.cfg.vm_cfg.num_patches
            else:
                target_h = target_w = self.cfg.vm_cfg.num_patches
            if target_h != original_grid_size or target_w != original_grid_size:
                pos_grid = position_embedding_weight.reshape(
                    original_grid_size, original_grid_size, self.cfg.vm_cfg.hidden_size
                )
                zoomed = zoom(pos_grid.astype(np.float32),
                              (target_h / original_grid_size, target_w / original_grid_size, 1.0),
                              order=1)
                position_embedding_weight = zoomed.reshape(target_h * target_w, self.cfg.vm_cfg.hidden_size)

        # Reshape "wc->nhwc" and cast to the activation dtype (float32 in RELAY, bfloat16 in SIMA_QUANTIZED).
        pos_weight = position_embedding_weight.astype(activation_dtype(quantizable))
        pos_weight = pos_weight.reshape(1, 1, pos_weight.shape[0], pos_weight.shape[1])
        position_embedding = builder.create_constant_node(pos_weight)
        embeddings = builder.create_add_node(embeddings, position_embedding)
        return embeddings

    def _build_sima_encoder(self, builder, base_name: str, input_node: NodeOrHandle, quantizable: bool) -> NodeOrHandle:
        epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
        layer_norm1 = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.layer_norm1", input_node,
            -1, epsilon
        )
        self_attn = self._build_sima_encoder_attention(builder, f"{base_name}.self_attn", layer_norm1)
        add1 = builder.create_add_node(input_node, self_attn)
        layer_norm2 = build_two_stage_layer_norm(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.layer_norm2", add1,
            -1, epsilon
        )
        mlp = self._build_sima_encoder_mlp(builder, f"{base_name}.mlp", layer_norm2, quantizable)
        add2 = builder.create_add_node(add1, mlp)
        return add2

    def _build_sima_encoder_attention(self, builder, base_name: str, input_node: NodeOrHandle) -> NodeOrHandle:
        num_heads = self.cfg.vm_cfg.num_attention_heads
        head_dim = self.cfg.vm_cfg.hidden_size // num_heads

        scaled_q_projs = build_matmul_and_split_heads(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.q_proj", input_node, num_heads, self.cfg.vm_cfg.seq_len,
            post_matmul_scale=head_dim ** -0.5
        )
        k_projs = build_matmul_and_split_heads(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.k_proj", input_node, num_heads, self.cfg.vm_cfg.seq_len
        )
        v_projs = build_matmul_and_split_heads(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.v_proj", input_node, num_heads, self.cfg.vm_cfg.seq_len
        )

        attn_outputs = list()
        for i, scaled_q_proj, k_proj, v_proj in zip(
            range(num_heads), scaled_q_projs, k_projs, v_projs
        ):
            attn_weights = builder.create_einsum_node(
                scaled_q_proj, k_proj, equation="nhwc,nhqc->nhwq", layout="NHWC",
            )

            softmax = builder.create_softmax_node(attn_weights, axis=3)

            attn_outputs.append(
                builder.create_einsum_node(
                    softmax, v_proj, equation="nhwc,nhcq->nhwq", layout="NHWC",
                )
            )
        return build_merge_heads_and_matmul(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.out_proj", attn_outputs, num_heads
        )

    def _build_sima_encoder_mlp(self, builder, base_name: str, input_node: NodeOrHandle, quantizable: bool) -> NodeOrHandle:
        fc1 = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.fc1", input_node
        )
        act = build_activation(
            builder, fc1, self.cfg.vm_cfg.hidden_act, quantizable
        )
        fc2 = build_conv(
            builder, self.get_hf_param, self.check_hf_param,
            f"{base_name}.fc2", act)
        return fc2

    def _build_sima_rms_norm_with_offset(
        self, builder, base_name: str, input_node: NodeOrHandle, epsilon: float, weight_offset: float
    ) -> NodeOrHandle:
        weight_tensor = load_tensor_from_source(f"{base_name}.weight", self.get_hf_param, self.check_hf_param)
        weight_tensor += weight_offset
        return builder.create_rms_norm_node(input_node, epsilon, weight_tensor)

    def _build_sima_mm_projector(self, builder, base_name: str, input_node: NodeOrHandle, quantizable: bool) -> NodeOrHandle:
        match self.cfg.model_type:
            case VlmArchType.VLM_LFM2_VL:
                # NHWC: (1, 1, seq_len, hidden) → (1, num_patches_h, num_patches_w, hidden)
                if isinstance(self.cfg.vm_cfg.num_patches, list):
                    num_patches_h = self.cfg.vm_cfg.num_patches[0]
                else:
                    num_patches_h = self.cfg.vm_cfg.num_patches
                reshaped = builder.create_slice_concat_node(
                    input_node, axis=1,
                    split_axis=2,
                    split_block=num_patches_h,
                    split_repeat=1
                )
                factor = self.cfg.mm_cfg.downsample_factor
                unshuffled = build_space_to_depth(builder, reshaped, factor)
                projector_input = unshuffled
                if self.cfg.mm_cfg.projector_use_layernorm:
                    epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
                    projector_input = build_two_stage_layer_norm(
                        builder, self.get_hf_param, self.check_hf_param,
                        f"{base_name}.layer_norm", projector_input, -1, epsilon
                    )
                fc1 = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.linear_1", projector_input
                )
                act = build_activation(builder, fc1, self.cfg.mm_cfg.hidden_act, quantizable)
                last = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.linear_2", act
                )
            case VlmArchType.VLM_LLAVA:
                fc1 = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.linear_1", input_node
                )
                act = build_activation(
                    builder, fc1, self.cfg.mm_cfg.hidden_act, quantizable
                )
                last = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.linear_2", act
                )
            case VlmArchType.VLM_GEMMA3:
                # NHWC layout, split on axis W, concat on axis H
                reshape1 = builder.create_slice_concat_node(
                    input_node, axis=1,
                    split_axis=2,
                    split_block=self.cfg.vm_cfg.num_patches,
                    split_repeat=1
                )
                tokens_per_side = int(self.cfg.mm_cfg.mm_tokens_per_image ** 0.5)
                kernel_shape = tuple([self.cfg.vm_cfg.num_patches // tokens_per_side] * 2)
                avgpool = builder.create_avgpool2d_node(
                    reshape1, kernel_shape=kernel_shape, strides=kernel_shape
                )
                epsilon = float(np.float32(self.cfg.vm_cfg.layer_norm_eps))
                norm = self._build_sima_rms_norm_with_offset(
                    builder, f"{base_name}.mm_soft_emb_norm", avgpool, epsilon, 1.0
                )
                last = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.proj", norm, reshape_str="io->oihw",
                    src_weight_name="multi_modal_projector.mm_input_projection_weight"
                )
            case VlmArchType.VLM_PALIGEMMA:
                last = build_conv(
                    builder, self.get_hf_param, self.check_hf_param,
                    f"{base_name}.linear", input_node
                )
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
