import numpy as np

from dataclasses import dataclass
from typing import ClassVar

from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.model.base import BaseModel, LayerConfiguration
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.onnx_builder import OnnxNode


@dataclass
class WhisperDecoderInitModel(BaseModel):
    """Implementation Whisper's decoder layers to process the input tokens and encoder outputs.

    Attributes:
        layer_idx: Transformer layer index.
        enable_log_probe: Whether to also output filtered logits from the final layer.
    """
    layer_idx: int
    enable_log_probe: bool = False
    # 4 init tokens used in the init model:
    #   1. <|startoftranscript|>,
    #   2. `language`,
    #   3. <|transcribe|>
    #   4. <|notimestamps|>
    num_tokens: ClassVar[int] = 4
    token_idx: ClassVar[int] = 0

    def __post_init__(self):
        assert 0 <= self.layer_idx < self.cfg.decoder_layers

    @property
    def enable_filter_sharing(self) -> bool:
        return self.use_filter_sharing

    def gen_model_sdk_files_directly(
        self, layer_cfg: LayerConfiguration, log_level: int, quantizable: bool
    ):
        shapes = {
            "input": (1, 1, self.num_tokens, self.cfg.d_model),
            "audio_features": (1, 1, self.cfg.max_source_positions, self.cfg.d_model),
        }

        graph = ModelGraph(self, shapes, quantizable)
        outputs = self._build_sima_nodes(graph, list(graph.inputs.values()))
        graph.save(outputs)

    def _build_sima_nodes(self, graph, inputs):
        pre = WhisperDecoderPreModel(
            self.cfg,
            self.model_name,
            hf_model=self.hf_model,
            num_tokens=self.num_tokens,
            layer_idx=self.layer_idx,
        )
        final_layer = self.layer_idx == self.cfg.decoder_layers - 1
        num_tokens = 1 if final_layer else self.num_tokens
        cache = WhisperDecoderCacheModel(
            self.cfg,
            self.model_name,
            num_tokens=num_tokens,
            token_idx=self.num_tokens - num_tokens,
            use_future_token_mask=False,
        )
        post = WhisperDecoderPostModel(
            self.cfg,
            self.model_name,
            hf_model=self.hf_model,
            num_tokens=num_tokens,
            layer_idx=self.layer_idx,
            skip_encoder_kv_proj=False,
            output_encoder_kv_cache=True,
            enable_log_probe=self.enable_log_probe,
        )
        pre_inputs = [inputs[0]]
        if self.layer_idx == 0:
            positions = graph.parameter("model.decoder.embed_positions.weight")[: self.num_tokens]
            pre_inputs.append(
                graph.constant(positions.reshape(1, 1, self.num_tokens, self.cfg.d_model))
            )
        pre_outputs = pre._build_sima_nodes(graph, pre_inputs)
        query, key, value = pre_outputs[:3]
        residual = pre_outputs[pre.positioned_residual_output_idx] if self.layer_idx == 0 else inputs[0]
        if final_layer:
            query = graph.slice(query, [self.num_tokens - 1], [self.num_tokens], [1], [2])
            residual = graph.slice(
                residual, [self.num_tokens - 1], [self.num_tokens], [1], [2]
            )
        attn = cache._build_sima_nodes(graph, [query, key, value])[0]
        outputs = post._build_sima_nodes(graph, [residual, attn, inputs[1]])
        # Hidden/token, optional logits, self K/V, encoder K/V.
        return [*outputs[:-2], key, value, *outputs[-2:]]

    def gen_onnx_files(self):
        base_name = f"model.decoder.layers.{self.layer_idx}"
        self.create_onnx_builder()
        self._onnx_builder.create_input_node("input", (1, self.cfg.d_model, 1, self.num_tokens))
        self._onnx_builder.create_input_node(
            "audio_features", (1, self.cfg.d_model, 1, self.cfg.max_source_positions)
        )
        output_nodes = self._build_onnx_nodes(base_name, self._onnx_builder.input_nodes)

        if self.layer_idx < self.cfg.decoder_layers - 1:
            # Decoder layer output.
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[0]),
                (1, self.cfg.d_model, 1, self.num_tokens)
            )
        else:
            # Argmax output.
            self._onnx_builder.create_output_node(
                self._onnx_builder.get_node_output_name(output_nodes[0]), (1, 1, 1, 1), np.int64
            )
            if self.enable_log_probe:
                self._onnx_builder.create_output_node(
                    self._onnx_builder.get_node_output_name(output_nodes[1]),
                    (1, self.cfg.vocab_size, 1, 1)
                )
        cache_output_idx = 2 if self.layer_idx == self.cfg.decoder_layers - 1 and self.enable_log_probe else 1
        # Self-attention key projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[cache_output_idx]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        # Self-attention value projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[cache_output_idx + 1]),
            (1, self.cfg.d_model, 1, self.num_tokens)
        )
        # Cross-attention key projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[cache_output_idx + 2]),
            (
                1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions
            )
        )
        # Cross-attention value projections.
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(output_nodes[cache_output_idx + 3]),
            (
                1, self.cfg.decoder_head_dim, self.cfg.decoder_attention_heads,
                self.cfg.max_source_positions
            )
        )
        self._onnx_builder.create_and_save_model()

        # Set to None to deallocate the memory.
        self._onnx_builder = None

    def _build_onnx_nodes(self, base_name: str, input_nodes: list[OnnxNode]) -> list[OnnxNode]:
        assert self.token_idx == 0
        pre_model = WhisperDecoderPreModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=self.num_tokens, layer_idx=self.layer_idx
        )
        pre_model._onnx_builder = self._onnx_builder
        if self.layer_idx < self.cfg.decoder_layers - 1:
            num_tokens = self.num_tokens
            token_idx = self.token_idx
        else:
            num_tokens = 1
            token_idx = self.token_idx + self.num_tokens - 1
        cache_model = WhisperDecoderCacheModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=num_tokens, token_idx=token_idx,
            use_future_token_mask=False
        )
        cache_model._onnx_builder = self._onnx_builder
        post_model = WhisperDecoderPostModel(
            self.cfg, self.model_name, onnx_path=self.onnx_path, sima_path=self.sima_path,
            hf_model=self.hf_model, num_tokens=num_tokens, layer_idx=self.layer_idx,
            skip_encoder_kv_proj=False, output_encoder_kv_cache=True,
            enable_log_probe=self.enable_log_probe
        )
        post_model._onnx_builder = self._onnx_builder

        if self.layer_idx == 0:
            pre_input_nodes = [input_nodes[0], self._build_position_embeddings()]
        else:
            pre_input_nodes = [input_nodes[0]]
        pre_output_nodes = pre_model._build_onnx_nodes(base_name, pre_input_nodes)
        residual_input_node = (
            pre_output_nodes[WhisperDecoderPreModel.positioned_residual_output_idx]
            if self.layer_idx == 0
            else pre_input_nodes[0]
        )

        if self.layer_idx < self.cfg.decoder_layers - 1:
            cache_input_nodes = pre_output_nodes
        else:
            slice_begin = self.token_idx + self.num_tokens - 1
            slice_end = slice_begin + 1
            slice_axis = 3
            last_token_q_proj = self._onnx_builder.build_op(
                f"{base_name}.last_token.slice_q_proj",
                [
                    pre_output_nodes[0],
                    np.array([slice_begin], dtype=np.int32),
                    np.array([slice_end], dtype=np.int32),
                    np.array([slice_axis], dtype=np.int32),
                ],
                "Slice"
            )
            cache_input_nodes = [last_token_q_proj, pre_output_nodes[1], pre_output_nodes[2]]
        cache_output_nodes = cache_model._build_onnx_nodes(base_name, cache_input_nodes)

        if self.layer_idx < self.cfg.decoder_layers - 1:
            post_input_nodes = [residual_input_node, cache_output_nodes[0], input_nodes[-1]]
        else:
            slice_begin = self.token_idx + self.num_tokens - 1
            slice_end = slice_begin + 1
            slice_axis = 3
            last_token_residual = self._onnx_builder.build_op(
                f"{base_name}.last_token.slice_residual",
                [
                    residual_input_node,
                    np.array([slice_begin], dtype=np.int32),
                    np.array([slice_end], dtype=np.int32),
                    np.array([slice_axis], dtype=np.int32),
                ],
                "Slice"
            )
            post_input_nodes = [last_token_residual, cache_output_nodes[0], input_nodes[-1]]
        post_output_nodes = post_model._build_onnx_nodes(base_name, post_input_nodes)
        output_nodes = [
            # Decoder layer output or argmax output.
            post_output_nodes[0],
        ]
        if self.layer_idx == self.cfg.decoder_layers - 1 and self.enable_log_probe:
            output_nodes.append(post_output_nodes[1])
            encoder_kv_output_idx = 2
        else:
            encoder_kv_output_idx = 1
        output_nodes.extend([
            # Self-attention key projections.
            pre_output_nodes[1],
            # Self-attention value projections.
            pre_output_nodes[2],
            # Cross-attention key projections.
            post_output_nodes[encoder_kv_output_idx],
            # Cross-attention value projections.
            post_output_nodes[encoder_kv_output_idx + 1],
        ])
        return output_nodes

    def _build_position_embeddings(self) -> OnnxNode:
        position_embeddings = self.get_hf_param("model.decoder.embed_positions.weight")
        assert isinstance(position_embeddings, np.ndarray)
        position_embeddings = position_embeddings[:self.num_tokens]
        position_embeddings = position_embeddings.T.reshape(1, self.cfg.d_model, 1, self.num_tokens)
        return self._onnx_builder.create_initializer(
            "model.decoder.init_embed_positions", position_embeddings
        )
