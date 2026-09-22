from dataclasses import dataclass

import numpy as np

from afe.apis.defines import TensorDRAMLayout, gen2_target
from afe.backends.backends import Backend
from afe.ir.defines import Status
from afe.ir.serializer import save_awesomenet
from afe.ir.tensor_type import ScalarType, TensorType

from sima_lmm.model.base import (
    BaseModel,
    LayerConfiguration,
    TensorTessellateParameters,
)
from sima_lmm.model.sima_builder import SimaBuilder, activation_type


@dataclass
class LanguageDFlashStateResolverModel(BaseModel):
    """Resolve any accepted Gated DeltaNet prefix from grouped intermediates."""

    block_size: int

    def __post_init__(self):
        assert self.cfg.lm_cfg.linear_attn_cfg is not None
        assert self.block_size in (4, 8, 16)

    @property
    def enable_filter_sharing(self) -> bool:
        return self.cfg.pipeline_cfg.enable_filter_sharing

    def gen_onnx_files(self):
        linear_cfg = self.cfg.lm_cfg.linear_attn_cfg
        heads = linear_cfg.num_value_heads
        key_dim = linear_cfg.key_head_dim
        value_dim = linear_cfg.value_head_dim

        self.create_onnx_builder()
        state = self._onnx_builder.create_input_node(
            "linear_delta_state_s1", (1, value_dim, heads, key_dim)
        )
        key = self._onnx_builder.create_input_node(
            "key", (1, key_dim, heads, self.block_size)
        )
        value = self._onnx_builder.create_input_node(
            "v_new", (1, value_dim, heads, self.block_size)
        )
        decay_row = self._onnx_builder.create_input_node(
            "decay_row", (1, self.block_size, heads, 1)
        )
        decay_row = self._onnx_builder.build_op(
            "resolver.decay_row.token_major",
            [decay_row],
            "Transpose",
            perm=[0, 3, 2, 1],
        )

        base_decay = self._onnx_builder.build_op(
            "resolver.base_decay",
            [
                decay_row,
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )
        state_base = self._onnx_builder.build_op(
            "resolver.state_base", [state, base_decay], "Mul"
        )
        update_tail = self._onnx_builder.build_op(
            "resolver.update_tail",
            [
                decay_row,
                np.array([1], dtype=np.int64),
                np.array([self.block_size], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ],
            "Slice",
        )
        zero = self._onnx_builder.build_op(
            "resolver.zero", [base_decay, base_decay], "Sub"
        )
        update_weights = self._onnx_builder.build_op(
            "resolver.update_weights", [zero, update_tail], "Concat", axis=3
        )
        weighted_value = self._onnx_builder.build_op(
            "resolver.weighted_value", [value, update_weights], "Mul"
        )
        weighted_value = self._onnx_builder.build_op(
            "resolver.weighted_value.token_major",
            [weighted_value],
            "Transpose",
            perm=[0, 3, 2, 1],
        )
        state_update = self._onnx_builder.build_op(
            "resolver.state_update",
            [weighted_value, key],
            "Einsum",
            equation="nchw,nqhc->nwhq",
        )
        state_out = self._onnx_builder.build_op(
            "resolver.state", [state_base, state_update], "Add"
        )
        self._onnx_builder.create_output_node(
            self._onnx_builder.get_node_output_name(state_out),
            (1, value_dim, heads, key_dim),
        )
        self._onnx_builder.create_and_save_model()
        self._onnx_builder = None

    def gen_model_sdk_files_directly(
        self,
        layer_cfg: LayerConfiguration,
        log_level: int,
        quantizable: bool,
    ):
        graph = self._build_sima_nodes(quantizable)
        save_awesomenet(
            graph,
            self.model_name + (".fp32" if quantizable else ""),
            str(self.sima_model_sdk_path),
        )

    def _build_sima_nodes(self, quantizable: bool):
        linear_cfg = self.cfg.lm_cfg.linear_attn_cfg
        heads = linear_cfg.num_value_heads
        key_dim = linear_cfg.key_head_dim
        value_dim = linear_cfg.value_head_dim
        dtype = activation_type(quantizable)

        input_specs = (
            ("linear_delta_state_s1", (1, heads, key_dim, value_dim)),
            ("key", (1, heads, self.block_size, key_dim)),
            ("v_new", (1, heads, self.block_size, value_dim)),
            ("decay_row", (1, heads, 1, self.block_size)),
        )
        builder = SimaBuilder(
            Status.RELAY if quantizable else Status.SIMA_QUANTIZED, gen2_target
        )
        model_inputs = [
            builder.create_placeholder_node(name, TensorType(dtype, shape))
            for name, shape in input_specs
        ]
        builder.begin_subnet(model_inputs)
        state, key, value, decay_row = [
            builder.create_placeholder_node(name, TensorType(dtype, shape))
            for name, shape in input_specs
        ]

        base_decay = builder.create_slice_node(decay_row, [0], [1], [1], [3])
        state_base = builder.create_mul_node(state, base_decay)
        update_tail = builder.create_slice_node(
            decay_row, [1], [self.block_size], [1], [3]
        )
        zero = builder.create_subtract_node(base_decay, base_decay)
        update_weights = builder.create_concat_node([zero, update_tail], 3)
        update_weights = builder.create_transpose_node(update_weights, [0, 1, 3, 2])
        weighted_value = builder.create_mul_node(value, update_weights)
        state_update = builder.create_einsum_node(
            key,
            weighted_value,
            equation="nhcw,nhcq->nhwq",
            layout="NHWC",
        )
        builder.create_add_node(state_base, state_update)

        mla_node = builder.finish_subnet("MLA_0")
        if dtype != ScalarType.float32:
            builder.create_cast_node(mla_node, ScalarType.float32, backend=Backend.EV)
        return builder.finish(self.model_name)

    def get_mla_input_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        linear_cfg = self.cfg.lm_cfg.linear_attn_cfg
        return {
            3: TensorTessellateParameters(
                tile_shape=(0, 0, 0, 0),
                enable_mla=True,
                dram_layout=TensorDRAMLayout.HWC16,
                dram_shape=(
                    1,
                    linear_cfg.num_value_heads,
                    self.block_size,
                    self.block_size,
                ),
            )
        }

    def get_mla_output_tessellate_params(self) -> dict[int, TensorTessellateParameters]:
        return {}
