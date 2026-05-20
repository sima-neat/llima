import numpy as np

from functools import cached_property
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader, ReaderField, ReaderTensor, dequantize
from llama_cpp import Llama
from ml_dtypes import bfloat16

import sima_lmm.gguf.ggml_quant as _ggml_quant


GGUF_CONFIG_MAP : dict = {
    "general": {
        "architecture": "model_type",
        "file_type": "data_type",
    },
    "llm" : {
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",

        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "head_dim",
        "attention.value_length": "head_dim",
        "attention.key_length_swa": "sliding_head_dim",
        "attention.value_length_swa": "sliding_head_dim",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "attention.sliding_window": "sliding_window",
        "attention.shared_kv_layers": "num_kv_shared_layers",
        "attention.sliding_window_pattern": "sliding_window_pattern",

        "rope.freq_base": "rope_theta",
        "rope.freq_base_swa": "rope_local_base_freq",
        "rope.dimension_count": "rope_dimension_count",
        "rope.dimension_count_swa": "sliding_rope_dimension_count",
        "rope.scaling.type": "rope_type",
        "rope.scaling.factor": "factor",
        "rope.scaling.original_context_length": "original_max_position_embeddings",
        "shortconv.l_cache": "conv_L_cache",
        "final_logit_softcapping": "final_logit_softcapping",
        "embedding_length_per_layer_input": "hidden_size_per_layer_input",
    },
    "tokenizer": {
        "ggml.model": "tokenizer_type",
    },
}


DEFAULT_HF_GGUF_WEIGHT_MAP : dict = {
    # HF name : GGUF name
    "model.embed_tokens.weight": "token_embd.weight",
    "model.embed_tokens_per_layer.weight": "per_layer_token_embd.weight",
    "model.per_layer_model_projection.weight": "per_layer_model_proj.weight",
    "model.per_layer_projection_norm.weight": "per_layer_proj_norm.weight",
    "model.norm.weight" : "output_norm.weight",
    "model.embedding_norm.weight" : "token_embd_norm.weight",
    "lm_head.weight": "output.weight",
    "model.layers.": "blk.",
    ".input_layernorm.weight": ".attn_norm.weight",
    ".operator_norm.weight": ".attn_norm.weight",
    ".self_attn.q_proj.weight": ".attn_q.weight",
    ".self_attn.q_norm.weight": ".attn_q_norm.weight",
    ".self_attn.q_layernorm.weight": ".attn_q_norm.weight",
    ".self_attn.k_proj.weight": ".attn_k.weight",
    ".self_attn.k_layernorm.weight": ".attn_k_norm.weight",
    ".self_attn.k_norm.weight": ".attn_k_norm.weight",
    ".self_attn.v_proj.weight": ".attn_v.weight",
    ".self_attn.qkv_proj.weight": ".attn_qkv.weight",
    ".self_attn.o_proj.weight": ".attn_output.weight",
    ".self_attn.out_proj.weight": ".attn_output.weight",
    ".post_attention_layernorm.weight": ".post_attention_norm.weight",
    ".per_layer_input_gate.weight": ".inp_gate.weight",
    ".per_layer_projection.weight": ".proj.weight",
    ".post_per_layer_input_norm.weight": ".post_norm.weight",
    ".layer_scalar": ".layer_output_scale.weight",
    ".mlp.down_proj.weight": ".ffn_down.weight",
    ".mlp.gate_proj.weight": ".ffn_gate.weight",
    ".mlp.up_proj.weight": ".ffn_up.weight",
    ".mlp.gate_up_proj.weight": ".ffn_up.weight",
    ".feed_forward.w1.weight": ".ffn_gate.weight",
    ".feed_forward.w2.weight": ".ffn_down.weight",
    ".feed_forward.w3.weight": ".ffn_up.weight",
    ".ffn_norm.weight": ".ffn_norm.weight",
    ".pre_feedforward_layernorm.weight": ".ffn_norm.weight",
    ".post_feedforward_layernorm.weight": ".post_ffw_norm.weight",
    ".conv.in_proj.weight": ".shortconv.in_proj.weight",
    ".conv.conv.weight": ".shortconv.conv.weight",
    ".conv.out_proj.weight": ".shortconv.out_proj.weight",
}


def unpack_gguf_tensor(
        gguf_tensor: ReaderTensor, force_float: bool
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Unpack a tensor from a ReaderTensor structure.

    For MLA natively supported MLA types, returns the decoded tensor data.
    Tensor data will be converted to a supported encoding.
    If force_float flag is set to True, tensor data will be dequantized.

    Decoded tensor data is tuple containing either floating-point or
    block-quantized values.  If floating-point, the first tuple value is None
    and the second tuple value is the floating-point array.  If quantized,
    the first tuple value is an array of quantization scales of shape (N*M, 1)
    and the second tuple value is an integer array of quantized values of
    shape (N, M*B) where B is the quantization's block size.

    Args:
        gguf_tensor: A ReaderTensor data structure containing tensor information.
        force_float: Flag controlling whether the values will be converted to
            floating point values.

    Returns: A tuple of scales, values Numpy arrays.  For floating point values,
        scales is None.

    """
    scales = None

    match (gguf_tensor.tensor_type, force_float):
        # F32, F16 and BF16 are natively supported on MLA as bfloat16
        case (GGMLQuantizationType.F32, _):
            weights = np.asarray(gguf_tensor.data, dtype=np.float32)
        case (GGMLQuantizationType.F16, _):
            weights = np.asarray(gguf_tensor.data, dtype=np.float16)
        case (GGMLQuantizationType.BF16, _):
            weights = np.reshape(
                np.frombuffer(gguf_tensor.data.tobytes(), dtype=bfloat16),
                newshape=tuple(gguf_tensor.shape)
            )
        # Q8_0 and Q4_0 are natively supported on MLA
        case (GGMLQuantizationType.Q8_0, False):
            scales, weights = _ggml_quant.unpack_q8_0(gguf_tensor.data.tobytes())
        case (GGMLQuantizationType.Q4_0, False):
            scales, weights = _ggml_quant.unpack_q4(
                gguf_tensor.data.tobytes(), _ggml_quant.QuantizationSymmetry.SYMMETRIC
            )
        # Q5_0 is natively supported on MLA as Q8_0
        case (GGMLQuantizationType.Q5_0, False):
            scales, weights = _ggml_quant.unpack_q5(
                gguf_tensor.data.tobytes(), _ggml_quant.QuantizationSymmetry.SYMMETRIC
            )
            weights = weights.astype(dtype=np.int8)
        # Asymmetric block quantizations (Q4_1, etc.) are converted to Q8_0
        case (GGMLQuantizationType.Q4_1 | GGMLQuantizationType.Q5_1 |
              GGMLQuantizationType.Q8_1, False):
            scales, weights = _ggml_quant.unpack_base_quant(
                gguf_tensor.data.tobytes(),
                gguf_tensor.tensor_type.name
            )
            scales, weights = _ggml_quant.requantize_symmetric(scales, weights, 8, np.int8)

        # Superblock quantizations are converted to Q8_0 with block size 16 or 32
        case (GGMLQuantizationType.Q6_K | GGMLQuantizationType.Q5_K | GGMLQuantizationType.Q4_K |
              GGMLQuantizationType.Q3_K | GGMLQuantizationType.Q2_K, False):
            k_super, k_block, k_quant = _ggml_quant.unpack_k_quant(
                gguf_tensor.data.tobytes(),
                gguf_tensor.tensor_type.name
            )
            scales, weights = _ggml_quant.convert_to_simple_block_quant(k_super, k_block, k_quant)
        case _:
            weights = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)

    # Reverse weights to (rows, cols) order
    shape = gguf_tensor.shape[::-1]
    return scales, weights.reshape(shape)


class GgufReaderProxy:
    _gguf_reader: GGUFReader
    _tensor_map: dict[str, int]

    def __init__(self, path: Path | str):
        self._gguf_reader = GGUFReader(path)
        self._tensor_map = {x.name: idx for idx, x in enumerate(self._gguf_reader.tensors)}

    def load_tensor(
            self, tensor_name: str, force_float: bool
    ) -> tuple[np.ndarray | None, np.ndarray]:
        assert tensor_name in self._tensor_map, f"Tensor {tensor_name} not exist in tensor map."
        gguf_tensor = self._gguf_reader.get_tensor(self._tensor_map[tensor_name])
        return unpack_gguf_tensor(gguf_tensor, force_float)

    def get_fields(self) -> dict[str, ReaderField]:
        return dict(self._gguf_reader.fields.items())

    def get_tensor_info(self) -> dict:
        tensor_info = {}
        for tensor_name, idx in self._tensor_map.items():
            t = self._gguf_reader.get_tensor(idx)
            tensor_info[tensor_name] = {
                'gguf_type': t.tensor_type,
                'shape': tuple(t.shape[::-1])  # shape is converted to (rows, columns) order
            }
        return tensor_info


class GgufModel:
    """LLM model from a GGUF file.

    Attributes:
        file_path: The path of a GGUF model file.
        tensor_info: The dictionary of model weights information.  The entries are in format:
            weight_name: {"gguf_type": GGMLQuantizationType, "shape": tuple[int, ...]}
        gguf_proxy: An instance for accessing the GGUF data via GGUFReader interface.
        _hf_gguf_weight_map: Mapping from HF weight name to GGUF weight name.
        _need_permute_qk_proj: If set True, the q_proj and k_proj weights and biases are permuted to
            match the HuggingFace order.
    """
    file_path: Path
    tensor_info: dict[str, dict]
    gguf_proxy: GgufReaderProxy
    _hf_gguf_weight_map: dict[str, str]
    _need_permute_qk_proj: bool = False

    def __init__(self, gguf_path: str | Path):
        file_path = Path(gguf_path)
        assert file_path.is_file(), f"GGUF model file {gguf_path} not exist."

        self.file_path = file_path
        self.gguf_proxy = GgufReaderProxy(file_path)
        self.tensor_info = self.gguf_proxy.get_tensor_info()

        # Models like Phi merges the ffn_up and ffn_gate weights into ffn_up. Need to remove the
        # up_proj entry in the dictionary to avoid using the weights with doubled dimensions.
        if (
            "blk.0.ffn_up.weight" in self.tensor_info
            and "blk.0.ffn_gate.weight" not in self.tensor_info
        ):
            self._hf_gguf_weight_map = DEFAULT_HF_GGUF_WEIGHT_MAP.copy()
            del self._hf_gguf_weight_map[".mlp.up_proj.weight"]
        else:
            self._hf_gguf_weight_map = DEFAULT_HF_GGUF_WEIGHT_MAP

        # The GGUF stores the q_proj and k_proj weights/bias in different order for different
        # architecture.
        # For example, gemma, phi, qwen and lfm2 keep the weights in the same order as our
        # implementation. However, llama uses an interleaved order. Mistral is using the llama
        # architecture.
        # See the source code for full list.
        # https://github.com/ggml-org/llama.cpp/blob/230d1169e5bfe04a013b2e20f4662ee56c2454b0/src/llama-model.cpp#L7431
        model_config = self.model_config
        if model_config["model_type"] in ("llama", "mistral"):
            self._need_permute_qk_proj = True

    def __getstate__(self):
        """Get object state for the pickle protocol."""
        state = self.__dict__.copy()
        del state['gguf_proxy'] # GGUFReader can't be pickled
        return state

    def __setstate__(self, state: dict):
        """Set object state for the pickle protocol."""
        self.__dict__.update(state)

        # Reconstruct object that was not pickled
        self.gguf_proxy = GgufReaderProxy(self.file_path)

    def convert_hf_weight_name(self, name: str) -> str:
        for hf, gguf in self._hf_gguf_weight_map.items():
            if hf in name:
                name = name.replace(hf, gguf)
        return name

    @property
    def is_gguf(self) -> bool:
        return True

    def load_np_param(
        self, name: str, force_float: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        s, w = self.load_weight(name, is_hf_name=True, force_float=force_float)
        return w if s is None else (s, w)

    def load_weight(
            self, name: str, is_hf_name: bool = True, force_float: bool = False
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if is_hf_name:
            name = self.convert_hf_weight_name(name)
        scales, weights = self.gguf_proxy.load_tensor(name, force_float)
        if self._need_permute_qk_proj:
            if name.endswith(("attn_q.weight", "attn_q.bias")):
                scales, weights = self._permute_qk_proj(
                    scales, weights, self.model_config["num_attention_heads"]
                )
            elif name.endswith(("attn_k.weight", "attn_k.bias")):
                scales, weights = self._permute_qk_proj(
                    scales, weights, self.model_config["num_key_value_heads"]
                )
        return scales, weights

    def param_exists(self, hf_name: str) -> bool:
        gguf_name = self.convert_hf_weight_name(hf_name)
        return gguf_name in self.tensor_info

    def load_all_params(self):
        # No action needed
        pass

    def unload_all_params(self):
        # No action needed
        pass

    def _permute_qk_proj(
        self, scales: np.ndarray | None, weights: np.ndarray, num_heads: int
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if scales is not None:
            original_scales_shape = scales.shape
            scales = scales.reshape(*weights.shape[0:-1], -1)
            scales = (
                scales
                .reshape(num_heads, -1, 2, *scales.shape[1:])
                .swapaxes(1, 2)
                .reshape(original_scales_shape)
            )
        weights = (
            weights
            .reshape(num_heads, -1, 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )
        return scales, weights

    @property
    def language_model_param_base_name(self) -> str:
        # Assume "model" is the base name
        return "model"

    @cached_property
    def model_config(self) -> dict:
        """Convert GGUF model config to HF style.
        """
        # Retrieve parameters from the GGUF file's fields
        cfg = dict()
        for key, value in self.gguf_proxy.get_fields().items():
            section, param = key.split(".", maxsplit=1)
            mapped_key = None
            if section in GGUF_CONFIG_MAP and param in GGUF_CONFIG_MAP[section]:
                mapped_key = GGUF_CONFIG_MAP[section][param]
            elif param in GGUF_CONFIG_MAP["llm"]:
                mapped_key = GGUF_CONFIG_MAP["llm"][param]
            if mapped_key:
                # Handle special case for num_key_value_heads
                if mapped_key == "num_key_value_heads":
                    if isinstance(value.contents(), (list, np.ndarray)):
                        # If it's an array, take the max value (e.g., [0, 0, 8] -> 8)
                        cfg[mapped_key] = int(np.max(value.contents()))
                    else:
                        # Otherwise, just use the plain value
                        cfg[mapped_key] = value.contents()
                else:
                    # Assign all other mapped values directly
                    cfg[mapped_key] = value.contents()

        # Keep the base MLP width; shared Gemma4 layers widen it later.
        if isinstance(cfg.get("intermediate_size"), (list, np.ndarray)):
            intermediate_sizes = cfg["intermediate_size"]
            unique_intermediate_sizes = sorted(set(intermediate_sizes))
            cfg["intermediate_size"] = min(intermediate_sizes)
            if (len(unique_intermediate_sizes) == 2
                and unique_intermediate_sizes[1] == 2 * unique_intermediate_sizes[0]
            ):
                cfg["use_double_wide_mlp"] = True

        #configure sliding window attention configs
        if "sliding_window" in cfg:
            sliding_window = cfg["sliding_window"]
            if sliding_window is None:
                cfg["sliding_window"] = 0
                cfg["swa_enable"] = False
                swa_ratio = 0
            elif sliding_window > 0:
                cfg["swa_enable"] = True
                if "gemma3" in cfg["model_type"]:
                    swa_ratio = 5
                elif "phi3" in cfg["model_type"]:
                    # Phi3 has a sliding window that is much larger than the max_num_tokens
                    # used during compilation, hence disable the sliding window.
                    swa_ratio = 0
                    cfg["swa_enable"] = False
        else:
            cfg["sliding_window"] = 0
            cfg["swa_enable"] = False
            swa_ratio = 0

        # Extract layer_types based on the swa_ratio and tensor names
        layer_types = []
        if "num_hidden_layers" in cfg:
            # Get the set of all tensor names from the model
            tensor_names = self.tensor_info.keys()
            sliding_window_pattern = cfg.get("sliding_window_pattern")

            for layer_idx in range(cfg["num_hidden_layers"]):
                # Check for a tensor name that only exists in conv layers
                conv_tensor_name = f"blk.{layer_idx}.shortconv.conv.weight"
                if conv_tensor_name in tensor_names:
                    layer_types.append("conv")
                else:
                    if sliding_window_pattern is not None:
                        if sliding_window_pattern[layer_idx]:
                            layer_types.append("sliding_attention")
                        else:
                            layer_types.append("full_attention")
                    elif (layer_idx + 1) % (swa_ratio + 1) == 0:
                        layer_types.append("full_attention")
                    else:
                        layer_types.append("sliding_attention")

        cfg["layer_types"] = layer_types

        # Insert parameters that are not provided in the GGUF file
        assert "vocab_size" not in cfg
        _, token_embedding_tensor = self.load_weight("token_embd.weight", is_hf_name=False)
        cfg["vocab_size"] = token_embedding_tensor.shape[0]

        assert "model_type" in cfg
        hf_arch = cfg["model_type"] + "ForCausalLM"
        cfg["architectures"] = [hf_arch[0].upper() + hf_arch[1:]]
        cfg["tokenizer_type"] = "gguf"

        # GGUF file for Microsoft/Phi models stores longrope scaling factors in tensor section.
        if "phi" in cfg["model_type"]:
            _, rope_factors_long = self.load_weight("rope_factors_long.weight", is_hf_name=False)
            _, rope_factors_short = self.load_weight("rope_factors_short.weight", is_hf_name=False)
            cfg["rope_scaling"] = {
                "long_factor": rope_factors_long.tolist(),
                "short_factor": rope_factors_short.tolist(),
                "type": "longrope"
            }
        # For gemma3 models, setting rope_local_base_freq as per llama-cpp
        if "gemma3" == cfg["model_type"]:
            cfg["rope_local_base_freq"] = 10000
        elif "gemma4" == cfg["model_type"]:
            # GGUF stores the full-attention head width; normalize to the HF-style
            # proportional-RoPE config used by the rest of the codebase.
            cfg["rope_type"] = "proportional"
            if "rope_dimension_count" in cfg:
                cfg["rope_dimension_count"] //= 4
            
        # Llama 3.x GGUF files do not store rope scaling metadata; hardcode known values.
        # Llama 3.2 uses factor 32.0, Llama 3.1 uses factor 8.0.
        # Mistral also uses the llama architecture in GGUF; detect via general.name and restore
        # the HF model_type so downstream arch detection matches the HF golden configs.
        if cfg["model_type"] == "llama":
            fields = self.gguf_proxy.get_fields()
            name = fields["general.name"].contents() if "general.name" in fields else ""
            if "Mistral" in name:
                cfg["model_type"] = "mistral"
            basename = fields["general.basename"].contents() if "general.basename" in fields else ""
            if "Llama-3.1" in basename:
                cfg["rope_scaling"] = {
                    "rope_type": "llama3",
                    "factor": 8.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                }
            elif "Llama-3.2" in basename:
                cfg["rope_scaling"] = {
                    "rope_type": "llama3",
                    "factor": 32.0,
                    "low_freq_factor": 1.0,
                    "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 8192,
                }

        # Calculate head dim if not provided in the GGUF file.
        if "head_dim" not in cfg:
            assert "hidden_size" in cfg
            assert "num_attention_heads" in cfg
            assert cfg["hidden_size"] % cfg["num_attention_heads"] == 0
            cfg["head_dim"] = cfg["hidden_size"] // cfg["num_attention_heads"]
        return cfg

    @staticmethod
    def load_model(gguf_path: str, verbose: bool = False) -> Llama:
        return Llama(model_path=gguf_path, verbose=verbose)

    def execute_llama_cpp(
        self,
        query: str,
        max_tokens_response: int = 8192,
        device: str = 'cpu',
    ) -> dict:
        """
        Execute the GGUF model by llama-cpp.

        Args:

        Returns:
            A dict contaning prompt, input_ids, image, output tokens and text.
        """
        assert device == "cpu"

        llama = self.load_model(str(self.file_path))
        response = llama.create_completion(
            prompt=query,
            max_tokens=max_tokens_response,
        )

        res = None
        if response['choices']:
            res = response['choices'][0]['text']
        return res


if __name__ == "__main__":
    import sys
    gguf_file = sys.argv[1]

    model = GgufModel(gguf_file)
    cfg = model.model_config
    print(cfg)

    for tensor_name, tensor_info in model.tensor_info.items():
        shape = tensor_info["shape"]
        tensor_type = str(tensor_info["gguf_type"])
        print(f"{tensor_name}, shape = {shape}, weight type: {tensor_type}")

    # Run inference
    query = "Why is the sky blue?"
    out_text = model.execute_llama_cpp(query)
    print(f"User query: {query}")
    print(f"Model output: {out_text}")
