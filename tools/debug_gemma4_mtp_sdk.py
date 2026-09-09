#!/usr/bin/env python3
"""Compare Gemma4 MTP Hugging Face outputs with compiled Model SDK graphs.

The Hugging Face reference and AFE Model SDK usually live in different Python
environments.  This tool therefore has two subcommands:

* ``capture`` runs one greedy MTP round with Hugging Face and writes an NPZ.
* ``replay`` feeds that capture through the saved assistant ``.sima`` graphs.

The replay uses the target's compiled int8 embedding table and dynamically
quantizes the captured target K/V tensors with the same ml_kernels operation
used during LLiMa execution.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_PROMPT = (
    "Explain speculative decoding in plain language, then give three reasons "
    "why it can improve single-user inference latency."
)


def _as_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _accepted_prefix(drafts: np.ndarray, target_predictions: np.ndarray) -> int:
    accepted = 0
    for draft, target in zip(drafts, target_predictions):
        if int(draft) != int(target):
            break
        accepted += 1
    return accepted


def _format_tokens(tokens: np.ndarray | list[int]) -> str:
    return "[" + ", ".join(str(int(token)) for token in tokens) + "]"


def capture_hf(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_path = str(args.target_model.resolve())
    assistant_path = str(args.assistant_model.resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        target_path,
        fix_mistral_regex=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    ).to(args.device)

    print(f"Loading target from {target_path}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        device_map={"": args.device},
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    target.eval()

    with torch.inference_mode():
        target_output = target(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            return_shared_kv_states=True,
            return_dict=True,
        )

    prompt_tokens = int(inputs["input_ids"].shape[1])
    prompt_next_logits = target_output.logits[:, -1, :]
    current_token = int(prompt_next_logits.argmax(dim=-1).item())
    predecessor_hidden = target_output.hidden_states[-1][:, -1:, :]
    shared_kv_states = target_output.shared_kv_states
    if not shared_kv_states:
        raise RuntimeError("Target did not return Gemma4 shared K/V states")

    print(f"Loading assistant from {assistant_path}", flush=True)
    assistant = AutoModelForCausalLM.from_pretrained(
        assistant_path,
        device_map={"": args.device},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    assistant.eval()
    if assistant.config.model_type != "gemma4_assistant":
        raise ValueError(
            f"Expected gemma4_assistant, got {assistant.config.model_type!r}"
        )

    layer_values: list[np.ndarray | None] = [None] * len(assistant.model.layers)

    def save_layer(index: int) -> Callable:
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            layer_values[index] = _as_numpy(value[:, -1, :]).reshape(-1)

        return hook

    handles = [
        layer.register_forward_hook(save_layer(index))
        for index, layer in enumerate(assistant.model.layers)
    ]
    centroid_values: list[np.ndarray] = []

    def save_centroids(_module, _inputs, output):
        centroid_values.append(_as_numpy(output[:, -1, :]).reshape(-1))

    handles.append(assistant.masked_embedding.centroids.register_forward_hook(save_centroids))

    position_ids = torch.tensor(
        [[prompt_tokens]], dtype=torch.long, device=args.device
    )
    attention_mask = torch.ones(
        (1, prompt_tokens + 1), dtype=inputs["attention_mask"].dtype, device=args.device
    )
    probe = torch.zeros(
        (1, 1, assistant.hidden_size),
        dtype=assistant.dtype,
        device=args.device,
    )
    frequencies: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for layer_type in sorted(set(assistant.config.text_config.layer_types)):
            cos, sin = assistant.model.rotary_emb(probe, position_ids, layer_type)
            half = cos.shape[-1] // 2
            frequencies[f"freq_real_{layer_type}"] = _as_numpy(cos[..., :half]).reshape(-1)
            frequencies[f"freq_imag_{layer_type}"] = _as_numpy(sin[..., :half]).reshape(-1)

    input_token = torch.tensor([[current_token]], dtype=torch.long, device=args.device)
    hidden = predecessor_hidden
    draft_tokens: list[int] = []
    round_inputs: list[np.ndarray] = []
    round_embeddings: list[np.ndarray] = []
    round_layers: list[np.ndarray] = []
    round_final_states: list[np.ndarray] = []
    round_projected: list[np.ndarray] = []
    round_logits: list[np.ndarray] = []
    input_token_ids: list[int] = []

    try:
        for _ in range(args.num_draft_tokens):
            with torch.inference_mode():
                embedding = target.get_input_embeddings()(input_token)
                assistant_input = torch.cat([embedding, hidden], dim=-1)
                layer_values[:] = [None] * len(layer_values)
                output = assistant(
                    inputs_embeds=assistant_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    shared_kv_states=shared_kv_states,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )

            token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            if any(value is None for value in layer_values):
                raise RuntimeError("Failed to capture every assistant layer output")
            input_token_ids.append(int(input_token.item()))
            round_embeddings.append(_as_numpy(embedding).reshape(-1))
            round_inputs.append(_as_numpy(assistant_input).reshape(-1))
            round_layers.append(np.stack(layer_values))  # type: ignore[arg-type]
            round_final_states.append(
                _as_numpy(output.hidden_states[-1][:, -1, :]).reshape(-1)
            )
            round_projected.append(_as_numpy(output.last_hidden_state).reshape(-1))
            round_logits.append(_as_numpy(output.logits[:, -1, :]).reshape(-1))
            draft_tokens.append(token)
            input_token = torch.tensor([[token]], dtype=torch.long, device=args.device)
            hidden = output.last_hidden_state
    finally:
        for handle in handles:
            handle.remove()

    verification_ids = torch.tensor(
        [[current_token, *draft_tokens]], dtype=torch.long, device=args.device
    )
    verification_position_ids = torch.arange(
        prompt_tokens,
        prompt_tokens + verification_ids.shape[1],
        dtype=torch.long,
        device=args.device,
    ).unsqueeze(0)
    verification_mask = torch.ones(
        (1, prompt_tokens + verification_ids.shape[1]),
        dtype=inputs["attention_mask"].dtype,
        device=args.device,
    )
    with torch.inference_mode():
        verification_output = target(
            input_ids=verification_ids,
            attention_mask=verification_mask,
            position_ids=verification_position_ids,
            past_key_values=target_output.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    verification_logits = _as_numpy(verification_output.logits[0])
    target_predictions = verification_logits.argmax(axis=-1).astype(np.int64)
    draft_array = np.asarray(draft_tokens, dtype=np.int64)
    accepted = _accepted_prefix(draft_array, target_predictions)

    token_ordering = _as_numpy(assistant.masked_embedding.token_ordering).astype(np.int64)
    output_data: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "prompt": np.asarray(args.prompt),
        "prompt_input_ids": _as_numpy(inputs["input_ids"]).astype(np.int64).reshape(-1),
        "prompt_token_count": np.asarray(prompt_tokens, dtype=np.int64),
        "query_position": np.asarray(prompt_tokens, dtype=np.int64),
        "current_token": np.asarray(current_token, dtype=np.int64),
        "predecessor_hidden": _as_numpy(predecessor_hidden).reshape(-1),
        "input_token_ids": np.asarray(input_token_ids, dtype=np.int64),
        "hf_round_embeddings": np.stack(round_embeddings),
        "hf_round_inputs": np.stack(round_inputs),
        "hf_layer_outputs": np.stack(round_layers),
        "hf_final_states": np.stack(round_final_states),
        "hf_projected_states": np.stack(round_projected),
        "hf_centroid_logits": np.stack(centroid_values),
        "hf_logits": np.stack(round_logits),
        "hf_draft_tokens": draft_array,
        "hf_prompt_next_logits": _as_numpy(prompt_next_logits).reshape(-1),
        "hf_verification_logits": verification_logits,
        "target_predictions": target_predictions,
        "token_ordering": token_ordering,
        "layer_types": np.asarray(assistant.config.text_config.layer_types),
        "shared_full_keys": _as_numpy(shared_kv_states["full_attention"][0]),
        "shared_full_values": _as_numpy(shared_kv_states["full_attention"][1]),
        "shared_sliding_keys": _as_numpy(shared_kv_states["sliding_attention"][0]),
        "shared_sliding_values": _as_numpy(shared_kv_states["sliding_attention"][1]),
        **frequencies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_data)

    decoded = tokenizer.convert_ids_to_tokens([current_token, *draft_tokens])
    print(f"Capture: {args.output}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Current + HF drafts: {_format_tokens([current_token, *draft_tokens])}")
    print(f"Token pieces: {decoded}")
    print(f"Target predictions: {_format_tokens(target_predictions)}")
    print(
        f"HF accepted draft prefix: {accepted}/{len(draft_tokens)} "
        f"({accepted / max(1, len(draft_tokens)):.1%})"
    )


def _tensor_stats(reference: np.ndarray, actual: np.ndarray) -> str:
    reference = np.asarray(reference, dtype=np.float32).reshape(-1)
    actual = np.asarray(actual, dtype=np.float32).reshape(-1)
    if reference.shape != actual.shape:
        return f"shape mismatch {reference.shape} vs {actual.shape}"
    delta = actual - reference
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    cosine = float(np.dot(reference, actual) / denominator) if denominator else 1.0
    reference_norm = float(np.linalg.norm(reference))
    actual_norm = float(np.linalg.norm(actual))
    norm_ratio = actual_norm / reference_norm if reference_norm else 1.0
    return (
        f"cos={cosine:.7f} norm_ratio={norm_ratio:.5g} mean_abs={np.mean(np.abs(delta)):.6g} "
        f"max_abs={np.max(np.abs(delta)):.6g}"
    )


@dataclass
class ModelPackage:
    assistant_dir: Path
    target_dir: Path
    assistant_name: str
    target_name: str
    assistant_config: dict[str, Any]
    assistant_vlm_config: dict[str, Any]
    target_config: dict[str, Any]
    target_vlm_config: dict[str, Any]

    @staticmethod
    def discover(root: Path) -> "ModelPackage":
        assistant: tuple[Path, dict[str, Any]] | None = None
        target: tuple[Path, dict[str, Any]] | None = None
        for config_path in root.rglob("sima_files/devkit/config.json"):
            config = json.loads(config_path.read_text())
            model_dir = config_path.parents[2]
            if config.get("model_type") == "gemma4_assistant":
                assistant = (model_dir, config)
            elif config.get("model_type") == "gemma4_text":
                target = (model_dir, config)
        if assistant is None or target is None:
            raise FileNotFoundError(
                f"Could not find both Gemma4 target and assistant under {root}"
            )
        assistant_dir, assistant_config = assistant
        target_dir, target_config = target
        assistant_sdk = assistant_dir / "sima_files" / "sdk"
        marker = next(assistant_sdk.glob("*_language_n1_pre_layer0.sima"), None)
        if marker is None:
            raise FileNotFoundError(f"Missing assistant n1 SDK graphs in {assistant_sdk}")
        assistant_name = marker.name.removesuffix("_language_n1_pre_layer0.sima")
        target_embedding = next(
            (target_dir / "sima_files" / "devkit").glob("*_language_embeddings.bin"),
            None,
        )
        if target_embedding is None:
            raise FileNotFoundError("Missing target embedding table")
        target_name = target_embedding.name.removesuffix("_language_embeddings.bin")
        return ModelPackage(
            assistant_dir=assistant_dir,
            target_dir=target_dir,
            assistant_name=assistant_name,
            target_name=target_name,
            assistant_config=assistant_config,
            assistant_vlm_config=json.loads(
                (assistant_dir / "sima_files" / "devkit" / "vlm_config.json").read_text()
            ),
            target_config=target_config,
            target_vlm_config=json.loads(
                (target_dir / "sima_files" / "devkit" / "vlm_config.json").read_text()
            ),
        )


class RuntimeEmbeddingTable:
    def __init__(self, package: ModelPackage):
        from ml_dtypes import bfloat16

        self._bfloat16 = bfloat16
        devkit = package.target_dir / "sima_files" / "devkit"
        values_path = devkit / f"{package.target_name}_language_embeddings.bin"
        scales_path = devkit / f"{package.target_name}_language_embedding_scales.bin"
        self.vocab_size = int(package.target_config["vocab_size"])
        self.hidden_size = int(package.target_config["hidden_size"])
        expected_int8_size = self.vocab_size * self.hidden_size
        if scales_path.is_file() and values_path.stat().st_size == expected_int8_size:
            self.quantized = True
            self.values = np.memmap(
                values_path,
                dtype=np.int8,
                mode="r",
                shape=(self.vocab_size, self.hidden_size),
            )
            self.scales = np.memmap(
                scales_path,
                dtype=bfloat16,
                mode="r",
                shape=(self.vocab_size,),
            )
        else:
            self.quantized = False
            self.values = np.memmap(
                values_path,
                dtype=bfloat16,
                mode="r",
                shape=(self.vocab_size, self.hidden_size),
            )
            self.scales = None

    def row(self, token_id: int) -> np.ndarray:
        if token_id < 0 or token_id >= self.vocab_size:
            raise IndexError(f"Embedding token {token_id} is outside the vocabulary")
        if not self.quantized:
            return np.asarray(self.values[token_id], dtype=np.float32)
        scale = np.float32(self.scales[token_id])
        values = np.asarray(self.values[token_id], dtype=np.float32) * (scale / np.float32(127.0))
        return values.astype(self._bfloat16).astype(np.float32)


class GraphRunner:
    def __init__(self, sdk_dir: Path, mode: str):
        from afe.apis.model import Model
        from afe.core.configs import RunConfigs
        from afe.ir.defines import TensorValue, get_expected_tensor_value
        from afe.ir.tensor_type import ScalarType

        self.sdk_dir = sdk_dir
        self.mode = mode
        self._model_class = Model
        self._cache: dict[str, Any] = {}

        # DynamicDequantOp currently returns BF16 in the host evaluator even
        # when a quantizable FP graph declares float32. Cast that boundary to
        # its declared type while otherwise executing the saved RELAY graph.
        run_config = RunConfigs(fast_mode=False)

        def fp_executor(node, inputs, node_outputs):
            output = node.ir.run(inputs, run_config)
            expected_type = node.ir.get_type().output
            if isinstance(output, np.ndarray) and isinstance(expected_type, TensorValue):
                expected = get_expected_tensor_value(expected_type)
                if expected.scalar == ScalarType.float32 and output.dtype != np.float32:
                    output = output.astype(np.float32)
            node_outputs[node.name] = output

        self._fp_executor = fp_executor

    def run(self, graph_name: str, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        from afe import load_awesomenet

        if graph_name not in self._cache:
            if self.mode == "fp32":
                file_name = f"{graph_name}.fp32.sima"
                if not (self.sdk_dir / file_name).is_file():
                    raise FileNotFoundError(self.sdk_dir / file_name)
                self._cache[graph_name] = load_awesomenet(file_name, str(self.sdk_dir))
            else:
                file_name = f"{graph_name}.sima"
                self._cache[graph_name] = self._model_class.load(
                    file_name,
                    str(self.sdk_dir),
                    include_unquantized_net=False,
                )
        graph = self._cache[graph_name]
        if self.mode == "fp32":
            return graph.run(inputs, node_callable=self._fp_executor)
        return graph.execute(inputs, fast_mode=False, use_jax=False)


def _find_cache_graph(
    sdk_dir: Path, assistant_name: str, sliding: bool, visible_tokens: int
) -> tuple[str, int]:
    cache_kind = "sliding_cache" if sliding else "cache"
    prefix = f"{assistant_name}_language_n1_{cache_kind}_token"
    candidates: list[tuple[int, str]] = []
    for path in sdk_dir.glob(f"{prefix}*.sima"):
        match = re.search(r"token(\d+)\.sima$", path.name)
        if match:
            width = int(match.group(1)) + 1
            if width >= visible_tokens:
                candidates.append((width, path.name.removesuffix(".sima")))
    if not candidates:
        raise FileNotFoundError(
            f"No {cache_kind} graph can hold {visible_tokens} visible tokens"
        )
    width, graph_name = min(candidates)
    return graph_name, width


def _quantize_and_pad_kv(
    tensor: np.ndarray, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from afe.ir.operation_functions import dynamic_dequant
    from ml_dtypes import bfloat16
    from ml_kernels.np_operators import ideal_dynamic_quantize

    tensor_bf16 = np.asarray(tensor, dtype=bfloat16)
    quantized, scale = ideal_dynamic_quantize(tensor_bf16, per_token=True)
    dequantized = dynamic_dequant(quantized, scale).astype(np.float32)
    padded = np.zeros((*quantized.shape[:-2], width, quantized.shape[-1]), dtype=np.int8)
    padded_scale = np.zeros((*scale.shape[:-2], width, 1), dtype=np.float32)
    length = quantized.shape[-2]
    padded[..., :length, :] = quantized
    padded_scale[..., :length, :] = scale.astype(np.float32)
    return padded, padded_scale, dequantized


@dataclass
class CacheInputs:
    graph_name: str
    keys: np.ndarray
    key_scales: np.ndarray
    values: np.ndarray
    value_scales: np.ndarray
    mask: np.ndarray


def _prepare_cache_inputs(
    sdk_dir: Path,
    assistant_name: str,
    sliding: bool,
    keys: np.ndarray,
    values: np.ndarray,
) -> tuple[CacheInputs, str, str]:
    visible_tokens = int(keys.shape[-2])
    graph_name, width = _find_cache_graph(
        sdk_dir, assistant_name, sliding, visible_tokens
    )
    quantized_keys, key_scales, dequantized_keys = _quantize_and_pad_kv(keys, width)
    quantized_values, value_scales, dequantized_values = _quantize_and_pad_kv(values, width)
    mask = np.full((1, 1, 1, width), -np.inf, dtype=np.float32)
    mask[..., :visible_tokens] = 0.0
    return (
        CacheInputs(
            graph_name=graph_name,
            keys=quantized_keys,
            key_scales=key_scales,
            values=quantized_values,
            value_scales=value_scales,
            mask=mask,
        ),
        _tensor_stats(keys, dequantized_keys),
        _tensor_stats(values, dequantized_values),
    )


@dataclass
class AssistantRound:
    token: int
    projected: np.ndarray
    layer_outputs: list[np.ndarray]
    centroid_logits: np.ndarray
    full_logits: np.ndarray


def _select_masked_token(
    full_logits: np.ndarray,
    centroid_logits: np.ndarray,
    token_ordering: np.ndarray,
    num_centroids: int,
    top_k: int,
) -> int:
    ordering = token_ordering.reshape(num_centroids, -1)
    top_centroids = np.argpartition(centroid_logits, -top_k)[-top_k:]
    candidates = ordering[top_centroids].reshape(-1)
    return int(candidates[np.argmax(full_logits[candidates])])


def _run_assistant_round(
    runner: GraphRunner,
    package: ModelPackage,
    input_vector: np.ndarray,
    frequencies: dict[str, tuple[np.ndarray, np.ndarray]],
    caches: dict[str, CacheInputs],
    token_ordering: np.ndarray,
) -> AssistantRound:
    lm_cfg = package.assistant_vlm_config["lm_cfg"]
    num_splits = int(lm_cfg["lm_head_num_splits"])
    num_centroids = int(lm_cfg["assistant_num_centroids"])
    centroid_top_k = int(lm_cfg["assistant_centroid_intermediate_top_k"])
    layer_types = package.assistant_config["text_config"]["layer_types"]
    state = np.asarray(input_vector, dtype=np.float32).reshape(1, 1, 1, -1)
    layer_outputs: list[np.ndarray] = []
    final_outputs: list[np.ndarray] | None = None

    for layer_index, layer_type in enumerate(layer_types):
        freq_real, freq_imag = frequencies[layer_type]
        text_config = package.assistant_config["text_config"]
        rope_config = text_config["rope_parameters"][layer_type]
        head_dim = (
            text_config.get("global_head_dim", text_config["head_dim"])
            if layer_type == "full_attention"
            else text_config["head_dim"]
        )
        rotary_width = int(
            head_dim * rope_config.get("partial_rotary_factor", 1.0) // 2
        )
        freq_real = freq_real.reshape(-1)[:rotary_width]
        freq_imag = freq_imag.reshape(-1)[:rotary_width]
        pre_name = (
            f"{package.assistant_name}_language_n1_pre_layer{layer_index}"
        )
        query = runner.run(
            pre_name,
            {
                "input": state,
                "freq_real": freq_real.reshape(1, 1, 1, -1),
                "freq_imag": freq_imag.reshape(1, 1, 1, -1),
            },
        )[0]
        cache = caches[layer_type]
        attention = runner.run(
            cache.graph_name,
            {
                "input": query,
                "cached_keys": cache.keys,
                "cached_keys_scale": cache.key_scales,
                "attn_mask": cache.mask,
                "cached_values": cache.values,
                "cached_values_scale": cache.value_scales,
            },
        )[0]
        post_name = (
            f"{package.assistant_name}_language_n1_post_layer{layer_index}"
        )
        post_outputs = runner.run(
            post_name,
            {"input": state, "self_attn": attention},
        )
        if layer_index < len(layer_types) - 1:
            state = np.asarray(post_outputs[0], dtype=np.float32)
            layer_outputs.append(state.reshape(-1))
        else:
            final_outputs = post_outputs

    if final_outputs is None or len(final_outputs) != num_splits + 2:
        raise RuntimeError("Unexpected final assistant post graph outputs")
    full_logits = np.concatenate(
        [np.asarray(output, dtype=np.float32).reshape(-1) for output in final_outputs[:num_splits]]
    )
    centroid_logits = np.asarray(final_outputs[num_splits], dtype=np.float32).reshape(-1)
    projected = np.asarray(final_outputs[num_splits + 1], dtype=np.float32).reshape(-1)
    token = _select_masked_token(
        full_logits,
        centroid_logits,
        token_ordering,
        num_centroids,
        centroid_top_k,
    )
    return AssistantRound(
        token=token,
        projected=projected,
        layer_outputs=layer_outputs,
        centroid_logits=centroid_logits,
        full_logits=full_logits,
    )


def replay_sdk(args: argparse.Namespace) -> None:
    package = ModelPackage.discover(args.model_package.resolve())
    with np.load(args.capture.resolve(), allow_pickle=False) as capture_file:
        capture = {key: capture_file[key] for key in capture_file.files}

    if int(capture["format_version"]) != 1:
        raise ValueError("Unsupported capture format")
    sdk_dir = package.assistant_dir / "sima_files" / "sdk"
    compiled_ordering_path = (
        package.assistant_dir / "sima_files" / "devkit" / "gemma4_token_ordering.npy"
    )
    compiled_ordering = np.load(compiled_ordering_path).reshape(-1).astype(np.int64)
    reference_ordering = capture["token_ordering"].reshape(-1).astype(np.int64)
    if not np.array_equal(compiled_ordering, reference_ordering):
        mismatch = int(np.flatnonzero(compiled_ordering != reference_ordering)[0])
        raise ValueError(f"Compiled token ordering differs from HF at index {mismatch}")

    embedding_table = RuntimeEmbeddingTable(package)
    pipeline_cfg = package.target_vlm_config["pipeline_cfg"]
    print(f"Model package: {args.model_package.resolve()}")
    print(
        "Target settings: "
        f"quantize_embeddings={pipeline_cfg.get('quantize_embeddings')} "
        f"quantize_kv_cache={pipeline_cfg.get('quantize_kv_cache')}"
    )
    print("Token ordering: exact HF/compiled match")

    input_token_ids = capture["input_token_ids"].astype(np.int64)
    embedding_stats = []
    for index, token_id in enumerate(input_token_ids):
        embedding_stats.append(
            _tensor_stats(
                capture["hf_round_embeddings"][index],
                embedding_table.row(int(token_id)),
            )
        )
    print("Embedding quantization by HF draft round:")
    for index, stats in enumerate(embedding_stats):
        print(f"  {index}: token={int(input_token_ids[index])} {stats}")

    full_cache, full_key_stats, full_value_stats = _prepare_cache_inputs(
        sdk_dir,
        package.assistant_name,
        False,
        capture["shared_full_keys"],
        capture["shared_full_values"],
    )
    sliding_cache, sliding_key_stats, sliding_value_stats = _prepare_cache_inputs(
        sdk_dir,
        package.assistant_name,
        True,
        capture["shared_sliding_keys"],
        capture["shared_sliding_values"],
    )
    caches = {
        "full_attention": full_cache,
        "sliding_attention": sliding_cache,
    }
    print("Shared KV dynamic quantization:")
    print(f"  full keys:     {full_key_stats}")
    print(f"  full values:   {full_value_stats}")
    print(f"  sliding keys:  {sliding_key_stats}")
    print(f"  sliding values:{sliding_value_stats}")

    frequencies = {
        "full_attention": (
            capture["freq_real_full_attention"],
            capture["freq_imag_full_attention"],
        ),
        "sliding_attention": (
            capture["freq_real_sliding_attention"],
            capture["freq_imag_sliding_attention"],
        ),
    }
    hf_drafts = capture["hf_draft_tokens"].astype(np.int64)
    target_predictions = capture["target_predictions"].astype(np.int64)
    num_teacher_rounds = min(args.teacher_rounds, len(hf_drafts))
    modes = ["quantized"]
    if args.include_fp32:
        modes.insert(0, "fp32")

    for mode in modes:
        print(f"Teacher-forced {mode} graph parity:")
        runner = GraphRunner(sdk_dir, mode)
        for round_index in range(num_teacher_rounds):
            result = _run_assistant_round(
                runner,
                package,
                capture["hf_round_inputs"][round_index],
                frequencies,
                caches,
                compiled_ordering,
            )
            print(
                f"  round {round_index}: token={result.token} "
                f"HF={int(hf_drafts[round_index])}"
            )
            for layer_index, actual in enumerate(result.layer_outputs):
                reference = capture["hf_layer_outputs"][round_index, layer_index]
                print(
                    f"    layer {layer_index} hidden: {_tensor_stats(reference, actual)}"
                )
            print(
                "    projected hidden: "
                + _tensor_stats(capture["hf_projected_states"][round_index], result.projected)
            )
            print(
                "    centroid logits:  "
                + _tensor_stats(capture["hf_centroid_logits"][round_index], result.centroid_logits)
            )

    recurrent_rounds = min(args.recurrent_rounds, len(hf_drafts))
    if recurrent_rounds:
        print("Runtime-faithful quantized recurrence with HF target hidden/KV:")
        runner = GraphRunner(sdk_dir, "quantized")
        token = int(capture["current_token"])
        hidden = capture["predecessor_hidden"].astype(np.float32)
        recurrent_tokens: list[int] = []
        for round_index in range(recurrent_rounds):
            embedding = embedding_table.row(token)
            assistant_input = np.concatenate([embedding, hidden])
            result = _run_assistant_round(
                runner,
                package,
                assistant_input,
                frequencies,
                caches,
                compiled_ordering,
            )
            recurrent_tokens.append(result.token)
            print(
                f"  round {round_index}: token={result.token} "
                f"HF={int(hf_drafts[round_index])} "
                f"projected={_tensor_stats(capture['hf_projected_states'][round_index], result.projected)}"
            )
            token = result.token
            hidden = result.projected

        recurrent_array = np.asarray(recurrent_tokens, dtype=np.int64)
        accepted = _accepted_prefix(recurrent_array, target_predictions)
        hf_accepted = _accepted_prefix(hf_drafts, target_predictions)
        print(f"HF drafts:        {_format_tokens(hf_drafts)}")
        print(f"SDK drafts:       {_format_tokens(recurrent_array)}")
        print(f"Target predicts:  {_format_tokens(target_predictions)}")
        print(
            f"Accepted prefix: SDK={accepted}/{recurrent_rounds}, "
            f"HF={hf_accepted}/{len(hf_drafts)}"
        )
        if np.array_equal(recurrent_array, hf_drafts[:recurrent_rounds]):
            print("Result: quantized assistant recurrence matches Hugging Face tokens")
        else:
            first = int(np.flatnonzero(recurrent_array != hf_drafts[:recurrent_rounds])[0])
            print(f"Result: first quantized assistant token divergence is round {first}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture one Hugging Face MTP round")
    capture.add_argument("--target-model", type=Path, required=True)
    capture.add_argument("--assistant-model", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--prompt", default=DEFAULT_PROMPT)
    capture.add_argument("--num-draft-tokens", type=int, default=6)
    capture.add_argument("--device", default="cpu")
    capture.add_argument("--thinking", action="store_true")
    capture.set_defaults(func=capture_hf)

    replay = subparsers.add_parser("replay", help="Replay a capture through Model SDK")
    replay.add_argument("--model-package", type=Path, required=True)
    replay.add_argument("--capture", type=Path, required=True)
    replay.add_argument("--teacher-rounds", type=int, default=1)
    replay.add_argument("--recurrent-rounds", type=int, default=6)
    replay.add_argument("--include-fp32", action="store_true")
    replay.set_defaults(func=replay_sdk)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "num_draft_tokens", 1) < 1:
        raise SystemExit("--num-draft-tokens must be positive")
    if getattr(args, "teacher_rounds", 0) < 0:
        raise SystemExit("--teacher-rounds must be non-negative")
    if getattr(args, "recurrent_rounds", 0) < 0:
        raise SystemExit("--recurrent-rounds must be non-negative")
    args.func(args)


if __name__ == "__main__":
    main()
