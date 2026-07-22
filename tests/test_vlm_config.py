from dataclasses import asdict
from pathlib import Path
import json

import pytest

from sima_lmm.config.vlm_config import (
    ModelFormat,
    PipelineConfig,
    VlmConfig,
    group_cache_model_indices,
    single_cache_model_indices,
)
from sima_lmm.gguf.gguf_conversion import GgufModel
from transformers import AutoConfig
from tests.conftest import require_readable_path

REFERENCE_CONFIGS_PATH = Path(__file__).parent / "reference_configs"

_EXAMPLE_CONFIGS = [
    # (hf_model_folder,                                    golden_config,                          image_resolution)
    # Llama
    ("models--meta-llama--Llama-2-7b-chat-hf",            "llama2_vlm_config.json",               None),
    ("models--meta-llama--Llama-3.1-8B-Instruct",         "llama3.1_vlm_config.json",             None),
    ("models--meta-llama--Llama-3.2-3B-Instruct",         "llama3.2_vlm_config.json",             None),
    # Gemma
    ("models--google--gemma-2-2b-it",                     "gemma2_vlm_config.json",               None),
    ("models--google--gemma-3-4b-it",                     "gemma3_vlm_config.json",               None),
    ("models--google--gemma-4-E4B-it",                    "gemma4_e4b_it_vlm_config.json",        [480, 480]),
    # Mistral
    ("models--mistralai--Mistral-7B-Instruct-v0.3",       "mistral_vlm_config.json",              None),
    # Phi
    ("models--microsoft--Phi-3.5-mini-instruct",          "phi3.5_vlm_config.json",               None),
    ("models--microsoft--Phi-4-mini-instruct",            "phi4_vlm_config.json",                 None),
    # Qwen
    ("models--Qwen--Qwen2.5-3B-Instruct",                 "qwen2.5_vlm_config.json",              None),
    ("models--Qwen--Qwen3-4B",                            "qwen3_vlm_config.json",                None),
    # LiquidAI
    ("models--LiquidAI--LFM2-1.2B",                       "lfm2_vlm_config.json",                 None),
    # VLMs
    ("models--stribomon--gemma3-siglip448",                "gemma3_siglip448_vlm_config.json",     None),
    ("models--Qwen--Qwen2.5-VL-3B-Instruct",              "qwen2.5_vl_vlm_config.json",           [448, 448]),
    ("models--Qwen--Qwen3-VL-4B-Instruct",                "qwen3_vl_vlm_config.json",             [448, 448]),
    ("models--LiquidAI--LFM2-VL-1.6B",                    "lfm2_vl_vlm_config.json",              None),
    ("models--LiquidAI--LFM2.5-VL-450M",                  "lfm2.5_vl_450m_vlm_config.json",       None),
    ("models--llava-hf--llava-1.5-7b-hf",                 "llava1.5_vlm_config.json",             None),
]


_GGUF_CONFIGS = [
    # (gguf_folder,                                                golden_config)
    # Llama
    ("models--unsloth--Llama-3.1-8B-Instruct-GGUF",              "llama3.1_vlm_config.json"),
    ("models--unsloth--Llama-3.2-3B-Instruct-GGUF",              "llama3.2_vlm_config.json"),
    # Gemma
    ("models--unsloth--gemma-4-E4B-it-GGUF",                     "gemma4_e4b_it_vlm_config.json"),
    # Mistral
    ("models--bartowski--Mistral-7B-Instruct-v0.3-GGUF",         "mistral_vlm_config.json"),
    # Phi
    ("models--bartowski--Phi-3.5-mini-instruct-GGUF",            "phi3.5_vlm_config.json"),
    #Qwen
    ("models--Qwen--Qwen2.5-3B-Instruct-GGUF",                "qwen2.5_vlm_config.json"),
    ("models--Qwen--Qwen3-4B-GGUF",                           "qwen3_vlm_config.json"),
    # LiquidAI
     ("models--LiquidAI--LFM2-1.2B-GGUF",                         "lfm2_vlm_config.json"),
]


@pytest.mark.premerge
def test_embedding_scale_is_absent_without_embedding_quantization():
    assert PipelineConfig().embeddings_scale is None


@pytest.mark.premerge
def test_default_max_num_tokens():
    assert PipelineConfig().max_num_tokens == 8192


@pytest.mark.parametrize(
    ("max_num_tokens", "expected_long_context_mask"),
    [(2047, None), (2048, 1024), (4096, 1024)],
)
@pytest.mark.premerge
def test_long_context_future_token_mask_resolution(
    max_num_tokens: int, expected_long_context_mask: int | None
):
    config = PipelineConfig()
    config.set_max_num_tokens(max_num_tokens)
    config.set_future_token_mask_size(128)

    assert config.future_token_mask_size == 128
    assert config.long_context_future_token_mask_size == expected_long_context_mask
    assert config.get_future_token_mask_size("full_attention") == (
        expected_long_context_mask or 128
    )
    assert config.get_future_token_mask_size("sliding_attention") == 128


@pytest.mark.premerge
def test_invalid_future_token_mask_is_rejected_for_long_context():
    config = PipelineConfig(max_num_tokens=2048)

    with pytest.raises(ValueError, match="greater than zero"):
        config.set_future_token_mask_size(0)


@pytest.mark.premerge
def test_legacy_pipeline_config_uses_stored_mask_for_all_attention_types():
    config = PipelineConfig(max_num_tokens=2048, future_token_mask_size=128)

    assert config.long_context_future_token_mask_size is None
    assert config.get_future_token_mask_size("full_attention") == 128
    assert config.get_future_token_mask_size("sliding_attention") == 128


def _load_reference_config(filename: str) -> VlmConfig:
    config = json.loads((REFERENCE_CONFIGS_PATH / filename).read_text())
    return VlmConfig.load(config)


def _layer_indices(config: VlmConfig, part: str) -> list[int]:
    return [layer.part_idx for layer in config.get_layer_ids() if layer.part == part]


@pytest.mark.parametrize(
    ("sliding_window", "expected_transition"),
    [(512, 384), (1024, 896)],
)
@pytest.mark.premerge
def test_gemma3_automatic_sliding_cache_transition(
    sliding_window: int, expected_transition: int
):
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = sliding_window
    config.config_pipeline(None, None, 2048, 128, 128)

    assert expected_transition in _layer_indices(config, "group_cache")
    assert _layer_indices(config, "group_sliding_cache") == []


@pytest.mark.premerge
def test_shared_sliding_cache_includes_full_and_sliding_mask_buckets():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 512
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == [0, 128, 256, 384, 896, 1920]
    assert _layer_indices(config, "single_cache") == [127, 255, 383, 511, 1023, 2047]


@pytest.mark.premerge
def test_shared_sliding_cache_transition_does_not_add_execution_offset():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 1000
    config.config_pipeline(None, None, 2048, 128, 128)

    transition = 872
    assert transition not in config.pipeline_cfg.input_token_group_offsets
    assert transition in _layer_indices(config, "group_cache")


@pytest.mark.premerge
def test_gemma4_keeps_separate_sliding_cache_models():
    config = _load_reference_config("gemma4_e4b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == [896, 1920]
    assert _layer_indices(config, "single_cache") == [1023, 2047]
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]
    assert _layer_indices(config, "single_sliding_cache") == [127, 255, 383, 511]


@pytest.mark.premerge
def test_sliding_cache_uses_non_default_base_mask_at_long_context():
    config = _load_reference_config("gemma4_e4b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 256)

    assert _layer_indices(config, "group_cache") == [896, 1920]
    assert _layer_indices(config, "single_cache") == [1023, 2047]
    assert _layer_indices(config, "group_sliding_cache") == [128, 384]
    assert _layer_indices(config, "single_sliding_cache") == [255, 511]


@pytest.mark.premerge
def test_long_context_mask_covers_partial_final_bucket():
    config = PipelineConfig()
    config.set_max_num_tokens(2500)
    config.set_group_size(128)
    config.set_future_token_mask_size(128)

    assert group_cache_model_indices(config)[-1] == 2372
    assert single_cache_model_indices(config)[-1] == 2499


@pytest.mark.premerge
def test_group_configuration_is_automatic_and_serializable():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.config_pipeline(None, None, 512, 128, 128)

    restored = VlmConfig.load(json.loads(json.dumps(asdict(config))))
    assert restored.pipeline_cfg.input_token_group_size == 128
    assert restored.pipeline_cfg.input_token_group_offsets == [0, 128, 256, 384]


@pytest.mark.premerge
def test_sliding_attention_rejects_group_at_least_as_large_as_window():
    config = _load_reference_config("gemma3_vlm_config.json")

    with pytest.raises(ValueError, match="smaller than sliding_window"):
        config.config_pipeline(None, None, 2048, 1024, 128)


def _find_gguf_file(hf_models_path: Path, folder: str) -> Path:
    """Find Q4_0 gguf file in folder, fall back to Q8_0."""
    base = require_readable_path(hf_models_path / folder)
    for quant in ("Q4_0", "Q8_0"):
        matches = list(base.glob(f"*[Qq]4_0.gguf")) if quant == "Q4_0" else list(base.glob(f"*[Qq]8_0.gguf"))
        if matches:
            return require_readable_path(matches[0])
    pytest.skip(f"No Q4_0 or Q8_0 gguf file found in {base}")


@pytest.mark.premerge
@pytest.mark.parametrize("test_config", _EXAMPLE_CONFIGS, ids=lambda x: x[0])
def test_gen_vlm_config(hf_models_path: Path, test_config: tuple):
    model_folder, golden_filename, image_resolution = test_config

    model_path = require_readable_path(hf_models_path / model_folder)
    hf_config = AutoConfig.from_pretrained(model_path)
    model_config = hf_config.to_dict()

    vlm_cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, model_path, model_config, image_resolution)

    vlm_dict = asdict(vlm_cfg)
    golden_dict = json.loads((REFERENCE_CONFIGS_PATH / golden_filename).read_text())
    vlm_dict.pop("model_name", None)
    golden_dict.pop("model_name", None)
    assert vlm_dict == golden_dict


def _strip_gguf_fields(d: dict) -> dict:
    """Remove fields that are expected to differ between HF and GGUF configs."""
    if d.get("lm_cfg"):
        # data_type reflects quantization format (e.g. mostly-Q4_0) in GGUF vs bfloat16 in HF
        d["lm_cfg"].pop("data_type", None)
    return d


@pytest.mark.premerge
@pytest.mark.parametrize("test_config", _GGUF_CONFIGS, ids=lambda x: x[0])
def test_gen_vlm_config_gguf(hf_models_path: Path, test_config: tuple):
    gguf_folder, golden_filename = test_config

    gguf_path = _find_gguf_file(hf_models_path, gguf_folder)
    model = GgufModel(gguf_path)
    model_cfg = model.model_config

    vlm_cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_GGUF, gguf_path, model_cfg)

    vlm_dict = _strip_gguf_fields(asdict(vlm_cfg))
    golden_dict = _strip_gguf_fields(json.loads((REFERENCE_CONFIGS_PATH / golden_filename).read_text()))
    # model_name is derived from the folder name and not part of the architecture config
    # mm_cfg,vm_cfg is None for gguf models (Vision Encoder not supported)
    vlm_dict.pop("model_name", None)
    vlm_dict.pop("mm_cfg",None)
    vlm_dict.pop("vm_cfg",None)
    vlm_dict["lm_cfg"].pop("model_type", None)

    golden_dict.pop("model_name", None)
    golden_dict.pop("mm_cfg",None)
    golden_dict.pop("vm_cfg",None)
    golden_dict["lm_cfg"].pop("model_type", None)

    # rms_norm_eps is stored as float32 in GGUF, causing minor precision loss vs HF float64
    rms_norm_eps_gguf = vlm_dict["lm_cfg"].pop("rms_norm_eps", None)
    rms_norm_eps_golden = golden_dict["lm_cfg"].pop("rms_norm_eps", None)
    if rms_norm_eps_gguf is not None and rms_norm_eps_golden is not None:
        assert abs(rms_norm_eps_gguf - rms_norm_eps_golden) < 1e-08, \
            f"rms_norm_eps mismatch: {rms_norm_eps_gguf} vs {rms_norm_eps_golden}"

    assert vlm_dict == golden_dict
