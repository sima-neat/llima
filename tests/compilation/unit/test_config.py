import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sima_lmm.config.vlm_config import (
    ModelFormat,
    PipelineConfig,
    VlmArchType,
    VlmConfig,
    group_cache_model_indices,
    single_cache_model_indices,
)
from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenPrecision
from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model import VisionLanguageModel
from sima_lmm.model import vision_language_model
from sima_lmm.model.language_model import LanguageModel
from sima_lmm.host.configuration_helper import read_configuration_file


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

REFERENCE_CONFIGS_PATH = (
    Path(__file__).parents[1] / "configuration" / "references"
)


def _load_reference_config(filename: str) -> VlmConfig:
    config = json.loads((REFERENCE_CONFIGS_PATH / filename).read_text())
    return VlmConfig.load(config)


def _layer_indices(config: VlmConfig, part: str) -> list[int]:
    return [layer.part_idx for layer in config.get_layer_ids() if layer.part == part]


def test_whisper_finds_generation_config_in_cache_root(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "revision"
    snapshot_path.mkdir(parents=True)
    generation_config = {
        "lang_to_id": {
            "<|en|>": 50259,
            "<|de|>": 50261,
        }
    }
    (snapshot_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = WhisperConfig.from_hf_config(
        str(tmp_path), {"architectures": ["WhisperForConditionalGeneration"]}
    )

    assert config.num_languages == 2


@pytest.mark.parametrize(
    ("model_suppress_tokens", "generation_config", "expected_suppress_tokens"),
    [
        ([1], {"suppress_tokens": [2, 3]}, [2, 3]),
        ([1], {}, [1]),
        (None, {"suppress_tokens": [2, 3]}, [2, 3]),
        (None, {}, []),
        ([1], {"suppress_tokens": None}, []),
    ],
)
def test_whisper_resolves_suppress_tokens_from_generation_config(
    tmp_path,
    model_suppress_tokens,
    generation_config,
    expected_suppress_tokens,
):
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = WhisperConfig.from_hf_config(
        tmp_path,
        {
            "architectures": ["WhisperForConditionalGeneration"],
            "suppress_tokens": model_suppress_tokens,
        },
    )

    assert config.suppress_tokens == expected_suppress_tokens


def test_embedding_quantization_is_supported_for_non_gemma4_vlm(monkeypatch, tmp_path):
    config = _load_reference_config("qwen3_vl_vlm_config.json")
    hf_model = SimpleNamespace(config={})
    vlm_helper = object()

    monkeypatch.setattr(
        vision_language_model, "model_file_type", lambda _path: ModelFormat.FORMAT_HF
    )
    monkeypatch.setattr(
        vision_language_model.LocalHuggingFaceModel,
        "create_from_directory",
        lambda **_kwargs: hf_model,
    )
    monkeypatch.setattr(
        vision_language_model.VlmConfig,
        "from_hf_config",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        vision_language_model, "VlmHelper", lambda *_args, **_kwargs: vlm_helper
    )

    model = VisionLanguageModel.from_hf_cache(
        model_name="qwen3-vl",
        hf_cache_path=tmp_path / "model",
        onnx_path=tmp_path / "onnx",
        sima_path=tmp_path / "sima",
        max_num_tokens=1024,
        quantize_embeddings=True,
    )

    assert model.cfg.is_multimodal
    assert model.cfg.pipeline_cfg.quantize_embeddings


def test_default_max_num_tokens():
    assert PipelineConfig().max_num_tokens == 4096


@pytest.mark.parametrize("image_size", ([224, 224], [960, 672]))
def test_vision_model_names_are_always_per_layer(image_size: list[int]):
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    model_name = "gemma4_vision"

    config.vm_cfg.image_size = image_size
    config.config_pipeline(None, None, 2048, 128, 128)
    assert config.get_vision_model_names(model_name) == [
        f"{model_name}_layer{layer_idx}"
        for layer_idx in range(config.vm_cfg.num_hidden_layers)
    ]
    assert _layer_indices(config, "vision") == list(
        range(config.vm_cfg.num_hidden_layers)
    )


def test_vision_model_names_omit_unused_llava_layer():
    config = _load_reference_config("gemma3_siglip448_vlm_config.json")
    config.model_type = VlmArchType.VLM_LLAVA
    config.config_pipeline(None, None, 2048, 128, 128)

    assert config.get_vision_model_names("llava_vision") == [
        f"llava_vision_layer{layer_idx}"
        for layer_idx in range(config.vm_cfg.num_hidden_layers - 1)
    ]
    assert _layer_indices(config, "vision") == list(
        range(config.vm_cfg.num_hidden_layers - 1)
    )


@pytest.mark.parametrize(
    ("max_num_tokens", "expected_long_context_mask"),
    [(1024, None), (2048, None), (3072, 1024), (4096, 1024)],
)
def test_long_context_future_token_mask_resolution(
    max_num_tokens: int, expected_long_context_mask: int | None
):
    config = PipelineConfig()
    config.set_max_num_tokens(max_num_tokens)
    config.set_future_token_mask_size(128)

    assert config.future_token_mask_size == 128
    assert config.long_context_future_token_mask_size == expected_long_context_mask
    assert config.get_cache_mask_size("full_attention", 2048, is_group=False) == 128
    assert config.get_cache_mask_size("full_attention", 2049, is_group=False) == (
        expected_long_context_mask or 128
    )
    assert (
        config.get_cache_mask_size(
            "sliding_attention", max_num_tokens, is_group=False
        )
        == 128
    )


def test_invalid_future_token_mask_is_rejected_for_long_context():
    config = PipelineConfig(max_num_tokens=2048)

    with pytest.raises(ValueError, match="greater than zero"):
        config.set_future_token_mask_size(0)


def test_speculative_decoding_rejects_sliding_attention():
    config = _load_reference_config("gemma3_vlm_config.json")

    with pytest.raises(
        ValueError,
        match="EAGLE3 speculative decoding does not support sliding-window attention",
    ):
        config.lm_cfg.set_speculative_decoding_config({})


def test_speculative_decoding_rejects_linear_attention():
    config = _load_reference_config("qwen3.5_vlm_config.json")

    with pytest.raises(
        ValueError,
        match="EAGLE3 speculative decoding does not support linear-attention layers",
    ):
        config.lm_cfg.set_speculative_decoding_config({})


@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_dflash_accepts_qwen35_hybrid_attention(block_size: int):
    config = _load_reference_config("qwen3.5_vlm_config.json")

    config.lm_cfg.set_speculative_decoding_config(
        {
            "method": "dflash",
            "speculative_budget": block_size,
            "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29],
            "mask_token_id": 248077,
        }
    )

    assert config.lm_cfg.speculative_decoding_cfg.speculative_budget == block_size


def _dflash_draft_config() -> VlmConfig:
    config = _load_reference_config("qwen3.5_vlm_config.json")
    config.model_type = VlmArchType.LLM_QWEN3
    config.vm_cfg = None
    config.mm_cfg = None
    config.lm_cfg.model_type = "qwen3"
    config.lm_cfg.num_hidden_layers = 6
    config.lm_cfg.layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    config.lm_cfg.linear_attn_cfg = None
    config.lm_cfg.attn_cfg.swa_enable = True
    config.lm_cfg.attn_cfg.sliding_window = 4096
    config.lm_cfg.set_speculative_decoding_config(
        {
            "method": "dflash",
            "is_draft": True,
            "speculative_budget": 8,
            "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29],
            "mask_token_id": 248077,
        }
    )
    config.config_pipeline(None, None, 2048, 128, 128)
    return config


def test_dflash_draft_layers_keep_prefill_and_verification_widths_separate(tmp_path):
    config = _dflash_draft_config()

    assert _layer_indices(config, "group_pre") == []
    assert _layer_indices(config, "group_post") == []
    assert _layer_indices(config, "group_dflash_context") == list(range(6))
    assert _layer_indices(config, "single_dflash_context") == list(range(6))
    assert _layer_indices(config, "single_pre") == list(range(6))
    assert _layer_indices(config, "single_post") == list(range(6))

    model = LanguageModel(
        config,
        "draft",
        onnx_path=tmp_path / "onnx",
        sima_path=tmp_path / "sima",
        hf_model=SimpleNamespace(),
    )
    pre = model._get_part_model("pre", 8, layer_idx=0)
    post = model._get_part_model("post", 8, layer_idx=0)
    assert pre.num_tokens == 8
    assert pre._layer_base_name == "layers.0"
    assert post._layer_base_name == "layers.0"
    assert model._get_part_model("dflash_context", 128, layer_idx=0).num_tokens == 128


def test_dflash_draft_deduplicates_equal_group_and_block_widths():
    config = _dflash_draft_config()
    config.config_pipeline(None, None, 2048, 8, 8)

    assert _layer_indices(config, "group_dflash_context") == []
    assert _layer_indices(config, "single_dflash_context") == list(range(6))
    assert _layer_indices(config, "group_draft_fc") == []
    assert _layer_indices(config, "single_draft_fc") == [0]


def test_dflash_final_post_uses_paired_target_head(tmp_path):
    config = _dflash_draft_config()
    quantized_head = (
        np.array([[0.5, 0.25]], dtype=np.float32),
        np.array([[2, 4, 8, 12]], dtype=np.int8),
        2,
    )
    target_weights = SimpleNamespace(
        param_exists=lambda name: name == "lm_head.weight",
        load_np_param=lambda _name: quantized_head,
    )
    model = LanguageModel(
        config,
        "draft",
        onnx_path=tmp_path / "onnx",
        sima_path=tmp_path / "sima",
        hf_model=SimpleNamespace(),
        dflash_target_hf_model=target_weights,
    )

    post = model._get_part_model("post", 8, layer_idx=5)
    assert post._dflash_target_output_embed_name() == "lm_head.weight"
    np.testing.assert_array_equal(
        post._get_dflash_target_param("lm_head.weight", dequantize=True),
        np.array([[1, 2, 2, 3]], dtype=np.float32),
    )


def test_configuration_identifies_speculative_draft_model(tmp_path):
    configuration_file = tmp_path / "precision.py"
    configuration_file.write_text(
        "def get_layer_configuration(model_properties, _layer):\n"
        "    precision = ('A_BF16_W_INT8' if model_properties['is_draft_model'] "
        "else 'BF16')\n"
        "    return {'precision': precision}\n"
    )

    layer_ids = [LayerID("single_pre", 0)]
    target_cfg = SimpleNamespace(
        lm_cfg=SimpleNamespace(num_hidden_layers=32, speculative_decoding_cfg=None),
        get_layer_ids=lambda: layer_ids,
    )
    draft_cfg = SimpleNamespace(
        lm_cfg=SimpleNamespace(
            num_hidden_layers=6,
            speculative_decoding_cfg=SimpleNamespace(is_draft=True),
        ),
        get_layer_ids=lambda: layer_ids,
    )
    target = read_configuration_file(SimpleNamespace(cfg=target_cfg), configuration_file)
    draft = read_configuration_file(SimpleNamespace(cfg=draft_cfg), configuration_file)

    assert set(target["precision"].values()) == {FileGenPrecision.BF16}
    assert set(draft["precision"].values()) == {FileGenPrecision.A_BF16_W_INT8}


def test_dflash_routes_linear_and_sliding_cache_graphs_at_block_width(
    monkeypatch, tmp_path
):
    target_cfg = _load_reference_config("qwen3.5_vlm_config.json")
    target_cfg.lm_cfg.set_speculative_decoding_config(
        {"method": "dflash", "speculative_budget": 8}
    )
    target = LanguageModel(
        target_cfg,
        "target",
        onnx_path=tmp_path / "target_onnx",
        sima_path=tmp_path / "target_sima",
        hf_model=SimpleNamespace(language_model_param_base_name="model"),
    )
    assert target._get_part_model("pre", 8, layer_idx=0)._layer_base_name == (
        "model.layers.0"
    )
    target_models = []
    monkeypatch.setattr(
        target,
        "gen_files_from_model_list",
        lambda models, *_args: target_models.extend(models),
    )
    target.gen_files(
        object(),
        gen_config={
            "precision": {LayerID("single_linear", 0): FileGenPrecision.BF16}
        },
    )
    assert len(target_models) == 1
    assert target_models[0][0].num_tokens == 8

    draft_cfg = _dflash_draft_config()
    draft = LanguageModel(
        draft_cfg,
        "draft",
        onnx_path=tmp_path / "draft_onnx",
        sima_path=tmp_path / "draft_sima",
        hf_model=SimpleNamespace(),
    )
    draft_models = []
    monkeypatch.setattr(
        draft,
        "gen_files_from_model_list",
        lambda models, *_args: draft_models.extend(models),
    )
    draft.gen_files(
        object(),
        gen_config={
            "precision": {
                LayerID("single_sliding_cache", 127): FileGenPrecision.BF16
            }
        },
    )
    assert draft_models[0][0].num_tokens == 8


def test_dflash_pair_validation_accepts_published_contract():
    target_cfg = _load_reference_config("qwen3.5_vlm_config.json")
    target_cfg.lm_cfg.num_hidden_layers = 32
    target_cfg.config_pipeline(None, None, 2048, 128, 128)
    target = SimpleNamespace(cfg=target_cfg)
    draft_cfg = _dflash_draft_config()
    draft_hf_cfg = {
        "model_type": "qwen3",
        "num_target_layers": 32,
        "dflash_config": {"block_size": 16},
    }

    VisionLanguageModel._validate_dflash_pair(
        target,
        draft_cfg,
        draft_hf_cfg,
        8,
        [1, 5, 9, 13, 17, 21, 25, 29],
        248077,
    )


def test_dflash_pair_validation_accepts_generic_target_taps():
    target_cfg = _load_reference_config("qwen3.5_vlm_config.json")
    target_cfg.lm_cfg.num_hidden_layers = 32
    target_cfg.config_pipeline(None, None, 2048, 128, 128)
    draft_cfg = _dflash_draft_config()

    VisionLanguageModel._validate_dflash_pair(
        SimpleNamespace(cfg=target_cfg),
        draft_cfg,
        {
            "model_type": "qwen3",
            "num_target_layers": 32,
            "dflash_config": {"block_size": 16},
        },
        8,
        [0, 7, 15, 23],
        248077,
    )


def test_dflash_pair_validation_derives_llama_draft_topology():
    target_cfg = _load_reference_config("qwen3.5_vlm_config.json")
    target_cfg.lm_cfg.model_type = "llama"
    target_cfg.lm_cfg.num_hidden_layers = 32
    target_cfg.lm_cfg.layer_types = ["full_attention"] * 32
    target_cfg.lm_cfg.linear_attn_cfg = None
    target_cfg.config_pipeline(None, None, 2048, 128, 128)

    draft_cfg = _dflash_draft_config()
    draft_cfg.lm_cfg.num_hidden_layers = 5
    draft_cfg.lm_cfg.layer_types = ["full_attention"] * 5
    draft_cfg.lm_cfg.attn_cfg.swa_enable = False
    draft_cfg.lm_cfg.attn_cfg.sliding_window = None

    VisionLanguageModel._validate_dflash_pair(
        SimpleNamespace(cfg=target_cfg),
        draft_cfg,
        {
            "model_type": "qwen3",
            "num_target_layers": 32,
            "block_size": 10,
        },
        8,
        [1, 8, 15, 22, 29],
        128002,
    )


@pytest.mark.parametrize(
    ("draft_update", "message"),
    [
        ({"group_size": 64}, "language group sizes must match"),
        ({"max_num_tokens": 1024}, "draft cache must be at least as large"),
        ({"quantize_embeddings": True}, "embedding quantization modes must match"),
    ],
)
def test_dflash_pair_validation_rejects_incompatible_runtime_contract(
    draft_update, message
):
    target_cfg = _load_reference_config("qwen3.5_vlm_config.json")
    target_cfg.lm_cfg.num_hidden_layers = 32
    target_cfg.config_pipeline(None, None, 2048, 128, 128)
    draft_cfg = _dflash_draft_config()
    if "group_size" in draft_update:
        draft_cfg.config_pipeline(
            None, None, 2048, draft_update["group_size"], 128
        )
    if "max_num_tokens" in draft_update:
        draft_cfg.pipeline_cfg.max_num_tokens = draft_update["max_num_tokens"]
    if "quantize_embeddings" in draft_update:
        draft_cfg.pipeline_cfg.quantize_embeddings = draft_update[
            "quantize_embeddings"
        ]

    with pytest.raises(ValueError, match=message):
        VisionLanguageModel._validate_dflash_pair(
            SimpleNamespace(cfg=target_cfg),
            draft_cfg,
            {
                "model_type": "qwen3",
                "num_target_layers": 32,
                "dflash_config": {"block_size": 16},
            },
            8,
            [1, 5, 9, 13, 17, 21, 25, 29],
            248077,
        )


def test_linear_attention_validates_group_size():
    for group_size in (1, 4, 8, 16, 32, 128):
        config = _load_reference_config("qwen3.5_vlm_config.json")
        config.config_pipeline(None, None, 2048, group_size, 128)
        assert config.pipeline_cfg.input_token_group_size == group_size

    for group_size in (2, 12, 24, 48):
        config = _load_reference_config("qwen3.5_vlm_config.json")
        with pytest.raises(
            ValueError,
            match="language_group_size must be 1, 4, 8, 16, or a multiple of 32",
        ):
            config.config_pipeline(None, None, 2048, group_size, 128)


def test_legacy_pipeline_config_uses_stored_mask_for_all_attention_types():
    config = PipelineConfig(max_num_tokens=2048, future_token_mask_size=128)

    assert config.long_context_future_token_mask_size is None
    assert config.get_cache_mask_size("full_attention", 2048, is_group=False) == 128
    assert (
        config.get_cache_mask_size("sliding_attention", 2048, is_group=False)
        == 128
    )


@pytest.mark.parametrize(
    ("sliding_window", "expected_transition"),
    [(512, 384), (1024, 896)],
)
def test_gemma3_automatic_sliding_cache_transition(
    sliding_window: int, expected_transition: int
):
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = sliding_window
    config.config_pipeline(None, None, 2048, 128, 128)

    assert expected_transition in _layer_indices(config, "group_cache")
    assert _layer_indices(config, "group_sliding_cache") == []


def test_shared_sliding_cache_includes_full_and_sliding_mask_buckets():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 512
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(127, 2048, 128))


def test_shared_sliding_cache_transition_does_not_add_execution_offset():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 1000
    config.config_pipeline(None, None, 2048, 128, 128)

    transition = 872
    assert transition not in config.pipeline_cfg.input_token_group_offsets
    assert transition in _layer_indices(config, "group_cache")


def test_gemma4_keeps_separate_sliding_cache_models():
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(127, 2048, 128))
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]
    assert _layer_indices(config, "single_sliding_cache") == [127, 255, 383, 511]


def test_non_default_mask_only_buckets_single_cache_models_through_2k():
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 256)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(255, 2048, 256))
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]
    assert _layer_indices(config, "single_sliding_cache") == [255, 511]


@pytest.mark.parametrize("max_num_tokens", [128, 512, 2049, 2500])
def test_unaligned_max_num_tokens_is_rejected(max_num_tokens: int):
    config = PipelineConfig()

    with pytest.raises(ValueError, match="multiple of 1024"):
        config.set_max_num_tokens(max_num_tokens)


def test_long_context_mask_starts_after_2k():
    config = PipelineConfig()
    config.set_max_num_tokens(8192)
    config.set_group_size(128)
    config.set_future_token_mask_size(256)

    expected_group_indices = [*range(0, 2048, 128), *range(2944, 8065, 1024)]
    expected_single_indices = [
        *range(255, 2048, 256),
        *range(3071, 8192, 1024),
    ]
    assert group_cache_model_indices(config) == expected_group_indices
    assert single_cache_model_indices(config) == expected_single_indices


def test_long_context_mask_supports_groups_larger_than_base_mask():
    config = PipelineConfig()
    config.set_max_num_tokens(8192)
    config.set_group_size(320)
    config.set_future_token_mask_size(128)

    assert group_cache_model_indices(config) == [
        0,
        320,
        640,
        960,
        1280,
        1600,
        2752,
        3776,
        4800,
        5824,
        6848,
        7872,
    ]


def test_group_configuration_is_automatic_and_serializable():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.config_pipeline(None, None, 1024, 128, 128)

    restored = VlmConfig.load(json.loads(json.dumps(asdict(config))))
    assert restored.pipeline_cfg.input_token_group_size == 128
    assert restored.pipeline_cfg.input_token_group_offsets == list(range(0, 1024, 128))


def test_sliding_attention_rejects_group_at_least_as_large_as_window():
    config = _load_reference_config("gemma3_vlm_config.json")

    with pytest.raises(ValueError, match="smaller than sliding_window"):
        config.config_pipeline(None, None, 2048, 1024, 128)
