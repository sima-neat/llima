from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import load_file, save_file
import torch
from transformers import Lfm2Config, Lfm2Model

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.hf import lfm2_audio
from sima_lmm.hf.lfm2_audio import Lfm2AudioCheckpoint
from sima_lmm.model.language_part_base import LanguagePostBaseModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def test_audio_resume_signature_tracks_tensor_view_settings():
    from sima_lmm.host.compile_lfm2_audio import _graph_signature
    source = SimpleNamespace(tensor_aliases={"weight": "first.weight"}, rotary_head_dim=32)
    model = SimpleNamespace(cfg=VlmConfig(), hf_model=source)
    original = _graph_signature(model)
    source.tensor_aliases["weight"] = "second.weight"
    assert _graph_signature(model) != original
    source.tensor_aliases["weight"] = "first.weight"
    source.rotary_head_dim = 64
    assert _graph_signature(model) != original


def test_audio_resume_signature_tracks_module_helpers(monkeypatch):
    from pathlib import Path
    from sima_lmm.host.compile_lfm2_audio import _graph_signature
    model = SimpleNamespace(cfg=VlmConfig(), hf_model=SimpleNamespace())
    original = _graph_signature(model)
    read_bytes = Path.read_bytes

    def changed_helper(path):
        content = read_bytes(path)
        return content + b"\n# changed module helper\n" if path.name == "lfm2_audio_model.py" else content

    monkeypatch.setattr(Path, "read_bytes", changed_helper)
    assert _graph_signature(model) != original


@pytest.mark.parametrize("stage,completed", [("quantize", "SOURCE_TO_FP"), ("compile", "FP_TO_QUANT")])
def test_audio_partial_stages_reject_stale_or_missing_intermediates(tmp_path, stage, completed):
    from sima_lmm.host.compile_lfm2_audio import _require_stage_inputs
    path = tmp_path / "intermediate.sima"
    path.write_bytes(b"fixture")
    model = SimpleNamespace(sdk_fp_file_name=path, sdk_file_name=path)
    entry = {"stages": [completed]}
    _require_stage_inputs(model, entry, stage, matching=True)
    with pytest.raises(RuntimeError, match="matching"):
        _require_stage_inputs(model, entry, stage, matching=False)
    with pytest.raises(RuntimeError, match="matching"):
        _require_stage_inputs(model, {}, stage, matching=True)
    path.unlink()
    with pytest.raises(RuntimeError, match="matching"):
        _require_stage_inputs(model, entry, stage, matching=True)


@pytest.fixture
def audio_checkpoint(tmp_path):
    language_config = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "hidden_size": 32,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "conv_L_cache": 3,
        "conv_bias": False,
        "layer_types": ["conv", "full_attention"],
        "block_multiple_of": 32,
        "block_auto_adjust_ff_dim": True,
        "torch_dtype": "float32",
    }
    config = {
        "architectures": ["Lfm2AudioForConditionalGeneration"],
        "lfm": language_config,
        "encoder": {"n_layers": 17, "d_model": 512},
        "depthformer": {"layers": 6, "dim": 1024, "tie": True},
        "preprocessor": {"sample_rate": 16000},
        "codebooks": 8,
        "interleaved_n_text": 6,
        "interleaved_n_audio": 12,
    }
    with torch.random.fork_rng():
        torch.manual_seed(0)
        reference = Lfm2Model(Lfm2Config(**language_config)).state_dict()
    tensors = {f"lfm.{name}": value for name, value in reference.items()}
    # A naive prefix search would select this unrelated model.* component.
    tensors["audio_adapter.model.0.weight"] = torch.ones((3, 3))
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, tmp_path / "model.safetensors")
    sidecar = tmp_path / "audio_detokenizer"
    sidecar.mkdir()
    (sidecar / "config.json").write_text('{"model_type":"lfm2"}', encoding="utf-8")
    save_file({"lin.weight": torch.ones((2, 2))}, sidecar / "model.safetensors")
    return tmp_path, config, tensors


def test_audio_config_preserves_full_contract_and_builds_native_backbone(audio_checkpoint):
    path, config, _ = audio_checkpoint
    original = (path / "config.json").read_bytes()
    source = Lfm2AudioCheckpoint.from_directory(path)
    assert source.config == config
    assert (path / "config.json").read_bytes() == original
    backbone = source.language_backbone()
    native = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, path, backbone.config)
    assert native.lm_cfg.layer_types == ["conv", "full_attention"]
    assert native.lm_cfg.mlp_cfg.intermediate_size == 64
    assert backbone.language_model_param_base_name == "model"
    assert all(name.startswith("model.") for name in backbone.weight_map)
    # Existing output-head resolution must reuse the exact input embedding table.
    component = SimpleNamespace(cfg=native, check_hf_param=backbone.param_exists)
    assert LanguagePostBaseModel._get_output_embed_name(component) == "model.embed_tokens.weight"
    backbone.config["hidden_size"] = 999
    assert source.backbone_config["hidden_size"] == 32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_audio_backbone_namespace_preserves_every_tensor(audio_checkpoint, dtype):
    path, _, tensors = audio_checkpoint
    save_file({name: value.to(dtype) for name, value in tensors.items()}, path / "model.safetensors")
    backbone = Lfm2AudioCheckpoint.from_directory(path).language_backbone()
    for name in backbone.weight_map:
        expected = tensors["lfm." + name.removeprefix("model.")].to(dtype).float().numpy()
        np.testing.assert_array_equal(backbone.load_np_param(name).astype(np.float32), expected)
    assert not backbone.param_exists("audio_adapter.model.0.weight")
    np.testing.assert_array_equal(
        backbone.load_param("lm_head", -1, "weight").astype(np.float32),
        tensors["lfm.embed_tokens.weight"].to(dtype).float().numpy(),
    )


def test_audio_backbone_materializes_only_its_component(audio_checkpoint):
    path, _, _ = audio_checkpoint
    backbone = Lfm2AudioCheckpoint.from_directory(path).language_backbone()
    backbone.load_all_params()
    assert set(backbone.params) == set(backbone.weight_map)
    assert all(name.startswith("model.") for name in backbone.params)
    np.testing.assert_array_equal(
        backbone.load_np_param("model.embed_tokens.weight"),
        load_file(path / "model.safetensors")["lfm.embed_tokens.weight"].numpy(),
    )


def test_normal_loader_rejects_full_audio_before_nested_file_discovery(audio_checkpoint):
    path, _, _ = audio_checkpoint
    with pytest.raises(NotImplementedError, match="explicit experimental component compiler"):
        LocalHuggingFaceModel.create_from_directory(path)


def test_normal_language_checkpoint_loading_still_works(audio_checkpoint):
    path, config, tensors = audio_checkpoint
    (path / "config.json").write_text(json.dumps(config["lfm"]), encoding="utf-8")
    save_file(
        {"model." + name.removeprefix("lfm."): value
         for name, value in tensors.items() if name.startswith("lfm.")},
        path / "model.safetensors",
    )
    # Retain the nested config to exercise root-config precedence. The ordinary
    # language path has only one weight file; nested weight support is separate.
    (path / "audio_detokenizer/model.safetensors").unlink()
    source = LocalHuggingFaceModel.create_from_directory(path)
    assert source.config["model_type"] == "lfm2"
    np.testing.assert_array_equal(
        source.load_np_param("model.embed_tokens.weight"),
        tensors["lfm.embed_tokens.weight"].numpy(),
    )


def test_audio_inspection_does_not_read_tensor_values(audio_checkpoint, monkeypatch):
    path, _, _ = audio_checkpoint
    original_open = lfm2_audio.safe_open

    class HeadersOnly:
        def __init__(self, *args, **kwargs):
            self.weights = original_open(*args, **kwargs)

        def __enter__(self):
            self.weights.__enter__()
            return self

        def __exit__(self, *args):
            return self.weights.__exit__(*args)

        def keys(self):
            return self.weights.keys()

        def metadata(self):
            return self.weights.metadata()

        def get_slice(self, name):
            tensor = self.weights.get_slice(name)
            return SimpleNamespace(get_shape=tensor.get_shape, get_dtype=tensor.get_dtype)

        def get_tensor(self, _name):
            pytest.fail("Checkpoint inspection must not materialize tensor values")

    monkeypatch.setattr(lfm2_audio, "safe_open", HeadersOnly)
    assert Lfm2AudioCheckpoint.from_directory(path).tensor_shapes


def test_backbone_view_rejects_complete_reference_execution(audio_checkpoint):
    path, _, _ = audio_checkpoint
    backbone = Lfm2AudioCheckpoint.from_directory(path).language_backbone()
    with pytest.raises(NotImplementedError, match="language-component view"):
        backbone.execute_hf(messages=[])


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "architectures", ["Lfm2ForCausalLM"], "Expected architectures"),
        (None, "encoder", None, "encoder configuration object"),
        (None, "codebooks", 16, "eight codebooks"),
        (None, "codebooks", True, "positive integer codebooks"),
        (None, "interleaved_n_audio", 0, "positive integer interleaved_n_audio"),
        (None, "quantization_config", {"quant_method": "gptq"}, "Pre-quantized"),
        ("lfm", "model_type", "llama", "lfm.model_type"),
        ("lfm", "architectures", ["LlamaForCausalLM"], "lfm.architectures"),
        ("lfm", "quantization_config", {"quant_method": "gptq"}, "Pre-quantized"),
        ("lfm", "num_hidden_layers", 3, "layer_types must match"),
        ("lfm", "layer_types", ["conv", "sliding_attention"], "only conv and full_attention"),
        ("lfm", "num_attention_heads", 3, "head counts are inconsistent"),
        ("lfm", "conv_L_cache", 0, "positive integer conv_L_cache"),
        ("lfm", "conv_bias", True, "conv_bias=False"),
        ("lfm", "tie_word_embeddings", False, "tied input/output embeddings"),
    ],
)
def test_audio_rejects_unsupported_or_inconsistent_contracts(
    audio_checkpoint, section, key, value, message,
):
    path, config, _ = audio_checkpoint
    changed = deepcopy(config)
    target = changed if section is None else changed[section]
    target[key] = value
    (path / "config.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        Lfm2AudioCheckpoint.from_directory(path)


@pytest.mark.parametrize("failure", ["missing", "shape", "unexpected", "integer"])
def test_audio_validates_backbone_tensor_contract(audio_checkpoint, failure):
    path, _, tensors = audio_checkpoint
    name = "lfm.layers.1.self_attn.k_proj.weight"
    if failure == "missing":
        del tensors[name]
        message = "missing=.*k_proj"
    elif failure == "shape":
        tensors[name] = tensors[name][:-1].contiguous()
        message = "k_proj.weight has shape"
    elif failure == "unexpected":
        tensors["lfm.layers.2.extra.weight"] = torch.ones((2, 2))
        message = "unexpected=.*extra"
    else:
        tensors[name] = tensors[name].to(torch.int8)
        message = "must have floating-point weights"
    save_file(tensors, path / "model.safetensors")
    with pytest.raises(ValueError, match=message):
        Lfm2AudioCheckpoint.from_directory(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_depth_aliases_preserve_qkv_banks_and_rotary_norm_order(audio_checkpoint, dtype):
    path, config, tensors = audio_checkpoint
    config["depthformer"].update(layers=1, dim=128)
    base = "depthformer.layers.0"
    tensors.update({
        f"{base}.operator_norm.weight": torch.ones(128),
        f"{base}.ffn_norm.weight": torch.ones(128),
        f"{base}.feed_forward.w1.weight": torch.zeros((512, 128)),
        f"{base}.feed_forward.w2.weight": torch.zeros((128, 512)),
        f"{base}.feed_forward.w3.weight": torch.zeros((512, 128)),
        f"{base}.operator.qkv_proj.weight": torch.arange(192 * 128).reshape(192, 128).float(),
        f"{base}.operator.out_proj.weight": torch.eye(128),
        f"{base}.operator.bounded_attention.q_layernorm.weight": torch.tensor([1., 2., 3., 4.]),
        f"{base}.operator.bounded_attention.k_layernorm.weight": torch.tensor([5., 6., 7., 8.]),
    })
    tensors = {name: tensor.to(dtype) for name, tensor in tensors.items()}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, path / "model.safetensors")
    source = Lfm2AudioCheckpoint.from_directory(path).depth_backbone()
    packed = source.load_np_param("model.layers.0.self_attn.qkv_proj.weight").astype(np.float32)
    original = tensors[f"{base}.operator.qkv_proj.weight"].float().numpy()
    np.testing.assert_array_equal(packed[:4], original[[0, 2, 1, 3]])
    np.testing.assert_array_equal(packed[128:132], original[[128, 130, 129, 131]])
    np.testing.assert_array_equal(packed[160:], original[160:])
    np.testing.assert_array_equal(source.load_np_param("model.layers.0.self_attn.q_layernorm.weight").astype(np.float32), [1, 3, 2, 4])
    np.testing.assert_array_equal(source.load_np_param("model.layers.0.self_attn.k_layernorm.weight").astype(np.float32), [5, 7, 6, 8])
