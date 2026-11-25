"""Package native MLA executables and original host lookup tables for LFM2 Audio."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
import torch
from sima_lmm.hf.lfm2_audio import Lfm2AudioCheckpoint
from sima_lmm.host.compile_lfm2_audio import _graph_signature, _interfaces, _model_parameters, _sha256, _write_manifest
from sima_lmm.model.lfm2_audio_compiler import plan_audio_components


def audio_mel_filters() -> np.ndarray:
    """The checkpoint's 128-bin Slaney filterbank at 16 kHz and FFT size 512."""
    step = 200. / 3.
    log_step = np.log(6.4) / 27.
    upper = 15. + np.log(8000. / 1000.) / log_step
    mel = np.linspace(0., upper, 130)
    hz = step * mel
    logarithmic = mel >= 15.
    hz[logarithmic] = 1000. * np.exp(log_step * (mel[logarithmic] - 15.))
    frequencies = np.fft.rfftfreq(512, 1. / 16000.)
    weights = np.empty((128, 257), np.float32)
    for index in range(128):
        lower = (frequencies - hz[index]) / (hz[index + 1] - hz[index])
        upper = (hz[index + 2] - frequencies) / (hz[index + 2] - hz[index + 1])
        weights[index] = np.maximum(0., np.minimum(lower, upper))
    weights *= (2. / (hz[2:] - hz[:-2]))[:, None]
    return weights


def export_host_tables(checkpoint: Lfm2AudioCheckpoint, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source, decoder = checkpoint.audio_source(), checkpoint.detokenizer_source()
    descriptions = {
        "text_embeddings": (source, "lfm.embed_tokens.weight"),
        "audio_embeddings": (source, "audio_embedding.embedding.weight"),
        "decoder_embeddings": (decoder, "emb.emb.weight"),
        "window": (decoder, "istft.window"),
        **{f"depth_embeddings_{index}": (source, f"depth_embeddings.{index}.embedding.weight")
           for index in range(checkpoint.config["codebooks"])},
    }
    assets = {}
    for name, (weights, tensor_name) in descriptions.items():
        tensor = weights.load_np_param(tensor_name)
        path = output / f"{name}.bin"
        tensor.tofile(path)
        assets[name] = {"path": path.name, "shape": list(tensor.shape),
                        "dtype": tensor.dtype.name, "sha256": _sha256(path)}
        del tensor
    preprocessing = checkpoint.config["preprocessor"]
    expected = {"sample_rate": 16000, "window_size": .025, "window_stride": .01,
                "window": "hann", "features": 128, "n_fft": 512, "normalize": "per_feature",
                "log": True, "frame_splicing": 1, "pad_to": 0, "pad_value": 0.}
    if any(preprocessing.get(key) != value for key, value in expected.items()):
        raise ValueError("The native PCM frontend requires the pinned LFM2.5 Audio preprocessing contract")
    for name, tensor in {
        "mel_filters": audio_mel_filters(),
        "analysis_window": torch.hann_window(400, periodic=False).numpy(),
    }.items():
        path = output / f"{name}.bin"
        tensor.tofile(path)
        assets[name] = {"path": path.name, "shape": list(tensor.shape),
                        "dtype": tensor.dtype.name, "sha256": _sha256(path)}
    return assets


def prepare_runtime(checkpoint_path: Path, build_path: Path) -> Path:
    build_path = build_path.resolve()
    manifest_path = build_path / "audio_compile_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["complete_compute_build"]:
        raise ValueError("Runtime packaging requires every planned compiler unit")
    checkpoint = Lfm2AudioCheckpoint.from_directory(checkpoint_path)
    for name, expected in manifest["identity"]["files"].items():
        if _sha256(checkpoint.directory / name) != expected:
            raise ValueError(f"Checkpoint changed since compilation: {name}")
    plan = plan_audio_components(checkpoint, build_path, **manifest["identity"]["options"])
    runtime = build_path / "runtime"
    (runtime / "elf").mkdir(parents=True, exist_ok=True)
    descriptions = {}
    for unit in plan:
        model = unit.model
        entry = manifest["units"][model.model_name]
        if not entry["compiled"]:
            raise ValueError(f"Uncompiled runtime unit: {model.model_name}")
        if entry.get("graph_signature") != _graph_signature(model):
            raise ValueError(f"Graph source changed for {model.model_name}; resume compilation first")
        if entry.get("model_parameters") != _model_parameters(model):
            raise ValueError(f"Graph parameters changed for {model.model_name}; resume compilation first")
        # Reconstruct metadata through the same compiler source; never guess
        # physical KV strides from logical tensor dimensions.
        current = _interfaces(model)
        if "mla_inputs" not in entry.get("interfaces", {}):
            entry["interfaces"] = current
        current_config = json.loads(json.dumps(asdict(model.cfg)))
        if entry["native_config"] != current_config:
            raise ValueError(f"Compiler configuration changed for {model.model_name}; resume compilation first")
        archive_path = build_path / entry["artifact"]
        if _sha256(archive_path) != entry["archive"]["sha256"]:
            raise ValueError(f"Compiled archive changed: {model.model_name}")
        if entry["archive"]["segments"].get("MLA") != 1 or entry["archive"]["segments"].get("A65", 0):
            raise ValueError(f"Native audio runtime needs one MLA segment without A65 fallback: {model.model_name}")
        target = runtime / "elf" / f"{model.model_name}.elf"
        with tarfile.open(archive_path) as archive:
            executables = [member for member in archive.getmembers() if member.name.endswith("_mla.elf")]
            if len(executables) != 1:
                raise ValueError(f"Expected one MLA executable in {archive_path}")
            with archive.extractfile(executables[0]) as stream, target.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
        descriptions[model.model_name] = {
            "elf": str(target.relative_to(runtime)), "component": unit.component, "role": unit.role,
            "interfaces": current, "native_config": current_config,
        }
    assets = export_host_tables(checkpoint, runtime / "tables")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        shutil.copyfile(checkpoint.directory / name, runtime / name)
    description = {
        "schema_version": 1, "model_type": "lfm2_audio", "precision": manifest["identity"]["precision"],
        "options": manifest["identity"]["options"], "source_config": checkpoint.config,
        "detokenizer_config": checkpoint.detokenizer_source().config,
        "units": descriptions, "tables": assets,
        "lookup_precision": "original checkpoint dtype", "kv_precision": "bfloat16",
        "hardware_validated": False,
    }
    path = runtime / "audio_runtime.json"
    _write_manifest(path, description)
    manifest["runtime_bundle"] = str(path.relative_to(build_path))
    _write_manifest(manifest_path, manifest)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("build_path", type=Path)
    args = parser.parse_args()
    print(prepare_runtime(args.model_path, args.build_path))


if __name__ == "__main__":
    main()
