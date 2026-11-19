"""Compile explicit LFM2 Audio native components and retain an audited manifest."""

import argparse
from collections import Counter
from dataclasses import asdict, fields
import fnmatch
import gc
import hashlib
import inspect
from importlib import import_module
from importlib.metadata import version
import json
import logging
from pathlib import Path
import shutil
import tarfile
import time

from afe.ir.defines import get_expected_tensor_value
from afe.ir.serializer import load_awesomenet
from afe.ir.node import node_is_tuple
from sima_lmm.hf.lfm2_audio import Lfm2AudioCheckpoint
from sima_lmm.model.base import FileGenMode, FileGenPrecision
from sima_lmm.model.lfm2_audio_compiler import plan_audio_components
from sima_lmm.model.model_graph import ModelGraph


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _write_manifest(path: Path, manifest: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _interfaces(model) -> dict:
    net = load_awesomenet(model.sdk_file_name.name, str(model.sdk_file_name.parent))

    def tensor(name):
        spec = get_expected_tensor_value(net.nodes[name].get_type().output)
        return {"name": name, "shape": list(spec.shape), "dtype": str(spec.scalar)}

    inputs = [tensor(name) for name in net.input_node_names]
    output = net.nodes[net.output_node_name]
    names = output.input_node_names if node_is_tuple(output) else [output.name]
    interfaces = {"inputs": inputs, "outputs": [tensor(name) for name in names]}
    subnet = net.nodes["MLA_0"].ir

    def mla_tensors(names, overrides):
        results = []
        for index, name in enumerate(names):
            spec = get_expected_tensor_value(subnet.nodes[name].get_type().output)
            override = overrides.get(index, overrides.get(index - len(names)))
            dram_shape = getattr(override, "dram_shape", None) or spec.shape
            layout = str(getattr(override, "dram_layout", "HWC16"))
            results.append({"name": name, "shape": list(spec.shape), "dtype": str(spec.scalar),
                            "dram_shape": list(dram_shape), "dram_layout": layout})
        return results

    last = subnet.nodes[subnet.output_node_name]
    output_names = last.input_node_names if node_is_tuple(last) else [last.name]
    interfaces["mla_inputs"] = mla_tensors(subnet.input_node_names, model.get_mla_input_tessellate_params())
    interfaces["mla_outputs"] = mla_tensors(output_names, model.get_mla_output_tessellate_params())
    return interfaces


def _archive_metadata(path: Path) -> dict:
    with tarfile.open(path) as archive:
        description = next(
            (member for member in archive.getmembers() if member.name.endswith("_mpk.json")), None
        )
        if description is None:
            raise RuntimeError(f"Compiled archive has no MPK description: {path}")
        metadata = json.load(archive.extractfile(description))
        segments = Counter(plugin["processor"] for plugin in metadata["plugins"])
        if not segments.get("MLA") or not any(member.name.endswith(".elf") for member in archive.getmembers()):
            raise RuntimeError(f"Compiled archive has no MLA executable: {path}")
        return {"segments": dict(segments), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _graph_signature(model) -> str:
    # Helpers and tensor-view settings affect graphs independently of classes.
    digest = hashlib.sha256(b"lfm2-audio-source-signature-v2")
    modules = {ModelGraph.__module__, Lfm2AudioCheckpoint.__module__,
               plan_audio_components.__module__, "sima_lmm.model.lfm2_audio_model"}
    for obj in (model, model.hf_model):
        for cls in type(obj).__mro__:
            if cls.__module__.startswith("sima_lmm."):
                modules.add(cls.__module__)
    for name in sorted(modules):
        digest.update(name.encode())
        digest.update(Path(inspect.getfile(import_module(name))).read_bytes())
    tensor_view = {name: getattr(model.hf_model, name) for name in
                   ("tensor_aliases", "rotary_head_dim", "query_rows", "key_rows")
                   if hasattr(model.hf_model, name)}
    digest.update(json.dumps(tensor_view, sort_keys=True).encode())
    digest.update(json.dumps(asdict(model.cfg), sort_keys=True).encode())
    return digest.hexdigest()


def _model_parameters(model) -> dict:
    """Include instance topology arguments that are not part of VlmConfig."""
    excluded = {"cfg", "sima_path", "hf_model", "vlm_helper"}
    return {field.name: getattr(model, field.name) for field in fields(model)
            if field.name not in excluded}


def _require_stage_inputs(model, entry: dict, stage: str, matching: bool):
    """Never label old intermediates with a new graph identity."""
    prerequisite = {
        "quantize": (FileGenMode.SOURCE_TO_FP.name, model.sdk_fp_file_name),
        "compile": (FileGenMode.FP_TO_QUANT.name, model.sdk_file_name),
    }.get(stage)
    if prerequisite is None:
        return
    completed, path = prerequisite
    if not matching or completed not in entry.get("stages", []) or not path.is_file():
        raise RuntimeError(
            f"Stage {stage} requires --resume with a matching {completed} intermediate; "
            "run --stage all in a fresh output directory when graph inputs changed"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--component", choices=["all", "backbone", "encoder", "depth", "detokenizer"], default="all")
    parser.add_argument("--unit", action="append", default=[], help="Select exact names or shell-style patterns for debugging")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--audio-frames", type=int, default=256)
    parser.add_argument("--decoder-frames", type=int, default=12)
    parser.add_argument("--stage", choices=["export", "quantize", "compile", "all"], default="all")
    parser.add_argument("--list", action="store_true", help="Print the plan without generating model files")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-fp32", action="store_true", help="Keep floating graphs after successful compilation")
    args = parser.parse_args()
    source_path, output = args.model_path.resolve(), args.output.resolve()
    if output == source_path or output.is_relative_to(source_path):
        parser.error("Output must be outside the input checkpoint")
    checkpoint = Lfm2AudioCheckpoint.from_directory(source_path)
    options = dict(context_length=args.context_length, group_size=args.group_size,
                   audio_frames=args.audio_frames, decoder_frames=args.decoder_frames)
    plan = plan_audio_components(checkpoint, output, **options)
    selected = [unit for unit in plan if
                (args.component == "all" or unit.component == args.component)
                and (not args.unit or any(fnmatch.fnmatchcase(unit.model.model_name, pattern) for pattern in args.unit))]
    if not selected:
        parser.error("Selection contains no compiler units")
    for unit in selected:
        print(f"{unit.component:12} {unit.role:12} {unit.model.model_name}", flush=True)
    print(f"Selected {len(selected)} of {len(plan)} units; precision=A_BF16_W_INT8", flush=True)
    if args.list:
        return

    output.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in ("sima-lmm", "sima-frontend", "sima-mlc", "transformers", "torch")}
    identity = {
        "files": {name: _sha256(source_path / name) for name in
                  ("config.json", "model.safetensors", "audio_detokenizer/config.json", "audio_detokenizer/model.safetensors")},
        "options": options, "precision": FileGenPrecision.A_BF16_W_INT8.value, "versions": versions,
    }
    manifest_path = output / "audio_compile_manifest.json"
    manifest = {
        "schema_version": 1, "source_path": str(source_path), "identity": identity,
        "source_config": checkpoint.config, "detokenizer_config": checkpoint.detokenizer_source().config,
        "planned_units": [unit.model.model_name for unit in plan], "units": {},
        "complete_compute_build": False, "runtime_ready": False,
        "runtime_note": "Component archive bundle; audio session/runtime integration requires separate validation.",
    }
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            parser.error("Existing build has different inputs, options, versions, or precision; use a new output directory")
        manifest = previous
    elif manifest_path.exists():
        parser.error("Output manifest exists; use --resume or a new output directory")
    _write_manifest(manifest_path, manifest)
    policy = {"precision": FileGenPrecision.A_BF16_W_INT8}
    stages = {
        "export": [FileGenMode.SOURCE_TO_FP], "quantize": [FileGenMode.FP_TO_QUANT],
        "compile": [FileGenMode.MODEL_SDK_COMPILE],
        "all": [FileGenMode.SOURCE_TO_FP, FileGenMode.FP_TO_QUANT, FileGenMode.MODEL_SDK_COMPILE],
    }[args.stage]
    for index, unit in enumerate(selected, start=1):
        model = unit.model
        signature = _graph_signature(model)
        parameters = _model_parameters(model)
        entry = manifest["units"].setdefault(model.model_name, {
            "component": unit.component, "role": unit.role, "native_config": asdict(model.cfg),
            "stages": [], "compiled": False,
        })
        unit_resume = (args.resume and entry.get("graph_signature") == signature
                       and entry.get("model_parameters") == parameters)
        _require_stage_inputs(model, entry, args.stage, unit_resume)
        if not unit_resume:
            entry.update(stages=[], compiled=False, graph_signature=signature,
                         model_parameters=parameters, native_config=asdict(model.cfg))
            manifest["complete_compute_build"] = False
            manifest["runtime_ready"] = False
        if args.resume and entry["compiled"] and model.mpk_file_name.is_file():
            if _sha256(model.mpk_file_name) != entry["archive"]["sha256"]:
                raise RuntimeError(f"Compiled artifact changed: {model.mpk_file_name}")
            if "mla_inputs" not in entry.get("interfaces", {}):
                entry["interfaces"] = _interfaces(model)
                _write_manifest(manifest_path, manifest)
            print(f"[{index}/{len(selected)}] REUSE {model.model_name}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] BUILD {model.model_name}", flush=True)
        try:
            for stage in stages:
                if shutil.disk_usage(output).free < 1024 ** 3:
                    raise RuntimeError("Less than 1 GiB disk space remains; stopped before starting another compiler stage")
                started = time.monotonic()
                # A previous interrupted compiler can leave an archive that
                # exists but has never passed archive validation.
                stage_resume = unit_resume and (stage != FileGenMode.MODEL_SDK_COMPILE or entry["compiled"])
                model.gen_files(stage, layer_cfg=policy, log_level=logging.INFO, resume=stage_resume)
                if stage == FileGenMode.FP_TO_QUANT:
                    entry["interfaces"] = _interfaces(model)
                if stage == FileGenMode.MODEL_SDK_COMPILE:
                    entry["archive"] = _archive_metadata(model.mpk_file_name)
                    entry["artifact"] = str(model.mpk_file_name.relative_to(output))
                    entry["compiled"] = True
                if stage.name not in entry["stages"]:
                    entry["stages"].append(stage.name)
                entry.pop("error", None)
                print(f"[{index}/{len(selected)}] {stage.name} {time.monotonic() - started:.1f}s", flush=True)
                _write_manifest(manifest_path, manifest)
            if entry["compiled"] and not args.keep_fp32:
                for path in (model.sdk_fp_file_name, model.sdk_fp_file_name.with_suffix(".sima.json")):
                    path.unlink(missing_ok=True)
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"
            _write_manifest(manifest_path, manifest)
            raise
        finally:
            gc.collect()
    manifest["complete_compute_build"] = all(
        manifest["units"].get(unit.model.model_name, {}).get("compiled", False) for unit in plan
    )
    _write_manifest(manifest_path, manifest)
    print(f"Manifest: {manifest_path}; complete_compute_build={manifest['complete_compute_build']}", flush=True)


if __name__ == "__main__":
    main()
