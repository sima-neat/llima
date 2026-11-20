"""Generate streamed-MoE expert weight files for the relocatable expert template.

The compiler emits a reloc map for the expert template (``*_moe_expert_template*.json``):
for every relocatable weight and its scale it lists the reloc section, and the DRAM rows each
(input, output) channel range occupies inside that section. Every routed expert has the same
shapes as the template, so each expert's weights are quantized exactly like the template's
(the LoRA reloc path: ``quantize_lora_weight`` + ``populate_numpy``) and written into the same
section layout.

The compiler tiles the template differently per token width, so each width (decode n1,
prefill n128) gets its own record set.

Output (``<out>/n<N>/`` per token width):
    layer_<L>.bin   num_experts records of ``record_size`` bytes; record e is expert e. Each
                    record holds the reloc sections in reloc order, each at a 4 KiB aligned
                    offset, so the runtime can read a record straight into an MLA buffer and bind
                    its sections to the template's reloc ports.
    index.json      record size, section names/offsets/sizes in reloc order, the template's
                    weight names per section, and the source checkpoint.

Reloc order is the sorted section-name order, which is the order MLA-RT expects reloc ports in
(see ``MLAModelWithBuffer::update_reloc``).
"""

import argparse
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlc.compiler.arch_constants import ARCH_CONSTS

from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.host.compile_lora_adapter import (
    dense_matrix_to_conv_weight,
    populate_numpy,
    quantize_lora_weight,
)
from sima_lmm.model import FileGenPrecision
from sima_lmm.model.language_moe_streamed_model import (
    EXPERT_PROJECTIONS,
    TEMPLATE_EXPERT_IDX,
    TEMPLATE_LAYER_IDX,
    expert_param_base,
)

# Record sections start on 4 KiB boundaries: O_DIRECT reads and DMA-BUF offsets both need it.
SECTION_ALIGNMENT = 4096


@dataclass(frozen=True)
class SectionSpec:
    """One reloc section of the expert template."""

    name: str
    size: int            # bytes, from the reloc map's DRAM extent
    offset: int          # byte offset inside an expert record
    key: str             # reloc map key (weight name, or "scale." + weight name)
    projection: str      # gate_proj / up_proj / down_proj
    is_scale: bool


def find_template_map(compile_out: Path, num_tokens: int) -> Path:
    """Reloc map the compiler wrote for the expert template of one token width."""
    maps = sorted(compile_out.rglob(f"*_n{num_tokens}_moe_expert_template_stage1_mla_reloc.json"))
    if len(maps) != 1:
        raise FileNotFoundError(
            f"Expected one n{num_tokens} expert-template reloc map under {compile_out}, got {maps}"
        )
    return maps[0]


def _section_extents(weight_map: dict) -> dict[str, tuple[str, int]]:
    """Map each reloc key to (section name, byte size)."""
    extents = {}
    for key, entries in weight_map.items():
        sections = {e["section"] for e in entries}
        if len(sections) != 1:
            raise ValueError(f"{key}: expected one reloc section, got {sorted(sections)}")
        size = max(e["dram_start"] + e["dram_size"] for e in entries)
        if size % ARCH_CONSTS.num_bytes_per_row:
            raise ValueError(f"{key}: reloc size {size} is not a whole number of MLA rows")
        extents[key] = (sections.pop(), size)
    return extents


def layout_from_map(weight_map: dict, template_base: str) -> list[SectionSpec]:
    """Record layout: sections in reloc (sorted name) order, each 4 KiB aligned."""
    extents = _section_extents(weight_map)
    specs = []
    offset = 0
    for key, (section, size) in sorted(extents.items(), key=lambda kv: kv[1][0]):
        weight_name = key.removeprefix("scale.")
        projection = weight_name.removeprefix(f"{template_base}.").removesuffix(".weight")
        if projection not in EXPERT_PROJECTIONS:
            raise ValueError(f"Unexpected relocatable weight {weight_name} in the expert template")
        specs.append(SectionSpec(section, size, offset, key, projection, key.startswith("scale.")))
        offset += -(-size // SECTION_ALIGNMENT) * SECTION_ALIGNMENT
    return specs


def record_size(specs: list[SectionSpec]) -> int:
    last = specs[-1]
    return last.offset + -(-last.size // SECTION_ALIGNMENT) * SECTION_ALIGNMENT


def build_record(
    weights: dict[str, np.ndarray], weight_map: dict, specs: list[SectionSpec],
    precision: FileGenPrecision,
) -> np.ndarray:
    """One expert's record: every reloc section populated in the template's MLA layout."""
    record = np.zeros(record_size(specs), dtype=np.int8)
    by_projection = {}
    for spec in specs:
        by_projection.setdefault(spec.projection, {})[spec.is_scale] = spec
    for projection, pair in by_projection.items():
        qw, qs = quantize_lora_weight(dense_matrix_to_conv_weight(weights[projection]), precision)
        for is_scale, value, value_precision in (
            (False, qw, precision), (True, qs, FileGenPrecision.BF16)
        ):
            spec = pair[is_scale]
            rows = np.zeros(
                (spec.size // ARCH_CONSTS.num_bytes_per_row, ARCH_CONSTS.num_bytes_per_row),
                dtype=np.int8,
            )
            populate_numpy(rows, weight_map[spec.key], value, value_precision, is_scale=is_scale)
            record[spec.offset:spec.offset + spec.size] = rows.reshape(-1)
    return record


def _expert_weights(
    hf: LocalHuggingFaceModel, base_name: str, layer: int, expert: int
) -> dict[str, np.ndarray]:
    prefix = expert_param_base(base_name, layer, expert)
    weights = {}
    for projection in EXPERT_PROJECTIONS:
        # load_np_param also slices fused expert tensors (Qwen3.5-MoE).
        weights[projection] = hf.load_np_param(f"{prefix}.{projection}.weight").astype(np.float32)
    return weights


def _write_layer(args) -> tuple[int, int]:
    model_dir, out_dir, layer, num_experts, weight_map, specs, precision = args
    hf = LocalHuggingFaceModel.create_from_directory(model_dir)
    base_name = hf.language_model_param_base_name
    path = Path(out_dir) / f"layer_{layer}.bin"
    tmp = path.with_suffix(".bin.partial")
    with open(tmp, "wb") as f:
        for expert in range(num_experts):
            weights = _expert_weights(hf, base_name, layer, expert)
            f.write(build_record(weights, weight_map, specs, precision).tobytes())
    tmp.replace(path)
    return layer, path.stat().st_size


def generate(
    model_dir: Path, compile_out: Path, out_dir: Path, precision: FileGenPrecision,
    jobs: int, num_tokens: int, layers: list[int] | None = None,
) -> dict:
    """Write one token width's layer files and index.json into out_dir; return the index."""
    hf = LocalHuggingFaceModel.create_from_directory(model_dir)
    cfg = hf.config
    num_layers, num_experts = cfg["num_hidden_layers"], cfg["num_experts"]
    base_name = hf.language_model_param_base_name
    template_base = expert_param_base(base_name, TEMPLATE_LAYER_IDX, TEMPLATE_EXPERT_IDX)

    map_path = find_template_map(compile_out, num_tokens)
    weight_map = json.loads(map_path.read_text())
    specs = layout_from_map(weight_map, template_base)
    size = record_size(specs)

    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(num_layers)) if layers is None else layers
    work = [(str(model_dir), str(out_dir), L, num_experts, weight_map, specs, precision) for L in layers]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        for layer, nbytes in pool.imap_unordered(_write_layer, work):
            assert nbytes == size * num_experts, (layer, nbytes)
            print(f"layer {layer}: {nbytes / 2**20:.1f} MiB", flush=True)

    index = {
        "format": "sima-llima-expert-records/1",
        "source_model": str(model_dir),
        "precision": precision.value,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "record_size": size,
        "section_alignment": SECTION_ALIGNMENT,
        "template_map": map_path.name,
        "template_weight_base": template_base,
        "sections": [
            {"name": s.name, "offset": s.offset, "size": s.size, "key": s.key,
             "projection": s.projection, "is_scale": s.is_scale}
            for s in specs
        ],
        "layer_files": {str(L): f"layer_{L}.bin" for L in layers},
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_path", type=Path, help="HF checkpoint directory")
    parser.add_argument("compile_out", type=Path, help="llima-compile output with the templates")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Expert record directory; one n<N> subdirectory per token width")
    parser.add_argument("--num_tokens", type=int, nargs="+", default=[1, 128],
                        help="Token widths of the compiled expert templates")
    parser.add_argument("--precision", default=FileGenPrecision.A_BF16_W_INT4.value,
                        choices=[FileGenPrecision.A_BF16_W_INT4.value, FileGenPrecision.A_BF16_W_INT8.value])
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("--layers", type=int, nargs="*", help="Only these layers (default: all)")
    args = parser.parse_args()
    for num_tokens in args.num_tokens:
        generate(args.model_path, args.compile_out, args.output / f"n{num_tokens}",
                 FileGenPrecision(args.precision), args.jobs, num_tokens, args.layers)


if __name__ == "__main__":
    main()
