#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import argparse
import logging
from pathlib import Path
import json
import re

import numpy as np
import safetensors.numpy

from ml_kernels.types import bfloat16
from afe.ir.quantization_conv import block_quantize_weight_tensor
from afe.backends.mla.afe_to_n2a_compiler import n2a_compiler_utils
from mlc.compiler.arch_constants import ARCH_CONSTS
from mlc.kernel.conv2d_helper import pack_filter_4bit_v2
from mlc.compiler.data_type import DataType
from mlc.compiler.numpy_helpers import convert_tensor_to_mla
from mlc.compiler.tensor_format import TensorShape, TensorType

from sima_lmm.config.vlm_config import ModelFormat, model_file_type
from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenPrecision, sima_builder, LoraGenMode
from sima_lmm.hf.hf_transformer import find_file, LocalHuggingFaceModel
from sima_utils.logging.sima_logger import ScopedLogLevel, sima_log_dbg, sima_log_info
from sima_lmm.host.configuration_helper import (
    default_configuration_lora, read_configuration_file_lora
)


_ITEMSIZE = {
    FileGenPrecision.BF16: 2,
    FileGenPrecision.A_BF16_W_INT8: 1,
    FileGenPrecision.A_BF16_W_INT4: 0.5
}

_MODEL_ID_PATTERN = r"_(n\d+)_(pre|post)_layer(\d+)_"

_LOGGING_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}


def _find_lora_adapter_layers(tensor_dict: dict[str, np.ndarray]):
    max_layer_idx = -1
    for tensor_name in tensor_dict:
        res = re.search(r".layers.(\d+).", tensor_name)
        assert res and len(res.groups()) == 1
        idx = int(res.groups()[0])
        max_layer_idx = idx if idx > max_layer_idx else max_layer_idx
    return max_layer_idx + 1


def dense_matrix_to_conv_weight(m: np.ndarray) -> np.ndarray:
    assert len(m.shape) == 2
    return sima_builder.layout_array(m, "oi", "hwigo")


def get_lora_merged_weight_tensor(
    base_name: str, base_w: np.ndarray, adapter: dict[str, np.ndarray], scale: float
) -> np.ndarray:
    """
    Merge LoRA adapter to the base model according to PEFT:
    https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L1009

    w_merged = fp32(w_base) + (fp32(lora_B) @ fp32(lora_B)) * lora_scale
    """
    lora_base_name = sima_builder.derive_lora_name_from_base_model(base_name)
    lora_A_name = lora_base_name.replace(".weight", ".lora_A.weight")
    lora_B_name = lora_base_name.replace(".weight", ".lora_B.weight")
    w_A = adapter[lora_A_name].astype(np.float32)
    w_B = adapter[lora_B_name].astype(np.float32)
    lora_w = np.matmul(w_B,  w_A) * scale
    assert lora_w.shape == base_w.shape, (
        f"Cannot merge lora due to shape mismatch: "
        f"Expected {base_w.shape}, but the merged shape is {lora_w.shape}"
    )
    w_merged = base_w.astype(np.float32) + lora_w
    return w_merged.astype(base_w.dtype).astype(np.float32)


def quantize_lora_weight(
    weight: np.ndarray, precision: FileGenPrecision
) -> tuple[np.ndarray, list[np.float32] | np.float32]:
    """
    Quantize one LoRA weight.
    """
    is_bfloat16 = False
    match precision:
        case FileGenPrecision.A_BF16_W_INT4:
            nbits = 4
            c_block_size = 256
        case FileGenPrecision.A_BF16_W_INT8:
            nbits = 8
            # Currently not block quantized.
            c_block_size = None
        case FileGenPrecision.BF16:
            is_bfloat16 = True

    if is_bfloat16:
        quantized_weight = weight.astype(bfloat16)
        scales = None
    else:
        # Bfloat16 convolution with int8 or int4 weights
        quantized_weight, scales = block_quantize_weight_tensor(
            weights=weight, per_channel=True, bits=nbits, c_block_size=c_block_size
        )

    return quantized_weight, scales


def populate_numpy(
    numpy_value: np.ndarray, entries: list, qw: np.ndarray, precision: FileGenPrecision,
    is_scale: bool = False
):
    """
    Populate appropriate area in the numpy array using the provided DRAM entries.
    Args:
        numpy_value: Numpy value to populate.
        entries: List of entries from layer's weight mapping.
        qw: Quantized weight/scale which will be split and stored into numpy_value based on entries.
        precision: Layer precision.
        is_scale: Boolen whether value is a scale.
    Returns:
        The original numpy value is modified.
    """
    # Convert to compiler layout.
    if is_scale:
        qw = np.expand_dims(qw, axis=tuple(range(5 - qw.ndim)))
        qw = qw.astype(bfloat16)
    else:
        qw = n2a_compiler_utils.to_conv2d_weights_layout(qw)

    for entry in entries:
        # Collect channels.
        in_0, in_1, out_0, out_1 = entry["channel_range"]

        # Slice out from quantized weight and convert to MLA layout.
        weight_slice = qw[0, :, :, in_0:in_1, out_0:out_1]

        match precision:
            case FileGenPrecision.A_BF16_W_INT4:
                tensor_shape = TensorShape(TensorType.FILTER, DataType.INT8, weight_slice.shape)
                weight_slice = pack_filter_4bit_v2(tensor_shape, weight_slice)
                tensor_shape = TensorShape(TensorType.MATRIX, DataType.INT8, weight_slice.shape)
                weight_slice_mla = convert_tensor_to_mla(tensor_shape, weight_slice)[0]
            case FileGenPrecision.A_BF16_W_INT8:
                tensor_shape = TensorShape(TensorType.FILTER, DataType.INT8, weight_slice.shape)
                weight_slice_mla = convert_tensor_to_mla(tensor_shape, weight_slice)[0]
            case FileGenPrecision.BF16:
                tensor_shape = TensorShape(TensorType.FILTER, DataType.BF16_1, weight_slice.shape)
                weight_slice_mla = convert_tensor_to_mla(tensor_shape, weight_slice)[0]

        # Collect DRAM start and size.
        dram_start = entry["dram_start"] // ARCH_CONSTS.num_bytes_per_row
        dram_size = entry["dram_size"] // ARCH_CONSTS.num_bytes_per_row

        assert (numpy_value[dram_start:dram_start + dram_size, :] == 0).all(), (
            "DRAM space already populated"
        )

        # Populate appropriate area in numpy area.
        numpy_value[dram_start:dram_start + dram_size, :] = weight_slice_mla


def save_numpy(
    numpy_value: np.ndarray, output_path: Path, weight_map_path: str, section_name: str,
    is_base: bool = False
):
    """
    Save numpy value to output path using the weight map name and the section name. There will be a
    numpy file per section per model.
    Args:
        numpy_value: Numpy value to populate.
        output_path: Base output path.
        weight_map_path: The same name from the this file is used to store the numpy file.
        section_name: Filter/Scale section name.
        is_base: Whether a base weight is being saved.
    Returns:
        None. Generated numpy files are written to disk.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    weight_map_path = Path(weight_map_path)
    new_suffix = f".{section_name}{'_base_weight' if is_base else ''}.npy"
    numpy_path = str((output_path / weight_map_path.name).with_suffix(new_suffix))
    np.save(numpy_path, numpy_value)
    sima_log_info(f"Saved numpy {numpy_path}")


def process_bundled_weights(
    tensors: dict[str, np.ndarray], weight_shapes: dict[str, tuple[int]]
):
    """
    Process bundled weights in LoRA adapter. Currently, these are bundled names we know:
        "qkv_proj", "gate_up_proj". When a LoRA model is created by Model SDK, all bundled
        weights are split in graph generation and set to zero. Hence, when the adapter weights
        are compiled, all bundled weights must be split and compiled separately.

    This function modifies the `tensors` dict in-place by deleting bundled items
        and adding split items.

    Args:
        tensors: The dict of all LoRA weights stored in the LoRA adapter.
        weight_shapes: The dict of weight shapes of all relocatable weights derived from
            the reloc map produced by the compiler for the LoRA model. It contains the target
            shapes to split a bundled weight.

    Returns:
        None.
    """
    known_bundle_names = ["qkv_proj", "gate_up_proj"]
    all_names = list(tensors.keys())
    for name in all_names:
        for bundle in known_bundle_names:
            if bundle in name:
                w = tensors[name]
                match bundle:
                    case "qkv_proj" if "lora_A" in name:
                        splits = {
                            name.replace("qkv_proj", "q_proj"): w,
                            name.replace("qkv_proj", "k_proj"): w,
                            name.replace("qkv_proj", "v_proj"): w,
                        }
                    case "qkv_proj" if "lora_B" in name:
                        # TODO: Simplify split logic once all weight map files are available.
                        q_name = name.replace("qkv_proj", "q_proj")
                        k_name = name.replace("qkv_proj", "k_proj")
                        v_name = name.replace("qkv_proj", "v_proj")
                        q_key = next(
                            (layer for layer in weight_shapes if layer in q_name), None
                        )
                        if q_key:
                            q_size = weight_shapes[q_key][0]
                            k_key = next(
                                (layer for layer in weight_shapes if layer in k_name), None
                            )
                            k_size = weight_shapes[k_key][0]
                            splits = {
                                q_name: w[:q_size],
                                k_name: w[q_size:q_size + k_size],
                                v_name: w[q_size + k_size:]
                            }
                        else:
                            split_size = w.shape[0] // 3
                            splits = {
                                q_name: w[:split_size],
                                k_name.replace("bundle", "k_proj"): w[split_size:2 * split_size],
                                v_name.replace("bundle", "v_proj"): w[2 * split_size:]
                            }
                    case "gate_up_proj" if "lora_A" in name:
                        splits = {
                            name.replace("gate_up_proj", "gate_proj"): w,
                            name.replace("gate_up_proj", "up_proj"): w
                        }
                    case "gate_up_proj" if "lora_B" in name:
                        # TODO: Simplify split logic once all weight map files are available.
                        gate_name = name.replace("gate_up_proj", "gate_proj")
                        up_name = name.replace("gate_up_proj", "up_proj")
                        gate_key = next(
                            (layer for layer in weight_shapes if layer in gate_name), None
                        )
                        if gate_key:
                            gate_size = weight_shapes[gate_key][0]
                            splits = {
                                gate_name: w[:gate_size],
                                up_name: w[gate_size:]
                            }
                        else:
                            split_size = w.shape[0] // 2
                            splits = {
                                gate_name: w[:split_size],
                                up_name: w[split_size:]
                            }
                tensors.update(splits)
                del tensors[name]
                sima_log_dbg(f"Weight {name} is split into {splits.keys()} {w.shape}")


def get_reloc_weight_shapes(reloc_file_paths: str) -> dict[str, tuple[int, int]]:
    """
    Collect shapes of all quantized weights that are relocatable.

    Reloc maps are generated by the compiler when the LoRA model is compiled.
    All weight tensors declared relocatable will be stored, along with their scales,
    in the reloc map file.

    Args:
        reloc_file_paths: The list of reloc map files.
    Returns:
        A dict mapping from weight name to weight shape.
    """
    weight_shapes = {}
    for map_path in reloc_file_paths:
        with open(map_path, "r") as f:
            weight_map = json.load(f)

        for layer in weight_map:
            # Skip scales accompanying quantized weights.
            if "scale." in layer:
                continue

            # Only need to derive once for single and group map files.
            if layer in weight_shapes:
                continue

            entry_list = weight_map[layer]
            max_in, max_out = -1, -1
            for entry in entry_list:
                max_in = max(max_in, entry["channel_range"][1])
                max_out = max(max_out, entry["channel_range"][3])
            weight_shapes[layer] = (max_out, max_in)
    return weight_shapes


def compile_lora_adapter(
    base_path: Path, lora_path: Path, configuration_path: Path | None,
    output_path: Path, weight_map_path: Path
):
    """
    Top level API to compile all weights in a LoRA adapter.

    Args:
        base_path: The path to the directory of the base model.
        lora_path: The path to the directory of the LoRA adapter.
        configuration_path: The path to the configuration file used to compile the base model.
            It contains model and/or layer specific settings for quantization precision and
            lora graph generation mode.
        output_path: The path to the output directory.
        weight_map_path: Path to the compiler maps generated for the LoRA adapter.

    Returns:
        None. Generated numpy files are written to disk.
    """
    weight_map_paths = [str(p) for p in weight_map_path.rglob("*.json")]
    model_format = model_file_type(base_path)
    assert model_format == ModelFormat.FORMAT_HF
    base_model = LocalHuggingFaceModel.create_from_directory(directory=base_path)
    base_model.load_all_params()

    lora_config_file = find_file(lora_path, "adapter_config.json")
    with lora_config_file.open("r") as fp:
        lora_cfg = json.load(fp=fp)

    lora_rank = lora_cfg.get("r")
    lora_alpha = lora_cfg.get("lora_alpha")
    b_scale = lora_alpha / lora_rank
    sima_log_info(f"LoRA rank = {lora_rank}, alpha = {lora_alpha}, b_scale = {b_scale}")
    
    lora_weight_file = find_file(lora_path, "adapter_model.safetensors")
    adapter_dict = safetensors.numpy.load_file(lora_weight_file)

    num_adapter_layers = _find_lora_adapter_layers(adapter_dict)
    has_vision_adapter = False
    if configuration_path is None:
        gen_cfg = default_configuration_lora(num_adapter_layers, has_vision_adapter)
    else:
        gen_cfg = read_configuration_file_lora(num_adapter_layers, configuration_path)

    precision = gen_cfg["precision"]
    lora_mode = gen_cfg["lora"]

    # Derive all weight shapes from the reloc mapping files.
    weight_shapes = get_reloc_weight_shapes(weight_map_paths)

    # Split bundled adapter weights, if found, using derived shapes.
    process_bundled_weights(adapter_dict, weight_shapes)

    # Loop through reloc map files - quantize weights and store to a dictionary.
    for weight_map_path in weight_map_paths:

        # Extract model segment information and look up for quantization precision.
        res = re.search(_MODEL_ID_PATTERN, weight_map_path)
        assert res and len(res.groups()) == 3
        group, segment, part_idx = res.groups()
        part = "single" if group == "n1" else "group"
        layer_id = LayerID(f"{part}_{segment}", int(part_idx))
        curr_precision = precision[layer_id]
        curr_lora_mode = lora_mode[layer_id]

        with open(weight_map_path, "r") as f:
            weight_map = json.load(f)

        def _create_reloc_buffer(key: str) -> tuple[str, np.ndarray]:
            """
            Find the maximum DRAM size to create an array.
            Ensure filter section names are the same.
            """
            max_dram = -1
            section_name = None
            entry_list = weight_map[key]

            for entry in entry_list:
                max_dram = max(max_dram, entry["dram_start"] + entry["dram_size"])

                if section_name:
                    assert section_name == entry["section"], "Section name mismatch"
                else:
                    section_name = entry["section"]

            # Create numpy value that represents LoRA filter section based on MLA layout.
            assert max_dram % ARCH_CONSTS.num_bytes_per_row == 0, (
                f"Total weight reloc DRAM size must be a multiple of"
                f" {ARCH_CONSTS.num_bytes_per_row}"
            )
            return section_name, np.zeros(
                (max_dram // ARCH_CONSTS.num_bytes_per_row, ARCH_CONSTS.num_bytes_per_row),
                dtype=np.int8
            )

        for key in weight_map:
            if "scale." in key:
                continue

            w_name = key
            s_name = f"scale.{key}"
            w_section_name, numpy_w = _create_reloc_buffer(w_name)
            s_section_name, numpy_s = _create_reloc_buffer(s_name)

            # LoRA weights are set to int8 for int4 models because of accuracy loss for branched lora.
            if (
                curr_lora_mode == LoraGenMode.LORA_BRANCH and
                curr_precision == FileGenPrecision.A_BF16_W_INT4
            ):
                curr_precision = FileGenPrecision.A_BF16_W_INT8

            weight_size = weight_shapes[w_name][0] * weight_shapes[w_name][1] * _ITEMSIZE[curr_precision]
            assert numpy_w.nbytes == weight_size, (
                f"Size of numpy does not match weight size for layer {w_name}"
            )

            # Because both branched lora and merged lora are supported, the keys in the weight_map
            # may or may not contain "lora". When "lora" is not present in the key, it means that
            # the lora adapter is merged to the base model; hence, the original weight from the base
            # model is retrieved and added with the adapter weight to get the merged weight.
            if ".lora_" in w_name:
                assert curr_lora_mode == LoraGenMode.LORA_BRANCH
                w = adapter_dict[w_name]
                # Apply lora_alpha scaling to B matrix.
                if ".lora_B." in w_name:
                    w = b_scale * w
            else:
                assert curr_lora_mode == LoraGenMode.LORA_MERGED
                base_w = base_model.load_np_param(w_name)
                w = get_lora_merged_weight_tensor(w_name, base_w, adapter_dict, b_scale)

            sima_log_dbg(f"Reloc weight = {w_name} {w.shape}, lora mode = {curr_lora_mode}")

            conv_weight = dense_matrix_to_conv_weight(w)
            qw, qs = quantize_lora_weight(conv_weight, curr_precision)
            sima_log_dbg(
                f"--Quantized as {curr_precision.value}: weight shape = {qw.shape},"
                f" scale shape = {qs.shape if qs is not None else None}"
            )

            populate_numpy(numpy_w, weight_map[w_name], qw, curr_precision)
            populate_numpy(numpy_s, weight_map[s_name], qs, FileGenPrecision.BF16, is_scale=True)

            # TODO: produce only one numpy if filters are shared between group and single.
            save_numpy(numpy_w, output_path, weight_map_path, w_section_name)
            save_numpy(numpy_s, output_path, weight_map_path, s_section_name)
            if curr_lora_mode == LoraGenMode.LORA_MERGED:
                conv_base_weight = dense_matrix_to_conv_weight(base_w.astype(np.float32))
                base_qw, base_qs = quantize_lora_weight(conv_base_weight, curr_precision)
                base_numpy_w = np.zeros_like(numpy_w)
                base_numpy_s = np.zeros_like(numpy_s)
                populate_numpy(base_numpy_w, weight_map[w_name], base_qw, curr_precision)
                populate_numpy(base_numpy_s, weight_map[s_name], base_qs, FileGenPrecision.BF16, is_scale=True)
                save_numpy(base_numpy_w, output_path, weight_map_path, w_section_name, True)
                save_numpy(base_numpy_s, output_path, weight_map_path, s_section_name, True)


def main():
    parser = argparse.ArgumentParser(description="LoRA Adapter compiler")
    parser.add_argument(
        "base_path", type=Path,
        help="Path of base model directory"
    )
    parser.add_argument(
        "lora_path", type=Path,
        help="Path of LoRA directory"
    )
    parser.add_argument(
        "-w", "--weight_map_path", required=True, type=Path,
        help="Path to folder with weight maps produced after compiling the base model"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Directory to write output files into (default: lora_path)"
    )
    parser.add_argument(
        "--log_level", default="WARNING",
        help="Set the logging verbosity level (default: WARNING)"
    )
    parser.add_argument(
        "-c", "--configuration_file", type=Path,
        help="Configuration file with layer-specific quantization options"
    )

    args = parser.parse_args()

    if args.output is None:
        output_path = Path(args.lora_path.name)
    else:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

    with ScopedLogLevel(_LOGGING_LEVELS[args.log_level]):
        compile_lora_adapter(
            args.base_path, args.lora_path, args.configuration_file, output_path,
            Path(args.weight_map_path)
        )


if __name__ == "__main__":
    main()
