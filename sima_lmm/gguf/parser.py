#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
import struct
import math
from enum import IntEnum
from typing import BinaryIO
from dataclasses import dataclass

import numpy as np

from sima_lmm.gguf.ggml_quant import (
    SUPPORTED_GGML_NAMES, SUPPORTED_GGML_TYPES, unpack_base_quant
)


"""
This file contains infrastructure for parsing a GGUF file.  This interface
has been DEPRECATED and py-gguf module is used for getting the information
from GGUF file format.
"""

class MetadataValueType(IntEnum):
    """GGUF metadata value type.
    """
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12
    FLOAT16 = 13
    BFLOAT16 = 14


@dataclass(frozen=True)
class GgufParamFormat:
    """
    GGUF parameter format: data type, pack string, and number of bytes.
    """
    datatype: MetadataValueType
    pack_fmt: str
    num_bytes: int


METADATA_TYPES: dict[str, GgufParamFormat] = {
    "uint8": GgufParamFormat(MetadataValueType.UINT8, "B", 1),
    "int8": GgufParamFormat(MetadataValueType.INT8, "b", 1),
    "uint16": GgufParamFormat(MetadataValueType.UINT16, "H", 2),
    "int16": GgufParamFormat(MetadataValueType.INT16, "h", 2),
    "uint32": GgufParamFormat(MetadataValueType.UINT32, "I", 4),
    "int32": GgufParamFormat(MetadataValueType.INT32, "i", 4),
    "float32": GgufParamFormat(MetadataValueType.FLOAT32, "f", 4),
    "bool": GgufParamFormat(MetadataValueType.BOOL, "?", 1),
    "string": GgufParamFormat(MetadataValueType.STRING, "Q", 8),
    "array": GgufParamFormat(MetadataValueType.ARRAY, "IQ", 4+8),
    "uint64": GgufParamFormat(MetadataValueType.UINT64, "Q", 8),
    "int64": GgufParamFormat(MetadataValueType.INT64, "q", 8),
    "float64": GgufParamFormat(MetadataValueType.FLOAT64, "d", 8),
    "float16": GgufParamFormat(MetadataValueType.FLOAT16, "H", 2),
    "bfloat16": GgufParamFormat(MetadataValueType.BFLOAT16, "H", 2),
}

METADATA_TYPE_NAMES = {v.datatype: name for name, v in METADATA_TYPES.items()}


def _read_value(f: BinaryIO, data_type: int, endian: str = "<") -> int | float | bool | list | str:
    """Retrieve data value by data type from GGUF file.

    Args:
        f: The binary I/O stream opened for the GGUF file.
        data_type: Integer value of data type defined by GGUF.
        endian: Endianness. Default is "<" for little-endian.

    Returns:
        Retrieved data value.
    """
    assert data_type in METADATA_TYPE_NAMES.keys(), \
        f"Metadata value type {data_type} not supported."
    type_name = METADATA_TYPE_NAMES[data_type]
    t = METADATA_TYPES[type_name]
    fmt = t.pack_fmt
    nbytes = t.num_bytes

    if type_name == "string":
        length = struct.unpack(endian + fmt, f.read(nbytes))[0]
        return f.read(length).decode("utf-8")
    elif type_name == "array":
        data_type, count = struct.unpack(endian + fmt, f.read(nbytes))
        return [_read_value(f, data_type) for _ in range(count)]
    else:
        return struct.unpack(endian + fmt, f.read(nbytes))[0]


def parse_gguf(f: BinaryIO) -> tuple[dict, dict]:
    """Get metadata and tensor info from a GGUF file stream.

    Note that weights within a GGUF file are stored using row-major order,
    but dimensions in tensor info are stored in the order of columns, then rows.

    Args:
        f: The opened file stream of a GGUF file.

    Returns:
        Matadata dict and tensor info dict.
    """
    f.seek(0)
    assert f.read(4) == b"GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(4+8+8))
    assert version == 3, f"GGUF Version {version} not supported."

    params = {}
    for _ in range(n_kv):
        name = _read_value(f, METADATA_TYPES["string"].datatype)
        data_type = struct.unpack("<I", f.read(4))[0]
        params[name] = _read_value(f, data_type)

    tensorinfo = {}
    for _ in range(n_tensors):
        name = _read_value(f, METADATA_TYPES["string"].datatype)
        shape_len = _read_value(f, METADATA_TYPES["uint32"].datatype)
        shape = [_read_value(f, METADATA_TYPES["uint64"].datatype) for _ in range(shape_len)]
        ggml_type = _read_value(f, METADATA_TYPES["uint32"].datatype)
        rel_offset = _read_value(f, METADATA_TYPES["uint64"].datatype)

        tensorinfo[name] = {
            "ggml_type": ggml_type,
            "shape": shape[::-1],  # reverse to (rows, cols) order
            "rel_offset": rel_offset,
        }

    # The offset defined in tensor info is relative to the end of the header and is unaligned.
    # Compute the absolute file offset with alignment (default 32).
    start = f.tell()
    alignment = params.get("general.alignment", 32)
    for t in tensorinfo.values():
        offset = start + t["rel_offset"]
        offset += (-offset) % alignment
        t["offset"] = offset

    return params, tensorinfo


def load_tensor(
    f: BinaryIO, offset: int, ggml_type: int, shape: list
) -> tuple[np.ndarray | None, np.ndarray]:
    """Load a tensor from a GGUF file stream.

    Args:
        f: The opened file stream of a GGUF file.
        offset: The offset of the requested tensor within the file.
        ggml_type: The integer quantization type of the tensor.
        shape: The shape of the tensor.
    Returns:
        Scales, None if not present, and quantized tensor values.
    """
    assert ggml_type in SUPPORTED_GGML_NAMES, f"GGML quantization type {ggml_type} not implemented."
    ggml_name = SUPPORTED_GGML_NAMES[ggml_type]
    qconfig = SUPPORTED_GGML_TYPES[ggml_name]
    block_size = qconfig.blk_buffer_size
    elements_per_block = qconfig.n_elements
    assert shape[-1] % elements_per_block == 0

    f.seek(offset)
    num_elements = math.prod(shape)
    size = num_elements * block_size // elements_per_block
    data = f.read(size)
    scales, values = unpack_base_quant(data, ggml_name)

    values = values.reshape(shape)

    return scales, values
