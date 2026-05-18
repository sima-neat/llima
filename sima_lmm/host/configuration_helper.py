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
import sys
from pathlib import Path
import importlib
import types
from typing import Any


from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import (
    FileGenPrecision, VisionLanguageModel, LoraGenMode, GenConfiguration, LayerConfiguration
)


# Layer part encoding.  Map from internal names to configuration file bool and enum values.
_ENCODE_LAYER_PART: dict[str, tuple[bool, str]] = {
    "group_pre": (True, "PRE"),
    "single_pre": (False, "PRE"),
    "group_cache": (True, "CACHE"),
    "single_cache": (False, "CACHE"),
    "group_sliding_cache": (True, "CACHE"),
    "single_sliding_cache": (False, "CACHE"),
    "group_post": (True, "POST"),
    "single_post": (False, "POST"),
    "group_conv": (True, "POST"),
    "single_conv": (False, "POST"),
    "conv_post_final": (False, "POST"),
    "vision": (False, "VISION"),
    "group_per_layer": (True, "PER_LAYER"),
    "single_per_layer": (False, "PER_LAYER"),
}


def _abort(message):
    """
    Terminate with a user-facing error message.
    """
    print(message, file=sys.stderr)
    sys.exit(-1)


def _import_file(module_name: str, path: str | Path) -> types.ModuleType:
    """
    Import a Python file as a module with name module_name.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_layer_id(l: LayerID) -> dict[str, Any]:
    """
    Encode a LayerID to pass as input to the configuration file.
    """
    is_group, part = _ENCODE_LAYER_PART[l.part]
    index = l.part_idx
    return {"is_group": is_group, "part": part, "index": index}


def _decode_layer_configuration(c: dict) -> LayerConfiguration | None:
    """
    Validate and decode layer configuration that was returned by the configuration file.
    Return configuration for the layer, or None to skip the layer.
    """
    if not isinstance(c, dict):
        _abort(f"Layer configuration returned a {type(c)}; expected a dict")

    c_keys = set(c.keys())
    c_keys.difference_update({"compile", "precision", "lora"})
    if c_keys:
        _abort(f"Layer configuration returned unexpected keys: {str(c_keys)}")

    compile_flag = c.get("compile", True)
    if not isinstance(compile_flag, bool):
        _abort(
            f"Layer configuration returned a {type(compile_flag)} in 'compile'; expected a bool"
        )

    precision_flag = c.get("precision", "BF16")
    if not isinstance(precision_flag, str):
        _abort(
            f"Layer configuration returned a {type(precision_flag)} in 'precision'; expected a str"
        )
    try:
        precision_value = FileGenPrecision(precision_flag)
    except ValueError:
        _abort(
            f"Layer configuration returned unrecognized value '{precision_flag}' in 'precision'"
        )

    lora_flag = c.get("lora", "LORA_DISABLED")
    if not isinstance(lora_flag, str):
        _abort(
            f"Layer configuration returned a {type(lora_flag)} in 'lora'; expected a str"
        )
    try:
        lora_value = LoraGenMode(lora_flag)
    except ValueError:
        _abort(
            f"Layer configuration returned unrecognized value '{lora_flag}' in 'lora'"
        )

    if not compile_flag:
        return None

    return {"precision": precision_value, "lora": lora_value}


def _fetch_configuration(
    configuration_path: str, num_hidden_layers: int, layer_ids: list[LayerID]
) -> GenConfiguration:
    """
    Read configuration from the file at configuration_path.
    Returns the configuration as a mapping from LayerID to FileGenPrecision for
    layers to be compiled.
    """
    config_module = _import_file("private__configuration_file", configuration_path)

    model_properties = {"num_hidden_layers": num_hidden_layers}
    precision_configuration = {}
    lora_configuration = {}
    for layer_id in layer_ids:
        encoded_layer_id = _encode_layer_id(layer_id)
        try:
            encoded_layer_configuration = config_module.get_layer_configuration(
                model_properties, encoded_layer_id
            )
        except Exception as e:
            _abort(
                "The following error occurred while getting layer configuration\n"
                "from " + configuration_path + ":\n" +
                str(e)
            )
        layer_configuration = _decode_layer_configuration(encoded_layer_configuration)
        if layer_configuration is not None:
            precision_configuration[layer_id] = layer_configuration["precision"]
            lora_configuration[layer_id] = layer_configuration["lora"]

    return {"precision": precision_configuration, "lora": lora_configuration}


def read_configuration_file(
    model: VisionLanguageModel, configuration_path: Path
) -> GenConfiguration:
    """
    Read parameters for each layer from a configuration file.
    """
    num_hidden_layers = model.cfg.lm_cfg.num_hidden_layers
    layer_ids = model.cfg.get_layer_ids()
    configurations = _fetch_configuration(configuration_path, num_hidden_layers, layer_ids)
    return configurations


def default_configuration(
    model: VisionLanguageModel
) -> GenConfiguration:
    layer_ids = model.cfg.get_layer_ids()
    precision_dict = {l: FileGenPrecision.BF16 for l in layer_ids}
    lora_dict = {l: LoraGenMode.LORA_DISABLED for l in layer_ids}
    return {"precision": precision_dict, "lora": lora_dict}


def _get_lora_layer_ids(num_layers: int, adapter_is_multimodal: bool = False) -> list[LayerID]:
    """
    Get IDs of all adapter layers specified by the input number of layers.
    """
    layers = []
    layers.extend(LayerID("group_pre", n) for n in range(num_layers))
    layers.extend(LayerID("group_post", n) for n in range(num_layers - 1))
    layers.extend(LayerID("single_pre", n) for n in range(num_layers))
    layers.extend(LayerID("single_post", n) for n in range(num_layers))
    if adapter_is_multimodal:
        layers.extend([LayerID("vision", 0)])
    return layers


def default_configuration_lora(
    num_layers: int, is_multimodal: bool = False
) -> GenConfiguration:
    layer_ids = _get_lora_layer_ids(num_layers, is_multimodal)
    precision_dict = {l: FileGenPrecision.BF16 for l in layer_ids}
    lora_dict = {l: LoraGenMode.LORA_BRANCH for l in layer_ids}
    return {"precision": precision_dict, "lora": lora_dict}


def read_configuration_file_lora(
    num_layers: int, configuration_path: Path
) -> GenConfiguration:
    """
    Read parameters for each layer from a configuration file.
    """
    layer_ids = _get_lora_layer_ids(num_layers)
    configurations = _fetch_configuration(configuration_path, num_layers, layer_ids)
    return configurations
