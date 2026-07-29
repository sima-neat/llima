# LLiMa Compilation Configuration

## Pre-Quantized Inputs

Do not apply a generic precision configuration to a published pre-quantized
checkpoint or to the output of its matching repository-specific
`quantize.py`. The compressed checkpoint's encoded precision and
`quantization_config` are authoritative.

A configuration may select compiler units from a pre-quantized input for
debugging through its `compile` decisions. This selection does not override
the checkpoint's INT4/INT8 weights. Do not claim that a `precision` return
value requantizes those weights, replaces model-specific targets, or
reconstructs the repository's llm-compressor recipe.

The precision examples below are for an explicitly approved FP/BF16 fallback.
For a pre-quantized input, follow the installed LLiMa release when selecting
units and verify that the emitted unit list preserves the checkpoint's encoded
weight quantization. A partial compiler-debug output is normally undeployable.

## FP/BF16 Fallback

Use trusted Python configuration files to select precision and compiler units:

```bash
llima-compile <model-path> -c config.py -o <output-directory>
```

LLiMa imports and executes this file. Keep credentials and unrelated side
effects out of it.

Define:

```python
def get_layer_configuration(model_properties, layer):
    return {"precision": "BF16"}
```

LLiMa calls this function for every unit exposed by the model.

| Input | Meaning |
| --- | --- |
| `model_properties["num_hidden_layers"]` | Transformer layer count |
| `layer["part"]` | Usually `PRE`, `CACHE`, `POST`, `VISION`; may include `DRAFT_FC` or `PER_LAYER` |
| `layer["is_group"]` | `True` for multi-token/group, `False` for single-token |
| `layer["index"]` | Unit index |

`PRE`/`POST` indices usually map to transformer layers. `CACHE` indices may
identify cache or token-position variants. Treat units printed by
`llima-compile` as authoritative.

Return keys:

| Key | Values/default |
| --- | --- |
| `precision` | `BF16` (default), `A_BF16_W_INT8`, `A_BF16_W_INT4` |
| `compile` | `True` (default); `False` omits the unit |
| `lora` | Optional mode documented by the installed LLiMa version |

## Mixed Precision Fallback

This common policy uses BF16 vision, INT8 group/prefill, and INT4
single-token/decode:

```python
def get_layer_configuration(model_properties, layer):
    if layer["part"] == "VISION":
        precision = "BF16"
    elif layer["is_group"]:
        precision = "A_BF16_W_INT8"
    else:
        precision = "A_BF16_W_INT4"
    return {"precision": precision}
```

Use this for an explicit FP/BF16 requirement—such as maximum fidelity or
source-weight reproduction—or as an agreed fallback when no exact
pre-quantized checkpoint or matching custom fine-tune script exists. BF16
gives highest fidelity and largest output; INT8/INT4 reduce size and can change
accuracy, compilation time, TTFT, and TPS. Validate representative prompts and
images.

## Select Units

Compile only single-token `PRE`/`POST` units at indices 4–7:

```python
SELECTED_PARTS = {"PRE", "POST"}
SELECTED_INDICES = range(4, 8)


def get_layer_configuration(model_properties, layer):
    selected = (
        layer["part"] in SELECTED_PARTS
        and not layer["is_group"]
        and layer["index"] in SELECTED_INDICES
    )
    if not selected:
        return {"compile": False}
    return {"precision": "A_BF16_W_INT8"}
```

Adjust parts, `is_group`, and indices independently. Use
`model_properties["num_hidden_layers"]` for depth-relative selection, but do
not assume every part exposes every index.

Selective compilation is for debugging and focused tests. Omitting a required
unit normally makes output undeployable. Inspect the printed unit list before
expensive stages and retain the exact `config.py` in the compilation report.
For a pre-quantized input, use selection only; do not describe the
configuration as overriding its encoded weight quantization.
