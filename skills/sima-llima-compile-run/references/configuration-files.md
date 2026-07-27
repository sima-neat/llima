# LLiMa Compilation Configuration Files

## Use `-c config.py`

Pass a Python configuration file to control precision and select which
compiler units are generated:

```bash
llima-compile <model-path> \
  -c config.py \
  -o <output-directory>
```

LLiMa imports and executes this file. Use only trusted configuration files; do
not place credentials or unrelated side effects in them.

The file must define:

```python
def get_layer_configuration(model_properties, layer):
    return {"precision": "BF16"}
```

LLiMa calls the function once for every compiler unit exposed by the model.
`model_properties` contains:

```python
{"num_hidden_layers": 32}
```

`layer` contains:

- `part`: the logical component, normally `PRE`, `CACHE`, `POST`, or `VISION`;
- `is_group`: `True` for a multi-token/group variant and `False` otherwise;
  and
- `index`: the index of that compiler unit.

Some architectures also expose parts such as `DRAFT_FC` or `PER_LAYER`. Treat
the units printed by `llima-compile` as authoritative. For `PRE` and `POST`,
`index` normally corresponds to a transformer layer. Cache indices can
identify cache variants or token-position ranges rather than transformer
layers.

Return a dictionary containing:

- `precision`: `BF16`, `A_BF16_W_INT8`, or `A_BF16_W_INT4`;
- `compile`: `False` to omit the unit; defaults to `True`; and
- `lora`: an optional LoRA mode documented by the installed LLiMa version.

For every unit that will be compiled, return `precision` explicitly. Return
`{"compile": False}` for an omitted unit.

## Choose Precision

Use:

- `BF16` for the highest fidelity and largest compiled model;
- `A_BF16_W_INT8` for BF16 activations with INT8 weights; or
- `A_BF16_W_INT4` for BF16 activations with INT4 weights.

A practical mixed-precision policy is BF16 for vision, INT8 for group/prefill
units, and INT4 for single-token/decode units:

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

Quantization changes accuracy, memory use, compilation time, TTFT, and TPS.
Validate the resulting model on representative prompts and images.

## Compile Only Selected Parts

Compile only decode-time `PRE` and `POST` units:

```python
SELECTED_PARTS = {"PRE", "POST"}


def get_layer_configuration(model_properties, layer):
    if layer["part"] not in SELECTED_PARTS or layer["is_group"]:
        return {"compile": False}

    return {"precision": "A_BF16_W_INT4"}
```

Change `SELECTED_PARTS` to select `CACHE`, `VISION`, or another part exposed by
the model. Use `is_group` independently when both prefill and decode variants
of a part exist.

## Compile Only Selected Indices

Compile index 0 of every single-token/decode part:

```python
def get_layer_configuration(model_properties, layer):
    if layer["is_group"] or layer["index"] != 0:
        return {"compile": False}

    return {"precision": "A_BF16_W_INT4"}
```

Compile only single-token `PRE` and `POST` units for transformer layers 4
through 7:

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

Use `model_properties["num_hidden_layers"]` when selection depends on model
depth:

```python
def get_layer_configuration(model_properties, layer):
    last_index = model_properties["num_hidden_layers"] - 1
    if layer["part"] not in {"PRE", "POST"}:
        return {"compile": False}
    if layer["index"] not in {0, last_index}:
        return {"compile": False}

    return {"precision": "BF16"}
```

Do not assume every part has every index. LLiMa calls the function only for
units that exist for the selected architecture.

## Understand Partial Compilation

Selective compilation is useful for compiler debugging, focused testing, and
reducing iteration time. An output that omits required units is not a complete
model and normally cannot pass deployment or `llima run` validation.

For a deployable model, ensure the configuration includes every runtime unit
required by that architecture. Inspect the compiler's printed unit list before
the expensive stages begin and keep the exact `config.py` with the compilation
report.
