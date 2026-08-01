# Audit Model Compatibility

Use this audit before deciding that a new architecture, tensor transform, or
runtime branch is required. Answer the user's scoped question first, then list
independent downstream findings.

## Contents

- [Evidence boundaries](#evidence-boundaries)
- [Framework and config loading](#check-the-actual-framework-path)
- [Compatibility hooks](#trace-existing-compatibility-hooks)
- [Tokenizer transport](#trace-tokenizer-transport)
- [Claim discipline](#claim-discipline)

## Evidence Boundaries

Record evidence for each affected boundary. Mark unrelated boundaries not
applicable instead of investigating them.

| Boundary | Question | Minimum evidence |
| --- | --- | --- |
| Provenance and access | Are the exact model, revision, format, and required files available? | Pinned model ID/revision and exact file inventory |
| Compiler environment | Which Python and dependency versions actually execute the loader? | Executable path and imported package versions |
| Config loading | Does the current call load the exact `config.json`? | Reproduction using the repository call and arguments |
| Architecture normalization | Can LLiMa represent and validate the config contract? | Generated config or a precise rejection |
| Tensor resolution | Can existing names, aliases, and bundled-weight fallbacks resolve every required tensor? | Exact target tensor index plus resolver trace |
| Graph semantics | Do existing pre/cache/post or vision/projector graphs express the computation and state? | Numerical comparison against a pinned reference |
| Host tokenizer/processor | Can host preprocessing create the expected prompt and token IDs? | Exact reference prompt and token IDs |
| Runtime assets | Are all tokenizer/template/processor files packaged in a format the C++ runtime loads? | Generated DevKit inventory and loader trace |
| Modalix execution | Does the complete model load, generate, and exit cleanly? | Packaged tests and representative smoke result |

Do not collapse these into a single supported/unsupported result. For example,
`AutoConfig` loading can succeed while architecture recognition fails, and a
graph can support fused QKV after an existing resolver maps the name.

## Check the Actual Framework Path

Distinguish three sources of reference behavior:

1. a native implementation in the installed Transformers version;
2. custom code supplied by the pinned model repository; and
3. a secondary implementation such as vLLM or llama.cpp.

Use the exact model repository implementation as the oracle when upstream
Transformers has no native implementation. Treat secondary implementations as
corroboration, not a substitute for the pinned target.

Inspect the supported compiler environment rather than assuming repository
metadata declares every transitive dependency. This reproduces the native
mapping check and the repository's current
`AutoConfig.from_pretrained(config_file.parent)` call:

```bash
python - "<model-dir>" <<'PY'
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from sima_lmm.hf.hf_transformer import find_file

model_dir = Path(sys.argv[1])
config_file = find_file(
    directory=model_dir,
    filename="config.json",
    resolve=False,
)
if config_file is None:
    raise FileNotFoundError(f"No config.json found under {model_dir}")

model_type = json.loads(config_file.read_text())["model_type"]
print(sys.executable)
print(transformers.__version__)
print(f"{model_type=}")
print(f"native_config={model_type in CONFIG_MAPPING}")

config = AutoConfig.from_pretrained(config_file.parent)
print(f"loaded_config={type(config).__module__}.{type(config).__name__}")
PY
```

If `config.json` contains `auto_map`, reproduce the repository's current
`AutoConfig` or `AutoTokenizer` call both as written and, only for a reviewed
investigation, with the required remote-code option. Report these separately:

- whether custom code makes loading succeed;
- whether its declared dependency range matches the installed environment;
- whether executing checkpoint-supplied code is acceptable for the production
  compiler; and
- which later LLiMa boundary still requires work.

Do not add blanket `trust_remote_code=True` behavior or broaden the compiler's
code-execution boundary without explicit project authorization.

## Trace Existing Compatibility Hooks

Before writing a transform, inspect:

- `LocalHuggingFaceModel.param_exists`, `load_np_param`, and
  `language_model_param_base_name` in `sima_lmm/hf/hf_transformer.py`;
- `find_alternate_weight` in `sima_lmm/model/onnx_builder.py`;
- its ONNX and direct Model SDK call sites in `onnx_builder.py` and
  `sima_builder.py`;
- conditional name selection in the nearest language or vision graph; and
- existing configuration aliases in `sima_lmm/config/vlm_config.py`.

A useful initial search is:

```bash
rg -n \
  "find_alternate_weight|qkv_proj|gate_up_proj|check_hf_param|param_exists|load_np_param" \
  sima_lmm tests
```

Trace the exact requested name through the fallback. A resolver that supports
`self_attn.q_proj -> self_attn.qkv_proj` still needs an alias if the source uses
another module or block prefix. Reuse its validated split logic after resolving
the name.

## Trace Tokenizer Transport

Tokenizer compatibility is more than chat-template rendering. Inspect:

1. source tokenizer and custom-code files;
2. host discovery and `AutoTokenizer`/`AutoProcessor` loading in
   `sima_lmm/preproc/vlm_helper.py`;
3. copied/generated files in `BaseModel.gen_devkit_files`;
4. `sima_lmm/devkit/cpp/vlm_helper.*` asset selection; and
5. `sima_lmm/devkit/cpp/tokenizer.*` supported formats.

A host-only custom tokenizer is insufficient if compilation does not produce a
runtime-supported asset. Prefer a deterministic compiler-side conversion when
it can preserve exact token IDs and decoding; add runtime support only when the
contract cannot be represented by existing formats.

## Claim Discipline

- Do not infer an exact tensor layout from a model name or architecture class.
- Do not attribute a mirror or derivative checkpoint's config, weights, or
  tokenizer to the target model.
- If gated access prevents inspection, label conclusions provisional and list
  the exact files still required.
- Do not treat a successful CPU/reference run as compiler or Modalix evidence.
