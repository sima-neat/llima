"""Run the session lifecycle suite with both halves of a deployed speculative model."""

import os
import subprocess
import sys
from pathlib import Path

from sima_lmm.devkit.model_manager import DEFAULT_MODELS_ROOT, resolve_target_and_draft_paths


def main() -> int:
    configured = os.environ.get("SIMA_TEST_LLIMA_SPEC_MODEL")
    models = Path(os.environ.get("LLIMA_MODELS_PATH") or DEFAULT_MODELS_ROOT)
    bundle = models / (configured or "Llama-3.1-8B-Instruct-GPTQ-EAGLE3")
    if not configured and not bundle.exists():
        print(f"Skipping shared-draft session test: {bundle} is absent")
        return 77
    target, draft = resolve_target_and_draft_paths(bundle)
    if draft is None:
        raise ValueError(f"Expected a paired target and draft in {bundle}")
    binary = sys.argv[1]
    subprocess.run([binary, "--direct-pair", str(target), str(draft)], check=True)
    os.execv(binary, [binary, str(target), str(draft)])


if __name__ == "__main__":
    sys.exit(main())
