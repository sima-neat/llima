"""Run the direct and facade generation suites against a deployed target/draft pair."""

import json
import os
import subprocess
import sys
from pathlib import Path

from sima_lmm.devkit.model_manager import DEFAULT_MODELS_ROOT, resolve_target_and_draft_paths

BUNDLES = {
    "eagle3": "Llama-3.1-8B-Instruct-GPTQ-EAGLE3",
    "dflash": "Qwen3-4B-Autoround-DFlash-a16w4",
    "gemma4_mtp": "Gemma-4-E2B-it-TextOnly-GPTQ-MTP-a16w4",
}


def resolve(bundle: Path) -> tuple[str, Path, Path]:
    target, draft = resolve_target_and_draft_paths(bundle)
    if draft is None:
        raise ValueError(f"Expected a paired target and draft in {bundle}")
    config = json.loads((target / "devkit/vlm_config.json").read_text())
    method = config["lm_cfg"]["speculative_decoding_cfg"].get("method", "eagle3")
    return method, target, draft


def main(binary: str, method: str) -> int:
    models = Path(os.environ.get("LLIMA_MODELS_PATH") or DEFAULT_MODELS_ROOT)
    configured = os.environ.get(f"SIMA_TEST_LLIMA_{method.upper()}_MODEL")
    legacy = os.environ.get("SIMA_TEST_LLIMA_SPEC_MODEL")
    bundle = models / (configured or BUNDLES[method])
    if legacy and not configured and resolve(models / legacy)[0] == method:
        bundle = models / legacy
    if not configured and not bundle.exists():
        print(f"Skipping {method} generation test: {bundle} is absent")
        return 77
    actual, target, draft = resolve(bundle)
    if actual != method:
        raise ValueError(f"Expected {method}, found {actual} in {bundle}")
    subprocess.run([binary, "--direct-pair", str(target), str(draft)], check=True)
    os.execv(binary, [binary, str(target), str(draft)])


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
