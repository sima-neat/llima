#!/usr/bin/env python3
"""Check component dependencies and private-header boundaries in the runtime."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

PUBLIC_COMPONENTS = {
    "vision_language_model.hpp": "vlm",
    "chat.hpp": "vlm",
    "image_processor.hpp": "vlm",
    "whisper_model.hpp": "whisper",
    "vlm_config.hpp": "package",
    "whisper_config.hpp": "package",
    "file_provider.hpp": "package",
    "mla_model.hpp": "mla",
    "mla_buffer.hpp": "mla",
    "tokenizer.hpp": "text",
    "text_streamer.hpp": "text",
    "generation_types.hpp": "text",
    "tool_call_parser.hpp": "text",
    "reasoning_parser.hpp": "text",
    "setup.hpp": "common",
    "utils.hpp": "common",
    "eigen_types.hpp": "common",
    "nlohmann_optional.hpp": "common",
}

# Every component can include itself and common headers. Package metadata may
# retain MLA buffers for Whisper; it must not depend on model execution classes.
DEPENDENCIES = {
    "common": set(),
    "mla": set(),
    "package": {"mla"},
    "text": set(),
    "language": {"package", "text", "mla"},
    "vlm": {"language", "package", "text", "mla"},
    "whisper": {"package", "text", "mla"},
    "frontends": {"vlm", "whisper", "package", "text"},
    "bindings": {"frontends", "vlm", "whisper", "text"},
}
EXCEPTIONS = {
    ("runtime/src/mla/mla_model.cpp", "runtime/include/sima_lmm/file_provider.hpp"):
        "Deferred ELF loading uses generic file-provider hooks, not model metadata.",
    ("runtime/src/setup.cpp", "runtime/include/sima_lmm/mla_model.hpp"):
        "Startup connects the MLA backend and reads its environment settings.",
    ("runtime/src/setup.cpp", "runtime/include/sima_lmm/image_processor.hpp"):
        "Startup reads image-processor environment settings.",
}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[">]', re.MULTILINE)


def component(path: Path) -> str:
    if path.parts[:3] == ("runtime", "include", "sima_lmm"):
        return PUBLIC_COMPONENTS[path.name]
    if path.parts[:2] == ("runtime", "devkit"):
        return "bindings"
    return path.parts[2] if len(path.parts) > 3 else "common"


def check(root: Path) -> list[str]:
    public = root / "runtime/include/sima_lmm"
    private = root / "runtime/src"
    files = sorted([*public.glob("*.hpp"), *private.rglob("*.hpp"),
                    *private.rglob("*.cpp"), *root.glob("runtime/devkit/*.cpp")])
    errors = []
    for source in files:
        relative = source.relative_to(root)
        if relative.parts[:3] == ("runtime", "include", "sima_lmm") and source.name not in PUBLIC_COMPONENTS:
            errors.append(f"{relative}: classify this installed header in PUBLIC_COMPONENTS")
            continue
        origin = component(relative)
        if origin not in DEPENDENCIES:
            errors.append(f"{relative}: unknown runtime component {origin}")
            continue
        text = source.read_text(encoding="utf-8")
        for match in INCLUDE.finditer(text):
            name = match[1]
            candidates = [private / name, public / name.removeprefix("sima_lmm/"), source.parent / name]
            target = next((p.resolve() for p in candidates if p.is_file()), None)
            if target is None or not target.is_relative_to(root.resolve()):
                continue  # System and third-party headers are not component edges.
            target_relative = target.relative_to(root.resolve())
            line = text.count("\n", 0, match.start()) + 1
            location = f"{relative}:{line}"
            is_private = target.is_relative_to(private.resolve())
            if is_private and name != target.relative_to(private.resolve()).as_posix():
                errors.append(f"{location}: include private headers from runtime/src: {target_relative}")
            if is_private and (source.is_relative_to(public) or
                               origin in {"frontends", "bindings"} and component(target_relative) != "frontends"):
                errors.append(f"{location}: private model header {target_relative} is not part of the public API")
            if target.is_relative_to(public.resolve()) and target.name not in PUBLIC_COMPONENTS:
                errors.append(f"{location}: classify installed header {target_relative} in PUBLIC_COMPONENTS")
                continue
            dependency = component(target_relative)
            if dependency in {origin, "common"} | DEPENDENCIES[origin]:
                continue
            if (relative.as_posix(), target_relative.as_posix()) not in EXCEPTIONS:
                errors.append(f"{location}: {origin} must not depend on {dependency}: {target_relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().root.resolve()
    errors = check(root)
    for error in errors:
        print(error)
    if not errors:
        print("Runtime include boundaries passed.")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
