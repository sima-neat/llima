"""Package unchanged upstream tasks/scoring with an image-only dependency profile.

Run only during MoLE packaging, never on a DevKit. The upstream tag v0.7.3
currently declares distribution version 0.7.2; the commit is the source identity.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


REQUIREMENTS = [
    "accelerate>=0.29.1", "aiohttp", "datasets>=4.0.0", "dill", "evaluate>=0.4.0",
    "huggingface-hub>=1.5.0", "jinja2", "loguru", "numpy>=1.26.4", "pandas",
    "pillow", "pytablewriter", "pytz", "pyyaml", "requests", "sacrebleu>=1.5.0",
    "tenacity>=8.3.0", "torch>=2.1.0", "tqdm", "transformers>=4.39.2",
]


def make_profile(source: Path, output: Path, pin: dict) -> Path:
    version = pin["version"] + "+llima." + pin["revision"][:12]
    with zipfile.ZipFile(source) as wheel:
        contents = {name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")}
    metadata_name = next(name for name in contents if name.endswith(".dist-info/METADATA"))
    old_info = metadata_name.split("/")[0]
    metadata = BytesParser().parsebytes(contents[metadata_name])
    if metadata["Version"] != pin["version"]:
        raise ValueError("Upstream distribution version changed")
    for name in ("Requires-Dist", "Provides-Extra", "Version"):
        del metadata[name]
    metadata["Version"] = version
    for requirement in REQUIREMENTS:
        metadata["Requires-Dist"] = requirement
    contents[metadata_name] = metadata.as_bytes()
    # Record every upstream code/task asset hash. No upstream code is edited.
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
                     if name.startswith("lmms_eval/")}
    contents["lmms_eval/mole_profile.json"] = (json.dumps(
        {**pin, "profile": "llima-image", "source_files": source_hashes}, sort_keys=True
    ) + "\n").encode()
    new_info = f"lmms_eval-{version}.dist-info"
    contents = {name.replace(old_info + "/", new_info + "/", 1): data
                for name, data in contents.items() if not name.endswith("/RECORD")}
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in sorted(contents.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((new_info + "/RECORD", "", ""))
    contents[new_info + "/RECORD"] = record.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"lmms_eval-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            wheel.writestr(info, data)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin = json.loads((Path(__file__).resolve().parents[1] / "deps/manifest.json").read_text())["lmms-eval"]
    with tempfile.TemporaryDirectory(prefix="llima-lmms-") as tmp:
        subprocess.run([
            sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", tmp,
            f"git+{pin['repository']}@{pin['revision']}",
        ], check=True)
        wheels = list(Path(tmp).glob("lmms_eval-*.whl"))
        if len(wheels) != 1:
            raise ValueError("Expected one pinned lmms-eval wheel")
        print(make_profile(wheels[0], args.output, pin))


if __name__ == "__main__":
    main()
