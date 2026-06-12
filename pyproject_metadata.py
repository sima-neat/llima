import os
import json
from pathlib import Path
import subprocess
import yaml


def _version_info(version):
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise RuntimeError(f"ERROR: unable to parse package-version {version}")
    return {
        "major": parts[0],
        "minor": parts[1],
        "patch": parts[2],
    }


def _version_info_from_manifest():
    manifest = Path(__file__).parent / "deps" / "manifest.json"
    version = str(json.loads(manifest.read_text(encoding="utf-8")).get("package-version", "")).strip()
    return _version_info(version), version


def _tag_version_override():
    if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ.get("GITHUB_REF_NAME"):
        return os.environ["GITHUB_REF_NAME"].removeprefix("v")

    proc = subprocess.Popen(
        ['git', 'describe', '--tags', '--exact-match'],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    proc.wait()
    if proc.returncode == 0:
        return proc.stdout.read().decode('utf-8').rstrip().removeprefix("v")
    return None


def get_version(pkg_dir):
    vinfo, version = _version_info_from_manifest()
    tag_version = _tag_version_override()
    if tag_version:
        version = tag_version
        vinfo = _version_info(version)

    if "GIT_HASH" in os.environ:
        # In tox the .git directory is not available so do this instead
        vinfo["git_hash"] = os.environ["GIT_HASH"]
    else:
        proc = subprocess.Popen(['git','log','-1','--format=%h'], stdout=subprocess.PIPE)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("ERROR: unable to execute git log")
        vinfo['git_hash'] = proc.stdout.read().decode('utf-8').rstrip()

    if not os.path.exists(pkg_dir):
        os.makedirs(pkg_dir)

    with open(pkg_dir + "/VERSION","w") as fh:
        print(vinfo)
        print(pkg_dir)
        fh.write(yaml.dump(vinfo))

    return version


def get_package(env, pkg_name, pkg_version):
    if env + "_BRANCH" in os.environ and env + "_VERSION" in os.environ:
        branch = os.environ[env + "_BRANCH"]
        version = os.environ[env + "_VERSION"]

        if version == pkg_version or pkg_version == "latest":
            deps = {}

            if os.path.exists("DEPS"):
              # Append dependency to DEPS file
              with open("DEPS", "r") as fh:
                deps = yaml.load(fh.read(), Loader=yaml.BaseLoader)

            deps[env] = "%s==%s.dev0+%s" % (pkg_name, version, branch)

            with open("DEPS", "w") as fh:
              fh.write(yaml.dump(deps))

            return "%s==%s.dev0+%s" % (pkg_name, version, branch)

    if pkg_version == "latest":
        return "%s" % (pkg_name)
    else:
        return "%s==%s" % (pkg_name, pkg_version)


install_require = []


sdk_ext_require = [
    # Same as other SDK packages.
    "numpy==1.23.5; python_version>='3.10' and python_version<'3.12'",
    # Same as other SDK packages.
    "numpy==1.26.4; python_version>='3.12'",
    "onnx==1.17.0",  # Required for graph surgery
    "onnxruntime==1.21.1",
    "onnxsim-prebuilt",  # Required for graph surgery
    "safetensors",  # Required for HF LLM models
    "torch==2.3.1",  # Required to load GGUF file to transformers.
    "av",  # Required for audio preprocessing
    "ml_dtypes==0.3.1",  # Required for bfloat16 datatype
    "sentencepiece==0.2.0",  # Required for LLM tokenizer
    "tiktoken",  # Required for LLM tokenizer
    "blobfile",  # Required by tiktoken
    "pillow",  # Required for VLM image preprocessing
    "protobuf==4.25.7",  # Required for SentencePiece with extended vocabulary.
    "transformers==4.57.1",  # Required for load/verify HF models and use HF processor.
    "huggingface-hub>=0.34.0,<1.0", # Required to work with current transformers version.
    "gguf==0.17.1",  # Required for GGUF weights dequantization
    "llama_cpp_python==0.3.16",  # Required to run GGUF inference

    # Required for MoLE
    "datasets>=4.0.0",
    "fabric>=3.2.2",
    "lm-eval[hf]>=0.4.11",
    "loguru>=0.7.3",
    "msgpack>=1.1.1",
    "pytablewriter>=1.2.1",
    "pyzmq>=27.1.0",
    "rich>=14.1.0",
    "tqdm>=4.67.1",
    "matplotlib>=3.10.6",
]


sdk_require = [
    get_package("SIMA_UTILS", "sima-utils", "latest"),
    get_package("AWESOME_FRONT_END", "sima-frontend", "latest"),
    *sdk_ext_require
]


tests_require = [
    "flake8",
    "mypy",
    "pytest",
    "pytest-cov",
    "pytest-xdist",
    "pytest-timeout",
    "termcolor",
    "colored_traceback",
]


def dynamic_metadata(field, settings=None):
    if field == "version":
        return get_version("sima_lmm")

    if field == "dependencies":
        return install_require

    if field == "optional-dependencies":
        return {
            "sdk_ext": sdk_ext_require,
            "sdk": sdk_require,
            "dev": tests_require + sdk_require,
            "tests": tests_require,
        }
    return None
