"""Download/cache performance workloads and record their source identities."""

import json
import os
from urllib.request import Request, urlopen
from pathlib import Path

from huggingface_hub import hf_hub_download


def _hf_file(repo: str, filename: str, revision: str | None = None):
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision or "main"))
    parts = path.parts
    sha = parts[parts.index("snapshots") + 1]
    return path, {"repository": repo, "revision": sha, "file": filename}


def _github_file(repo: str, filename: str):
    root = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "mole" / repo.replace("/", "--")
    root.mkdir(parents=True, exist_ok=True)
    cached_revision = root / "revision"
    if os.getenv("HF_HUB_OFFLINE") == "1":
        revision = cached_revision.read_text().strip()
    else:
        request = Request(f"https://api.github.com/repos/{repo}/commits?path={filename}&per_page=1",
                          headers={"User-Agent": "llima-mole"})
        with urlopen(request, timeout=30) as response:
            revision = json.load(response)[0]["sha"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("GitHub dataset revision must be a full commit SHA")
    path = root / f"{revision}-{Path(filename).name}"
    if not path.exists():
        if os.getenv("HF_HUB_OFFLINE") == "1":
            raise FileNotFoundError(f"Dataset is not cached: {path}")
        with urlopen(f"https://raw.githubusercontent.com/{repo}/{revision}/{filename}", timeout=30) as response:
            data = response.read()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    cached_revision.write_text(revision)
    return path, {"repository": repo, "revision": revision, "file": filename}


def _json_schema(value):
    if isinstance(value, list):
        return [_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _json_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = {"dict": "object", "float": "number", "int": "integer",
                          "list": "array", "tuple": "array", "bool": "boolean"}.get(result["type"], result["type"])
    return result


def _sample(dataset: str, row: dict, index: int) -> dict:
    sample = {"id": str(row.get("id", row.get("key", row.get("fname", index)))),
              "category": row.get("category", dataset)}
    if dataset == "bfcl-v3":
        if len(row["question"]) != 1:
            raise ValueError("BFCL v3 simple/multiple workloads require one conversation turn")
        sample.update(messages=row["question"][0], tools=[
            {"type": "function", "function": _json_schema(function)} for function in row["function"]
        ])
        sample["category"] = row["id"].rsplit("_", 1)[0]
    elif dataset in ("alpacaeval", "ifeval"):
        content = row["instruction"] if dataset == "alpacaeval" else row["prompt"]
        sample.update(messages=[{"role": "user", "content": content}])
        sample["category"] = row.get("dataset", dataset)
        if dataset == "ifeval":
            sample["instruction_ids"] = row["instruction_id_list"]
    elif dataset == "dialogsum":
        sample.update(messages=[{"role": "user", "content": "Summarize the following conversation concisely:\n\n" + row["dialogue"]}])
    elif dataset == "gsm8k":
        sample.update(messages=[{"role": "user", "content": row["question"]}])
    else:
        raise ValueError(f"Unsupported performance dataset: {dataset}")
    if not sample["messages"] or sample["messages"][-1].get("role") != "user":
        raise ValueError(f"Sample {sample['id']} must end in a user message")
    return sample


def load_workload(dataset: str, limit: int | None = None):
    """Return normalized conversations and immutable source identities; never load model weights."""
    sources, rows = [], []
    if dataset == "bfcl-v3":
        paths = []
        revision = None
        for category in ("simple", "multiple"):
            path, identity = _hf_file("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                                       f"BFCL_v3_{category}.json", revision)
            paths.append((path, identity))
            revision = identity["revision"]
    elif dataset in ("alpacaeval", "ifeval", "gsm8k"):
        repo, filename = {
            "alpacaeval": ("tatsu-lab/alpaca_eval", "alpaca_eval.json"),
            "ifeval": ("google/IFEval", "ifeval_input_data.jsonl"),
            "gsm8k": ("openai/gsm8k", "main/test-00000-of-00001.parquet"),
        }[dataset]
        paths = [_hf_file(repo, filename)]
    elif dataset == "dialogsum":
        paths = [_github_file("cylnlp/dialogsum", "DialogSum_Data/dialogsum.test.jsonl")]
    else:
        raise ValueError(f"Unsupported performance dataset: {dataset}")
    for path, identity in paths:
        sources.append(identity)
        if path.suffix == ".parquet":
            import pyarrow.parquet as parquet
            contents = parquet.read_table(path).to_pylist()
        else:
            contents = (json.loads(path.read_text()) if path.read_text().lstrip().startswith("[")
                        else [json.loads(line) for line in path.read_text().splitlines() if line.strip()])
        for index, row in enumerate(contents):
            sample = _sample(dataset, row, index)
            sample["source"] = identity
            rows.append(sample)
    if len({sample["id"] for sample in rows}) != len(rows):
        raise ValueError("Dataset sample IDs must be unique")
    # Interleave BFCL categories so small runs exercise both simple and multiple tools.
    if dataset == "bfcl-v3":
        groups = [[row for row in rows if row["category"] == category] for category in ("simple", "multiple")]
        rows = [group[index] for index in range(max(map(len, groups))) for group in groups if index < len(group)]
    if not rows:
        raise ValueError("Dataset contains no requests")
    return rows[:limit], {"name": dataset, "sources": sources,
                         "selection": "all" if limit is None else "category_interleaved" if dataset == "bfcl-v3" else "first_n",
                         "available_samples": len(rows)}
