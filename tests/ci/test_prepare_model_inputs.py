from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path, PurePosixPath
from unittest import mock

import pytest

from tools.ci import prepare_model_inputs


pytestmark = pytest.mark.premerge


def source_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "base_url": None,
        "s3_bucket": "sima-neat-artifacts-production",
        "aws_region": "us-west-2",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_s3_source_requires_region() -> None:
    with pytest.raises(
        prepare_model_inputs.PreparationError,
        match="--aws-region is required",
    ):
        prepare_model_inputs.artifact_source_from_args(
            source_args(aws_region=None)
        )


def test_https_source_rejects_aws_region() -> None:
    with pytest.raises(
        prepare_model_inputs.PreparationError,
        match="valid only with --s3-bucket",
    ):
        prepare_model_inputs.artifact_source_from_args(
            source_args(
                base_url="https://artifacts.neat.sima.ai",
                s3_bucket=None,
            )
        )


def test_s3_source_builds_exact_object_uri() -> None:
    source = prepare_model_inputs.artifact_source_from_args(source_args())

    assert (
        source.object_uri("llima-gguf/org/model/latest/manifest.json")
        == "s3://sima-neat-artifacts-production/"
        "llima-gguf/org/model/latest/manifest.json"
    )


def test_https_source_remains_available_for_non_aws_use() -> None:
    source = prepare_model_inputs.artifact_source_from_args(
        source_args(
            base_url="https://artifacts.neat.sima.ai/",
            s3_bucket=None,
            aws_region=None,
        )
    )

    assert source.object_uri("llima-gguf/org/model/latest/model file.gguf") == (
        "https://artifacts.neat.sima.ai/"
        "llima-gguf/org/model/latest/model%20file.gguf"
    )


def test_s3_manifest_access_denial_has_no_http_fallback() -> None:
    source = prepare_model_inputs.ArtifactSource(
        s3_bucket="sima-neat-artifacts-production",
        aws_region="us-west-2",
    )
    denied = mock.Mock(
        returncode=1,
        stdout=b"",
        stderr=b"fatal error: An error occurred (AccessDenied)",
    )

    with (
        mock.patch.object(
            prepare_model_inputs.subprocess,
            "run",
            return_value=denied,
        ),
        mock.patch.object(prepare_model_inputs, "fetch_json") as fetch_json,
        pytest.raises(
            prepare_model_inputs.PreparationError,
            match="directly from S3.*AccessDenied",
        ),
    ):
        prepare_model_inputs.fetch_manifest(
            source,
            source.object_uri("llima-safetensors/org/model/latest/manifest.json"),
        )

    fetch_json.assert_not_called()


def test_s3_object_is_streamed_and_hashed(tmp_path: Path) -> None:
    payload = b"verified model payload"
    process = mock.Mock()
    process.stdout = io.BytesIO(payload)
    process.stderr = io.BytesIO()
    process.wait.return_value = 0
    source = prepare_model_inputs.ArtifactSource(
        s3_bucket="sima-neat-artifacts-production",
        aws_region="us-west-2",
    )
    cached_file = prepare_model_inputs.CachedFile(
        relative_path=PurePosixPath("model.gguf"),
        s3_key="llima-gguf/org/model/latest/model.gguf",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        download_uri=(
            "s3://sima-neat-artifacts-production/"
            "llima-gguf/org/model/latest/model.gguf"
        ),
    )
    partial = tmp_path / "model.partial"

    with mock.patch.object(
        prepare_model_inputs.subprocess,
        "Popen",
        return_value=process,
    ) as popen:
        size, sha256 = prepare_model_inputs.stream_s3_object(
            source,
            cached_file,
            partial,
        )

    assert partial.read_bytes() == payload
    assert size == len(payload)
    assert sha256 == hashlib.sha256(payload).hexdigest()
    command = popen.call_args.args[0]
    assert command[:4] == [
        "aws",
        "s3",
        "cp",
        cached_file.download_uri,
    ]
    assert command[-2:] == ["--region", "us-west-2"]
