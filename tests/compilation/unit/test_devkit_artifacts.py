from pathlib import Path

import pytest

from sima_lmm.model.base import _should_write_embedding_artifacts


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("resume", "embedding_exists", "scale_exists", "expected"),
    [
        (False, True, True, True),
        (True, True, True, False),
        (True, True, False, True),
        (True, False, True, True),
        (True, False, False, True),
    ],
)
def test_quantized_embedding_artifacts_are_written_as_a_pair(
    tmp_path: Path,
    resume: bool,
    embedding_exists: bool,
    scale_exists: bool,
    expected: bool,
) -> None:
    embeddings_file = tmp_path / "embeddings.bin"
    scales_file = tmp_path / "embedding_scales.bin"
    if embedding_exists:
        embeddings_file.touch()
    if scale_exists:
        scales_file.touch()

    assert _should_write_embedding_artifacts(
        resume, embeddings_file, scales_file
    ) is expected


def test_unquantized_embedding_resume_only_checks_embedding_table(tmp_path: Path) -> None:
    embeddings_file = tmp_path / "embeddings.bin"
    embeddings_file.touch()

    assert not _should_write_embedding_artifacts(True, embeddings_file)
