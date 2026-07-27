from collections.abc import Sequence

import numpy as np


def assert_outputs_close(
    reference: Sequence[np.ndarray],
    actual: Sequence[np.ndarray],
    tolerances: Sequence[float],
    *,
    concatenate: bool = False,
) -> None:
    if concatenate:
        reference = [np.concatenate([value.squeeze() for value in reference])]
        actual = [np.concatenate([value.squeeze() for value in actual])]

    assert len(reference) == len(actual), (
        f"Output count differs: reference={len(reference)}, actual={len(actual)}"
    )
    assert len(tolerances) in (1, len(reference)), (
        f"Expected one tolerance or one per output; got {len(tolerances)} "
        f"for {len(reference)} outputs"
    )

    for index, (reference_output, actual_output) in enumerate(
        zip(reference, actual, strict=True)
    ):
        tolerance = tolerances[0] if len(tolerances) == 1 else tolerances[index]
        absolute_tolerance = tolerance * float(np.max(np.abs(reference_output)))
        np.testing.assert_allclose(
            actual_output,
            reference_output,
            rtol=0.0,
            atol=absolute_tolerance,
            err_msg=(
                f"Output {index} differs with relative-to-maximum tolerance "
                f"{tolerance}"
            ),
        )
