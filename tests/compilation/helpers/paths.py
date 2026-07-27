import os
from pathlib import Path


def require_readable_path(path: Path, description: str | None = None) -> Path:
    """Return a required readable path or fail with an actionable exception."""
    label = description or str(path)
    try:
        exists = path.exists()
        is_dir = path.is_dir()
    except OSError as error:
        raise OSError(f"{label} is not accessible: {path}: {error}") from error

    if not exists:
        raise FileNotFoundError(f"{label} not found: {path}")

    access_mode = os.R_OK | (os.X_OK if is_dir else 0)
    if not os.access(path, access_mode):
        raise PermissionError(f"{label} is not readable: {path}")

    return path
