#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${LLIMA_MANIFEST:-${ROOT_DIR}/deps/manifest.json}"

package_version="$(
  python3 - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
data = json.loads(manifest.read_text(encoding="utf-8"))
version = str(data.get("package-version", "")).strip()
if not version:
    raise SystemExit(f"Missing or empty 'package-version' in {manifest}")
print(version)
PY
)"

if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
  tag_version="${GITHUB_REF_NAME}"
  printf '%s\n' "${tag_version#v}"
  exit 0
fi

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  tag_version="$(git -C "${ROOT_DIR}" describe --tags --exact-match 2>/dev/null || true)"
  if [[ -n "${tag_version}" ]]; then
    printf '%s\n' "${tag_version#v}"
    exit 0
  fi
fi

branch="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
if [[ -z "${branch}" ]] && git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  branch="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
fi
if [[ -z "${branch}" || "${branch}" == "HEAD" ]]; then
  branch="branch"
fi

branch_for_deb="$(
  printf '%s' "${branch}" |
    tr '[:upper:]' '[:lower:]' |
    sed -E 's#[^a-z0-9.+~-]+#-#g; s/^-+//; s/-+$//'
)"
if [[ -z "${branch_for_deb}" ]]; then
  branch_for_deb="branch"
fi

short_sha="${GITHUB_SHA:-}"
if [[ -n "${short_sha}" ]]; then
  short_sha="${short_sha:0:12}"
elif git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  short_sha="$(git -C "${ROOT_DIR}" rev-parse --short=12 HEAD 2>/dev/null || true)"
fi
if [[ -z "${short_sha}" ]]; then
  short_sha="unknown"
fi

printf '%s+%s.%s\n' "${package_version}" "${branch_for_deb}" "${short_sha}"
