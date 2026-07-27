#!/usr/bin/env bash
set -euo pipefail

MODE="changed"
BASE_REF="${FORMAT_BASE_REF:-}"
VERBOSE=0
CLANG_FORMAT_BIN="${CLANG_FORMAT_BIN:-clang-format}"

usage() {
  cat <<USAGE
Usage: scripts/check_format.sh [--changed-only|--all] [--base-ref <ref>] [--verbose]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --changed-only)
      MODE="changed"
      shift
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --base-ref)
      BASE_REF="${2:-}"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

resolve_base_ref() {
  if [[ -n "$BASE_REF" ]]; then
    echo "$BASE_REF"
    return 0
  fi

  if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    local remote_ref="origin/${GITHUB_BASE_REF}"
    if ! git rev-parse --verify --quiet "$remote_ref" >/dev/null; then
      git fetch --no-tags --depth=1 origin "${GITHUB_BASE_REF}:${remote_ref}" >/dev/null 2>&1 || true
    fi
    if git rev-parse --verify --quiet "$remote_ref" >/dev/null; then
      echo "$remote_ref"
      return 0
    fi
  fi

  # Local checks use staged changes by default. CI falls back to its preceding
  # commit when no pull-request base is available.
  if [[ -n "${CI:-}" && "${CI:-}" != "false" && "${CI:-}" != "0" ]]; then
    if git rev-parse --verify --quiet HEAD~1 >/dev/null; then
      echo "HEAD~1"
      return 0
    fi
  fi

  echo ""
}

collect_changed() {
  local base
  base="$(resolve_base_ref)"
  if [[ -n "$base" ]]; then
    git diff --name-only --diff-filter=ACMRTUXB "$base"...HEAD
  else
    git diff --name-only --diff-filter=ACMRTUXB --cached
  fi
}

is_cpp_file() {
  case "$1" in
    *.c | *.cc | *.cpp | *.cxx | *.h | *.hpp) return 0 ;;
    *) return 1 ;;
  esac
}

is_first_party_file() {
  case "$1" in
    third_party/*) return 1 ;;
    *) return 0 ;;
  esac
}

mapfile -t candidates < <(
  if [[ "$MODE" == "all" ]]; then
    git ls-files
  else
    collect_changed
  fi
)

files=()
for file in "${candidates[@]}"; do
  [[ -f "$file" ]] || continue
  if is_cpp_file "$file" && is_first_party_file "$file"; then
    files+=("$file")
  fi
done

if [[ ${#files[@]} -eq 0 ]]; then
  echo "[format] no first-party C/C++ files to check (${MODE} mode)"
  exit 0
fi

if ! command -v "$CLANG_FORMAT_BIN" >/dev/null 2>&1; then
  echo "ERROR: $CLANG_FORMAT_BIN is required. Install clang-format 18 and rerun." >&2
  exit 1
fi

echo "[format] checking ${#files[@]} files (${MODE} mode)"
failed=0
for file in "${files[@]}"; do
  if ! "$CLANG_FORMAT_BIN" --dry-run --Werror "$file" >/dev/null 2>&1; then
    echo "[format] needs formatting: $file"
    if [[ "$VERBOSE" -eq 1 ]]; then
      echo "[format] diff for $file:"
      "$CLANG_FORMAT_BIN" "$file" | diff -u "$file" - || true
    fi
    failed=1
  fi
done

if [[ $failed -ne 0 ]]; then
  echo "[format] failed. Run clang-format 18 on the files above." >&2
  exit 1
fi

echo "[format] OK"
