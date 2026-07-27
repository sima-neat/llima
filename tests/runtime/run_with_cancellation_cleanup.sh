#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $(basename "$0") COMMAND [ARG ...]" >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "ERROR: setsid is required for runtime-test cancellation cleanup." >&2
  exit 1
fi

child_pid=""

stop_child_group() {
  trap - EXIT INT TERM
  if [[ ! "${child_pid}" =~ ^[1-9][0-9]*$ ]] \
    || ! kill -0 -- "-${child_pid}" 2>/dev/null; then
    return
  fi

  echo "[runtime-tests] stopping active inference process group ${child_pid}" >&2
  kill -INT -- "-${child_pid}" 2>/dev/null || true
  while kill -0 -- "-${child_pid}" 2>/dev/null; do
    sleep 1
  done
}

on_exit() {
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    stop_child_group
  else
    trap - EXIT INT TERM
  fi
  exit "${status}"
}
trap on_exit EXIT INT TERM

setsid --wait "$@" &
child_pid=$!

set +e
wait "${child_pid}"
status=$?
set -e
exit "${status}"
