#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $(basename "$0") COMMAND [ARG ...]" >&2
  exit 2
fi

# TODO: Remove the per-test restart after mla-rt 2.1.3 fixes unreleased MLA buffers.
echo "[runtime-tests] restarting simaai-appcomplex.service for test isolation"
sudo -n systemctl restart simaai-appcomplex.service

for _ in {1..30}; do
  if systemctl is-active --quiet simaai-appcomplex.service; then
    daemon_pid="$(
      systemctl show \
        --property=MainPID \
        --value \
        simaai-appcomplex.service
    )"
    if [[ "${daemon_pid}" =~ ^[1-9][0-9]*$ ]]; then
      exec "$@"
    fi
  fi
  sleep 1
done

echo "ERROR: simaai-appcomplex.service did not become ready after restart." >&2
exit 1
