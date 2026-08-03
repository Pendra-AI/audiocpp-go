#!/usr/bin/env bash
# check-symbols.sh — fail-closed export audit for the built audiocpp shared lib.
#
# Asserts:
#   1. Every symbol in lib/expected-symbols.txt is exported.
#   2. NOTHING else is exported — in particular zero ggml_/engine/sentencepiece/
#      absl/protobuf symbols leak (they must stay statically hidden inside the
#      lib so they never collide with the worker's other in-process ggml
#      engines or another loader).
#
# Usage: scripts/check-symbols.sh <path-to-libaudiocpp.{dylib,so}>
set -euo pipefail

LIB="${1:?usage: check-symbols.sh <lib>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPECTED="$ROOT/lib/expected-symbols.txt"

if [[ ! -f "$LIB" ]]; then
  echo "FAIL: library not found: $LIB" >&2
  exit 1
fi

# Exported (defined, global) symbols, normalised to bare names (strip leading _).
case "$(uname -s)" in
  Darwin) exported="$(nm -gU "$LIB" | awk '{print $NF}' | sed 's/^_//')" ;;
  *)      exported="$(nm -D --defined-only "$LIB" | awk '{print $NF}')" ;;
esac

fail=0

# 1. Every expected symbol present.
while IFS= read -r sym; do
  [[ -z "$sym" ]] && continue
  if ! grep -qx "$sym" <<<"$exported"; then
    echo "FAIL: expected symbol missing from exports: $sym" >&2
    fail=1
  fi
done <"$EXPECTED"

# 2. No symbol outside the expected set is exported.
leaked="$(grep -vxF -f "$EXPECTED" <<<"$exported" | grep -v '^$' || true)"
if [[ -n "$leaked" ]]; then
  echo "FAIL: library leaks non-shim symbols:" >&2
  echo "$leaked" | sed 's/^/  /' >&2
  fail=1
fi

# 3. Explicit belt-and-braces check for the dependency families that MUST hide.
forbidden="$(grep -Ei '^(ggml_|gguf_|engine|sentencepiece|absl|_ZN4absl|protobuf|google)' <<<"$exported" || true)"
if [[ -n "$forbidden" ]]; then
  echo "FAIL: forbidden dependency symbols exported:" >&2
  echo "$forbidden" | sed 's/^/  /' >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "check-symbols: FAILED" >&2
  exit 1
fi

n="$(grep -c . "$EXPECTED")"
echo "check-symbols: OK — exactly $n shim symbols exported, no leaks."
