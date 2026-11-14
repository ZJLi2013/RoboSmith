#!/usr/bin/env bash
# Build HIPRT inside a hipRT_render checkout and write the env file later shells source.
#
#   bash scripts/setup/hiprt.sh /opt/hipRT_render
#   source /opt/hipRT_render/robosmith-hiprt.env
#
# The checkout belongs to this container: the script writes only its third_party/,
# .torch_ext/ and robosmith-hiprt.env. HIPRT itself is built by the checkout's
# instinct_render/build_hiprt.sh; a finished build is reused while its stamp
# (HIPRT commit, patch hashes, ROCm version) still matches.
set -euo pipefail

HIPRT_COMMIT=e3c01fce5860f5d74b959ae71abbbab6a5462aed
ROOT=$(cd "${1:?usage: hiprt.sh <hipRT_render checkout>}" && pwd)
BUILD="$ROOT/instinct_render/build_hiprt.sh"
SRC="$ROOT/third_party/HIPRT"
PREFIX="$ROOT/third_party/hiprt_install"
STAMP="$PREFIX/.robosmith-stamp"
ENV_FILE="$ROOT/robosmith-hiprt.env"
ROCM_VERSION_FILE="${ROCM_VERSION_FILE:-/opt/rocm/.info/version}"

if [[ ! -f "$BUILD" ]]; then
  echo "not a hipRT_render checkout: $ROOT" >&2
  exit 2
fi
if [[ -e "$SRC" ]]; then
  head=$(git -c safe.directory='*' -C "$SRC" rev-parse HEAD 2>/dev/null || echo none)
  if [[ "$head" != "$HIPRT_COMMIT" ]]; then
    echo "$SRC is at $head, not HIPRT $HIPRT_COMMIT; move it away and rerun" >&2
    exit 3
  fi
fi

want="hiprt $HIPRT_COMMIT
$(sha256sum "$ROOT"/instinct_render/patches/hiprt-*.patch | awk '{print $1}')
rocm $(cat "$ROCM_VERSION_FILE" 2>/dev/null || echo unknown)"

if [[ -f "$STAMP" && -f "$PREFIX/bin/libhiprt0300164.so" && "$(cat "$STAMP")" == "$want" ]]; then
  echo "HIPRT build matches $STAMP; reusing it"
else
  rm -rf "$PREFIX"
  if ! bash "$BUILD"; then
    rm -rf "$PREFIX"
    echo "HIPRT build failed; rerun to retry" >&2
    exit 1
  fi
  printf '%s\n' "$want" > "$STAMP"
fi

mkdir -p "$ROOT/.torch_ext"
cat > "$ENV_FILE" <<EOF
export PYTHONPATH="$ROOT\${PYTHONPATH:+:\$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$ROOT/.torch_ext"
export HIPRT_PATH="$SRC"
export HIPRT_PREFIX="$PREFIX"
EOF

bash -c "source '$ENV_FILE' && python3 -c 'import instinct_render.genesis'"
echo "ready: source $ENV_FILE"
