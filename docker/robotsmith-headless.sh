#!/usr/bin/env bash
set -euo pipefail

# Genesis 1.4 compiles for the real ISA, so no gfx override; and Genesis' own
# camera runs on Xvfb/GLX, not EGL.
unset HSA_OVERRIDE_GFX_VERSION
unset PYOPENGL_PLATFORM

export DISPLAY="${DISPLAY:-:99}"

if ! pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1; then
  Xvfb "${DISPLAY}" -screen 0 "${XVFB_SCREEN:-1280x1024x24}" -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
  sleep 2
fi

exec "$@"
