#!/usr/bin/env bash
# Render + encode pipeline for the valley shot.
#
#   ./tools/render.sh --look            # 3 cheap preview stills (480x270)
#   ./tools/render.sh --frames [N]      # render frames (default from valley.py)
#   ./tools/render.sh --encode          # frames -> MP4
#   ./tools/render.sh --all             # frames + encode
#
# Everything runs headless through bpy, so this works with no GPU, no display
# and no Blender install - just ./tools/setup_env.sh once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STUBS="${BLENDER_STUBS:-/tmp/blender-stubs}"
FRAMES="${VALLEY_OUT:-/tmp/valley_frames}"
OUT="$ROOT/render/mist-valley.mp4"

[ -x "$PY" ] || { echo "run ./tools/setup_env.sh first" >&2; exit 1; }

render() { LD_LIBRARY_PATH="$STUBS" "$PY" "$ROOT/valley.py" "$@"; }

encode() {
  mkdir -p "$(dirname "$OUT")"
  local ffmpeg
  ffmpeg=$("$PY" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
  # Frame 1 in the filename is "f_0001.png"; pattern must be %04d.
  "$ffmpeg" -y -framerate "${FPS:-24}" -i "$FRAMES/f_%04d.png" \
    -vf "scale=1280:720:flags=lanczos,noise=alls=6:allf=t+u,format=yuv420p" \
    -c:v libx264 -preset slow -crf 17 -tune film -movflags +faststart \
    "$OUT"
  echo "wrote $OUT"
}

case "${1:---all}" in
  --look)   VALLEY_PREVIEW=1 render ;;
  --frames) render ;;
  --encode) encode ;;
  --all)    render && encode ;;
  *) echo "usage: $0 [--look|--frames|--encode|--all]" >&2; exit 1 ;;
esac
