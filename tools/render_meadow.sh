#!/usr/bin/env bash
# Render + deliver the meadow shot.
#
#   ./tools/render_meadow.sh --look       # 3 cheap preview stills
#   ./tools/render_meadow.sh --frames     # full 1080p master frames
#   ./tools/render_meadow.sh --encode     # master -> 16:9 + 1:1 deliverables
#   ./tools/render_meadow.sh --all        # frames + encode
#
# The 1:1 square is cropped from the same 1920x1080 master, so the shot was
# framed with a safe centre square and both versions come from one render.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STUBS="${BLENDER_STUBS:-/tmp/blender-stubs}"
FRAMES="${MEADOW_OUT:-/tmp/meadow_frames}"
FPS="${FPS:-24}"
OUT="$ROOT/render"

[ -x "$PY" ] || { echo "run ./tools/setup_env.sh first" >&2; exit 1; }

ffmpeg_bin() { "$PY" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"; }

render() { LD_LIBRARY_PATH="$STUBS" "$PY" "$ROOT/meadow.py" "$@"; }

encode() {
  local ff; ff="$(ffmpeg_bin)"
  mkdir -p "$OUT"
  [ -f "$FRAMES/f_0001.png" ] || { echo "no frames in $FRAMES" >&2; exit 1; }

  # 16:9 master, 1080p. A whisper of temporal grain hides banding in the sky
  # gradient and stops the slow push-in looking like a still.
  "$ff" -y -framerate "$FPS" -i "$FRAMES/f_%04d.png" \
    -vf "noise=alls=4:allf=t+u,format=yuv420p" \
    -c:v libx264 -preset slow -crf 17 -tune film -profile:v high \
    -movflags +faststart "$OUT/meadow-16x9.mp4"

  # 1:1 square, cropped from the centre of the same master. Scaled to 1080x1080
  # so it uploads square without YouTube adding pillarbox bars.
  "$ff" -y -framerate "$FPS" -i "$FRAMES/f_%04d.png" \
    -vf "crop=1080:1080:(iw-1080)/2:0,noise=alls=4:allf=t+u,format=yuv420p" \
    -c:v libx264 -preset slow -crf 17 -tune film -profile:v high \
    -movflags +faststart "$OUT/meadow-1x1.mp4"

  # poster frame from the master
  "$ff" -y -i "$FRAMES/f_0180.png" -vf "noise=alls=4:allf=t+u" -frames:v 1 -update 1 "$OUT/meadow-poster.png"

  ls -lh "$OUT"/meadow-*
}

case "${1:---all}" in
  --look)   MEADOW_PREVIEW=1 render ;;
  --frames) render ;;
  --encode) encode ;;
  --all)    render && encode ;;
  *) echo "usage: $0 [--look|--frames|--encode|--all]" >&2; exit 1 ;;
esac
