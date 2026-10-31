#!/usr/bin/env bash
# Build the whole film: frames -> silent video -> narration mix -> mp4
set -euo pipefail
cd "$(dirname "$0")"

# ffmpeg: use $FFMPEG, else PATH, else the binary bundled with imageio-ffmpeg
if [ -z "${FFMPEG:-}" ]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG="$(command -v ffmpeg)"
  else
    FFMPEG="$(python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  fi
fi
export FFMPEG
echo "→ ffmpeg: $FFMPEG"

node src/render.mjs
node src/mix.mjs

"$FFMPEG" -y -i out/video-silent.mp4 -i out/voice.wav \
  -c:v copy -c:a aac -b:a 192k -ar 48000 -shortest \
  -movflags +faststart out/no-camera.mp4
echo "✓ out/no-camera.mp4"
