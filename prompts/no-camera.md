# "No Camera" — a 50-second explainer made of math

Made the same way every video in
[awesome-opus5-5-videos](https://github.com/yihui-dev/awesome-opus5-5-videos)
is made: the animation is written as code, not filmed and not generated as
footage. Remake on
[Skillry](https://skillry.dev/ai-videos/opus-5-5) · source:
[`../video/src/anim.js`](../video/src/anim.js)

## Prompt

> Make a 50-second, 1920×1080, 30 fps explainer called **NO CAMERA** about the
> fact that this very video is made of math. Write it as a single Canvas 2D file
> that exposes `drawFrame(ctx, t)` — no assets, no video, no images, no
> dependencies beyond a canvas. Six scenes, each one driven by time only:
>
> 1. **Cold open** — black. A caret types `render(0)`. Three lines slam in and
>    get struck through: NO CAMERA. / NO ACTORS. / NO FOOTAGE. They collapse
>    into "JUST CODE." beside a wall of thirty live frames ticking at 30 fps.
> 2. **One picture, repeated** — a filmstrip of nine frames, each showing the
>    same ball at a different moment on a sine path, with a glowing playhead
>    sweeping across and a `t = 0.00 → 1.00` axis underneath.
> 3. **The function** — a code panel types out `function frame(t)` while the
>    panel beside it paints the output: a ball riding the eased curve, leaving
>    ghosts, then orbiting satellites when it comes alive.
> 4. **Sound** — the real waveform of the narration, drawn from its actual
>    amplitude envelope, with a playhead locked to the voice and the syllables
>    marked as ticks on a timeline.
> 5. **The encoder** — a funnel of frame cards streaming into a glowing ENCODER
>    box and coming out as `no-camera.mp4`, with a live frame counter.
> 6. **Outro** — a gradient title, `1,507 frames · one canvas · zero footage`,
>    credits, and a final stamp: `render(t) → this.mp4`.
>
> Design: near-black `#05060B`, violet `#8B6BFF` → cyan `#3FE0E6`, Space Grotesk
> for headlines, Inter for body, JetBrains Mono for code and labels. Generous
> margins (content starts at x=160), a mono scene label top-left, a frame
> timecode top-right, burned-in captions of the narration, a hairline progress
> bar, film grain stepped at 15 Hz, a vignette and a slow 1.4% push-in.
>
> Voice: a young male explainer voice, crystal clear, generated from the script
> and then timed to the frame. No music.
>
> Render every frame deterministically and pipe the pixels into ffmpeg; make the
> same file play live in a browser with a scrubber.

## Narration (as generated)

1. This video has no camera. No actors, no footage. Just code, running at thirty frames a second.
2. Here's the trick. A video is really just a picture, repeated. The only thing that changes between frames is time.
3. So instead of filming, you write a function. Give it a moment, and it hands you back a frame. Move a circle with a sine wave, ease it into place, and suddenly it feels alive.
4. Sound works the same way. The voice you're hearing was generated from text, then every syllable was lined up on the timeline, so the picture hits exactly when I do.
5. Then the frames march into an encoder, thirty per second, and come out the other side as one single video file.
6. No studio. No crew. A few hundred lines of math, and a minute of video. That's how this one got made.

## Build notes

- Canvas: `@napi-rs/canvas` (Skia) in Node, the same `anim.js` in the browser.
- Encode: `libx264`, CRF 21, yuv420p, faststart, grain stepped to 15 Hz so the
  bitrate stays sane.
- Audio: `adelay` per clip onto the film's timeline, `loudnorm I=-15 LUFS`,
  AAC 192 kbps.
- Everything is deterministic: frame *n* of the MP4 is exactly `drawFrame(n/30)`.
