// Renders src/anim.js to a real MP4: every frame is drawn on a Skia canvas,
// streamed as raw RGBA into ffmpeg, then muxed with the narration.
//
//   node src/render.mjs                 → out/video-silent.mp4
//   node src/render.mjs --still 12.4    → out/still-12.40.png

import { createCanvas, GlobalFonts } from '@napi-rs/canvas';
import { spawn } from 'node:child_process';
import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { W, H, FPS, FRAMES, TOTAL, init, drawFrame } from './anim.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

// Register the same family aliases the browser player uses.
const fonts = [
  ['SpaceGrotesk-Bold.ttf', 'SG'],
  ['SpaceGrotesk-Medium.ttf', 'SGM'],
  ['Inter-Regular.ttf', 'Inter'],
  ['Inter-SemiBold.ttf', 'InterSB'],
  ['Inter-Bold.ttf', 'InterB'],
  ['JetBrainsMono-Regular.ttf', 'Mono'],
  ['JetBrainsMono-Medium.ttf', 'MonoM'],
];
for (const [file, alias] of fonts) {
  const p = join(root, 'fonts', file);
  if (!existsSync(p)) throw new Error(`missing font ${p}`);
  if (!GlobalFonts.registerFromPath(p, alias)) throw new Error(`failed to register ${file}`);
}

const canvas = createCanvas(W, H);
const ctx = canvas.getContext('2d');
init(ctx, (w, h) => createCanvas(w, h));

// ── single still, for checking the design ──────────────────────────────────
const stillIdx = process.argv.indexOf('--still');
if (stillIdx > -1) {
  const t = parseFloat(process.argv[stillIdx + 1] || '0');
  drawFrame(ctx, t);
  mkdirSync(join(root, 'out'), { recursive: true });
  const file = join(root, 'out', `still-${t.toFixed(2)}.png`);
  writeFileSync(file, canvas.toBuffer('image/png'));
  console.log(file);
  process.exit(0);
}

// ── full render ────────────────────────────────────────────────────────────
mkdirSync(join(root, 'out'), { recursive: true });
const outFile = join(root, 'out', 'video-silent.mp4');

const ff = spawn(
  FFMPEG,
  [
    '-y',
    '-f', 'rawvideo',
    '-pix_fmt', 'rgba',
    '-s', `${W}x${H}`,
    '-r', String(FPS),
    '-i', '-',
    '-an',
    '-vf', 'eq=contrast=1.05:saturation=1.10',
    '-c:v', 'libx264',
    '-preset', 'slow',
    '-crf', '21',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    outFile,
  ],
  { stdio: ['pipe', 'inherit', 'pipe'] }
);

ff.stderr.on('data', (d) => {
  const s = d.toString();
  if (/error|Invalid|failed/i.test(s)) process.stderr.write(s);
});

const pixelCount = W * H;

function writeFrame(i) {
  // @napi-rs/canvas exposes the raw RGBA buffer directly.
  const px = canvas.data();
  const buf = Buffer.isBuffer(px) ? px : Buffer.from(px.buffer, px.byteOffset, px.byteLength);
  if (buf.length !== pixelCount * 4) throw new Error(`bad frame size ${buf.length}`);
  if (!ff.stdin.write(buf)) {
    return new Promise((res) => ff.stdin.once('drain', res));
  }
  return Promise.resolve();
}

const t0 = Date.now();
for (let i = 0; i < FRAMES; i++) {
  const t = i / FPS;
  drawFrame(ctx, t);
  await writeFrame(i);
  if (i % 150 === 0 || i === FRAMES - 1) {
    const el = (Date.now() - t0) / 1000;
    const fps = (i + 1) / Math.max(el, 0.001);
    process.stdout.write(`\rframe ${i + 1}/${FRAMES}  t=${t.toFixed(2)}s  ${fps.toFixed(1)} f/s   `);
  }
}
ff.stdin.end();
await new Promise((res, rej) => ff.on('close', (c) => (c === 0 ? res() : rej(new Error('ffmpeg exit ' + c)))));
console.log(`\n✓ ${outFile}  (${TOTAL}s, ${FRAMES} frames)`);
