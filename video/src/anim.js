// ─────────────────────────────────────────────────────────────────────────────
//  NO CAMERA — an explainer where every frame is math.
//
//  There is no footage in this project. This file draws the whole film as a
//  pure function of time: drawFrame(ctx, t). The Node renderer calls it 1,507
//  times and pipes the pixels into ffmpeg; the browser player calls it with
//  requestAnimationFrame and plays the narration alongside.
//
//  Coordinate space is always 1920 × 1080, regardless of display size.
// ─────────────────────────────────────────────────────────────────────────────

import { ENV04 } from './audio.js';

export const W = 1920;
export const H = 1080;
export const FPS = 30;

// Measured lengths of the six narration clips (seconds).
const VO = [6.39, 6.71, 9.59, 8.63, 6.07, 6.39];
const LEAD = 0.45; // silence before the first word
const GAP = 0.6; // breath between lines
const TAIL = 3.0; // room after the last word

export const STARTS = [];
{
  let acc = LEAD;
  for (const d of VO) {
    STARTS.push(+acc.toFixed(3));
    acc += d + GAP;
  }
}
export const VOICE_END = +(STARTS[5] + VO[5]).toFixed(3);
export const TOTAL = +(VOICE_END + TAIL).toFixed(3);
export const FRAMES = Math.round(TOTAL * FPS);
export const DURATIONS = VO;

// Visual windows: each scene starts a little before its line and hands over
// 0.35 s before the next one, so cuts land on the breath, not mid-word.
const MID = STARTS.slice(1).map((s) => +(s - 0.35).toFixed(3));
const VSTART = [0, ...MID];
const VEND = [...MID, TOTAL];

// ── tiny math kit ───────────────────────────────────────────────────────────
const TAU = Math.PI * 2;
const clamp = (v, a = 0, b = 1) => (v < a ? a : v > b ? b : v);
const lerp = (a, b, t) => a + (b - a) * t;
const inv = (a, b, v) => clamp((v - a) / (b - a));
const smooth = (t) => t * t * (3 - 2 * t);
const eOutCubic = (t) => 1 - (1 - t) ** 3;
const eInCubic = (t) => t ** 3;
const eInOutCubic = (t) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2);
const eOutExpo = (t) => (t >= 1 ? 1 : 1 - 2 ** (-10 * t));
const eOutQuint = (t) => 1 - (1 - t) ** 5;
const eOutBack = (t, s = 1.7) => 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2;
const hash = (n) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

// ── palette ─────────────────────────────────────────────────────────────────
const C = {
  bg: '#05060B',
  bg2: '#0A0B14',
  ink: '#F3F4FB',
  dim: '#7B7F96',
  faint: '#3A3D4D',
  violet: '#8B6BFF',
  blue: '#4D7CFE',
  cyan: '#3FE0E6',
  pink: '#FF5D8F',
  lime: '#C9F55C',
  amber: '#FCD34D',
};

// Font families are registered by the host (Node or the browser) under these
// names, so the exact same font strings work in both.
const SG = 'SG'; // Space Grotesk Bold
const SGM = 'SGM'; // Space Grotesk Medium
const IN = 'Inter'; // Inter Regular
const INSB = 'InterSB'; // Inter SemiBold
const INB = 'InterB'; // Inter Bold
const MO = 'Mono'; // JetBrains Mono
const MOM = 'MonoM'; // JetBrains Mono Medium

// ── drawing helpers ─────────────────────────────────────────────────────────
function rr(ctx, x, y, w, h, r) {
  const k = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + k, y);
  ctx.lineTo(x + w - k, y);
  ctx.arcTo(x + w, y, x + w, y + k, k);
  ctx.lineTo(x + w, y + h - k);
  ctx.arcTo(x + w, y + h, x + w - k, y + h, k);
  ctx.lineTo(x + k, y + h);
  ctx.arcTo(x, y + h, x, y + h - k, k);
  ctx.lineTo(x, y + k);
  ctx.arcTo(x, y, x + k, y, k);
  ctx.closePath();
}

function fillRR(ctx, x, y, w, h, r, fill) {
  rr(ctx, x, y, w, h, r);
  ctx.fillStyle = fill;
  ctx.fill();
}

function strokeRR(ctx, x, y, w, h, r, stroke, lw = 1) {
  rr(ctx, x, y, w, h, r);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lw;
  ctx.stroke();
}

// Text with optional letter-spacing (drawn per glyph so it renders identically
// in Skia and the browser).
function text(ctx, str, x, y, o = {}) {
  const size = o.size ?? 32;
  const fam = o.fam ?? IN;
  const color = o.color ?? C.ink;
  const align = o.align ?? 'left';
  const spacing = o.spacing ?? 0;
  const alpha = o.alpha ?? 1;
  if (alpha <= 0.002) return;
  ctx.save();
  ctx.font = `400 ${size}px ${fam}`;
  ctx.fillStyle = color;
  ctx.textBaseline = o.baseline ?? 'alphabetic';
  ctx.globalAlpha *= alpha;
  if (!spacing) {
    ctx.textAlign = align;
    ctx.fillText(str, x, y);
  } else {
    const chars = [...str];
    const widths = chars.map((c) => ctx.measureText(c).width);
    const total = widths.reduce((a, b) => a + b, 0) + spacing * Math.max(0, chars.length - 1);
    let cx = align === 'center' ? x - total / 2 : align === 'right' ? x - total : x;
    ctx.textAlign = 'left';
    for (let i = 0; i < chars.length; i++) {
      ctx.fillText(chars[i], cx, y);
      cx += widths[i] + spacing;
    }
  }
  ctx.restore();
}

function measure(ctx, str, size, fam, spacing = 0) {
  ctx.save();
  ctx.font = `400 ${size}px ${fam}`;
  const chars = [...str];
  const w = chars.reduce((a, c) => a + ctx.measureText(c).width, 0) + spacing * Math.max(0, chars.length - 1);
  ctx.restore();
  return w;
}

function glowDot(ctx, x, y, r, color, blur = 30, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, TAU);
  ctx.fill();
  ctx.restore();
}

function linear(ctx, x0, y0, x1, y1, stops) {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  for (const [p, c] of stops) g.addColorStop(p, c);
  return g;
}

function radial(ctx, x, y, r0, r1, stops) {
  const g = ctx.createRadialGradient(x, y, r0, x, y, r1);
  for (const [p, c] of stops) g.addColorStop(p, c);
  return g;
}

// ── captions (narration text, chunked and timed by character count) ─────────
const NARRATION = [
  'This video has no camera. No actors, no footage. Just code, running at thirty frames a second.',
  "Here's the trick. A video is really just a picture, repeated. The only thing that changes between frames is time.",
  'So instead of filming, you write a function. Give it a moment, and it hands you back a frame. Move a circle with a sine wave, ease it into place, and suddenly it feels alive.',
  "Sound works the same way. The voice you're hearing was generated from text, then every syllable was lined up on the timeline, so the picture hits exactly when I do.",
  'Then the frames march into an encoder, thirty per second, and come out the other side as one single video file.',
  "No studio. No crew. A few hundred lines of math, and a minute of video. That's how this one got made.",
];

function chunk(line, max) {
  const words = line.split(' ');
  const out = [];
  let cur = '';
  for (const w of words) {
    if (cur && (cur + ' ' + w).length > max) {
      out.push(cur);
      cur = w;
    } else cur = cur ? cur + ' ' + w : w;
  }
  if (cur) out.push(cur);
  return out;
}

export const CAPTIONS = [];
NARRATION.forEach((line, i) => {
  const parts = chunk(line, 44);
  const weights = parts.map((p) => p.length + (/[.!?]$/.test(p) ? 12 : 0));
  const sum = weights.reduce((a, b) => a + b, 0);
  let acc = STARTS[i];
  parts.forEach((p, k) => {
    const d = (VO[i] * weights[k]) / sum;
    CAPTIONS.push({ s: acc, e: acc + d, text: p });
    acc += d;
  });
});

// Syllable-ish ticks for scene 4, taken from the real envelope's local peaks.
const SYLLABLES = (() => {
  const out = [];
  const t0 = 0.14;
  let last = -1;
  for (let i = 2; i < ENV04.length - 1; i++) {
    const v = ENV04[i];
    if (v > 0.3 && v >= ENV04[i - 1] && v > ENV04[i + 1]) {
      const t = t0 + (i / ENV04.length) * (VO[3] - t0);
      if (t - last > 0.11) {
        out.push({ t, v });
        last = t;
      }
    }
  }
  return out;
})();

// ── assets prepared once ────────────────────────────────────────────────────
let noiseTile = null;
let noisePattern = null;
let vignette = null;

export function init(ctx, createCanvas) {
  // film grain tile
  const size = 220;
  const cv = createCanvas(size, size);
  const c2 = cv.getContext('2d');
  const img = c2.createImageData(size, size);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255;
    img.data[i] = v;
    img.data[i + 1] = v;
    img.data[i + 2] = v;
    img.data[i + 3] = 20;
  }
  c2.putImageData(img, 0, 0);
  noiseTile = cv;
  if (ctx.createPattern) noisePattern = ctx.createPattern(cv, 'repeat');
  vignette = radial(ctx, W / 2, H / 2, H * 0.52, H * 1.05, [
    [0, 'rgba(0,0,0,0)'],
    [1, 'rgba(0,0,0,0.42)'],
  ]);
}

// ── background ──────────────────────────────────────────────────────────────
function background(ctx, t, hueShift = 0) {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);

  const ax = W * 0.22 + Math.sin(t * 0.13) * 90;
  const ay = H * 0.2 + Math.cos(t * 0.11) * 60;
  ctx.fillStyle = radial(ctx, ax, ay, 0, 1000, [
    [0, 'rgba(124,92,246,0.20)'],
    [0.55, 'rgba(124,92,246,0.05)'],
    [1, 'rgba(124,92,246,0)'],
  ]);
  ctx.fillRect(0, 0, W, H);

  const bx = W * 0.82 + Math.cos(t * 0.09 + 1.4) * 110;
  const by = H * 0.78 + Math.sin(t * 0.12 + 0.7) * 70;
  ctx.fillStyle = radial(ctx, bx, by, 0, 980, [
    [0, 'rgba(63,224,230,0.14)'],
    [0.55, 'rgba(63,224,230,0.04)'],
    [1, 'rgba(63,224,230,0)'],
  ]);
  ctx.fillRect(0, 0, W, H);

  // faint dotted field
  const gx = 96;
  const ox = (t * 5) % gx;
  ctx.fillStyle = 'rgba(255,255,255,0.035)';
  for (let y = 60; y < H; y += gx) {
    for (let x = -gx + ox; x < W + gx; x += gx) {
      ctx.beginPath();
      ctx.arc(x, y, 1.6, 0, TAU);
      ctx.fill();
    }
  }

  // slow top light sweep
  const s = ((t * 0.06) % 1) * 2400 - 240;
  ctx.fillStyle = linear(ctx, s - 400, 0, s + 400, 0, [
    [0, 'rgba(255,255,255,0)'],
    [0.5, 'rgba(255,255,255,0.022)'],
    [1, 'rgba(255,255,255,0)'],
  ]);
  ctx.fillRect(0, 0, W, H);
}

// ── chrome: scene label, timecode ───────────────────────────────────────────
function chrome(ctx, t, label, alpha) {
  if (alpha <= 0.01) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  const y = 92;
  // left: dot + label
  ctx.fillStyle = linear(ctx, 140, 0, 400, 0, [
    [0, C.violet],
    [1, C.cyan],
  ]);
  ctx.beginPath();
  ctx.arc(148, y - 6, 4, 0, TAU);
  ctx.fill();
  text(ctx, label, 168, y, { size: 15, fam: MO, color: 'rgba(240,241,250,0.62)', spacing: 3 });
  // right: timecode
  const f = Math.floor(t * FPS);
  const ff = String(f % FPS).padStart(2, '0');
  const ss = String(Math.floor(t) % 60).padStart(2, '0');
  const mm = String(Math.floor(t / 60)).padStart(2, '0');
  text(ctx, `${mm}:${ss}:${ff}`, W - 140, y, {
    size: 15,
    fam: MO,
    color: 'rgba(240,241,250,0.34)',
    align: 'right',
    spacing: 2,
  });
  ctx.restore();
}

// ── SCENE 1 — cold open: this video has no camera ───────────────────────────
function scene1(ctx, lt) {
  const beatA = inv(0.35, 1.0, lt); // caret + render(0)
  const lines = ['NO CAMERA.', 'NO ACTORS.', 'NO FOOTAGE.'];
  const lineIn = [1.15, 2.35, 3.5];
  const strikeCols = [C.violet, C.cyan, C.pink];

  if (lt < 4.35) {
    // typing "render(0)"
    const typed = clamp((lt - 0.3) / 0.9);
    const s = 'render(0)';
    const n = Math.floor(typed * s.length);
    const shown = s.slice(0, n);
    const tw = measure(ctx, shown, 26, MO);
    const caretOn = lt < 0.28 ? Math.sin(lt * 18) > 0 : Math.floor(lt * 3) % 2 === 0;
    text(ctx, shown, 160, 300, { size: 26, fam: MO, color: 'rgba(240,241,250,0.45)' });
    if (caretOn) {
      ctx.fillStyle = 'rgba(240,241,250,0.75)';
      ctx.fillRect(160 + tw + 4, 280, 2, 26);
    }
  }

  for (let i = 0; i < 3; i++) {
    const p = clamp((lt - lineIn[i]) / 0.42);
    if (p <= 0) continue;
    const out = clamp((lt - 4.15 - i * 0.05) / 0.35);
    const alpha = 1 - out;
    const dim = i < 2 ? 0.34 * clamp((lt - lineIn[i + 1]) / 0.3) : 0;
    const a = (1 - dim) * alpha;
    const y = 470 + i * 108;
    const dy = lerp(46, 0, eOutCubic(p));
    text(ctx, lines[i], 160, y + dy, {
      size: 96,
      fam: SG,
      color: C.ink,
      alpha: a * clamp(p * 3),
      spacing: -1.5,
    });
    // accent strike
    const sp = clamp((lt - lineIn[i] - 0.18) / 0.45);
    if (sp > 0 && i < 2) {
      const w = measure(ctx, lines[i], 96, SG, -1.5);
      ctx.save();
      ctx.globalAlpha *= a * eOutCubic(sp);
      ctx.strokeStyle = strikeCols[i];
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(162, y - 34);
      ctx.lineTo(162 + w * eOutCubic(sp), y - 34);
      ctx.stroke();
      ctx.restore();
    }
  }

  // "just code, running at thirty frames a second" — a wall of live frames
  const gp = clamp((lt - 4.25) / 0.7);
  if (gp > 0) {
    const ga = eOutCubic(gp);
    const la = eOutCubic(clamp((lt - 4.5) / 0.6));

    // left: the claim, in the same column the crossed-out lines used
    text(ctx, 'JUST', 160, 470, { size: 96, fam: SG, color: C.ink, alpha: la, spacing: -1.5 });
    text(ctx, 'CODE.', 160, 578, { size: 96, fam: SG, color: C.ink, alpha: la, spacing: -1.5 });
    text(ctx, 'running at thirty frames a second', 160, 650, {
      size: 22,
      fam: MO,
      color: 'rgba(240,241,250,0.55)',
      alpha: la,
      spacing: 0.5,
    });

    // right: 30 frames, each alive at its own phase
    const cols = 6;
    const rows = 5;
    const cw = 132;
    const ch = 86;
    const gx = 14;
    const gy = 14;
    const tw = cols * cw + (cols - 1) * gx;
    const th = rows * ch + (rows - 1) * gy;
    const sx = 900;
    const sy = (H - th) / 2 - 40;

    for (let r = 0; r < rows; r++) {
      for (let cIdx = 0; cIdx < cols; cIdx++) {
        const i = r * cols + cIdx;
        const st = 4.25 + (cIdx + r) * 0.03;
        const p = clamp((lt - st) / 0.45);
        if (p <= 0) continue;
        const x = sx + cIdx * (cw + gx);
        const y = sy + r * (ch + gy) + (1 - eOutCubic(p)) * 12;
        const a = eOutCubic(p) * ga;
        const accent = i % 7 === 0;
        ctx.save();
        ctx.globalAlpha *= a;
        fillRR(ctx, x, y, cw, ch, 8, accent ? 'rgba(139,107,255,0.10)' : 'rgba(255,255,255,0.045)');
        strokeRR(ctx, x, y, cw, ch, 8, accent ? 'rgba(160,130,255,0.45)' : 'rgba(255,255,255,0.11)');
        for (let b = 0; b < 3; b++) {
          const ph = hash(i * 5.1 + b * 2.3) * TAU;
          const live = 0.5 + 0.5 * Math.sin(lt * (2.4 + hash(i + b) * 2) + ph);
          const bh = 8 + live * (ch - 26);
          const bw = 12;
          const bx = x + cw / 2 + (b - 1) * (bw + 7) - bw / 2;
          ctx.fillStyle = accent
            ? linear(ctx, 0, y + ch - 10 - bh, 0, y + ch - 10, [
                [0, C.cyan],
                [1, C.violet],
              ])
            : 'rgba(233,236,255,0.26)';
          if (accent) {
            ctx.shadowColor = 'rgba(124,92,246,0.85)';
            ctx.shadowBlur = 16;
          }
          ctx.fillRect(bx, y + ch - 10 - bh, bw, bh);
          ctx.shadowBlur = 0;
        }
        ctx.restore();
      }
    }

    const shownFrames = Math.floor(lt * FPS);
    text(ctx, `${String(shownFrames).padStart(4, '0')} frames · 30 fps`, sx + tw, sy + th + 46, {
      size: 16,
      fam: MO,
      color: 'rgba(240,241,250,0.45)',
      align: 'right',
      spacing: 2,
      alpha: ga,
    });
  }
}

// ── SCENE 2 — one picture, repeated ─────────────────────────────────────────
function scene2(ctx, lt) {
  const head = 'a video is just a picture, repeated.';
  const words = head.split(' ');
  let wx = 160;
  ctx.save();
  words.forEach((w, i) => {
    const p = clamp((lt - 0.1 - i * 0.055) / 0.4);
    const a = eOutCubic(p);
    const wid = measure(ctx, w + ' ', 62, SG);
    if (a > 0) {
      text(ctx, w, wx, 250 + (1 - a) * 22, { size: 62, fam: SG, color: C.ink, alpha: a });
    }
    wx += wid;
  });
  ctx.restore();

  const sub = 'the only thing that changes between frames is time.';
  const sp = clamp((lt - 5.0) / 0.6);
  if (sp > 0) {
    text(ctx, sub, 160, 318, {
      size: 33,
      fam: IN,
      color: 'rgba(240,241,250,0.62)',
      alpha: eOutCubic(sp),
    });
  }

  // film strip
  const n = 9;
  const cw = 168;
  const gap = 20;
  const stripW = n * cw + (n - 1) * gap;
  const sx = (W - stripW) / 2;
  const sy = 452;
  const ph = inv(1.4, 4.4, lt); // playhead travel 0..1
  const playX = sx + ph * stripW;

  for (let i = 0; i < n; i++) {
    const st = 0.5 + i * 0.085;
    const p = clamp((lt - st) / 0.55);
    if (p <= 0) continue;
    const a = eOutCubic(p);
    const x = sx + i * (cw + gap);
    const passed = playX > x + cw * 0.5;
    const lit = clamp((playX - x) / 60) * (passed ? 1 : 0);
    const dimA = passed ? 1 : 0.5;

    ctx.save();
    ctx.globalAlpha *= a * dimA;
    fillRR(ctx, x, sy + (1 - a) * 26, cw, cw, 12, 'rgba(255,255,255,0.035)');
    strokeRR(ctx, x, sy + (1 - a) * 26, cw, cw, 12, `rgba(255,255,255,${0.14 + lit * 0.18})`, 1 + lit * 0.6);
    // the picture inside: a ball on a sine path through time
    const tNorm = i / (n - 1);
    const bx = x + cw / 2;
    const by = sy + (1 - a) * 26 + cw / 2 + Math.sin(tNorm * TAU) * 44;
    // path remnant
    ctx.strokeStyle = `rgba(255,255,255,${0.05 + lit * 0.07})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let k = 0; k <= 40; k++) {
      const u = k / 40;
      const px = x + 16 + u * (cw - 32);
      const py = sy + (1 - a) * 26 + cw / 2 + Math.sin(u * TAU) * 44;
      k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    // ball
    const col = i === 0 ? C.violet : lit > 0.3 ? C.cyan : 'rgba(255,255,255,0.75)';
    ctx.shadowColor = lit > 0.3 ? 'rgba(63,224,230,0.9)' : 'rgba(255,255,255,0.35)';
    ctx.shadowBlur = 20 * lit + 6;
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(bx, by, 13, 0, TAU);
    ctx.fill();
    ctx.restore();

    // sprocket holes
    ctx.save();
    ctx.globalAlpha *= a * dimA * 0.5;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    for (let k = 0; k < 2; k++) {
      fillRR(ctx, x + 26 + k * (cw - 60), sy - 16, 34, 8, 3, 'rgba(255,255,255,0.18)');
      fillRR(ctx, x + 26 + k * (cw - 60), sy + cw + 8, 34, 8, 3, 'rgba(255,255,255,0.18)');
    }
    ctx.restore();
  }

  // playhead
  const pa = clamp((lt - 1.4) / 0.3);
  if (pa > 0 && ph < 1) {
    const py0 = sy - 34;
    const py1 = sy + cw + 34;
    ctx.save();
    ctx.globalAlpha *= pa;
    ctx.fillStyle = linear(ctx, 0, py0, 0, py1, [
      [0, 'rgba(63,224,230,0)'],
      [0.2, 'rgba(63,224,230,0.95)'],
      [0.8, 'rgba(63,224,230,0.95)'],
      [1, 'rgba(63,224,230,0)'],
    ]);
    ctx.shadowColor = 'rgba(63,224,230,0.9)';
    ctx.shadowBlur = 24;
    ctx.fillRect(playX - 1.5, py0, 3, py1 - py0);
    ctx.restore();
  }

  // time axis
  const ax = sy + cw + 74;
  const axisA = clamp((lt - 1.2) / 0.6);
  ctx.save();
  ctx.globalAlpha *= axisA;
  ctx.fillStyle = 'rgba(255,255,255,0.13)';
  ctx.fillRect(sx, ax, stripW, 1);
  for (let i = 0; i < n; i++) {
    const x = sx + i * (cw + gap) + cw / 2;
    ctx.fillStyle = 'rgba(255,255,255,0.22)';
    ctx.fillRect(x, ax - 5, 1, 10);
  }
  text(ctx, 't = 0.00', sx, ax + 34, { size: 16, fam: MO, color: 'rgba(240,241,250,0.42)', spacing: 1 });
  text(ctx, 't = 1.00', sx + stripW, ax + 34, {
    size: 16,
    fam: MO,
    color: 'rgba(240,241,250,0.42)',
    align: 'right',
    spacing: 1,
  });
  const tv = clamp(ph);
  text(ctx, `t = ${tv.toFixed(2)}`, sx + stripW / 2, ax + 34, {
    size: 16,
    fam: MO,
    color: C.cyan,
    align: 'center',
    spacing: 1,
    alpha: clamp((lt - 1.6) / 0.4),
  });
  ctx.restore();
}

// ── SCENE 3 — you write a function ──────────────────────────────────────────
const CODE = [
  { parts: [['function', C.violet], [' ', null], ['frame', C.cyan], ['(', C.dim], ['t', C.ink], [') {', C.dim]] },
  { parts: [['  ', null], ['const', C.pink], [' ', null], ['x', C.ink], [' = ', C.dim], ['ease', C.cyan], ['(t', C.ink], [')', C.dim], [' * W;', C.dim]] },
  { parts: [['  ', null], ['const', C.pink], [' ', null], ['y', C.ink], [' = ', C.dim], ['midY', C.ink], [' + ', C.dim], ['sin', C.cyan], ['(t * TAU) * 140;', C.dim]] },
  { parts: [['  ', null], ['return', C.violet], [' ', null], ['paint', C.cyan], ['(x, y);', C.dim]] },
  { parts: [['}', C.dim]] },
];

function typedChars(ctx, budget) {
  let used = 0;
  const rows = [];
  for (const line of CODE) {
    const chars = [];
    for (const [s, c] of line.parts) for (const ch of s) chars.push({ ch, c });
    rows.push(chars);
  }
  return rows;
}
const CODE_ROWS = typedChars(null, 0);

function scene3(ctx, lt, nl) {
  const hA = eOutCubic(clamp((lt - 0.15) / 0.5));
  if (hA > 0) {
    text(ctx, 'instead of filming, you write a function.', 160, 222, {
      size: 46,
      fam: SG,
      color: C.ink,
      alpha: hA,
    });
  }

  // code card
  const cx = 140;
  const cy = 292;
  const cw = 742;
  const ch = 470;
  const appear = eOutCubic(clamp((lt - 0.1) / 0.5));
  ctx.save();
  ctx.globalAlpha *= appear;
  ctx.translate(0, (1 - appear) * 18);
  fillRR(ctx, cx, cy, cw, ch, 18, 'rgba(10,11,20,0.92)');
  strokeRR(ctx, cx, cy, cw, ch, 18, 'rgba(255,255,255,0.09)');
  // titlebar
  ctx.fillStyle = 'rgba(255,255,255,0.045)';
  rr(ctx, cx, cy, cw, 52, 18);
  ctx.fill();
  ctx.fillStyle = 'rgba(10,11,20,1)';
  ctx.fillRect(cx, cy + 38, cw, 14);
  ctx.fillStyle = 'rgba(255,255,255,0.045)';
  ctx.fillRect(cx, cy + 38, cw, 14);
  const dots = ['#FF5F57', '#FEBC2E', '#28C840'];
  dots.forEach((c, i) => {
    ctx.fillStyle = c;
    ctx.globalAlpha *= 0.75;
    ctx.beginPath();
    ctx.arc(cx + 26 + i * 20, cy + 26, 5.5, 0, TAU);
    ctx.fill();
    ctx.globalAlpha /= 0.75;
  });
  text(ctx, 'frame.js', cx + cw - 24, cy + 32, {
    size: 14,
    fam: MO,
    color: 'rgba(240,241,250,0.35)',
    align: 'right',
  });

  // typing
  const typeStart = 0.55;
  const perChar = 0.035;
  const totalChars = CODE_ROWS.reduce((a, r) => a + r.length, 0);
  const budget = clamp((lt - typeStart) / (totalChars * perChar)) * totalChars;
  let fed = 0;
  let caretX = cx + 40;
  let caretY = cy + 108;
  ctx.font = `400 24px ${MO}`;
  for (let r = 0; r < CODE_ROWS.length; r++) {
    let x = cx + 40;
    const y = cy + 108 + r * 46;
    for (const c of CODE_ROWS[r]) {
      if (fed >= budget) break;
      ctx.fillStyle = c.c ?? 'rgba(240,241,250,0.7)';
      ctx.fillText(c.ch, x, y);
      x += ctx.measureText(c.ch).width;
      fed++;
      caretX = x;
      caretY = y;
    }
  }
  if (lt > 0.4 && (budget < totalChars ? true : Math.floor(lt * 2.4) % 2 === 0)) {
    ctx.fillStyle = C.cyan;
    ctx.globalAlpha *= 0.8;
    ctx.fillRect(caretX + 2, caretY - 20, 2, 26);
    ctx.globalAlpha /= 0.8;
  }
  ctx.restore();

  // output panel
  const ox = 946;
  const oy = 292;
  const ow = 834;
  const oh = 470;
  const panelIn = eOutCubic(clamp((lt - 0.85) / 0.6));
  if (panelIn > 0) {
    ctx.save();
    ctx.globalAlpha *= panelIn;
    fillRR(ctx, ox, oy, ow, oh, 18, 'rgba(255,255,255,0.022)');
    strokeRR(ctx, ox, oy, ow, oh, 18, 'rgba(255,255,255,0.08)');
    ctx.restore();

    const outA = eOutCubic(clamp((lt - 3.05) / 0.6));
    ctx.save();
    ctx.globalAlpha *= outA;
    // dot grid
    ctx.fillStyle = 'rgba(255,255,255,0.055)';
    for (let y = oy + 30; y < oy + oh - 16; y += 40) {
      for (let x = ox + 30; x < ox + ow - 16; x += 40) {
        ctx.beginPath();
        ctx.arc(x, y, 1.3, 0, TAU);
        ctx.fill();
      }
    }
    text(ctx, 'frame(t)', ox + 22, oy + 32, { size: 14, fam: MO, color: 'rgba(240,241,250,0.4)' });
    rr(ctx, ox, oy, ow, oh, 18);
    ctx.clip();

    // progress of the rendered moment, looping
    const u = ((lt - 3.05) * 0.22) % 1;
    const midY = oy + oh / 2 + 6;
    const x0 = ox + 70;
    const x1 = ox + ow - 70;
    const pos = (uu) => {
      const x = lerp(x0, x1, smooth(uu));
      const y = midY - Math.cos(uu * TAU * 1.5) * 112;
      return [x, y];
    };
    // full path
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let k = 0; k <= 80; k++) {
      const [px, py] = pos(k / 80);
      k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    // traced path
    ctx.strokeStyle = linear(ctx, x0, 0, x1, 0, [
      [0, C.violet],
      [1, C.cyan],
    ]);
    ctx.lineWidth = 2.6;
    ctx.shadowColor = 'rgba(124,92,246,0.75)';
    ctx.shadowBlur = 16;
    ctx.beginPath();
    const step = 80 * u;
    for (let k = 0; k <= step; k++) {
      const [px, py] = pos(k / 80);
      k ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // ball + ghosts
    const alive = clamp((lt - 7.2) / 0.8);
    const ghosts = 7 + Math.round(alive * 9);
    for (let g = ghosts; g >= 0; g--) {
      const uu = u - g * 0.012 * (1 + alive);
      if (uu < 0) continue;
      const [px, py] = pos(uu % 1);
      const fade = (1 - g / (ghosts + 1)) * (0.16 + alive * 0.34);
      ctx.fillStyle = `rgba(120,225,255,${fade * 0.75})`;
      ctx.beginPath();
      ctx.arc(px, py, 11 - g * 0.7, 0, TAU);
      ctx.fill();
    }
    const [hx, hy] = pos(u % 1);
    ctx.globalAlpha *= 1;
    glowDot(ctx, hx, hy, 13, '#EAFDFF', 40);
    ctx.fillStyle = C.cyan;
    ctx.shadowColor = 'rgba(63,224,230,1)';
    ctx.shadowBlur = 44;
    ctx.beginPath();
    ctx.arc(hx, hy, 9, 0, TAU);
    ctx.fill();
    ctx.shadowBlur = 0;

    // satellites — "suddenly it feels alive"
    if (alive > 0) {
      for (let i = 0; i < 7; i++) {
        const ang = lt * (1.1 + i * 0.15) + (i * TAU) / 7;
        const rad = 34 + i * 9 + Math.sin(lt * 2 + i) * 6;
        const px = hx + Math.cos(ang) * rad;
        const py = hy + Math.sin(ang) * rad * 0.7;
        const col = [C.violet, C.cyan, C.pink][i % 3];
        ctx.globalAlpha *= 0.85 * alive;
        glowDot(ctx, px, py, 2.6 + (i % 3), col, 18);
        ctx.globalAlpha /= 0.85 * alive;
      }
    }
    ctx.restore();
  }
}

// ── SCENE 4 — sound works the same way ──────────────────────────────────────
function scene4(ctx, lt, nl) {
  const quote = "the voice you're hearing was generated from text.";
  const qp = clamp((lt - 0.35) / 1.5);
  const qOut = clamp((lt - 2.5) / 0.5);
  if (qp > 0 && qOut < 1) {
    const n = Math.floor(quote.length * qp);
    const shown = quote.slice(0, n);
    ctx.save();
    ctx.globalAlpha *= 1 - eInCubic(qOut);
    const wq = 980;
    const xq = (W - wq) / 2;
    const yq = 236;
    fillRR(ctx, xq, yq, wq, 84, 14, 'rgba(255,255,255,0.035)');
    strokeRR(ctx, xq, yq, wq, 84, 14, 'rgba(255,255,255,0.09)');
    text(ctx, '“' + shown + (Math.floor(lt * 2.5) % 2 ? '▌' : ' ') + '”', xq + 34, yq + 54, {
      size: 28,
      fam: IN,
      color: 'rgba(240,241,250,0.9)',
    });
    ctx.restore();
  }

  // waveform from the actual narration envelope
  const baseY = 600;
  const x0 = 150;
  const x1 = 1770;
  const width = x1 - x0;
  const grow = clamp((lt - 2.35) / 0.9);
  if (grow > 0) {
    const bars = 260;
    const bw = width / bars;
    const playT = clamp(nl / VO[3]);
    const headX = x0 + playT * width;
    const peak = 210;

    for (let i = 0; i < bars; i++) {
      const u = i / (bars - 1);
      const src = ENV04[Math.floor(u * (ENV04.length - 1))];
      const wobble = 0.86 + 0.14 * Math.sin(lt * 9 + i * 0.35);
      const amp = Math.pow(clamp(src * 1.12), 0.78) * peak * wobble;
      const x = x0 + i * bw;
      const appeared = clamp((grow - u * 0.35) / 0.6);
      if (appeared <= 0) continue;
      const hh = Math.max(2.5, amp * appeared);
      const heard = x < headX;
      if (heard) {
        ctx.fillStyle = linear(ctx, 0, baseY - peak, 0, baseY + peak, [
          [0, 'rgba(139,107,255,0.95)'],
          [0.5, 'rgba(63,224,230,0.95)'],
          [1, 'rgba(139,107,255,0.95)'],
        ]);
        ctx.shadowColor = 'rgba(63,224,230,0.45)';
        ctx.shadowBlur = 10;
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.12)';
        ctx.shadowBlur = 0;
      }
      fillRR(ctx, x, baseY - hh, Math.max(2, bw * 0.42), hh * 2, bw * 0.21, ctx.fillStyle);
    }
    ctx.shadowBlur = 0;

    // playhead
    if (playT > 0 && playT < 1) {
      ctx.save();
      ctx.fillStyle = linear(ctx, 0, baseY - peak - 30, 0, baseY + peak + 30, [
        [0, 'rgba(255,255,255,0)'],
        [0.5, 'rgba(255,255,255,0.9)'],
        [1, 'rgba(255,255,255,0)'],
      ]);
      ctx.shadowColor = 'rgba(255,255,255,0.8)';
      ctx.shadowBlur = 18;
      ctx.fillRect(headX - 1, baseY - peak - 30, 2, peak * 2 + 60);
      ctx.restore();
      glowDot(ctx, headX, baseY, 5, '#FFFFFF', 30);
    }

    // level readout
    const lvl = ENV04[Math.min(ENV04.length - 1, Math.max(0, Math.floor(playT * ENV04.length)))];
    text(ctx, 'GENERATED VOICE', x0, baseY - peak - 92, {
      size: 14,
      fam: MO,
      color: 'rgba(240,241,250,0.45)',
      spacing: 4,
    });
    // meter
    const mx = x0 + 202;
    for (let i = 0; i < 22; i++) {
      const on = lvl > i / 22;
      const col = on ? (i > 17 ? C.pink : i > 12 ? C.cyan : C.violet) : 'rgba(255,255,255,0.1)';
      fillRR(ctx, mx + i * 7, baseY - peak - 103, 4, 12, 2, col);
    }
    text(ctx, 'waveform', x1, baseY - peak - 92, {
      size: 14,
      fam: MO,
      color: 'rgba(240,241,250,0.35)',
      align: 'right',
      spacing: 2,
    });

    // syllable ticks on a timeline
    const tlY = baseY + peak + 76;
    const sylA = clamp((lt - 3.2) / 0.8);
    ctx.save();
    ctx.globalAlpha *= sylA;
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.fillRect(x0, tlY, width, 1);
    for (const s of SYLLABLES) {
      const x = x0 + (s.t / VO[3]) * width;
      const on = playT * VO[3] > s.t;
      const h = 6 + s.v * 16;
      ctx.fillStyle = on ? 'rgba(63,224,230,0.95)' : 'rgba(255,255,255,0.2)';
      ctx.fillRect(x, tlY - h, 2, h * 2 * 0.6);
    }
    text(ctx, 'syllables · lined up on the timeline', x0, tlY + 42, {
      size: 15,
      fam: MO,
      color: 'rgba(240,241,250,0.4)',
      spacing: 2,
    });
    ctx.restore();
  }
}

// ── SCENE 5 — frames become a file ──────────────────────────────────────────
function scene5(ctx, lt, nl) {
  const head = 'then the frames become one file.';
  const hp = clamp((lt - 0.12) / 0.5);
  if (hp > 0) {
    text(ctx, head, 160, 248, { size: 58, fam: SG, color: C.ink, alpha: eOutCubic(hp) });
  }

  const boxX = 800;
  const boxY = 330;
  const boxW = 320;
  const boxH = 340;
  const mouthY = boxY + boxH / 2;
  const boxA = eOutCubic(clamp((lt - 0.45) / 0.6));
  const done = clamp(lt / 5.4);

  // a funnel of frames, all of them heading for the same mouth
  const lanes = 4;
  for (let lane = 0; lane < lanes; lane++) {
    const startY = 300 + lane * 152;
    for (let k = 0; k < 6; k++) {
      const speed = 0.155 + lane * 0.012;
      const progRaw = (lt * speed + k / 6 + lane * 0.137) % 1;
      const x = lerp(-220, boxX + 30, progRaw);
      const y = lerp(startY, mouthY, progRaw ** 1.7);
      const sc = lerp(1, 0.44, progRaw);
      const w = 120 * sc;
      const h = 72 * sc;
      const a = boxA * clamp(progRaw * 9) * clamp((1 - progRaw) * 7);
      if (a <= 0.012) continue;
      ctx.save();
      ctx.globalAlpha *= a;
      // motion streak behind the card
      ctx.fillStyle = linear(ctx, x - 150, 0, x, 0, [
        [0, 'rgba(139,107,255,0)'],
        [1, 'rgba(139,107,255,0.08)'],
      ]);
      ctx.fillRect(x - 150, y - h / 2 + 6, 150, h - 12);
      fillRR(ctx, x, y - h / 2, w, h, 8 * sc, 'rgba(255,255,255,0.05)');
      strokeRR(ctx, x, y - h / 2, w, h, 8 * sc, 'rgba(255,255,255,0.14)');
      const by = y + Math.sin(x * 0.03 + lt * 4) * 12 * sc;
      ctx.fillStyle = lane % 2 ? 'rgba(63,224,230,0.85)' : 'rgba(139,107,255,0.9)';
      ctx.shadowColor = lane % 2 ? 'rgba(63,224,230,0.8)' : 'rgba(139,107,255,0.8)';
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.arc(x + w / 2, by, 5.5 * sc, 0, TAU);
      ctx.fill();
      ctx.restore();
    }
  }

  // the encoder
  ctx.save();
  ctx.globalAlpha *= boxA;
  ctx.translate(0, (1 - boxA) * 20);
  ctx.shadowColor = 'rgba(139,107,255,0.45)';
  ctx.shadowBlur = 46;
  fillRR(ctx, boxX, boxY, boxW, boxH, 22, 'rgba(12,13,24,0.96)');
  ctx.shadowBlur = 0;
  strokeRR(ctx, boxX, boxY, boxW, boxH, 22, 'rgba(139,107,255,0.5)', 1.6);
  text(ctx, 'ENCODER', boxX + boxW / 2, boxY + 66, {
    size: 25,
    fam: SG,
    color: C.ink,
    align: 'center',
    spacing: 2,
  });
  text(ctx, 'h.264 · 30 fps', boxX + boxW / 2, boxY + 96, {
    size: 13,
    fam: MO,
    color: 'rgba(240,241,250,0.45)',
    align: 'center',
    spacing: 2,
  });
  // three rasping bars
  for (let b = 0; b < 3; b++) {
    const by = boxY + 132 + b * 30;
    const lvl = 0.35 + 0.65 * Math.abs(Math.sin(lt * (2.2 + b * 0.7) + b));
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    fillRR(ctx, boxX + 40, by, boxW - 80, 10, 5, 'rgba(255,255,255,0.07)');
    ctx.fillStyle = linear(ctx, boxX + 40, 0, boxX + boxW - 40, 0, [
      [0, C.violet],
      [1, C.cyan],
    ]);
    fillRR(ctx, boxX + 40, by, (boxW - 80) * lvl, 10, 5, ctx.fillStyle);
  }
  text(ctx, `${Math.round(done * 100)}%`, boxX + boxW / 2, boxY + boxH - 56, {
    size: 19,
    fam: MO,
    color: C.cyan,
    align: 'center',
  });
  text(ctx, `${String(Math.floor(done * 1507)).padStart(4, '0')} / 1507 frames`, boxX + boxW / 2, boxY + boxH - 24, {
    size: 12,
    fam: MO,
    color: 'rgba(240,241,250,0.4)',
    align: 'center',
    spacing: 1,
  });
  // mouth glow
  ctx.fillStyle = radial(ctx, boxX, mouthY, 0, 90, [
    [0, 'rgba(63,224,230,0.30)'],
    [1, 'rgba(63,224,230,0)'],
  ]);
  ctx.fillRect(boxX - 90, mouthY - 90, 120, 180);
  ctx.restore();

  // output file card
  const fcA = eOutCubic(clamp((lt - 1.5) / 0.7));
  if (fcA > 0) {
    const fx = 1216;
    const fy = 470;
    const fw = 504;
    const fh = 196;
    ctx.save();
    ctx.globalAlpha *= fcA;
    ctx.translate((1 - fcA) * 34, 0);
    ctx.shadowColor = 'rgba(63,224,230,0.25)';
    ctx.shadowBlur = 60;
    fillRR(ctx, fx, fy, fw, fh, 18, 'rgba(16,18,32,0.96)');
    ctx.shadowBlur = 0;
    strokeRR(ctx, fx, fy, fw, fh, 18, 'rgba(63,224,230,0.32)');
    fillRR(ctx, fx + 30, fy + 34, 62, 74, 10, 'rgba(255,255,255,0.05)');
    strokeRR(ctx, fx + 30, fy + 34, 62, 74, 10, 'rgba(255,255,255,0.16)');
    text(ctx, 'MP4', fx + 61, fy + 80, {
      size: 17,
      fam: MOM,
      color: C.cyan,
      align: 'center',
      spacing: 1,
    });
    text(ctx, 'no-camera.mp4', fx + 118, fy + 66, { size: 26, fam: INSB, color: C.ink });
    const mb = (2.1 + done * 12.6).toFixed(1);
    text(ctx, `${mb} MB · 1920×1080 · ${Math.round(TOTAL)} s`, fx + 118, fy + 100, {
      size: 15,
      fam: MO,
      color: 'rgba(240,241,250,0.45)',
    });
    text(ctx, done > 0.98 ? 'done' : 'encoding…', fx + 30, fy + fh - 74, {
      size: 13,
      fam: MO,
      color: done > 0.98 ? C.lime : 'rgba(240,241,250,0.45)',
      spacing: 2,
    });
    ctx.fillStyle = 'rgba(255,255,255,0.09)';
    fillRR(ctx, fx + 30, fy + fh - 52, fw - 60, 4, 2, 'rgba(255,255,255,0.09)');
    ctx.fillStyle = linear(ctx, fx + 30, 0, fx + fw - 30, 0, [
      [0, C.violet],
      [1, C.cyan],
    ]);
    fillRR(ctx, fx + 30, fy + fh - 52, (fw - 60) * done, 4, 2, ctx.fillStyle);
    ctx.restore();
  }

  // connector
  const ar = clamp((lt - 1.2) / 0.6);
  if (ar > 0) {
    ctx.save();
    ctx.globalAlpha *= ar;
    ctx.strokeStyle = 'rgba(63,224,230,0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 8]);
    ctx.beginPath();
    ctx.moveTo(boxX + boxW + 8, mouthY);
    ctx.lineTo(1186, mouthY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(63,224,230,0.75)';
    ctx.beginPath();
    ctx.moveTo(1206, mouthY);
    ctx.lineTo(1192, mouthY - 8);
    ctx.lineTo(1192, mouthY + 8);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

// ── SCENE 6 — outro ─────────────────────────────────────────────────────────
function scene6(ctx, lt) {
  const titleIn = eOutExpo(clamp((lt - 0.25) / 1.3));
  const outAll = clamp((lt - 6.9) / 0.9);

  ctx.save();
  ctx.globalAlpha *= 1 - outAll;

  text(ctx, 'AN EXPLAINER IN ONE FILE', W / 2, 392, {
    size: 16,
    fam: MO,
    color: 'rgba(240,241,250,0.45)',
    align: 'center',
    spacing: 6,
    alpha: clamp((lt - 0.15) / 0.7),
  });

  // gradient title
  const sc = lerp(0.955, 1, titleIn);
  ctx.save();
  ctx.translate(W / 2, 540);
  ctx.scale(sc, sc);
  ctx.globalAlpha *= clamp(titleIn * 1.6);
  const size = 168;
  const tw = measure(ctx, 'NO CAMERA', size, SG, 2);
  const grad = linear(ctx, -tw / 2, -120, tw / 2, 120, [
    [0, '#B9A7FF'],
    [0.45, '#FFFFFF'],
    [1, '#7DF0F4'],
  ]);
  ctx.font = `400 ${size}px ${SG}`;
  ctx.textAlign = 'center';
  ctx.fillStyle = grad;
  ctx.shadowColor = 'rgba(139,107,255,0.55)';
  ctx.shadowBlur = 60;
  ctx.fillText('NO CAMERA', 0, 0);
  ctx.shadowBlur = 0;
  ctx.restore();

  text(ctx, '1,507 frames · one canvas · zero footage', W / 2, 626, {
    size: 28,
    fam: IN,
    color: 'rgba(240,241,250,0.72)',
    align: 'center',
    alpha: clamp((lt - 1.2) / 0.8),
  });

  // credits
  const rows = [
    ['voice', 'generated from text, timed to the frame'],
    ['visuals', 'canvas 2d math · rendered frame by frame'],
    ['source', 'github.com/yihui-dev/awesome-opus5-5-videos'],
  ];
  rows.forEach((r, i) => {
    const a = clamp((lt - 3.3 - i * 0.35) / 0.7);
    if (a <= 0) return;
    const y = 764 + i * 46;
    text(ctx, r[0], W / 2 - 190, y, {
      size: 15,
      fam: MO,
      color: 'rgba(240,241,250,0.38)',
      align: 'right',
      alpha: a,
      spacing: 2,
    });
    text(ctx, r[1], W / 2 - 152, y, {
      size: 17,
      fam: MO,
      color: 'rgba(240,241,250,0.72)',
      alpha: a,
    });
  });
  ctx.restore();

  // final stamp after the last word
  const st = clamp((lt - 7.2) / 0.9);
  if (st > 0) {
    text(ctx, 'render(t)  →  this.mp4', W / 2, 560, {
      size: 34,
      fam: MOM,
      color: 'rgba(240,241,250,0.8)',
      align: 'center',
      spacing: 2,
      alpha: st,
    });
  }
}

// ── captions + grain + progress ─────────────────────────────────────────────
function captions(ctx, t) {
  if (t > VOICE_END + 0.35) return;
  const cap = CAPTIONS.find((c) => t >= c.s && t < c.e);
  if (!cap) return;
  const fin = clamp((t - cap.s) / 0.16);
  const fout = clamp((cap.e - t) / 0.16);
  const a = Math.min(fin, fout);
  ctx.save();
  ctx.globalAlpha *= a;
  const y = 1002;
  ctx.font = `400 32px ${INSB}`;
  let size = 32;
  let w = measure(ctx, cap.text, size, INSB);
  if (w > 1360) {
    size = Math.floor((size * 1360) / w);
    w = measure(ctx, cap.text, size, INSB);
  }
  fillRR(ctx, W / 2 - w / 2 - 26, y - 37, w + 52, 58, 12, 'rgba(5,6,12,0.45)');
  text(ctx, cap.text, W / 2, y, {
    size,
    fam: INSB,
    color: 'rgba(245,246,252,0.95)',
    align: 'center',
  });
  ctx.restore();
}

function overlay(ctx, t) {
  // vignette
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, W, H);

  // grain — stepped at 15 Hz so the encoder has something to hold on to
  if (noisePattern) {
    const g = Math.floor(t * 15);
    const ox = -hash(g) * 220;
    const oy = -hash(g + 91) * 220;
    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.translate(ox, oy);
    ctx.fillStyle = noisePattern;
    ctx.fillRect(0, 0, W + 220, H + 220);
    ctx.restore();
  }

  // progress hairline
  const p = clamp(t / TOTAL);
  ctx.fillStyle = 'rgba(255,255,255,0.07)';
  ctx.fillRect(0, H - 3, W, 3);
  ctx.fillStyle = linear(ctx, 0, 0, W, 0, [
    [0, C.violet],
    [0.5, C.cyan],
    [1, C.violet],
  ]);
  ctx.fillRect(0, H - 3, W * p, 3);
}

// ── the film ────────────────────────────────────────────────────────────────
const SCENES = [
  { label: '01 — COLD OPEN', fn: scene1 },
  { label: '02 — ONE PICTURE, REPEATED', fn: scene2 },
  { label: '03 — THE FUNCTION', fn: scene3 },
  { label: '04 — SOUND', fn: scene4 },
  { label: '05 — THE ENCODER', fn: scene5 },
  { label: '06 — OUTRO', fn: scene6 },
];

const FADE = 0.32;

export function drawFrame(ctx, t) {
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalAlpha = 1;
  ctx.shadowBlur = 0;

  // very slow push-in, so the frame breathes
  const zoom = 1 + 0.014 * (t / TOTAL);
  ctx.translate(W / 2, H / 2);
  ctx.scale(zoom, zoom);
  ctx.translate(-W / 2, -H / 2);

  background(ctx, t);

  for (let i = 0; i < SCENES.length; i++) {
    const v0 = VSTART[i];
    const v1 = VEND[i];
    if (t < v0 - 0.001 || t > v1 + 0.001) continue;
    const lt = t - v0;
    const nl = t - STARTS[i];
    let a = 1;
    if (i > 0) a *= clamp((t - v0) / FADE);
    if (i < SCENES.length - 1) a *= clamp((v1 - t) / FADE);
    if (a <= 0.002) continue;
    ctx.save();
    ctx.globalAlpha = a;
    SCENES[i].fn(ctx, lt, nl);
    ctx.restore();
    chrome(ctx, t, SCENES[i].label, a * clamp((i - 0.2) / 0.6) * (i === 0 ? 0 : 1));
  }

  captions(ctx, t);
  overlay(ctx, t);

  // fade from / to black
  const fadeIn = 1 - clamp(t / 0.55);
  const fadeOut = clamp((t - (TOTAL - 1.15)) / 1.15);
  const black = Math.max(fadeIn, fadeOut);
  if (black > 0.001) {
    ctx.fillStyle = `rgba(0,0,0,${black})`;
    ctx.fillRect(-W, -H, W * 3, H * 3);
  }
  ctx.restore();
}
