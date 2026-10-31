// Lays the six narration clips onto the film's timeline and normalises them.
// The timings come from anim.js, so picture and voice can never drift apart.
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { STARTS, TOTAL } from './anim.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const FFMPEG = process.env.FFMPEG || 'ffmpeg';

const inputs = [];
const filters = [];
STARTS.forEach((start, i) => {
  inputs.push('-i', join(root, 'audio', `vo-0${i + 1}.mp3`));
  const ms = Math.round(start * 1000);
  filters.push(
    `[${i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,` +
      `adelay=${ms}|${ms}[a${i}]`
  );
});
const mix = STARTS.map((_, i) => `[a${i}]`).join('');
filters.push(
  `${mix}amix=inputs=${STARTS.length}:normalize=0:dropout_transition=0,` +
    `highpass=f=75,` +
    `loudnorm=I=-15:TP=-1.5:LRA=9,` +
    `aresample=48000,apad,atrim=0:${TOTAL}[out]`
);

const args = [
  '-y',
  ...inputs,
  '-filter_complex', filters.join(';'),
  '-map', '[out]',
  '-ar', '48000',
  '-c:a', 'pcm_s16le',
  join(root, 'out', 'voice.wav'),
];

const ff = spawn(FFMPEG, args, { stdio: ['ignore', 'inherit', 'pipe'] });
let err = '';
ff.stderr.on('data', (d) => (err += d.toString()));
ff.on('close', (c) => {
  if (c !== 0) {
    console.error(err.split('\n').slice(-12).join('\n'));
    process.exit(c);
  }
  console.log(`✓ out/voice.wav  (${TOTAL}s)`);
});
