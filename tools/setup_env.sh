#!/usr/bin/env bash
# Bootstraps a headless Blender (bpy) render environment inside a bare Debian
# container that has no GPU, no X server and no OpenGL stack.
#
# bpy's shared libraries still *link* against libGL / libXrender / libXi /
# libxkbcommon. Cycles renders on the CPU and never calls into them, so we
# generate empty shims that satisfy the dynamic linker, then self-heal by
# adding whatever symbol the loader complains about next. This lets a real
# Blender build run with zero display infrastructure - which is what makes
# path-traced animation possible in this sandbox.
#
#   ./tools/setup_env.sh                 # idempotent: venv + shims
#   LD_LIBRARY_PATH=/tmp/blender-stubs .venv/bin/python forest.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
STUBS="${BLENDER_STUBS:-/tmp/blender-stubs}"

if [ ! -x "$PY" ]; then
  python3 -m venv "$ROOT/.venv"
  "$PY" -m pip -q install --upgrade pip
  "$PY" -m pip -q install bpy==5.0.1 imageio-ffmpeg numpy
fi

mkdir -p "$STUBS"
SYMS="$STUBS/stub_symbols.txt"
STUB_SRC="$STUBS/stub.c"
touch "$SYMS"

# Names that must be emitted as data (function-pointer variables), not functions.
is_data_sym() { case "$1" in _X*_fn) return 0 ;; *) return 1 ;; esac; }

# --- xkbcommon: bpy's UI code wants the symbols under version V_0.5.0 --------
gen_xkb() {
  local map="$STUBS/xkb.map"
  cat > "$map" <<'EOF'
V_0.5.0 { global: *; };
EOF
  "$PY" - "$ROOT" "$STUBS" <<'PYEOF'
import re, subprocess, sys, pathlib
root, stubs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
lib = root / '.venv/lib/python3.11/site-packages/bpy'
nm = subprocess.run(['nm', '-D', '--undefined-only', *map(str, lib.glob('*.so'))],
                    capture_output=True, text=True).stdout
syms = sorted({m.group(1) + '@V_0.5.0' for m in re.finditer(r'\b(xkb_\w+)@V_0\.5\.0', nm)})
src = "\n".join("long %s(void){return 0;}" % s.split('@')[0] for s in syms) + "\n"
(stubs / 'xkb.c').write_text(src or "int xkb_placeholder;\n")
print(f"  xkbcommon shim: {len(syms)} symbols")
PYEOF
  gcc -shared -fPIC -w -o "$STUBS/libxkbcommon.so.0" "$STUBS/xkb.c" \
      -Wl,--version-script="$STUBS/xkb.map"
}

# --- GL / Xrender / Xi / SM / ICE shims --------------------------------------
gen_gl() {
  "$PY" - "$ROOT" "$SYMS" "$STUB_SRC" <<'PYEOF'
import re, subprocess, sys, pathlib
root, syms_file, src_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
lib = root / '.venv/lib/python3.11/site-packages/bpy'
nm = subprocess.run(['nm', '-D', '--undefined-only', *map(str, lib.glob('*.so'))],
                    capture_output=True, text=True).stdout
# Every undefined X*/gl*/egl* symbol is a candidate for shimming.
found = {m.group(1) for m in re.finditer(r'^\s*U (\S+)$', nm, re.M)
         if re.match(r'^(X[A-Z]|XFixes|Xkb|Xi[A-Z]|_Xi|gl[A-Z]|glX|gl_|egl|EGL|GL)', m.group(1))}
syms = set(syms_file.read_text().split()) | found
syms_file.write_text("\n".join(sorted(syms)) + "\n")
lines = []
for s in sorted(syms):
    if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', s):
        continue
    data = s.startswith('_X') and s.endswith('_fn')
    lines.append("void *%s;" % s if data else "long %s(void){return 0;}" % s)
src_file.write_text("\n".join(lines) + "\n")
print(f"  GL shim: {len(lines)} symbols")
PYEOF
  gcc -shared -fPIC -w -o "$STUBS/glstub.so" "$STUB_SRC"
  # Libs the loader expects to exist. Real libX11/libXext/libxcb are kept: the
  # image ships them and Xrender/Xfixes/Xi depend on them.
  for f in libGL.so.1 libGLX.so.0 libGLdispatch.so.0 libOpenGL.so.0 libEGL.so.1 \
           libGLU.so.1 libXrender.so.1 libXi.so.6 libSM.so.6 libICE.so.6 \
           libXfixes.so.3 libXrandr.so.2 libXinerama.so.1 libXcursor.so.1 \
           libXxf86vm.so.1; do
    cp "$STUBS/glstub.so" "$STUBS/$f"
  done
}

gen_xkb
for attempt in $(seq 1 30); do
  gen_gl
  out=$(LD_LIBRARY_PATH="$STUBS" "$PY" -c "import bpy; print('bpy', bpy.app.version_string)" 2>&1) || true
  if grep -q '^bpy ' <<<"$out"; then
    echo "OK $(grep '^bpy ' <<<"$out")"
    echo "run with: LD_LIBRARY_PATH=$STUBS $PY <script.py>"
    exit 0
  fi
  sym=$(grep -oP 'undefined symbol: \K[^,)]*' <<<"$out" | head -1)
  if [ -z "$sym" ]; then
    echo "Unrecoverable:" >&2
    tail -5 <<<"$out" >&2
    exit 1
  fi
  echo "  + shim $sym"
  printf '%s\n' "$sym" >> "$SYMS"
done
echo "gave up after 30 attempts" >&2
exit 1
