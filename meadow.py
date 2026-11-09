"""
Golden meadow at sunrise - a bright, photoreal path-traced shot.

Framed for a safe centre square so one 1920x1080 master serves both a 16:9
upload and a 1:1 crop (no double render).

Rendered headless with Blender/Cycles on CPU. Every blade of grass, every
flower, the trees and the hills are procedural geometry.

    MEADOW_PREVIEW=1 ./tools/render.sh --look
    ./tools/render.sh --all
"""
import math
import os
import random

import bpy
from mathutils import Vector, noise

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
RES_X = int(os.environ.get("MEADOW_RES_X", 1920))
RES_Y = int(os.environ.get("MEADOW_RES_Y", 1080))
SAMPLES = int(os.environ.get("MEADOW_SAMPLES", 6))
FRAMES = int(os.environ.get("MEADOW_FRAMES", 360))
FPS = int(os.environ.get("MEADOW_FPS", 24))
SEED = int(os.environ.get("MEADOW_SEED", 3))
OUT_DIR = os.environ.get("MEADOW_OUT", "/tmp/meadow_frames")
PREVIEW = bool(int(os.environ.get("MEADOW_PREVIEW", 0)))
SINGLE = int(os.environ.get("MEADOW_SINGLE", 0))
STILLS = [int(v) for v in os.environ.get("MEADOW_STILLS", "").split(",") if v.strip()]

if PREVIEW:
    RES_X, RES_Y, SAMPLES = 640, 360, 12

rng = random.Random(SEED)

# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = float(os.environ.get("MEADOW_ADAPT", 0.05))
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = int(os.environ.get("MEADOW_BOUNCES", 4))
scene.cycles.diffuse_bounces = int(os.environ.get("MEADOW_DIFF", 2))
scene.cycles.glossy_bounces = 2
scene.cycles.transmission_bounces = 1
scene.cycles.transparent_max_bounces = int(os.environ.get("MEADOW_TRANS", 4))
scene.cycles.volume_bounces = 0
scene.cycles.sample_clamp_indirect = 8.0
scene.render.use_persistent_data = True   # static geometry: BVH built once
scene.cycles.use_light_tree = True
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.film_transparent = False
scene.render.use_motion_blur = False       # static scene + slow camera: not needed
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"
scene.render.filepath = os.path.join(OUT_DIR, "f_")

scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = 0.30           # bright, positive morning

# ----------------------------------------------------------------------------
# sky: warm dawn gradient + sun disc
# ----------------------------------------------------------------------------
world = bpy.data.worlds.new("Dawn")
scene.world = world
world.use_nodes = True
wnt = world.node_tree
wbg = wnt.nodes["Background"]
wbg.inputs["Strength"].default_value = 1.4

wcoord = wnt.nodes.new("ShaderNodeTexCoord")
wsep = wnt.nodes.new("ShaderNodeSeparateXYZ")
wnt.links.new(wcoord.outputs["Generated"], wsep.inputs[0])
# Sky: deep blue overhead, a warm band only near the sun, plus soft streaky
# clouds. Generated coords on a Background node are the view vector, so Z=0.5
# is exactly the horizon.
sky_sep = wnt.nodes.new("ShaderNodeSeparateXYZ")
wnt.links.new(wcoord.outputs["Generated"], sky_sep.inputs[0])

sky_map = wnt.nodes.new("ShaderNodeMapRange")
sky_map.inputs["From Min"].default_value = 0.0
sky_map.inputs["From Max"].default_value = 0.42
sky_map.clamp = True
wnt.links.new(sky_sep.outputs["Z"], sky_map.inputs["Value"])

sky_ramp = wnt.nodes.new("ShaderNodeValToRGB")
sky_ramp.color_ramp.elements[0].position = 0.0
sky_ramp.color_ramp.elements[0].position = 0.0
sky_ramp.color_ramp.elements[0].color = (1.0, 0.84, 0.64, 1.0)   # warm haze at the horizon
sky_ramp.color_ramp.elements[1].position = 0.09
sky_ramp.color_ramp.elements[1].color = (0.80, 0.87, 0.98, 1.0)  # pale transition
sky_ramp.color_ramp.elements.new(0.55)
sky_ramp.color_ramp.elements[2].color = (0.26, 0.55, 1.0, 1.0)   # clear sky blue
sky_ramp.color_ramp.elements.new(1.0)
sky_ramp.color_ramp.elements[3].color = (0.09, 0.32, 0.85, 1.0)  # deeper overhead
wnt.links.new(sky_map.outputs["Result"], sky_ramp.inputs["Fac"])

# soft streaky cloud deck, only above the horizon
cnoise = wnt.nodes.new("ShaderNodeTexNoise")
cnoise.inputs["Scale"].default_value = 2.4
cnoise.inputs["Detail"].default_value = 6.0
cnoise.inputs["Roughness"].default_value = 0.62
cmap = wnt.nodes.new("ShaderNodeMapping")
cmap.inputs["Scale"].default_value = (1.0, 0.35, 1.0)   # stretch into streaks
wnt.links.new(wcoord.outputs["Generated"], cmap.inputs["Vector"])
wnt.links.new(cmap.outputs["Vector"], cnoise.inputs["Vector"])

cloud_ramp = wnt.nodes.new("ShaderNodeValToRGB")
cloud_ramp.color_ramp.elements[0].position = 0.52
cloud_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
cloud_ramp.color_ramp.elements[1].position = 0.70
cloud_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
wnt.links.new(cnoise.outputs["Fac"], cloud_ramp.inputs["Fac"])

# keep clouds off the horizon line
hz = wnt.nodes.new("ShaderNodeMapRange")
hz.inputs["From Min"].default_value = 0.0
hz.inputs["From Max"].default_value = 0.05
wnt.links.new(sky_sep.outputs["Z"], hz.inputs["Value"])
cloud_mask = wnt.nodes.new("ShaderNodeMath")
cloud_mask.operation = "MULTIPLY"
wnt.links.new(cloud_ramp.outputs["Color"], cloud_mask.inputs[0])
wnt.links.new(hz.outputs["Result"], cloud_mask.inputs[1])

sky_mix = wnt.nodes.new("ShaderNodeMix")
sky_mix.data_type = "RGBA"
sky_mix.inputs["Factor"].default_value = 0.0
wnt.links.new(cloud_mask.outputs[0], sky_mix.inputs["Factor"])
wnt.links.new(sky_ramp.outputs["Color"], sky_mix.inputs[6])
sky_mix.inputs[7].default_value = (0.98, 0.96, 0.94, 1.0)   # sunlit cloud white
wnt.links.new(sky_mix.outputs[2], wbg.inputs["Color"])

# ----------------------------------------------------------------------------
# sun: low, warm, just above the treeline
# ----------------------------------------------------------------------------
SUN_DIR = Vector((0.12, 0.98, 0.145)).normalized()

sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 9.5
sun_data.color = (1.0, 0.70, 0.38)
sun_data.angle = math.radians(2.2)   # slightly soft: golden hour, not noon
sun = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (-SUN_DIR).to_track_quat("-Z", "Y").to_euler()

# sun disc so the light has a visible source behind the grass
bpy.ops.mesh.primitive_uv_sphere_add(radius=11.0, location=SUN_DIR * 900.0)
disc = bpy.context.object
disc.name = "sun_disc"
for c in list(disc.users_collection):
    c.objects.unlink(disc)
sun_mat = bpy.data.materials.new("sun_disc")
sun_mat.use_nodes = True
smt = sun_mat.node_tree
smt.nodes.remove(smt.nodes["Principled BSDF"])
emit = smt.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (1.0, 0.80, 0.48, 1.0)
emit.inputs["Strength"].default_value = 55.0
smt.links.new(emit.outputs["Emission"], smt.nodes["Material Output"].inputs["Surface"])
disc.data.materials.append(sun_mat)
for p in disc.data.polygons:
    p.use_smooth = True
scene.collection.objects.link(disc)

# cool sky fill so shadows stay blue and open, never muddy
fill_data = bpy.data.lights.new("skyfill", "AREA")
fill_data.energy = 300.0
fill_data.size = 400.0
fill_data.color = (0.60, 0.75, 1.0)
fill = bpy.data.objects.new("skyfill", fill_data)
scene.collection.objects.link(fill)
fill.location = (0.0, -30.0, 90.0)
fill.rotation_euler = (math.radians(30), 0, 0)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def new_collection(name):
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def link(obj, col):
    col.objects.link(obj)
    return obj


def principled(name, base, rough, metal=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return mat, b


# Camera dolly path (start -> end), used to keep geometry out of the lens.
CAM_A = (-1.1, 40.0)
CAM_B = (1.5, 38.6)


def dist_to_cam_path(x, y):
    ax, ay = CAM_A
    bx, by = CAM_B
    vx, vy = bx - ax, by - ay
    wx, wy = x - ax, y - ay
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / (vx * vx + vy * vy)))
    px, py = ax + vx * t, ay + vy * t
    return math.hypot(x - px, y - py)


def ground_height(x, y):
    """Gentle rolling meadow: long swells, no drama."""
    return (noise.noise(Vector((x * 0.018, y * 0.018, 0.0))) * 2.6
            + noise.noise(Vector((x * 0.07, y * 0.07, 5.0))) * 0.55
            + noise.noise(Vector((x * 0.3, y * 0.3, 9.0))) * 0.10)


# ----------------------------------------------------------------------------
# meadow floor
# ----------------------------------------------------------------------------
floor_col = new_collection("floor")
bpy.ops.mesh.primitive_grid_add(x_subdivisions=160, y_subdivisions=160, size=420)
floor = bpy.context.object
floor.name = "meadow"
floor.location = (0, 120.0, 0.0)
for v in floor.data.vertices:
    v.co.z = ground_height(v.co.x, v.co.y + 120.0)
for p in floor.data.polygons:
    p.use_smooth = True
for c in floor.users_collection:
    c.objects.unlink(floor)
link(floor, floor_col)

floor_mat, floor_b = principled("meadow_ground", (0.10, 0.16, 0.05), 0.95)
floor_b.inputs["Base Color"].default_value = (0.13, 0.19, 0.06, 1.0)
if "Specular IOR Level" in floor_b.inputs:
    floor_b.inputs["Specular IOR Level"].default_value = 0.08
fnt = floor_mat.node_tree
fcoord = fnt.nodes.new("ShaderNodeTexCoord")
# patchy grass colour: sun-bleached yellow-green vs deep green
fn = fnt.nodes.new("ShaderNodeTexNoise")
fn.inputs["Scale"].default_value = 0.35
fn.inputs["Detail"].default_value = 6.0
fnt.links.new(fcoord.outputs["Object"], fn.inputs["Vector"])
framp = fnt.nodes.new("ShaderNodeValToRGB")
framp.color_ramp.elements[0].position = 0.38
framp.color_ramp.elements[0].color = (0.075, 0.125, 0.035, 1)
framp.color_ramp.elements[1].position = 0.66
framp.color_ramp.elements[1].color = (0.20, 0.24, 0.075, 1)
fnt.links.new(fn.outputs["Fac"], framp.inputs["Fac"])
fnt.links.new(framp.outputs["Color"], floor_b.inputs["Base Color"])
floor.data.materials.append(floor_mat)


# ----------------------------------------------------------------------------
# grass: instanced blade clumps. Static geometry -> BVH cached across frames.
# ----------------------------------------------------------------------------
grass_col = new_collection("grass")
grass, grass_b = principled("grass_blade", (0.16, 0.24, 0.045), 0.45)
try:
    grass_b.inputs["Transmission Weight"].default_value = 0.26  # backlit glow
except KeyError:
    pass
# colour varies along the blade: darker at the root, bright toward the tip
gnt = grass.node_tree
gcoord = gnt.nodes.new("ShaderNodeTexCoord")
gsep = gnt.nodes.new("ShaderNodeSeparateXYZ")
gnt.links.new(gcoord.outputs["Generated"], gsep.inputs[0])
gmap = gnt.nodes.new("ShaderNodeMapRange")
gmap.inputs["From Min"].default_value = 0.0
gmap.inputs["From Max"].default_value = 1.0
gnt.links.new(gsep.outputs["Z"], gmap.inputs["Value"])
gramp = gnt.nodes.new("ShaderNodeValToRGB")
gramp.color_ramp.elements[0].position = 0.0
gramp.color_ramp.elements[0].color = (0.075, 0.115, 0.022, 1)  # shaded root
gramp.color_ramp.elements[1].position = 1.0
gramp.color_ramp.elements[1].color = (0.34, 0.36, 0.055, 1)    # sunlit golden tip
gnt.links.new(gmap.outputs["Result"], gramp.inputs["Fac"])
gnt.links.new(gramp.outputs["Color"], grass_b.inputs["Base Color"])


def blade_clump_mesh(name, blades=7, height=1.0):
    """A tuft of tapered, slightly curved blades - the workhorse of the meadow."""
    verts, faces = [], []
    for i in range(blades):
        a = i / blades * math.tau + rng.uniform(-0.5, 0.5)
        lean = rng.uniform(0.10, 0.45)
        h = height * rng.uniform(0.55, 1.0)
        w = rng.uniform(0.012, 0.022)
        dx, dy = math.cos(a), math.sin(a)
        b = len(verts)
        # 3 segments so the blade can curve
        segs = 3
        for s in range(segs + 1):
            t = s / segs
            z = h * t
            px = dx * (lean * h * t * t) + dx * 0.02 * math.sin(t * 3.0)
            py = dy * (lean * h * t * t) + dy * 0.02 * math.sin(t * 3.0)
            ww = w * (1.0 - 0.85 * t)
            verts.append((px - dy * ww, py + dx * ww, z))
            verts.append((px + dy * ww, py - dx * ww, z))
        for s in range(segs):
            b0 = b + s * 2
            faces.append((b0, b0 + 1, b0 + 3, b0 + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(grass)
    me.shade_smooth()
    return me


def blade_geometry(blades=7, height=1.0):
    """One tuft: narrow curved blades that taper to a point.

    Blade width is the single biggest quality lever here - wide quads read as
    paper shards at 1080p, fine blades read as grass. Each blade also curves
    along its length and twists slightly, so it catches the light unevenly.
    """
    V, F = [], []
    for i in range(blades):
        a = i / blades * math.tau + rng.uniform(-0.55, 0.55)
        lean = rng.uniform(0.15, 0.65)
        h = height * rng.uniform(0.45, 1.0)
        w0 = rng.uniform(0.0045, 0.0095)     # fine blades
        dx, dy = math.cos(a), math.sin(a)
        # sideways splay direction for the twist
        twist = rng.uniform(-0.5, 0.5)
        segs = 4
        base = len(V)
        for sg in range(segs + 1):
            t = sg / segs
            # arc: sweeps out then droops back down at the tip
            out = lean * h * (t ** 1.7)
            z = h * (t - 0.35 * t ** 3.4)
            px = dx * out
            py = dy * out
            # width tapers to a point
            w = w0 * (1.0 - t) ** 0.85 + 0.0005
            # twist makes the blade turn its face to the sky along its length
            ta = twist * t
            nx = -dy * math.cos(ta) + dx * math.sin(ta)
            ny = dx * math.cos(ta) + dy * math.sin(ta)
            V.append((px - nx * w, py - ny * w, z))
            V.append((px + nx * w, py + ny * w, z))
        for sg in range(segs):
            b0 = base + sg * 2
            F.append((b0, b0 + 1, b0 + 3, b0 + 2))
    return V, F


def build_merged(name, n_tufts, radius, inner, scale_rng, material):
    """One mesh holding thousands of tufts. A single object means Cycles builds
    one small BVH instead of tens of thousands - the difference between a
    14-minute still and a fast frame."""
    V, F = [], []
    for i in range(n_tufts):
        r = radius * (rng.random() ** 1.55)
        r = max(r, inner)
        a = rng.uniform(0, math.tau)
        x = math.cos(a) * r
        y = 40.0 + math.sin(a) * r * 0.85
        if dist_to_cam_path(x, y) < 4.2:
            continue          # keep a clear pocket around the camera
        z = ground_height(x, y)
        rot = rng.uniform(0, math.tau)
        s = rng.uniform(*scale_rng)
        ca, sa = math.cos(rot), math.sin(rot)
        verts, faces = blade_geometry(blades=7 + i % 4, height=1.0)
        base = len(V)
        for (vx, vy, vz) in verts:
            vx, vy, vz = vx * s, vy * s, vz * s
            rx = vx * ca - vy * sa
            ry = vx * sa + vy * ca
            V.append((x + rx, y + ry, z + vz))
        F.extend([(a0 + base, a1 + base, a2 + base, a3 + base) for (a0, a1, a2, a3) in faces])
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    me.materials.append(material)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    link(ob, grass_col)
    return ob


build_merged("grass_near", 6000, 26.0, 0.5, (0.8, 1.7), grass)
build_merged("grass_mid", 7000, 90.0, 26.0, (0.6, 1.25), grass)
build_merged("grass_far", 3000, 170.0, 90.0, (0.5, 0.9), grass)


# ----------------------------------------------------------------------------
# wildflowers: bright accents - the "positive" part of the image
# ----------------------------------------------------------------------------
flower_col = new_collection("flowers")


def flower_mesh(name, petal_colour, petals=6):
    mat, b = principled(name + "_mat", petal_colour, 0.45)
    try:
        b.inputs["Transmission Weight"].default_value = 0.12
    except KeyError:
        pass
    verts, faces = [], []
    # stem
    verts += [(-0.006, 0, 0), (0.006, 0, 0), (-0.004, 0, 0.30), (0.004, 0, 0.30)]
    faces += [(0, 1, 3, 2)]
    # petals: fanned quads around the head
    for i in range(petals):
        a = i / petals * math.tau
        ca, sa = math.cos(a), math.sin(a)
        r0, r1 = 0.008, 0.055
        b0 = len(verts)
        verts += [
            (ca * r0 - sa * 0.014, sa * r0 + ca * 0.014, 0.30),
            (ca * r0 - sa * -0.014, sa * r0 + ca * -0.014, 0.30),
            (ca * r1, sa * r1, 0.335),
            (ca * r1 * 0.92, sa * r1 * 0.92, 0.352),
        ]
        faces.append((b0, b0 + 1, b0 + 3, b0 + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(mat)
    return me


PALETTE = [
    ("poppy", (0.75, 0.10, 0.03)),      # warm red
    ("buttercup", (0.95, 0.72, 0.06)),  # yellow
    ("cornflower", (0.20, 0.32, 0.85)), # blue
    ("daisy", (0.95, 0.94, 0.88)),      # white
    ("clover", (0.72, 0.28, 0.62)),     # pink-purple
]
flower_meshes = [flower_mesh(n, c, petals=5 + i % 3) for i, (n, c) in enumerate(PALETTE)]

def flower_geometry(petals=6):
    V, F = [], []
    V += [(-0.006, 0, 0), (0.006, 0, 0), (-0.004, 0, 0.30), (0.004, 0, 0.30)]
    F += [(0, 1, 3, 2)]
    for i in range(petals):
        a = i / petals * math.tau
        ca, sa = math.cos(a), math.sin(a)
        r0, r1 = 0.008, 0.055
        b0 = len(V)
        V += [
            (ca * r0 - sa * 0.014, sa * r0 + ca * 0.014, 0.30),
            (ca * r0 + sa * 0.014, sa * r0 - ca * 0.014, 0.30),
            (ca * r1, sa * r1, 0.335),
            (ca * r1 * 0.92, sa * r1 * 0.92, 0.352),
        ]
        F.append((b0, b0 + 1, b0 + 3, b0 + 2))
    return V, F


for pi, (pname, pcol) in enumerate(PALETTE):
    mat, _b = principled(pname + "_mat", pcol, 0.45)
    try:
        _b.inputs["Transmission Weight"].default_value = 0.12
    except KeyError:
        pass
    V, F = [], []
    verts0, faces0 = flower_geometry(petals=5 + pi % 3)
    for i in range(260):
        r = 55.0 * (rng.random() ** 1.5)
        a = rng.uniform(0, math.tau)
        x = math.cos(a) * r
        y = 40.0 + math.sin(a) * r * 0.9
        # clump in patches so colour appears in drifts, not evenly
        if noise.noise(Vector((x * 0.09 + pi * 5.0, y * 0.09, 2.0))) < 0.18:
            continue
        if dist_to_cam_path(x, y) < 6.0:
            continue
        z = ground_height(x, y)
        sc = rng.uniform(1.0, 2.0)
        base = len(V)
        for (vx, vy, vz) in verts0:
            V.append((x + vx * sc, y + vy * sc, z + vz * sc))
        F.extend([(a0 + base, a1 + base, a2 + base, a3 + base) for (a0, a1, a2, a3) in faces0])
    me = bpy.data.meshes.new(f"flowers_{pname}")
    me.from_pydata(V, [], F)
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new(f"flowers_{pname}", me)
    link(ob, flower_col)


# ----------------------------------------------------------------------------
# trees and distant hills: soft, sunlit, closing the horizon
# ----------------------------------------------------------------------------
tree_col = new_collection("trees")
leaf, leaf_b = principled("leaf", (0.05, 0.10, 0.022), 0.75)
_lnt = leaf.node_tree
_lc = _lnt.nodes.new("ShaderNodeTexCoord")
_ln = _lnt.nodes.new("ShaderNodeTexNoise")
_ln.inputs["Scale"].default_value = 3.0
_ln.inputs["Detail"].default_value = 6.0
_lnt.links.new(_lc.outputs["Object"], _ln.inputs["Vector"])
_lr = _lnt.nodes.new("ShaderNodeValToRGB")
_lr.color_ramp.elements[0].position = 0.35
_lr.color_ramp.elements[0].color = (0.018, 0.042, 0.010, 1)   # deep shade inside crown
_lr.color_ramp.elements[1].position = 0.65
_lr.color_ramp.elements[1].color = (0.085, 0.150, 0.028, 1)   # sunlit leaf
_lnt.links.new(_ln.outputs["Fac"], _lr.inputs["Fac"])
_lnt.links.new(_lr.outputs["Color"], leaf_b.inputs["Base Color"])
try:
    leaf_b.inputs["Transmission Weight"].default_value = 0.18
except KeyError:
    pass
bark, bark_b = principled("trunk", (0.055, 0.040, 0.030), 0.85)


def tree(x, y, height, spread, layers=1):
    """A broadleaf tree: tapered trunk, a few boughs, and a canopy of lobes at
    uneven heights with noise displacement so it doesn't read as a lollipop."""
    z = ground_height(x, y)
    bpy.ops.mesh.primitive_cone_add(
        vertices=9, radius1=height * 0.035, radius2=height * 0.016,
        depth=height * 0.62, location=(x, y, z + height * 0.31))
    tr = bpy.context.object
    tr.rotation_euler = (math.radians(rng.uniform(-3, 3)), math.radians(rng.uniform(-3, 3)), 0)
    tr.data.materials.append(bark)
    for c in list(tr.users_collection):
        c.objects.unlink(tr)
    link(tr, tree_col)

    # two or three visible boughs
    for _ in range(rng.randint(2, 3)):
        ang = rng.uniform(0, math.tau)
        elev = math.radians(rng.uniform(28, 48))
        ln = spread * rng.uniform(0.7, 1.2)
        p0 = Vector((x, y, z + height * rng.uniform(0.42, 0.58)))
        d = Vector((math.cos(ang) * math.cos(elev), math.sin(ang) * math.cos(elev), math.sin(elev)))
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=height * 0.012, depth=ln,
                                            location=p0 + d * (ln * 0.5))
        br = bpy.context.object
        br.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        br.data.materials.append(bark)
        for c in list(br.users_collection):
            c.objects.unlink(br)
        link(br, tree_col)

    # Canopy: ONE dense displaced crown. Many small lobes created ring
    # artefacts and cost ~50x more geometry; a single high-detail sphere with
    # strong multi-scale displacement reads as foliage and renders fast.
    crown_z = z + height * 0.70
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0,
                                          location=(x, y, crown_z))
    cn = bpy.context.object
    cn.scale = (spread, spread, height * 0.34)
    cn.rotation_euler = (0, 0, rng.uniform(0, math.tau))
    for band, scale, strength in ((0, 0.42, 0.34), (1, 0.16, 0.11)):
        tex = bpy.data.textures.new(f"foliage_{len(bpy.data.textures)}", "CLOUDS")
        tex.noise_scale = spread * scale
        tex.noise_depth = 4
        d = cn.modifiers.new("disp", "DISPLACE")
        d.texture = tex
        d.strength = spread * strength
        d.texture_coords = "LOCAL"
    cn.data.materials.append(leaf)
    for p_ in cn.data.polygons:
        p_.use_smooth = True
    for c in list(cn.users_collection):
        c.objects.unlink(cn)
    link(cn, tree_col)


# a loose treeline well behind the meadow so the sun sits just above it
for i in range(16):
    a = rng.uniform(-0.85, 0.95) + math.pi * 0.5
    r = rng.uniform(145.0, 240.0)
    x = math.cos(a) * r * 1.4
    y = 40.0 + abs(math.sin(a)) * r
    tree(x, y, rng.uniform(13.0, 21.0), rng.uniform(4.5, 7.5))

# two soft hill layers far away
hill, hill_b = principled("hill", (0.055, 0.095, 0.075), 0.95)
if "Specular IOR Level" in hill_b.inputs:
    hill_b.inputs["Specular IOR Level"].default_value = 0.05
for li, (dist, hbase, amp) in enumerate([(560.0, 20.0, 16.0), (980.0, 38.0, 26.0)]):
    verts, faces = [], []
    N = 160
    for i in range(N):
        x = -900.0 + 1800.0 * i / (N - 1)
        n = noise.noise(Vector((x * 0.006, li * 7.0, 0.0)))
        n2 = noise.noise(Vector((x * 0.02, li * 3.0 + 4.0, 0.0)))
        z = hbase + amp * (0.6 * n + 0.4 * n2)
        verts += [(x, dist, z), (x, dist, -20.0)]
        if i:
            a = (i - 1) * 2
            faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new(f"hill_{li}")
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(f"hill_{li}", me)
    ob.data.materials.append(hill)
    link(ob, tree_col)


# ----------------------------------------------------------------------------
# drifting seeds / pollen: tiny bright specks catching the light
# ----------------------------------------------------------------------------
mote_col = new_collection("motes")
mote_mat, mote_b = principled("mote", (0.95, 0.92, 0.80), 0.4)
mote_mat, mote_b = principled("mote", (0.95, 0.92, 0.80), 0.4)
try:
    mote_b.inputs["Emission Color"].default_value = (1.0, 0.95, 0.82, 1.0)
    mote_b.inputs["Emission Strength"].default_value = 0.6
except KeyError:
    pass
V, F = [], []
for i in range(300):
    cx = rng.uniform(-12, 12)
    cy = rng.uniform(30, 52)
    cz = rng.uniform(0.3, 4.0)
    base = len(V)
    # tiny tetra - cheaper than a sphere and reads the same at this size
    V += [(cx + 0.012, cy, cz), (cx - 0.006, cy + 0.010, cz), (cx - 0.006, cy - 0.010, cz), (cx, cy, cz + 0.014)]
    F += [(base, base + 1, base + 2), (base, base + 1, base + 3),
          (base + 1, base + 2, base + 3), (base, base + 2, base + 3)]
me = bpy.data.meshes.new("motes")
me.from_pydata(V, [], F)
me.update()
me.materials.append(mote_mat)
link(bpy.data.objects.new("motes", me), mote_col)


# ----------------------------------------------------------------------------
# distance haze: a thin scattering shell far out, so hills and treeline separate
# ----------------------------------------------------------------------------
haze_col = new_collection("haze")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 500.0, 400.0))
hz_box = bpy.context.object
hz_box.scale = (2600.0, 1400.0, 1600.0)
for c in list(hz_box.users_collection):
    c.objects.unlink(hz_box)
link(hz_box, haze_col)
haze_mat = bpy.data.materials.new("haze")
haze_mat.use_nodes = True
hnt = haze_mat.node_tree
hnt.nodes.remove(hnt.nodes["Principled BSDF"])
hvol = hnt.nodes.new("ShaderNodeVolumePrincipled")
hvol.inputs["Color"].default_value = (0.72, 0.82, 1.0, 1.0)
hvol.inputs["Density"].default_value = 0.0007
hvol.inputs["Anisotropy"].default_value = 0.4
hnt.links.new(hvol.outputs["Volume"], hnt.nodes["Material Output"].inputs["Volume"])
hz_box.data.materials.append(haze_mat)

# ----------------------------------------------------------------------------
# camera: low, wide, slow forward drift - eye-level with the grass
# ----------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 40.0
cam_data.sensor_width = 36.0
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = 2.8
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# Aim at a point just above the treeline with the sun beyond it, so we look
# *toward* the light: grass rim-lights and the sky carries the top of frame.
LOOK = Vector((2.0, 200.0, 9.0))


def set_camera(f):
    t = f / max(FRAMES - 1, 1)
    e = t * t * (3 - 2 * t)        # smoothstep ease
    cam.location = Vector((
        -1.1 + 2.6 * e,
        40.0 - 1.4 * e,                       # slow push in
        ground_height(0.0, 40.0) + 2.15 + 0.12 * e,
    ))
    target = LOOK + Vector((0.5 * e, 0.0, 0.25 * e))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    # focus on the near grass so the meadow falls away softly
    cam_data.dof.focus_distance = (cam.location - Vector((0.2, 43.5, ground_height(0.2, 43.5) + 0.5))).length


scene.frame_start = 1
scene.frame_end = FRAMES
for f in range(1, FRAMES + 1):
    set_camera(f)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.dof.keyframe_insert("focus_distance", frame=f)


# ----------------------------------------------------------------------------
# compositor: gentle bloom + warm lift
# ----------------------------------------------------------------------------
comp = bpy.data.node_groups.new("comp", "CompositorNodeTree")
scene.compositing_node_group = comp
comp.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

rl = comp.nodes.new("CompositorNodeRLayers")
glare = comp.nodes.new("CompositorNodeGlare")
glare.inputs["Type"].default_value = "Fog Glow"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Threshold"].default_value = 1.1
glare.inputs["Strength"].default_value = 0.30
glare.inputs["Size"].default_value = 0.5
out_node = comp.nodes.new("NodeGroupOutput")
comp.links.new(rl.outputs["Image"], glare.inputs["Image"])
comp.links.new(glare.outputs["Image"], out_node.inputs[0])


# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------
if os.environ.get("MEADOW_DEBUG_SKY"):
    print("--- world links ---")
    for l in wnt.links:
        print(f"  {l.from_node.name}.{l.from_socket.name} -> {l.to_node.name}.{l.to_socket.name}")
    p = wbg.inputs["Color"]
    print("Background Color linked:", p.is_linked, "strength:", wbg.inputs["Strength"].default_value)
if os.environ.get("MEADOW_NO_CLOUDS"):
    for l in list(wnt.links):
        if l.to_node == sky_mix and l.to_socket.name == "Factor":
            wnt.links.remove(l)
    sky_mix.inputs["Factor"].default_value = 0.0

os.makedirs(OUT_DIR, exist_ok=True)
print(f"[meadow] {RES_X}x{RES_Y} samples={SAMPLES} frames={FRAMES} preview={PREVIEW}")

if SINGLE:
    scene.frame_set(SINGLE)
    scene.render.filepath = os.path.join(OUT_DIR, f"still_{SINGLE:04d}.png")
    bpy.ops.render.render(write_still=True)
elif PREVIEW:
    for f in (STILLS or [1, FRAMES // 2, FRAMES]):
        scene.frame_set(f)
        scene.render.filepath = os.path.join(OUT_DIR, f"preview_{f:03d}.png")
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
