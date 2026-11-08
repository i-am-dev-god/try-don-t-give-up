"""
Mist valley at dawn - a path-traced landscape short rendered in headless Cycles.

The valley is a procedural heightfield: a basin that rises into far ridges, with
conifers scattered over the slopes and a low sun burning through the gap. Depth
comes from atmospheric layering, so the shot reads as a landscape rather than a
set. Camera drifts slowly along the rim.

    VALLEY_PREVIEW=1 ./tools/render.sh     # 480x270 look checks
    ./tools/render.sh --frames             # full animation -> MP4
"""
import math
import os
import random

import bpy
from mathutils import Vector, noise

# ----------------------------------------------------------------------------
# knobs (override with VALLEY_* env vars)
# ----------------------------------------------------------------------------
RES_X = int(os.environ.get("VALLEY_RES_X", 1280))
RES_Y = int(os.environ.get("VALLEY_RES_Y", 720))
SAMPLES = int(os.environ.get("VALLEY_SAMPLES", 48))
FRAMES = int(os.environ.get("VALLEY_FRAMES", 96))
FPS = 24
SEED = int(os.environ.get("VALLEY_SEED", 5))
OUT_DIR = os.environ.get("VALLEY_OUT", "/tmp/valley_frames")
PREVIEW = bool(int(os.environ.get("VALLEY_PREVIEW", 0)))
SINGLE = int(os.environ.get("VALLEY_SINGLE", 0))
STILLS = [int(v) for v in os.environ.get("VALLEY_STILLS", "").split(",") if v.strip()]

if PREVIEW:
    RES_X, RES_Y, SAMPLES = 480, 270, 16

rng = random.Random(SEED)

# ----------------------------------------------------------------------------
# scene + render settings
# ----------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = float(os.environ.get("VALLEY_ADAPT", 0.02))
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 1
scene.cycles.transmission_bounces = 1
scene.cycles.transparent_max_bounces = 4
scene.cycles.volume_bounces = 1
scene.cycles.volume_step_rate = float(os.environ.get("VALLEY_VOL_STEP", 4.0))
scene.cycles.volume_max_steps = int(os.environ.get("VALLEY_VOL_STEPS", 96))
scene.cycles.sample_clamp_indirect = 6.0
# The scene is almost entirely heightfield + cones; instancing keeps the BVH small.
scene.render.use_persistent_data = True
scene.cycles.use_light_tree = True
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.use_motion_blur = not PREVIEW and bool(int(os.environ.get("VALLEY_MBLUR", 1)))
scene.render.motion_blur_shutter = 0.45
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(OUT_DIR, "f_")

scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Punchy"
scene.view_settings.exposure = -0.45

# Sky dome: warm at the horizon, deep blue overhead.
world = bpy.data.worlds.new("Dawn")
scene.world = world
world.use_nodes = True
wnt = world.node_tree
wbg = wnt.nodes["Background"]
wbg.inputs["Strength"].default_value = 1.0

wcoord = wnt.nodes.new("ShaderNodeTexCoord")
wsep = wnt.nodes.new("ShaderNodeSeparateXYZ")
wnt.links.new(wcoord.outputs["Generated"], wsep.inputs[0])
wh = wnt.nodes.new("ShaderNodeMapRange")
wh.inputs["From Min"].default_value = 0.50
wh.inputs["From Max"].default_value = 0.62
wh.clamp = True
wnt.links.new(wsep.outputs["Z"], wh.inputs["Value"])
wramp = wnt.nodes.new("ShaderNodeValToRGB")
wramp.color_ramp.elements[0].position = 0.0
wramp.color_ramp.elements[0].color = (1.0, 0.58, 0.30, 1.0)     # horizon
wramp.color_ramp.elements[0].color = (0.92, 0.66, 0.46, 1.0)
wramp.color_ramp.elements[1].position = 1.0
wramp.color_ramp.elements[1].color = (0.020, 0.065, 0.24, 1.0)   # zenith
wnt.links.new(wh.outputs["Result"], wramp.inputs["Fac"])
wnt.links.new(wramp.outputs["Color"], wbg.inputs["Color"])


CAM_X_HINT = -26.0
CAM_Y_HINT = -78.0


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


# ----------------------------------------------------------------------------
# terrain: valley basin rising into far ridges
# ----------------------------------------------------------------------------
def smoothstep(a, b, t):
    t = max(0.0, min(1.0, (t - a) / (b - a)))
    return t * t * (3 - 2 * t)


def terrain_height(x, y):
    """Long valley running along +Y: floor dips, walls climb, ridges recede."""
    # longitudinal floor profile: near rim high, basin low, far ridge high
    floor = 34.0 - 52.0 * smoothstep(-120.0, 200.0, y) + 165.0 * smoothstep(330.0, 920.0, y)
    # cross-section: the valley walls
    wall = 0.0
    ax = abs(x)
    if ax > 46.0:
        wall = 0.0026 * (ax - 46.0) ** 1.9
    # ridges and erosion detail
    big = noise.noise(Vector((x * 0.0022, y * 0.0022, 0.3))) * 46.0
    mid = noise.noise(Vector((x * 0.008, y * 0.008, 5.0))) * 14.0
    fine = noise.noise(Vector((x * 0.04, y * 0.04, 11.0))) * 2.2
    return floor + wall + big + mid + fine


# sampled once: the tree scatter reuses it to sit on the surface
terrain_col = new_collection("terrain")
bpy.ops.mesh.primitive_grid_add(x_subdivisions=240, y_subdivisions=240, size=1800)
terrain = bpy.context.object
terrain.name = "terrain"
terrain.location = (0, 300.0, 0.0)
for v in terrain.data.vertices:
    p = terrain.matrix_world @ v.co if False else Vector((v.co.x, v.co.y + 300.0, 0.0))
    v.co.z = terrain_height(p.x, p.y)
for p in terrain.data.polygons:
    p.use_smooth = True
for c in terrain.users_collection:
    c.objects.unlink(terrain)
link(terrain, terrain_col)

terrain_mat, terrain_b = principled("terrain", (0.014, 0.022, 0.012), 0.9)
tnt = terrain_mat.node_tree
tcoord = tnt.nodes.new("ShaderNodeTexCoord")
tn = tnt.nodes.new("ShaderNodeTexNoise")
tn.inputs["Scale"].default_value = 0.35
tn.inputs["Detail"].default_value = 8.0
tnt.links.new(tcoord.outputs["Object"], tn.inputs["Vector"])
tramp = tnt.nodes.new("ShaderNodeValToRGB")
tramp.color_ramp.elements[0].position = 0.40
tramp.color_ramp.elements[0].color = (0.004, 0.006, 0.004, 1)  # rock / shadow
tramp.color_ramp.elements[1].position = 0.62
tramp.color_ramp.elements[1].color = (0.011, 0.024, 0.010, 1)  # mossy grass
tnt.links.new(tn.outputs["Fac"], tramp.inputs["Fac"])
terrain_b.inputs["Base Color"].default_value = (0.016, 0.023, 0.014, 1.0)
if "Specular IOR Level" in terrain_b.inputs:
    terrain_b.inputs["Specular IOR Level"].default_value = 0.02  # no grazing-angle sky mirror
terrain_b.inputs["Roughness"].default_value = 1.0
# moss/soil mottle multiplied on top of a dark base
_mul = tnt.nodes.new("ShaderNodeMix")
_mul.data_type = "RGBA"
_mul.blend_type = "MULTIPLY"
_mul.inputs["Factor"].default_value = 0.6
tnt.links.new(tramp.outputs["Color"], _mul.inputs[6])
_mul.inputs[7].default_value = (1.0, 1.0, 1.0, 1.0)
tnt.links.new(_mul.outputs[2], terrain_b.inputs["Base Color"])
tbump = tnt.nodes.new("ShaderNodeBump")
tbump.inputs["Strength"].default_value = 0.3
tn2 = tnt.nodes.new("ShaderNodeTexNoise")
tn2.inputs["Scale"].default_value = 6.0
tn2.inputs["Detail"].default_value = 8.0
tnt.links.new(tcoord.outputs["Object"], tn2.inputs["Vector"])
tnt.links.new(tn2.outputs["Fac"], tbump.inputs["Height"])
tnt.links.new(tbump.outputs["Normal"], terrain_b.inputs["Normal"])
terrain.data.materials.append(terrain_mat)


# ----------------------------------------------------------------------------
# conifers: trunk + drooping foliage tiers. Silhouette is what matters here.
# ----------------------------------------------------------------------------
tree_col = new_collection("conifers")
tree_mat, tree_b = principled("conifer", (0.018, 0.030, 0.016), 0.55)
_tt = tree_mat.node_tree
_tc = _tt.nodes.new("ShaderNodeTexCoord")
_tn = _tt.nodes.new("ShaderNodeTexNoise")     # needle-scale breakup
_tn.inputs["Scale"].default_value = 55.0
_tn.inputs["Detail"].default_value = 6.0
_tt.links.new(_tc.outputs["Object"], _tn.inputs["Vector"])
_tb = _tt.nodes.new("ShaderNodeBump")
_tb.inputs["Strength"].default_value = 0.6
_tt.links.new(_tn.outputs["Fac"], _tb.inputs["Height"])
_tt.links.new(_tb.outputs["Normal"], tree_b.inputs["Normal"])
_tc2 = _tt.nodes.new("ShaderNodeTexNoise")    # blotchy light/dark foliage
_tc2.inputs["Scale"].default_value = 4.0
_tt.links.new(_tc.outputs["Object"], _tc2.inputs["Vector"])
_tr = _tt.nodes.new("ShaderNodeValToRGB")
_tr.color_ramp.elements[0].color = (0.008, 0.014, 0.008, 1)
_tr.color_ramp.elements[1].color = (0.030, 0.048, 0.024, 1)
_tt.links.new(_tc2.outputs["Fac"], _tr.inputs["Fac"])
_tt.links.new(_tr.outputs["Color"], tree_b.inputs["Base Color"])
# A little translucency: backlit needles glow at the silhouette edge, which is
# what makes a forest read as foliage instead of cardboard.
try:
    tree_b.inputs["Transmission Weight"].default_value = 0.10
except KeyError:
    pass
if "Sheen Weight" in tree_b.inputs:
    tree_b.inputs["Sheen Weight"].default_value = 0.35
    tree_b.inputs["Sheen Roughness"].default_value = 0.4


def conifer_mesh(name, tiers=22, sides=9):
    """Unit fir. Each tier is a drooping skirt of foliage: a cone whose skirt hangs
    lower than its apex, with per-branch jitter, so the silhouette reads as ragged
    needles rather than a stack of clean cones."""
    verts, faces = [], []
    for t in range(tiers):
        f = t / (tiers - 1)
        z0 = 0.05 + 0.85 * f
        # widest low down, tapering to a slender leader
        r = 0.23 * (1.0 - 0.93 * f) ** 1.25 + 0.008
        h = 0.16 + 0.14 * f
        jitter = 1.0 + 0.30 * noise.noise(Vector((f * 13.0, 0.5, 0.0)))
        r *= jitter
        base = len(verts)
        verts.append((0.0, 0.0, z0 + h))  # apex
        for k in range(sides):
            a = k / sides * math.tau + f * 2.3
            # branch tips sag and vary in reach -> ragged skirt edge
            sag = 0.055 * (1.0 + 0.7 * math.sin(k * 2.1 + t))
            rr = r * (0.72 + 0.55 * ((k * 37 + t * 11) % 7) / 7.0)
            # per-vertex noise: ragged needle-edge silhouette instead of a cone
            j = noise.noise(Vector((math.cos(a) * 3.0, math.sin(a) * 3.0, f * 22.0 + k * 0.7)))
            rr *= 1.0 + 0.42 * j
            sag += 0.05 * j
            verts.append((math.cos(a) * rr, math.sin(a) * rr, z0 - sag))
        for k in range(sides):
            faces.append((base, base + 1 + k, base + 1 + (k + 1) % sides))
        # underside of the skirt (dark, occluded - reads as depth inside the tree)
        base2 = len(verts)
        verts.append((0.0, 0.0, z0 - 0.02))
        for k in range(sides):
            a = k / sides * math.tau + f * 2.3
            rr = r * (0.72 + 0.55 * ((k * 37 + t * 11) % 7) / 7.0)
            verts.append((math.cos(a) * rr, math.sin(a) * rr, z0 - 0.055 * (1.0 + 0.7 * math.sin(k * 2.1 + t))))
        for k in range(sides):
            faces.append((base2, base2 + 1 + (k + 1) % sides, base2 + 1 + k))
    # trunk
    tb = len(verts)
    verts += [(-0.02, -0.02, -0.02), (0.02, -0.02, -0.02), (0.02, 0.02, -0.02), (-0.02, 0.02, -0.02)]
    for k in range(4):
        faces.append((tb + k, tb + (k + 1) % 4, tb + (k + 1) % 4, tb + k))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(tree_mat)
    return me


fir_meshes = [conifer_mesh(f"fir_{i}", tiers=18 + (i % 3) * 4, sides=8 + i % 3) for i in range(7)]


def plant(count, y_range, x_range, height_range, tag, corridor=0.0, clump=0.0):
    """Scatter conifers, but keep a clear corridor along the valley axis and use
    noise clumping so the forest has edges and clearings instead of an even carpet."""
    placed_any = 0
    for i in range(count):
        y = rng.uniform(*y_range)
        x = rng.uniform(*x_range)
        if corridor and abs(x) < corridor * (0.55 + 0.45 * min(1.0, y / 800.0)):
            continue
        if clump and noise.noise(Vector((x * 0.006, y * 0.006, 3.0))) < clump:
            continue
        z = terrain_height(x, y)
        obj = bpy.data.objects.new(f"{tag}_{i}", fir_meshes[i % len(fir_meshes)])
        obj.location = (x, y, z - 0.3)
        obj.rotation_euler = (0, 0, rng.uniform(0, math.tau))
        h = rng.uniform(*height_range) * rng.choice([0.55, 0.8, 1.0, 1.0, 1.35, 1.7])
        w = h * rng.uniform(0.45, 0.95)
        obj.scale = (w * rng.uniform(0.85, 1.2), w * rng.uniform(0.85, 1.2), h)
        link(obj, tree_col)
        placed_any += 1
    return placed_any


# near rim: big silhouettes at the frame edges, sightline kept clear
# near rim: big silhouettes framing the left and right of frame
plant(120, (-70.0, 120.0), (-230.0, 230.0), (17.0, 30.0), "near", corridor=52.0, clump=-0.20)
# valley flanks: forest that climbs the walls
plant(520, (70.0, 340.0), (-320.0, 320.0), (14.0, 26.0), "mid", corridor=60.0, clump=-0.12)
# immediate foreground: close, out-of-focus trunks framing the frame edges
for side in (-1, 1):
    for k in range(7):
        x = side * rng.uniform(34.0, 130.0)
        y = rng.uniform(-92.0, -58.0)
        z = terrain_height(x, y)
        obj = bpy.data.objects.new(f"fore_{side}_{k}", fir_meshes[k % len(fir_meshes)])
        obj.location = (x, y, z - 0.3)
        obj.rotation_euler = (0, 0, rng.uniform(0, math.tau))
        h = rng.uniform(22.0, 46.0)
        obj.scale = (h * 0.45, h * 0.45, h)
        link(obj, tree_col)
# far ridge: dense, hazy texture
plant(900, (330.0, 820.0), (-620.0, 620.0), (20.0, 38.0), "far", corridor=70.0, clump=-0.25)




# ----------------------------------------------------------------------------
# birds: a loose flock crossing the valley. Simple wing shapes, motion-blurred.
# ----------------------------------------------------------------------------
bird_col = new_collection("birds")
bird_mat, bird_b = principled("bird", (0.008, 0.009, 0.011), 0.9)


def bird_mesh(name):
    """Two swept wing planes, shallow V, plus a tiny body."""
    v = [(0.0, 0.18, 0.0), (0.0, -0.22, 0.0)]
    f = []
    # left wing
    v += [(-0.55, 0.02, 0.10), (-0.30, -0.10, 0.03)]
    f.append((0, 2, 3, 1))
    v += [(0.55, 0.02, 0.10), (0.30, -0.10, 0.03)]
    f.append((0, 1, 5, 4))
    me = bpy.data.meshes.new(name)
    me.from_pydata(v, [], f)
    me.update()
    me.materials.append(bird_mat)
    return me


bm = bird_mesh("bird")
for i in range(9):
    b = bpy.data.objects.new(f"bird_{i}", bm)
    b.location = (rng.uniform(-120, 90), rng.uniform(180, 420), rng.uniform(70, 130))
    b.rotation_euler = (math.radians(90), 0, rng.uniform(0, math.tau))
    sc = rng.uniform(2.2, 3.4)
    b.scale = (sc, sc, sc)
    link(b, bird_col)
    # slow drift across frame + a wing flap via rotation
    for fr in (1, FRAMES):
        b.keyframe_insert("location", frame=fr)
        b.keyframe_insert("rotation_euler", frame=fr)
        b.rotation_euler = (math.radians(90), math.radians(-25 * math.sin(fr * 0.5)), 0)
    b.keyframe_insert("rotation_euler", frame=1)
    b.keyframe_insert("rotation_euler", frame=FRAMES)




# ----------------------------------------------------------------------------
# undergrowth: grass tufts and boulders across the near rim. This is what stops
# the foreground reading as an empty beach.
# ----------------------------------------------------------------------------
under_col = new_collection("undergrowth")
grass, grass_b = principled("grass", (0.026, 0.040, 0.016), 0.85)
if "Specular IOR Level" in grass_b.inputs:
    grass_b.inputs["Specular IOR Level"].default_value = 0.15
try:
    grass_b.inputs["Transmission Weight"].default_value = 0.20
except KeyError:
    pass


def tuft_mesh(name, blades=9):
    """A clump of tapered grass blades splayed in a fan."""
    verts, faces = [], []
    for i in range(blades):
        a = i / blades * math.tau + rng.uniform(-0.4, 0.4)
        lean = rng.uniform(0.12, 0.5)
        h = rng.uniform(0.5, 1.0)
        w = rng.uniform(0.02, 0.045)
        dx, dy = math.cos(a), math.sin(a)
        base = len(verts)
        verts += [
            (-dy * w, dx * w, 0.0),
            (dy * w, -dx * w, 0.0),
            (dx * lean * h, dy * lean * h, h),
        ]
        faces.append((base, base + 1, base + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.materials.append(grass)
    return me


tufts = [tuft_mesh(f"tuft_{i}") for i in range(8)]
for i in range(5200):
    # cluster around the camera rim, densest close in
    r = 300.0 * (rng.random() ** 2.6)
    a = rng.uniform(0, math.tau)
    x = CAM_X_HINT + math.cos(a) * r
    y = CAM_Y_HINT + math.sin(a) * r * 0.8
    if y < -95.0 or abs(x) > 200.0:
        continue
    obj = bpy.data.objects.new(f"grass_{i}", tufts[i % len(tufts)])
    obj.location = (x, y, terrain_height(x, y) - 0.05)
    obj.rotation_euler = (0, 0, rng.uniform(0, math.tau))
    sc = rng.uniform(1.1, 3.6)
    obj.scale = (sc, sc, sc * rng.uniform(0.7, 1.4))
    link(obj, under_col)

rock, rock_b = principled("rock", (0.024, 0.026, 0.024), 0.95)
if "Specular IOR Level" in rock_b.inputs:
    rock_b.inputs["Specular IOR Level"].default_value = 0.05
for i in range(90):
    r = 200.0 * (rng.random() ** 1.4)
    a = rng.uniform(0, math.tau)
    x = CAM_X_HINT + math.cos(a) * r
    y = CAM_Y_HINT + math.sin(a) * r * 0.8
    if y < -95.0:
        continue
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=(x, y, terrain_height(x, y) - 0.2))
    rk = bpy.context.object
    s0 = rng.uniform(0.3, 2.4)
    rk.scale = (s0 * rng.uniform(0.8, 1.5), s0 * rng.uniform(0.8, 1.5), s0 * rng.uniform(0.45, 0.9))
    rk.rotation_euler = (rng.uniform(0, math.tau), rng.uniform(0, math.tau), rng.uniform(0, math.tau))
    rk.data.materials.append(rock)
    for p in rk.data.polygons:
        p.use_smooth = True
    for cc in rk.users_collection:
        cc.objects.unlink(rk)
    link(rk, under_col)


# ----------------------------------------------------------------------------
# mist: dense in the basin, thinning upward and with depth
# ----------------------------------------------------------------------------
mist_col = new_collection("mist")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 320.0, 40.0))
fog = bpy.context.object
fog.name = "mist_domain"
fog.scale = (1800.0, 1400.0, 220.0)
for c in fog.users_collection:
    c.objects.unlink(fog)
link(fog, mist_col)

mist_mat = bpy.data.materials.new("mist")
mist_mat.use_nodes = True
mnt = mist_mat.node_tree
mnt.nodes.remove(mnt.nodes["Principled BSDF"])
fout = mnt.nodes["Material Output"]

vol = mnt.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Color"].default_value = (0.50, 0.68, 1.0, 1.0)
vol.inputs["Anisotropy"].default_value = 0.62
vol.inputs["Absorption Color"].default_value = (0.10, 0.20, 0.44, 1.0)
mnt.links.new(vol.outputs["Volume"], fout.inputs["Volume"])

geo = mnt.nodes.new("ShaderNodeNewGeometry")
sepz = mnt.nodes.new("ShaderNodeSeparateXYZ")
mnt.links.new(geo.outputs["Position"], sepz.inputs[0])

# Height band: the mist sits in the valley, roughly 0-45 m, and fades out above.
sepy = mnt.nodes.new("ShaderNodeSeparateXYZ")
mnt.links.new(geo.outputs["Position"], sepy.inputs[0])
# Distance ramp: almost no mist within ~60 m of the lens, thickening with depth.
dist = mnt.nodes.new("ShaderNodeMapRange")
dist.inputs["From Min"].default_value = -80.0
dist.inputs["From Max"].default_value = 420.0
dist.inputs["To Min"].default_value = 0.12
dist.inputs["To Max"].default_value = 1.0
dist.clamp = True
mnt.links.new(sepy.outputs["Y"], dist.inputs["Value"])

band = mnt.nodes.new("ShaderNodeMapRange")
band.inputs["From Min"].default_value = -12.0
band.inputs["From Max"].default_value = 80.0
band.inputs["To Min"].default_value = 0.0013
band.inputs["To Max"].default_value = 0.00018
band.clamp = True
mnt.links.new(sepz.outputs["Z"], band.inputs["Value"])

# Drifting billows.
coord_v = mnt.nodes.new("ShaderNodeTexCoord")
n1 = mnt.nodes.new("ShaderNodeTexNoise")
n1.inputs["Scale"].default_value = 0.008
n1.inputs["Detail"].default_value = 3.0
n1.inputs["Roughness"].default_value = 0.55
mnt.links.new(coord_v.outputs["Object"], n1.inputs["Vector"])
n2 = mnt.nodes.new("ShaderNodeTexNoise")
n2.inputs["Scale"].default_value = 0.03
n2.inputs["Detail"].default_value = 3.0
mnt.links.new(coord_v.outputs["Object"], n2.inputs["Vector"])

vary = mnt.nodes.new("ShaderNodeMapRange")
vary.inputs["From Min"].default_value = 0.30
vary.inputs["From Max"].default_value = 0.70
vary.inputs["To Min"].default_value = 0.10
vary.inputs["To Max"].default_value = 2.4
vary.clamp = True
mnt.links.new(n1.outputs["Fac"], vary.inputs["Value"])
vary2 = mnt.nodes.new("ShaderNodeMapRange")
vary2.inputs["To Min"].default_value = 0.7
vary2.inputs["To Max"].default_value = 1.3
mnt.links.new(n2.outputs["Fac"], vary2.inputs["Value"])

m0 = mnt.nodes.new("ShaderNodeMath")
m0.operation = "MULTIPLY"
mnt.links.new(band.outputs["Result"], m0.inputs[0])
mnt.links.new(dist.outputs["Result"], m0.inputs[1])
m1 = mnt.nodes.new("ShaderNodeMath")
m1.operation = "MULTIPLY"
mnt.links.new(m0.outputs[0], m1.inputs[0])
mnt.links.new(vary.outputs["Result"], m1.inputs[1])
m2 = mnt.nodes.new("ShaderNodeMath")
m2.operation = "MULTIPLY"
mnt.links.new(m1.outputs[0], m2.inputs[0])
mnt.links.new(vary2.outputs["Result"], m2.inputs[1])
MIST_SCALE = float(os.environ.get("VALLEY_MIST", 1.0))
m3 = mnt.nodes.new("ShaderNodeMath")
m3.operation = "MULTIPLY"
m3.inputs[1].default_value = MIST_SCALE
mnt.links.new(m2.outputs[0], m3.inputs[0])
mnt.links.new(m3.outputs[0], vol.inputs["Density"])
fog.data.materials.append(mist_mat)


# ----------------------------------------------------------------------------
# sun: low over the far ridge, sitting inside the frame
# ----------------------------------------------------------------------------
sun_col = new_collection("sun")
SUN_POS = Vector((58.0, 900.0, 232.0))

bpy.ops.mesh.primitive_uv_sphere_add(radius=17.0, location=SUN_POS)
disc = bpy.context.object
disc.name = "sun_disc"
for c in disc.users_collection:
    c.objects.unlink(disc)
link(disc, sun_col)
sun_mat = bpy.data.materials.new("sun_disc")
sun_mat.use_nodes = True
smt = sun_mat.node_tree
smt.nodes.remove(smt.nodes["Principled BSDF"])
emit = smt.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (1.0, 0.62, 0.34, 1.0)
emit.inputs["Strength"].default_value = 90.0
smt.links.new(emit.outputs["Emission"], smt.nodes["Material Output"].inputs["Surface"])
disc.data.materials.append(sun_mat)
for p in disc.data.polygons:
    p.use_smooth = True

sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 7.5
sun_data.color = (1.0, 0.56, 0.26)
sun_data.angle = math.radians(1.2)
sun_obj = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun_obj)
sun_obj.location = SUN_POS
sun_obj.rotation_euler = (Vector((0.0, 0.0, 20.0)) - SUN_POS).normalized().to_track_quat("-Z", "Y").to_euler()

fill_data = bpy.data.lights.new("skyfill", "AREA")
fill_data.energy = 260.0
fill_data.size = 1200.0
fill_data.color = (0.24, 0.46, 1.0)
fill = bpy.data.objects.new("skyfill", fill_data)
scene.collection.objects.link(fill)
fill.location = (0.0, 200.0, 260.0)
fill.rotation_euler = (math.radians(18), 0, 0)


# ----------------------------------------------------------------------------
# camera: on the near rim, low, drifting sideways. Long lens compresses layers.
# ----------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 40.0
cam_data.sensor_width = 36.0
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = 5.6
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

CAM_START = Vector((-26.0, -78.0, 46.0))
TARGET = Vector((6.0, 330.0, 12.0))


def set_camera(f):
    t = f / max(FRAMES - 1, 1)
    e = t * t * (3 - 2 * t)
    cam.location = CAM_START + Vector((9.0 * e, 2.0 * e, 1.6 * e))
    look = TARGET + Vector((0.6 * math.sin(f * 0.05), 0, 0.05 * math.sin(f * 0.08)))
    cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.dof.focus_distance = (cam.location - Vector((0.0, 120.0, 6.0))).length


scene.frame_start = 1
scene.frame_end = FRAMES
for f in range(1, FRAMES + 1):
    set_camera(f)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.dof.keyframe_insert("focus_distance", frame=f)


# ----------------------------------------------------------------------------
# compositor: bloom around the sun
# ----------------------------------------------------------------------------
comp = bpy.data.node_groups.new("comp", "CompositorNodeTree")
scene.compositing_node_group = comp
comp.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

rl = comp.nodes.new("CompositorNodeRLayers")
glare = comp.nodes.new("CompositorNodeGlare")
glare.inputs["Type"].default_value = "Fog Glow"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Threshold"].default_value = 0.9
glare.inputs["Strength"].default_value = 0.45
glare.inputs["Size"].default_value = 0.7
out_node = comp.nodes.new("NodeGroupOutput")
comp.links.new(rl.outputs["Image"], glare.inputs["Image"])
comp.links.new(glare.outputs["Image"], out_node.inputs[0])


# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[valley] {RES_X}x{RES_Y} samples={SAMPLES} frames={FRAMES} preview={PREVIEW}")

if SINGLE:
    scene.frame_set(SINGLE)
    scene.render.filepath = os.path.join(OUT_DIR, f"still_{SINGLE:04d}.png")
    bpy.ops.render.render(write_still=True)
elif PREVIEW:
    for f in (STILLS or [1, FRAMES // 2, FRAMES]):
        scene.frame_set(f)
        scene.render.filepath = os.path.join(OUT_DIR, f"preview_{f:02d}.png")
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
