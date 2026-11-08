"""
Foggy forest at dawn - a path-traced short rendered with headless Blender/Cycles.

This is old-growth conifer forest, built entirely from code: tapered lofted
trunks with root flare and procedural bark, a dense dark treeline behind that
gives the sun something to cut through, branch clusters of instanced leaf cards
for the canopy, fern clumps in the foreground, and a height-graded scattering
volume for the mist and light shafts.

    FOREST_PREVIEW=1 ./tools/render.sh   # cheap 480x270 look checks
    ./tools/render.sh --frames           # full animation -> MP4
"""
import math
import os
import random

import bpy
from mathutils import Vector, noise

# ----------------------------------------------------------------------------
# knobs (override with FOREST_* env vars)
# ----------------------------------------------------------------------------
RES_X = int(os.environ.get("FOREST_RES_X", 1280))
RES_Y = int(os.environ.get("FOREST_RES_Y", 720))
SAMPLES = int(os.environ.get("FOREST_SAMPLES", 48))
FRAMES = int(os.environ.get("FOREST_FRAMES", 96))
FPS = 24
SEED = int(os.environ.get("FOREST_SEED", 11))
OUT_DIR = os.environ.get("FOREST_OUT", "/tmp/forest_frames")
PREVIEW = bool(int(os.environ.get("FOREST_PREVIEW", 0)))
SINGLE = int(os.environ.get("FOREST_SINGLE", 0))
STILLS = [int(v) for v in os.environ.get("FOREST_STILLS", "").split(",") if v.strip()]

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
scene.cycles.adaptive_threshold = 0.02
scene.cycles.use_denoising = True
scene.cycles.denoiser = "OPENIMAGEDENOISE"
scene.cycles.max_bounces = 5
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 1
scene.cycles.transmission_bounces = 2
scene.cycles.transparent_max_bounces = 6
scene.cycles.volume_bounces = 1
scene.cycles.sample_clamp_indirect = 8.0
scene.render.resolution_x = RES_X
scene.render.resolution_y = RES_Y
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.render.film_transparent = False
scene.render.use_motion_blur = not PREVIEW
scene.render.motion_blur_shutter = 0.4
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"
scene.render.filepath = os.path.join(OUT_DIR, "f_")

scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = 0.0

world = bpy.data.worlds.new("Dawn")
scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.26, 0.38, 0.60, 1.0)  # cool overcast sky
bg.inputs["Strength"].default_value = 0.16


def new_collection(name):
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col


def link(obj, col):
    col.objects.link(obj)
    return obj


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def principled(name, base, rough, metal=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    return mat, b


def ground_height(x, y):
    return (noise.noise(Vector((x * 0.09, y * 0.09, 0.0))) * 0.55
            + noise.noise(Vector((x * 0.35, y * 0.35, 3.0))) * 0.14)


CAM_Y = -7.0  # camera starts here; everything is laid out relative to it

# ----------------------------------------------------------------------------
# ground: wet conifer litter, mossy in patches
# ----------------------------------------------------------------------------
ground_col = new_collection("ground")
bpy.ops.mesh.primitive_grid_add(x_subdivisions=80, y_subdivisions=80, size=240)
ground = bpy.context.object
ground.name = "forest_floor"
for v in ground.data.vertices:
    v.co.z = ground_height(v.co.x, v.co.y)
smooth(ground)
for c in ground.users_collection:
    c.objects.unlink(ground)
link(ground, ground_col)

gm = bpy.data.materials.new("floor")
gm.use_nodes = True
nt = gm.node_tree
bsdf = nt.nodes["Principled BSDF"]
coord = nt.nodes.new("ShaderNodeTexCoord")

big = nt.nodes.new("ShaderNodeTexNoise")
big.inputs["Scale"].default_value = 0.45
big.inputs["Detail"].default_value = 9.0
big.inputs["Roughness"].default_value = 0.7
nt.links.new(coord.outputs["Object"], big.inputs["Vector"])
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].position = 0.40
ramp.color_ramp.elements[1].position = 0.68
ramp.color_ramp.elements[0].color = (0.020, 0.018, 0.013, 1)   # wet needles
ramp.color_ramp.elements[1].color = (0.040, 0.052, 0.024, 1)   # moss
nt.links.new(big.outputs["Fac"], ramp.inputs["Fac"])

# fallen needles as a fine directional streak layer
needle = nt.nodes.new("ShaderNodeTexNoise")
needle.inputs["Scale"].default_value = 90.0
needle.inputs["Detail"].default_value = 4.0
nt.links.new(coord.outputs["Object"], needle.inputs["Vector"])
nmix = nt.nodes.new("ShaderNodeMix")
nmix.data_type = "RGBA"
nmix.blend_type = "OVERLAY"
nmix.inputs["Factor"].default_value = 0.35
nt.links.new(ramp.outputs["Color"], nmix.inputs[6])
nt.links.new(needle.outputs["Fac"], nmix.inputs[7])
nt.links.new(nmix.outputs[2], bsdf.inputs["Base Color"])

# wet sheen: low-ish roughness with a broad specular lobe
bsdf.inputs["Roughness"].default_value = 0.62
if "Specular IOR Level" in bsdf.inputs:
    bsdf.inputs["Specular IOR Level"].default_value = 0.45

nb = nt.nodes.new("ShaderNodeTexNoise")
nb.inputs["Scale"].default_value = 55.0
nb.inputs["Detail"].default_value = 10.0
nt.links.new(coord.outputs["Object"], nb.inputs["Vector"])
bump = nt.nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.6
nt.links.new(nb.outputs["Fac"], bump.inputs["Height"])
nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
ground.data.materials.append(gm)


# ----------------------------------------------------------------------------
# bark: vertical furrows, lichen mottle, two-scale bump
# ----------------------------------------------------------------------------
bark, bark_b = principled("bark", (0.034, 0.028, 0.023), 0.8)
bnt = bark.node_tree
bcoord = bnt.nodes.new("ShaderNodeTexCoord")
mapping = bnt.nodes.new("ShaderNodeMapping")
mapping.inputs["Scale"].default_value = (6.0, 6.0, 0.45)  # stretch along the trunk
bnt.links.new(bcoord.outputs["Object"], mapping.inputs["Vector"])

furrow = bnt.nodes.new("ShaderNodeTexNoise")
furrow.inputs["Scale"].default_value = 3.4
furrow.inputs["Detail"].default_value = 12.0
furrow.inputs["Roughness"].default_value = 0.72
bnt.links.new(mapping.outputs["Vector"], furrow.inputs["Vector"])
bramp = bnt.nodes.new("ShaderNodeValToRGB")
bramp.color_ramp.elements[0].position = 0.34
bramp.color_ramp.elements[0].color = (0.008, 0.007, 0.006, 1)  # crevice
bramp.color_ramp.elements[1].position = 0.72
bramp.color_ramp.elements[1].color = (0.075, 0.058, 0.044, 1)  # ridge
bnt.links.new(furrow.outputs["Fac"], bramp.inputs["Fac"])

lichen = bnt.nodes.new("ShaderNodeTexNoise")
lichen.inputs["Scale"].default_value = 0.9
lichen.inputs["Detail"].default_value = 6.0
bnt.links.new(bcoord.outputs["Object"], lichen.inputs["Vector"])
lramp = bnt.nodes.new("ShaderNodeValToRGB")
lramp.color_ramp.elements[0].position = 0.60
lramp.color_ramp.elements[1].position = 0.68
bnt.links.new(lichen.outputs["Fac"], lramp.inputs["Fac"])
lmix = bnt.nodes.new("ShaderNodeMix")
lmix.data_type = "RGBA"
bnt.links.new(lramp.outputs["Color"], lmix.inputs["Factor"])
bnt.links.new(bramp.outputs["Color"], lmix.inputs[6])
lmix.inputs[7].default_value = (0.070, 0.080, 0.058, 1)
bnt.links.new(lmix.outputs[2], bark_b.inputs["Base Color"])

bbump = bnt.nodes.new("ShaderNodeBump")
bbump.inputs["Strength"].default_value = 1.0
bbump.inputs["Distance"].default_value = 0.05
bnt.links.new(furrow.outputs["Fac"], bbump.inputs["Height"])
coarse = bnt.nodes.new("ShaderNodeTexNoise")
coarse.inputs["Scale"].default_value = 1.4
coarse.inputs["Detail"].default_value = 6.0
bnt.links.new(bcoord.outputs["Object"], coarse.inputs["Vector"])
cbump = bnt.nodes.new("ShaderNodeBump")
cbump.inputs["Strength"].default_value = 0.7
cbump.inputs["Distance"].default_value = 0.35
bnt.links.new(coarse.outputs["Fac"], cbump.inputs["Height"])
bnt.links.new(bbump.outputs["Normal"], cbump.inputs["Normal"])
bnt.links.new(cbump.outputs["Normal"], bark_b.inputs["Normal"])


# ----------------------------------------------------------------------------
# trees
# ----------------------------------------------------------------------------
trunk_col = new_collection("trees")


def loft_rings(rings, name, sides=12):
    verts, faces = [], []
    for center, radius, side in rings:
        rad = side.cross(Vector((0, 0, 1))).normalized()
        for k in range(sides):
            a = k / sides * math.tau
            verts.append(center + rad * (math.cos(a) * radius) + side * (math.sin(a) * radius))
    for i in range(len(rings) - 1):
        for k in range(sides):
            a = i * sides + k
            b = i * sides + (k + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    me = bpy.data.meshes.new(name)
    me.from_pydata([v[:] for v in verts], [], faces)
    me.update()
    return me


def make_trunk(name, x, y, height, radius, sides=12):
    lean = Vector((rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), 0.0))
    curve = Vector((rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0.0))
    base_z = ground_height(x, y) - 0.25
    rings, N = [], 24
    for i in range(N):
        t = i / (N - 1)
        z = base_z + height * t
        off = lean * (t ** 1.5) * height * 0.03 + curve * math.sin(t * 2.0) * height * 0.018
        flare = 1.0 + 2.2 * math.exp(-t * height / 1.5)  # buttressed base
        r = radius * flare * (1.0 - 0.6 * t ** 1.5)
        r *= 1.0 + 0.14 * noise.noise(Vector((t * 7.0, x, y)))
        rings.append((Vector((x + off.x, y + off.y, z)), r, Vector((1, 0, 0))))
    obj = bpy.data.objects.new(name, loft_rings(rings, name, sides))
    obj.data.materials.append(bark)
    smooth(obj)
    link(obj, trunk_col)
    return obj, rings


def make_branch(x, y, z, radius, length, azimuth):
    elev = math.radians(rng.uniform(-8, 18))
    d = Vector((math.cos(azimuth) * math.cos(elev),
                math.sin(azimuth) * math.cos(elev),
                math.sin(elev)))
    rings, N = [], 8
    for i in range(N):
        t = i / (N - 1)
        p = Vector((x, y, z)) + d * (length * t) + Vector((0, 0, -length * 0.1 * t * t))
        rings.append((p, radius * (1.0 - 0.85 * t), Vector((0, 0, 1)).cross(d).normalized()))
    obj = bpy.data.objects.new("branch", loft_rings(rings, "branch", 7))
    obj.data.materials.append(bark)
    smooth(obj)
    link(obj, trunk_col)


# HERO trunks up front: fat, close, dark silhouettes that frame the shot.
hero_l, _ = make_trunk("hero_left", -6.4, CAM_Y + 9.5, 34.0, 0.75, sides=16)
hero_r, _ = make_trunk("hero_right", 5.9, CAM_Y + 7.0, 36.0, 0.62, sides=16)

# HERO mid-ground trunks: these carry the canopy the camera looks through.
placed = []
for i in range(26):
    ang = rng.uniform(-math.pi, math.pi)
    dist = rng.uniform(7.0, 30.0)
    x = math.sin(ang) * dist * 0.9
    y = CAM_Y + 10.0 + math.cos(ang) * dist
    if abs(x) < 2.6 and y < CAM_Y + 16.0:
        continue  # keep the sightline to the light open
    height = rng.uniform(22.0, 34.0)
    radius = rng.uniform(0.28, 0.55)
    obj, rings = make_trunk(f"trunk_{i}", x, y, height, radius)
    placed.append((Vector((x, y, 0.0)), height, radius))
    for _ in range(rng.randint(2, 4)):
        t = rng.uniform(0.45, 0.85)
        p = rings[int(t * (len(rings) - 1))][0]
        make_branch(p.x, p.y, p.z, radius * rng.uniform(0.3, 0.5),
                    rng.uniform(2.0, 6.0), rng.uniform(0, math.tau))

# FAR treeline: dense, thin, near-silhouette. This is the dark wall the sun
# breaks through - without it the light shafts have nothing to read against.
for i in range(150):
    ang = rng.uniform(-math.pi, math.pi)
    dist = rng.uniform(26.0, 78.0)
    x = math.sin(ang) * dist
    y = CAM_Y + 12.0 + math.cos(ang) * dist
    height = rng.uniform(24.0, 40.0)
    make_trunk(f"far_{i}", x, y, height, rng.uniform(0.16, 0.36), sides=7)


# ----------------------------------------------------------------------------
# canopy: branch clusters carrying instanced leaf cards
# ----------------------------------------------------------------------------
canopy_col = new_collection("canopy")
leaf, leaf_b = principled("canopy", (0.026, 0.042, 0.016), 0.62)
try:
    leaf_b.inputs["Transmission Weight"].default_value = 0.35  # backlit translucency
except KeyError:
    pass
lnt = leaf.node_tree
lcoord = lnt.nodes.new("ShaderNodeTexCoord")
ln = lnt.nodes.new("ShaderNodeTexNoise")
ln.inputs["Scale"].default_value = 2.2
ln.inputs["Detail"].default_value = 8.0
lnt.links.new(lcoord.outputs["Object"], ln.inputs["Vector"])
lramp = lnt.nodes.new("ShaderNodeValToRGB")
lramp.color_ramp.elements[0].position = 0.35
lramp.color_ramp.elements[0].color = (0.014, 0.026, 0.010, 1)
lramp.color_ramp.elements[1].position = 0.68
lramp.color_ramp.elements[1].color = (0.055, 0.080, 0.028, 1)
lnt.links.new(ln.outputs["Fac"], lramp.inputs["Fac"])
lnt.links.new(lramp.outputs["Color"], leaf_b.inputs["Base Color"])


def leaf_card_mesh(name, size=0.34):
    w, l = size * 0.5, size
    verts = [(0.0, 0.0, 0.0), (w, l * 0.4, w * 0.3), (0.0, l, 0.0), (-w, l * 0.4, w * 0.3)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    me.update()
    return me


card = leaf_card_mesh("leafcard")
card.materials.append(leaf)

# Clusters hang where the hero trunks branch, plus a general high layer.
cluster_centres = []
for pos, height, radius in placed:
    for _ in range(rng.randint(1, 3)):
        cluster_centres.append((
            Vector((pos.x + rng.uniform(-4, 4), pos.y + rng.uniform(-4, 4),
                    rng.uniform(0.55, 0.92) * height)),
            rng.uniform(2.0, 4.6),
        ))
for _ in range(60):
    a = rng.uniform(-math.pi, math.pi)
    d = rng.uniform(4.0, 46.0)
    cluster_centres.append((
        Vector((math.sin(a) * d, CAM_Y + 12.0 + math.cos(a) * d, rng.uniform(16.0, 30.0))),
        rng.uniform(2.4, 5.2),
    ))

for ci, (c, rad) in enumerate(cluster_centres):
    for j in range(rng.randint(18, 42)):
        o = bpy.data.objects.new(f"leaf_{ci}_{j}", card)
        off = Vector((rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1) * 0.55))
        off.normalize()
        o.location = c + off * rad * rng.uniform(0.15, 1.0)
        o.rotation_euler = (rng.uniform(0, math.tau), rng.uniform(0, math.tau),
                            rng.uniform(0, math.tau))
        o.scale = (rng.uniform(0.8, 1.8),) * 3
        link(o, canopy_col)


# ----------------------------------------------------------------------------
# undergrowth: fern clumps + deadfall logs
# ----------------------------------------------------------------------------
fern_col = new_collection("ferns")
fern, fern_b = principled("fern", (0.034, 0.066, 0.020), 0.5)
try:
    fern_b.inputs["Transmission Weight"].default_value = 0.25
except KeyError:
    pass


def build_frond_mesh(name, leaflets=15, length=1.9, width=0.62):
    verts, faces = [], []
    for i in range(leaflets):
        t = i / (leaflets - 1)
        z = math.sin(t * 1.9) * length * 0.65
        y = t * length
        w = (width * (1.0 - t) ** 0.55 * 0.5 + 0.012) * (1.0 + 0.22 * math.sin(t * 9.0))
        droop = 0.3 * t * t
        b = len(verts)
        verts += [(0.0, y - 0.04, z),
                  (w, y + 0.14, z + 0.05 - droop),
                  (0.0, y + 0.04, z + 0.02),
                  (-w, y + 0.14, z + 0.05 - droop)]
        faces.append((b, b + 1, b + 2, b + 3))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    return me


frond_mesh = build_frond_mesh("frond")
frond_mesh.materials.append(fern)
frond_mesh.shade_smooth()

clump = 0
for c in range(55):
    ang = rng.uniform(0, math.tau)
    dist = rng.uniform(1.0, 22.0)
    cx = math.sin(ang) * dist * 0.9
    cy = CAM_Y + 2.0 + math.cos(ang) * dist
    if abs(cx) < 1.1 and cy < CAM_Y + 6.0:
        cx += 2.1
    scale = rng.uniform(0.6, 1.6)
    for _ in range(rng.randint(4, 10)):
        obj = bpy.data.objects.new(f"fern_{clump}", frond_mesh)
        clump += 1
        ox, oy = rng.uniform(-0.35, 0.35) * scale, rng.uniform(-0.35, 0.35) * scale
        obj.location = (cx + ox, cy + oy, ground_height(cx + ox, cy + oy) - 0.05)
        obj.rotation_euler = (math.radians(rng.uniform(-24, 5)),
                              math.radians(rng.uniform(-18, 18)),
                              rng.uniform(0, math.tau))
        obj.scale = (scale, scale * rng.uniform(0.85, 1.2), scale)
        link(obj, fern_col)

# fallen logs: scale cues and foreground occlusion
log_mat, log_b = principled("log", (0.030, 0.024, 0.019), 0.85)
for i in range(7):
    x = rng.uniform(-14, 14)
    y = CAM_Y + rng.uniform(3.0, 24.0)
    length = rng.uniform(4.0, 11.0)
    r = rng.uniform(0.18, 0.38)
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=r, depth=length,
                                        location=(x, y, ground_height(x, y) + r * 0.7))
    lg = bpy.context.object
    lg.rotation_euler = (math.radians(90 + rng.uniform(-14, 14)), 0,
                         rng.uniform(0, math.tau))
    lg.data.materials.append(log_mat)
    smooth(lg)
    for cc in lg.users_collection:
        cc.objects.unlink(lg)
    link(lg, fern_col)


# ----------------------------------------------------------------------------
# fog: height-graded scattering with drifting density noise
# ----------------------------------------------------------------------------
fog_col = new_collection("fog")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 20, 20.0))
fog = bpy.context.object
fog.name = "fog_domain"
fog.scale = (420.0, 420.0, 60.0)
for c in fog.users_collection:
    c.objects.unlink(fog)
link(fog, fog_col)

fog_mat = bpy.data.materials.new("fog")
fog_mat.use_nodes = True
fnt = fog_mat.node_tree
fnt.nodes.remove(fnt.nodes["Principled BSDF"])
fout = fnt.nodes["Material Output"]

vol = fnt.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Color"].default_value = (0.68, 0.78, 1.0, 1.0)
vol.inputs["Anisotropy"].default_value = 0.72
vol.inputs["Absorption Color"].default_value = (0.34, 0.44, 0.62, 1.0)
fnt.links.new(vol.outputs["Volume"], fout.inputs["Volume"])

geo = fnt.nodes.new("ShaderNodeNewGeometry")
sep = fnt.nodes.new("ShaderNodeSeparateXYZ")
fnt.links.new(geo.outputs["Position"], sep.inputs[0])
height = fnt.nodes.new("ShaderNodeMapRange")
height.inputs["From Min"].default_value = 0.0
height.inputs["From Max"].default_value = 18.0
height.inputs["To Min"].default_value = 0.0055  # thick morning mist
height.inputs["To Max"].default_value = 0.0016  # thin upper air
height.clamp = True
fnt.links.new(sep.outputs["Z"], height.inputs["Value"])

coord_v = fnt.nodes.new("ShaderNodeTexCoord")
n1 = fnt.nodes.new("ShaderNodeTexNoise")
n1.inputs["Scale"].default_value = 0.05
n1.inputs["Detail"].default_value = 2.0
fnt.links.new(coord_v.outputs["Object"], n1.inputs["Vector"])
n2 = fnt.nodes.new("ShaderNodeTexNoise")
n2.inputs["Scale"].default_value = 0.16
n2.inputs["Detail"].default_value = 3.0
fnt.links.new(coord_v.outputs["Object"], n2.inputs["Vector"])

vary = fnt.nodes.new("ShaderNodeMapRange")
vary.inputs["From Min"].default_value = 0.32
vary.inputs["From Max"].default_value = 0.7
vary.inputs["To Min"].default_value = 0.3
vary.inputs["To Max"].default_value = 2.0
vary.clamp = True
fnt.links.new(n1.outputs["Fac"], vary.inputs["Value"])
vary2 = fnt.nodes.new("ShaderNodeMapRange")
vary2.inputs["To Min"].default_value = 0.65
vary2.inputs["To Max"].default_value = 1.35
fnt.links.new(n2.outputs["Fac"], vary2.inputs["Value"])

m1 = fnt.nodes.new("ShaderNodeMath")
m1.operation = "MULTIPLY"
fnt.links.new(height.outputs["Result"], m1.inputs[0])
fnt.links.new(vary.outputs["Result"], m1.inputs[1])
m2 = fnt.nodes.new("ShaderNodeMath")
m2.operation = "MULTIPLY"
fnt.links.new(m1.outputs[0], m2.inputs[0])
fnt.links.new(vary2.outputs["Result"], m2.inputs[1])
fnt.links.new(m2.outputs[0], vol.inputs["Density"])
fog.data.materials.append(fog_mat)


# ----------------------------------------------------------------------------
# light: hard low sun cutting across the corridor + faint cool bounce
# ----------------------------------------------------------------------------
sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 5.0
sun_data.color = (1.0, 0.60, 0.30)
sun_data.angle = math.radians(0.6)
sun = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun)
SUN_POS = Vector((26.0, 30.0, 26.0))
SUN_AIM = Vector((-4.0, CAM_Y + 14.0, 2.0))
sun.location = SUN_POS
sun.rotation_euler = (SUN_AIM - SUN_POS).normalized().to_track_quat("-Z", "Y").to_euler()

fill_data = bpy.data.lights.new("skylight", "AREA")
fill_data.energy = 90.0
fill_data.size = 26.0
fill_data.color = (0.45, 0.62, 0.95)
fill = bpy.data.objects.new("skylight", fill_data)
scene.collection.objects.link(fill)
fill.location = (-14.0, CAM_Y + 6.0, 20.0)
fill.rotation_euler = (math.radians(28), math.radians(-16), 0)


# ----------------------------------------------------------------------------
# camera: low, slow dolly-in, slight rise, hint of handheld drift
# ----------------------------------------------------------------------------
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 32.0
cam_data.sensor_width = 36.0
cam_data.dof.use_dof = True
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

TARGET = Vector((0.5, CAM_Y + 22.0, 8.6))


def set_camera(f):
    t = f / max(FRAMES - 1, 1)
    e = t * t * (3 - 2 * t)
    cam.location = Vector((
        0.45 * math.sin(e * 1.3),
        CAM_Y + 2.4 * e,
        1.45 + 0.22 * e + 0.012 * math.sin(f * 0.85),
    ))
    look = TARGET + Vector((0.4 * math.sin(f * 0.26), 0, 0.14 * math.sin(f * 0.18 + 1.0)))
    cam.rotation_euler = (look - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam_data.dof.focus_distance = (cam.location - Vector((0.6, CAM_Y + 4.0, 0.6))).length
    cam_data.dof.aperture_fstop = 3.2 - 1.0 * e


scene.frame_start = 1
scene.frame_end = FRAMES
for f in range(1, FRAMES + 1):
    set_camera(f)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.dof.keyframe_insert("aperture_fstop", frame=f)


# ----------------------------------------------------------------------------
# compositor: fog glow so the sun blooms through the branches
# ----------------------------------------------------------------------------
comp = bpy.data.node_groups.new("comp", "CompositorNodeTree")
scene.compositing_node_group = comp
comp.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")

rl = comp.nodes.new("CompositorNodeRLayers")
glare = comp.nodes.new("CompositorNodeGlare")
glare.inputs["Type"].default_value = "Fog Glow"
glare.inputs["Quality"].default_value = "High"
glare.inputs["Threshold"].default_value = 1.2
glare.inputs["Strength"].default_value = 0.4
glare.inputs["Size"].default_value = 0.6
out_node = comp.nodes.new("NodeGroupOutput")
comp.links.new(rl.outputs["Image"], glare.inputs["Image"])
comp.links.new(glare.outputs["Image"], out_node.inputs[0])


# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[forest] {RES_X}x{RES_Y} samples={SAMPLES} frames={FRAMES} preview={PREVIEW}")

if SINGLE:
    scene.frame_set(SINGLE)
    scene.render.filepath = os.path.join(OUT_DIR, f"still_{SINGLE:04d}.png")
    bpy.ops.render.render(write_still=True)
elif PREVIEW:
    for f in (STILLS or [1, FRAMES // 2, FRAMES]):
        tag = f"{f:02d}"
        scene.frame_set(f)
        scene.render.filepath = os.path.join(OUT_DIR, f"preview_{tag}.png")
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
