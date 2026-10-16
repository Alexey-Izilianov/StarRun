"""StarRun reference scene: desert, minute 8. Run: blender -b -P scene.py -- out.png"""
import math
import random
import sys

import bpy
from mathutils import Vector

OUT = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else "//ref.png"
RES = (1920, 1080)
rng = random.Random(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


# ---------- materials ----------
def mat(name, color, rough=0.5, metal=0.0, emit=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emit
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.25
    return m


def add_bump(m, scale, strength, kind="noise"):
    nt = m.node_tree
    if kind == "noise":
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = 6
        out = n.outputs["Fac"]
    else:  # panel seams
        n = nt.nodes.new("ShaderNodeTexBrick")
        n.inputs["Scale"].default_value = scale
        n.inputs["Mortar Size"].default_value = 0.01
        n.offset = 0.0
        out = n.outputs["Fac"]
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    nt.links.new(out, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])


def random_tint(m, amount):
    """Per-object brightness jitter so a crowd of copies does not look cloned."""
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    base = tuple(b.inputs["Base Color"].default_value)
    info = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(c * (1 - amount) for c in base[:3]) + (1,)
    ramp.color_ramp.elements[1].color = tuple(min(1, c * (1 + amount)) for c in base[:3]) + (1,)
    nt.links.new(info.outputs["Random"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])


def grime(m, dirt, scale=3.0, low=0.45, high=0.75):
    """Mix base colour toward a dirt colour by a noise mask (dust, rust, wear)."""
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    base = tuple(b.inputs["Base Color"].default_value)
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = 8
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = low
    ramp.color_ramp.elements[0].color = base
    ramp.color_ramp.elements[1].position = high
    ramp.color_ramp.elements[1].color = (*dirt, 1)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])


M = {
    "armor": mat("armor", (0.16, 0.18, 0.11), 0.5, 0.25),
    "pauldron": mat("pauldron", (0.30, 0.31, 0.25), 0.45, 0.3),
    "suit": mat("suit", (0.03, 0.03, 0.035), 0.7),
    "visor": mat("visor", (1.0, 0.35, 0.05), 0.2, emit=6.0),
    "gun": mat("gun", (0.035, 0.035, 0.04), 0.35, 0.8),
    "shell": mat("shell", (0.06, 0.03, 0.016), 0.38, coat=0.8),
    "plate": mat("plate", (0.24, 0.10, 0.035), 0.35, coat=0.8),
    "leg": mat("leg", (0.05, 0.03, 0.018), 0.45, coat=0.4),
    "warrior": mat("warrior", (0.40, 0.06, 0.025), 0.3, coat=0.8),
    "blade": mat("blade", (0.62, 0.52, 0.36), 0.25, coat=0.9),
    "tank_plate": mat("tank_plate", (0.13, 0.10, 0.07), 0.45, coat=0.6),
    "tank_edge": mat("tank_edge", (0.55, 0.42, 0.24), 0.35, coat=0.6),
    "sac": mat("sac", (0.22, 0.45, 0.02), 0.15, emit=0.6, coat=1.0),
    "solar": mat("solar", (0.02, 0.035, 0.08), 0.15, 0.6),
    "blood": mat("blood", (0.25, 0.9, 0.02), 0.06, emit=0.6),
    "blood_dark": mat("blood_dark", (0.05, 0.25, 0.01), 0.1),
    "crystal": mat("crystal", (0.10, 0.85, 1.0), 0.1, emit=6.0),
    "tracer": mat("tracer", (1.0, 0.55, 0.12), 0.5, emit=40.0),
    "marker": mat("marker", (1.0, 0.07, 0.02), 0.5, emit=10.0),
    "wall": mat("wall", (0.46, 0.45, 0.41), 0.75),
    "roof": mat("roof", (0.10, 0.105, 0.11), 0.55, 0.4),
    "stripe": mat("stripe", (0.75, 0.22, 0.03), 0.5),
    "glass": mat("glass", (0.01, 0.015, 0.02), 0.06, 0.6),
    "concrete": mat("concrete", (0.34, 0.32, 0.29), 0.85),
    "sandbag": mat("sandbag", (0.30, 0.24, 0.13), 0.9),
    "cactus": mat("cactus", (0.07, 0.14, 0.06), 0.55),
    "cactus_cut": mat("cactus_cut", (0.42, 0.48, 0.22), 0.7),
    "flower": mat("flower", (0.9, 0.25, 0.30), 0.5),
    "rock": mat("rock", (0.24, 0.17, 0.11), 0.85),
    "bush": mat("bush", (0.16, 0.11, 0.06), 0.9),
    "canister": mat("canister", (0.22, 0.23, 0.24), 0.35, 0.7),
    "dna_glow": mat("dna_glow", (0.10, 0.85, 1.0), 0.2, emit=6.0),
}
for k, amt in (("shell", 0.3), ("plate", 0.35), ("warrior", 0.3), ("tank_plate", 0.2), ("rock", 0.35), ("cactus", 0.3), ("bush", 0.3)):
    random_tint(M[k], amt)
for k in ("shell", "plate", "warrior", "tank_plate", "leg"):
    add_bump(M[k], 45, 0.2)
add_bump(M["rock"], 6, 0.7)
add_bump(M["sandbag"], 25, 0.3)
add_bump(M["armor"], 30, 0.05)
grime(M["wall"], (0.34, 0.24, 0.14), 2.5, 0.5, 0.8)
grime(M["roof"], (0.26, 0.17, 0.09), 1.6, 0.45, 0.85)
add_bump(M["roof"], 1.2, 0.35, kind="panels")
add_bump(M["wall"], 1.5, 0.15, kind="panels")
add_bump(M["solar"], 4.0, 0.4, kind="panels")


# ---------- mesh helpers ----------
def _finish(ob, material, smooth=True):
    ob.data.materials.append(material)
    if smooth:
        bpy.ops.object.shade_smooth()
    return ob


def ellipsoid(loc, size, material, segs=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=segs // 2, radius=1, location=loc)
    ob = bpy.context.object
    ob.scale = size
    return _finish(ob, material)


def box(loc, size, material, bevel=0.03, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=2, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if bevel:
        mod = ob.modifiers.new("bev", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.data.materials.append(material)
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
    return ob


def limb(p1, p2, r1, r2, material, verts=10):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r1, radius2=r2, depth=d.length, location=(p1 + p2) / 2)
    ob = bpy.context.object
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = d.to_track_quat("Z", "Y")
    return _finish(ob, material)


def ribbed(ob, n, depth=0.18):
    """Pull every second radial vertex inward: real ribs for cacti."""
    for v in ob.data.vertices:
        r = math.hypot(v.co.x, v.co.y)
        if r < 1e-4:
            continue
        idx = round(math.atan2(v.co.y, v.co.x) / (2 * math.pi / n))
        if idx % 2:
            v.co.x *= 1 - depth
            v.co.y *= 1 - depth
    return ob


def join(parts, name):
    bpy.ops.object.select_all(action="DESELECT")
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.join()
    ob = bpy.context.object
    ob.name = name
    return ob


TEMPLATES = bpy.data.collections.new("templates")


def template(ob):
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for c in ob.users_collection:
        c.objects.unlink(ob)
    TEMPLATES.objects.link(ob)
    return ob


def place(tpl, loc, yaw=0.0, scale=1.0, roll=0.0):
    ob = tpl.copy()
    scene.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler = (roll, 0, yaw)
    ob.scale = (scale, scale, scale)
    return ob


def legs(parts, body_r, z_hip, spread, length, r, material, lift=0.16):
    """Six insect legs: hip -> raised knee -> foot on the ground, splayed fore and aft."""
    for i, y in enumerate((0.10, 0.0, -0.10)):
        fwd = (1 - i) * 0.16
        for s in (-1, 1):
            hip = (s * body_r, y * spread, z_hip)
            knee = (s * (body_r + length * 0.5), y * spread + fwd, z_hip + lift)
            foot = (s * (body_r + length), y * spread + fwd * 2.2, 0.0)
            parts.append(limb(hip, knee, r, r * 0.8, material, verts=8))
            parts.append(limb(knee, foot, r * 0.8, r * 0.25, material, verts=8))


def segmented_abdomen(parts, y0, length, width, height, z, n, shell, plate):
    """Abdomen as overlapping segments: dark shell with a lighter dorsal plate on each."""
    for i in range(n):
        t = i / (n - 1)
        k = math.sin(math.pi * (0.25 + 0.75 * t)) * 0.35 + 0.65
        y = y0 - t * length
        parts.append(ellipsoid((0, y, z), (width * k, length / n * 0.85, height * k), shell))
        parts.append(ellipsoid((0, y + 0.02, z + height * k * 0.55), (width * k * 0.75, length / n * 0.7, height * k * 0.5), plate))


# ---------- bugs (forward = +Y) ----------
def make_worker():
    p = [ellipsoid((0, -0.27, 0.27), (0.23, 0.30, 0.18), M["shell"])]
    for i, (y, w) in enumerate(((-0.10, 0.17), (-0.24, 0.20), (-0.38, 0.17))):  # dorsal plates
        p.append(ellipsoid((0, y, 0.38 - i * 0.02), (w, 0.10, 0.07), M["plate"]))
    p.append(ellipsoid((0, 0.06, 0.28), (0.14, 0.15, 0.12), M["shell"]))
    p.append(ellipsoid((0, 0.07, 0.36), (0.10, 0.12, 0.05), M["plate"]))
    p.append(ellipsoid((0, 0.26, 0.26), (0.12, 0.11, 0.09), M["shell"]))
    for s in (-1, 1):  # curved mandibles
        a, b, c = (s * 0.07, 0.32, 0.24), (s * 0.10, 0.44, 0.22), (s * 0.02, 0.53, 0.20)
        p.append(limb(a, b, 0.035, 0.025, M["leg"], verts=8))
        p.append(limb(b, c, 0.025, 0.003, M["leg"], verts=8))
        p.append(limb((s * 0.04, 0.33, 0.30), (s * 0.18, 0.55, 0.42), 0.012, 0.004, M["leg"], verts=6))  # antennae
    legs(p, 0.10, 0.27, 1.1, 0.40, 0.026, M["leg"])
    return template(join(p, "worker"))


def make_warrior():
    p = []
    segmented_abdomen(p, -0.22, 0.75, 0.26, 0.2, 0.40, 5, M["shell"], M["warrior"])
    p.append(ellipsoid((0, 0.06, 0.46), (0.19, 0.24, 0.17), M["shell"]))
    p.append(ellipsoid((0, 0.08, 0.58), (0.14, 0.2, 0.07), M["warrior"]))
    p.append(ellipsoid((0, 0.36, 0.44), (0.14, 0.15, 0.11), M["shell"]))
    for s in (-1, 1):
        sh = (s * 0.16, 0.18, 0.52)
        el = (s * 0.42, 0.40, 1.00)
        tip = (s * 0.26, 1.15, 0.45)
        p.append(limb(sh, el, 0.07, 0.055, M["warrior"]))
        blade = limb(el, tip, 0.11, 0.004, M["blade"], verts=10)
        bpy.ops.object.transform_apply(rotation=True)
        p.append(blade)
        p.append(limb((s * 0.10, 0.10, 0.62), (s * 0.08, -0.05, 0.85), 0.04, 0.003, M["blade"], verts=6))  # back spikes
    legs(p, 0.14, 0.40, 1.5, 0.62, 0.042, M["leg"], lift=0.26)
    return template(join(p, "warrior"))


def make_tanker():
    p = []
    # overlapping armour bands, widest in the middle
    for i in range(6):
        t = i / 5
        y = 0.55 - t * 1.6
        w = 0.62 + 0.25 * math.sin(math.pi * t)
        p.append(ellipsoid((0, y, 0.55), (w, 0.36, 0.42 + 0.12 * math.sin(math.pi * t)), M["tank_plate"]))
        p.append(ellipsoid((0, y - 0.05, 0.62 + 0.12 * math.sin(math.pi * t)), (w * 0.92, 0.30, 0.38), M["tank_edge"] if i % 2 else M["tank_plate"]))
    # front ram shield with horns
    p.append(ellipsoid((0, 0.95, 0.55), (0.85, 0.22, 0.5), M["tank_edge"]))
    p.append(ellipsoid((0, 1.00, 0.55), (0.78, 0.20, 0.44), M["tank_plate"]))
    for s in (-1, 1):
        p.append(limb((s * 0.45, 1.05, 0.8), (s * 0.75, 1.55, 1.15), 0.12, 0.01, M["blade"], verts=8))
    for i in range(4):
        y = 0.3 - i * 0.42
        p.append(limb((0, y, 1.05), (0, y - 0.22, 1.45 - i * 0.05), 0.10, 0.008, M["blade"], verts=6))
    legs(p, 0.6, 0.5, 3.4, 0.7, 0.10, M["leg"], lift=0.32)
    return template(join(p, "tanker"))


def make_spitter():
    p = [ellipsoid((0, -0.45, 0.48), (0.36, 0.46, 0.38), M["sac"])]
    for i in range(3):  # chitin plates over the acid sac, glowing gaps between them
        y = -0.22 - i * 0.24
        p.append(ellipsoid((0, y, 0.62), (0.36 - abs(i - 1) * 0.05, 0.11, 0.30), M["plate"], segs=16))
    p.append(ellipsoid((0, 0.04, 0.34), (0.16, 0.18, 0.13), M["shell"]))
    p.append(ellipsoid((0, 0.05, 0.42), (0.11, 0.14, 0.06), M["plate"]))
    p.append(ellipsoid((0, 0.26, 0.32), (0.11, 0.12, 0.09), M["shell"]))
    legs(p, 0.11, 0.30, 1.2, 0.46, 0.03, M["leg"])
    return template(join(p, "spitter"))


# ---------- marine (forward = +Y) ----------
def make_marine():
    A, P, S, G = M["armor"], M["pauldron"], M["suit"], M["gun"]
    p = []
    for s, fy in ((-1, -0.12), (1, 0.16)):  # legs in a firing stance
        x = s * 0.19
        p.append(box((x, fy + 0.04, 0.10), (0.13, 0.21, 0.10), A, 0.05))          # boot
        p.append(box((x, fy, 0.40), (0.115, 0.13, 0.22), S, 0.04))               # shin
        p.append(box((x, fy + 0.10, 0.42), (0.11, 0.05, 0.20), A, 0.04))          # shin guard
        p.append(ellipsoid((x, fy + 0.12, 0.64), (0.10, 0.08, 0.09), P))           # knee pad
        p.append(box((x, fy * 0.5, 0.82), (0.13, 0.15, 0.18), A, 0.05))          # thigh plate
    p += [
        box((0, 0, 1.02), (0.30, 0.20, 0.10), S, 0.04),                            # belt
        box((0, 0.02, 1.34), (0.38, 0.26, 0.30), A, 0.11),                         # torso
        box((0, 0.25, 1.38), (0.28, 0.05, 0.20), P, 0.04, rot=(math.radians(-8), 0, 0)),  # chest plate
        box((0, -0.34, 1.36), (0.28, 0.13, 0.30), A, 0.07),                        # backpack
        box((0, 0.0, 1.70), (0.16, 0.16, 0.06), S, 0.03),                          # neck
        ellipsoid((0, 0.03, 1.90), (0.18, 0.21, 0.17), A),                         # helmet
        box((0, 0.02, 2.06), (0.035, 0.17, 0.035), P, 0.015),                      # helmet ridge
        box((0, 0.21, 1.88), (0.13, 0.035, 0.035), M["visor"], 0.0),
    ]
    for s in (-1, 1):
        p.append(limb((s * 0.14, -0.47, 1.12), (s * 0.14, -0.47, 1.62), 0.075, 0.075, G, verts=12))  # tanks
        pd = ellipsoid((s * 0.46, -0.02, 1.60), (0.21, 0.25, 0.13), P)              # pauldron, tilted out
        pd.rotation_euler = (0, s * math.radians(35), 0)
        p.append(pd)
        low = ellipsoid((s * 0.52, -0.02, 1.46), (0.18, 0.22, 0.08), A)
        low.rotation_euler = (0, s * math.radians(30), 0)
        p.append(low)
    # arms bring the rifle up on the right side
    p.append(limb((-0.46, 0.0, 1.42), (-0.24, 0.30, 1.22), 0.095, 0.085, A))
    p.append(limb((-0.24, 0.30, 1.22), (0.04, 0.62, 1.28), 0.085, 0.07, A))
    p.append(limb((0.46, 0.0, 1.42), (0.30, 0.12, 1.18), 0.095, 0.085, A))
    p.append(limb((0.30, 0.12, 1.18), (0.16, 0.30, 1.30), 0.085, 0.07, A))
    gx = 0.10
    p.append(box((gx, 0.42, 1.34), (0.06, 0.40, 0.085), G, 0.015))               # receiver
    p.append(limb((gx, 0.80, 1.36), (gx, 1.30, 1.36), 0.03, 0.03, G, verts=10))   # barrel
    p.append(box((gx, 0.90, 1.26), (0.05, 0.18, 0.05), G, 0.01))                 # under-barrel launcher
    p.append(box((gx, 0.40, 1.20), (0.04, 0.07, 0.12), G, 0.01))                 # magazine
    p.append(box((gx, 0.30, 1.46), (0.035, 0.12, 0.035), G, 0.01))               # sight
    p.append(box((gx, 0.02, 1.32), (0.05, 0.10, 0.07), G, 0.01))                 # stock
    return join(p, "marine")


# ---------- props ----------
def make_cactus(h, arms):
    p = [ribbed(limb((0, 0, 0), (0, 0, h), 0.34, 0.30, M["cactus"], verts=28), 28), ellipsoid((0, 0, h), (0.30, 0.30, 0.24), M["cactus"])]
    for ang, z, up in arms:
        dx, dy = math.cos(ang), math.sin(ang)
        a = (dx * 0.2, dy * 0.2, z)
        b = (dx * 0.72, dy * 0.72, z + 0.18)
        c = (dx * 0.78, dy * 0.78, z + up)
        p.append(ribbed(limb(a, b, 0.21, 0.2, M["cactus"], verts=20), 20))
        p.append(ellipsoid(b, (0.2, 0.2, 0.2), M["cactus"]))
        p.append(ribbed(limb(b, c, 0.2, 0.18, M["cactus"], verts=20), 20))
        p.append(ellipsoid(c, (0.18, 0.18, 0.14), M["cactus"]))
    return join(p, "cactus")


def make_barrel(r):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1)
    ob = bpy.context.object
    ribbed(ob, 32, 0.12)
    ob.scale = (r, r, r * 0.85)
    ob.location.z = r * 0.55
    _finish(ob, M["cactus"])
    parts = [ob]
    for i in range(5):
        a = i / 5 * 6.28
        parts.append(ellipsoid((math.cos(a) * r * 0.25, math.sin(a) * r * 0.25, r * 1.38), (r * 0.12, r * 0.12, r * 0.08), M["flower"]))
    return join(parts, "barrel")


def make_rock(r):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=r)
    ob = bpy.context.object
    tex = bpy.data.textures.new("rocknoise", "VORONOI")
    tex.noise_scale = 0.6
    mod = ob.modifiers.new("d", "DISPLACE")
    mod.texture = tex
    mod.strength = r * 0.4
    bpy.ops.object.modifier_apply(modifier=mod.name)
    ob.scale = (1.0, 0.8, 0.5)
    return _finish(ob, M["rock"], smooth=False)


def make_bush():
    p = []
    for _ in range(14):
        a, tilt, ln = rng.uniform(0, 6.28), rng.uniform(0.5, 1.2), rng.uniform(0.3, 0.55)
        tip = (math.cos(a) * math.sin(tilt) * ln, math.sin(a) * math.sin(tilt) * ln, math.cos(tilt) * ln)
        p.append(limb((0, 0, 0), tip, 0.025, 0.004, M["bush"], verts=5))
    return template(join(p, "bush"))


def make_splat(r, pieces=9):
    parts = []
    for i in range(pieces):
        rr = r * (1.0 if i == 0 else rng.uniform(0.12, 0.35))
        off = (0, 0) if i == 0 else (rng.uniform(-1.8, 1.8) * r, rng.uniform(-1.8, 1.8) * r)
        for layer, (scale, z, m) in enumerate(((1.15, 0.012, M["blood_dark"]), (1.0, 0.02, M["blood"]))):
            bpy.ops.mesh.primitive_circle_add(vertices=28, radius=rr * scale, fill_type="NGON", location=(off[0], off[1], z))
            ob = bpy.context.object
            ph = random.Random(i * 31 + 5)
            f1, f2, f3 = ph.uniform(0, 6.28), ph.uniform(0, 6.28), ph.uniform(0, 6.28)
            for v in ob.data.vertices:  # smooth lobed outline instead of jagged shards
                a = math.atan2(v.co.y, v.co.x)
                k = 1 + 0.22 * math.sin(3 * a + f1) + 0.14 * math.sin(5 * a + f2) + 0.08 * math.sin(9 * a + f3)
                v.co.x *= k
                v.co.y *= k
            ob.data.materials.append(m)
            parts.append(ob)
    return join(parts, "splat")


def make_crystal():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=1)
    ob = bpy.context.object
    ob.scale = (0.08, 0.08, 0.15)
    ob.data.materials.append(M["crystal"])
    return template(ob)


def make_house(loc, yaw):
    """Prefab colonist module: walls with window/door gaps, flat roof with gear."""
    w, d, h, t = 4.2, 2.8, 2.6, 0.15
    p = []
    for s in (-1, 1):
        y = s * d
        for x0, x1 in ((-w, -2.6), (-1.6, 0.9), (1.9, w)):
            p.append(box(((x0 + x1) / 2, y, h / 2), ((x1 - x0) / 2, t, h / 2), M["wall"], 0.02))
        for x0, x1 in ((-2.6, -1.6), (0.9, 1.9)):
            p.append(box(((x0 + x1) / 2, y, 0.45), ((x1 - x0) / 2, t, 0.45), M["wall"], 0.02))
            p.append(box(((x0 + x1) / 2, y, 2.35), ((x1 - x0) / 2, t, 0.25), M["wall"], 0.02))
            p.append(box(((x0 + x1) / 2, y, 1.5), ((x1 - x0) / 2, 0.03, 0.6), M["glass"], 0.0))
    p.append(box((-w, 0, h / 2), (t, d, h / 2), M["wall"], 0.02))
    p.append(box((w, -1.6, h / 2), (t, d - 1.6 + 0.4, h / 2), M["wall"], 0.02))
    p.append(box((w, 1.95, h / 2), (t, 0.85, h / 2), M["wall"], 0.02))
    p.append(box((w, 0.55, 2.35), (t, 0.55, 0.25), M["wall"], 0.02))
    # roof: slab, raised rim, stripe, AC units, solar panel, antenna
    p.append(box((0, 0, h + 0.1), (w + 0.3, d + 0.3, 0.12), M["roof"], 0.04))
    for s in (-1, 1):
        p.append(box((0, s * (d + 0.25), h + 0.28), (w + 0.3, 0.06, 0.08), M["roof"], 0.02))
        p.append(box((s * (w + 0.25), 0, h + 0.28), (0.06, d + 0.3, 0.08), M["roof"], 0.02))
    p.append(box((0, -d - 0.31, h + 0.08), (w + 0.3, 0.03, 0.08), M["stripe"], 0.0))
    p.append(box((-2.6, 1.2, h + 0.48), (0.65, 0.55, 0.32), M["canister"], 0.05))
    p.append(box((-1.2, 1.2, h + 0.40), (0.45, 0.45, 0.24), M["canister"], 0.04))
    p.append(box((1.6, -0.9, h + 0.32), (1.4, 1.0, 0.04), M["solar"], 0.02, rot=(math.radians(12), 0, 0)))
    p.append(box((1.6, -0.9, h + 0.30), (1.46, 1.06, 0.03), M["canister"], 0.01, rot=(math.radians(12), 0, 0)))
    p.append(limb((3.3, 1.8, h + 0.2), (3.3, 1.8, h + 1.6), 0.05, 0.03, M["gun"]))
    ob = join(p, "house")
    ob.location = loc
    ob.rotation_euler = (0, 0, yaw)
    return ob


# ---------- ground ----------
QS_CENTER = Vector((6.6, -2.3, 0))
QS_R = 2.6


def make_ground():
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=300, y_subdivisions=260, size=1)
    g = bpy.context.object
    g.scale = (50, 44, 1)
    bpy.ops.object.transform_apply(scale=True)
    tex = bpy.data.textures.new("dunes", "CLOUDS")
    tex.noise_scale = 5.0
    mod = g.modifiers.new("dunes", "DISPLACE")
    mod.texture = tex
    mod.strength = 0.6
    mod.mid_level = 0.5
    bpy.ops.object.modifier_apply(modifier=mod.name)
    for v in g.data.vertices:  # quicksand: shallow bowl with a slight lip
        dist = (Vector((v.co.x, v.co.y, 0)) - QS_CENTER).length
        if dist < QS_R * 1.25:
            t = dist / (QS_R * 1.25)
            v.co.z += -0.35 * (1 - t) ** 2 + 0.06 * math.sin(t * math.pi) ** 4
    bpy.ops.object.shade_smooth()

    m = bpy.data.materials.new("sand")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    L = nt.links.new
    geo = nt.nodes.new("ShaderNodeNewGeometry")

    big = nt.nodes.new("ShaderNodeTexNoise")
    big.inputs["Scale"].default_value = 0.09
    big.inputs["Detail"].default_value = 8
    L(geo.outputs["Position"], big.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.38
    ramp.color_ramp.elements[0].color = (0.30, 0.16, 0.065, 1)
    ramp.color_ramp.elements[1].position = 0.62
    ramp.color_ramp.elements[1].color = (0.50, 0.30, 0.13, 1)
    L(big.outputs["Fac"], ramp.inputs["Fac"])

    # gravel / hardpan patches
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 9.0
    L(geo.outputs["Position"], vor.inputs["Vector"])
    patch = nt.nodes.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 0.25
    patch.inputs["Detail"].default_value = 4
    L(geo.outputs["Position"], patch.inputs["Vector"])
    pmask = nt.nodes.new("ShaderNodeMapRange")
    pmask.inputs["From Min"].default_value = 0.55
    pmask.inputs["From Max"].default_value = 0.62
    L(patch.outputs["Fac"], pmask.inputs["Value"])
    gravel_col = nt.nodes.new("ShaderNodeMix")
    gravel_col.data_type = "RGBA"
    gravel_col.inputs["A"].default_value = (0.22, 0.13, 0.07, 1)
    gravel_col.inputs["B"].default_value = (0.36, 0.25, 0.15, 1)
    L(vor.outputs["Distance"], gravel_col.inputs["Factor"])
    g_mul = nt.nodes.new("ShaderNodeMath")
    g_mul.operation = "MULTIPLY"
    g_mul.inputs[1].default_value = 0.8
    L(pmask.outputs["Result"], g_mul.inputs[0])
    with_gravel = nt.nodes.new("ShaderNodeMix")
    with_gravel.data_type = "RGBA"
    L(g_mul.outputs[0], with_gravel.inputs["Factor"])
    L(ramp.outputs["Color"], with_gravel.inputs["A"])
    L(gravel_col.outputs["Result"], with_gravel.inputs["B"])

    # quicksand mask and swirl
    dist = nt.nodes.new("ShaderNodeVectorMath")
    dist.operation = "DISTANCE"
    dist.inputs[1].default_value = QS_CENTER
    L(geo.outputs["Position"], dist.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = QS_R * 0.8
    mr.inputs["From Max"].default_value = QS_R * 1.1
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    L(dist.outputs["Value"], mr.inputs["Value"])
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = QS_CENTER
    L(geo.outputs["Position"], sub.inputs[0])
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    L(sub.outputs["Vector"], sep.inputs[0])
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    L(sep.outputs["X"], comb.inputs["X"])
    L(sep.outputs["Y"], comb.inputs["Y"])
    rings = nt.nodes.new("ShaderNodeTexWave")
    rings.wave_type = "RINGS"
    rings.rings_direction = "SPHERICAL"
    rings.inputs["Scale"].default_value = 0.4
    rings.inputs["Distortion"].default_value = 5.0
    rings.inputs["Detail"].default_value = 3
    rings.inputs["Detail Scale"].default_value = 1.5
    L(comb.outputs[0], rings.inputs["Vector"])
    qs_col = nt.nodes.new("ShaderNodeMix")
    qs_col.data_type = "RGBA"
    qs_col.inputs["A"].default_value = (0.12, 0.07, 0.03, 1)
    qs_col.inputs["B"].default_value = (0.26, 0.16, 0.07, 1)
    L(rings.outputs["Fac"], qs_col.inputs["Factor"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    L(mr.outputs["Result"], mix.inputs["Factor"])
    L(with_gravel.outputs["Result"], mix.inputs["A"])
    L(qs_col.outputs["Result"], mix.inputs["B"])
    L(mix.outputs["Result"], b.inputs["Base Color"])
    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.inputs["To Min"].default_value = 0.95
    rough.inputs["To Max"].default_value = 0.3
    L(mr.outputs["Result"], rough.inputs["Value"])
    L(rough.outputs["Result"], b.inputs["Roughness"])

    # wind ripples + grain + quicksand rings as bump
    rip = nt.nodes.new("ShaderNodeTexWave")
    rip.inputs["Scale"].default_value = 0.6
    rip.inputs["Distortion"].default_value = 7.0
    rip.inputs["Detail"].default_value = 3.0
    L(geo.outputs["Position"], rip.inputs["Vector"])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 120
    h1 = nt.nodes.new("ShaderNodeMath")
    h1.operation = "MULTIPLY_ADD"
    h1.inputs[1].default_value = 0.35
    L(grain.outputs["Fac"], h1.inputs[0])
    L(rip.outputs["Fac"], h1.inputs[2])
    h2 = nt.nodes.new("ShaderNodeMath")
    h2.operation = "MULTIPLY_ADD"
    L(rings.outputs["Fac"], h2.inputs[0])
    L(mr.outputs["Result"], h2.inputs[1])
    L(h1.outputs[0], h2.inputs[2])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    L(h2.outputs[0], bump.inputs["Height"])
    L(bump.outputs["Normal"], b.inputs["Normal"])
    g.data.materials.append(m)
    return g


ground = make_ground()


def ground_z(x, y):
    """Ground height under (x, y); the ground mesh has an identity transform."""
    hit, loc, *_ = ground.ray_cast(Vector((x, y, 10)), Vector((0, 0, -1)))
    return loc.z if hit else 0.0


# ---------- build scene ----------
worker, warrior, tanker, spitter = make_worker(), make_warrior(), make_tanker(), make_spitter()
crystal, bush = make_crystal(), make_bush()

MARINE_POS = Vector((0.0, 0.0, 0))
marine = make_marine()
marine.location = (0, 0, ground_z(0, 0))
marine.rotation_euler = (0, 0, math.radians(-52))
marine.scale = (1.35, 1.35, 1.35)

HOUSE = (-8.4, 4.6)
make_house((*HOUSE, ground_z(*HOUSE) - 0.05), math.radians(6))
blockers = [(*HOUSE, 5.2), (QS_CENTER.x, QS_CENTER.y, QS_R + 0.2)]


def free(x, y, r=0.6):
    return all((x - bx) ** 2 + (y - by) ** 2 > (br + r) ** 2 for bx, by, br in blockers)


for (x, y, h, arms) in (
    (-4.4, -3.9, 3.2, [(0.2, 1.4, 0.9), (3.3, 1.9, 0.7)]),
    (3.0, -4.6, 2.6, [(2.9, 1.2, 0.8)]),
    (-10.0, -1.6, 3.6, [(0.4, 1.6, 1.0), (3.6, 1.2, 0.6), (1.8, 2.2, 0.5)]),
    (10.4, 7.6, 3.0, [(3.0, 1.5, 0.8)]),
    (-3.0, 9.2, 2.8, [(0.1, 1.3, 0.7)]),
):
    c = make_cactus(h, arms)
    c.location = (x, y, ground_z(x, y) - 0.05)
    c.rotation_euler = (0, 0, rng.uniform(0, 6.28))
    blockers.append((x, y, 1.0))
for (x, y, r) in ((-5.6, -4.8, 0.5), (-5.0, -4.2, 0.32), (1.9, -5.0, 0.42), (-10.8, 0.6, 0.45), (10.4, -4.4, 0.55), (11.0, -3.7, 0.33)):
    b = make_barrel(r)
    b.location = (x, y, ground_z(x, y) - 0.05)
    blockers.append((x, y, r + 0.2))

# chopped cactus: stump + fallen pieces near the marine (axe mechanic)
SX, SY = -2.7, 1.5
stump = ribbed(limb((0, 0, 0), (0, 0, 0.45), 0.32, 0.31, M["cactus"], verts=28), 28)
stump.location = (SX, SY, ground_z(SX, SY))
ellipsoid((SX, SY, stump.location.z + 0.46), (0.28, 0.28, 0.02), M["cactus_cut"])
for i, (dx, dy, ang) in enumerate(((0.9, -0.4, 0.4), (-0.7, 0.6, 2.1), (0.3, 1.0, 1.2))):
    piece = ribbed(limb((0, 0, 0), (0, 0, 0.9 - i * 0.2), 0.27, 0.25, M["cactus"], verts=24), 24)
    piece.location = (SX + dx, SY + dy, ground_z(SX + dx, SY + dy) + 0.22)
    piece.rotation_mode = "XYZ"
    piece.rotation_euler = (math.radians(88), 0, ang)
blockers.append((SX, SY, 1.2))

# rocks, pebbles, dry bushes
for _ in range(30):
    x, y = rng.uniform(-12, 12), rng.uniform(-7, 11)
    if free(x, y, 0.8) and (Vector((x, y, 0)) - MARINE_POS).length > 2.5:
        r = rng.uniform(0.2, 0.6)
        rk = make_rock(r)
        rk.location = (x, y, ground_z(x, y) - r * 0.15)
        rk.rotation_euler = (0, 0, rng.uniform(0, 6.28))
pebble = template(make_rock(0.08))
for _ in range(260):
    x, y = rng.uniform(-12, 12), rng.uniform(-7, 11)
    if free(x, y, 0.1):
        place(pebble, (x, y, ground_z(x, y) - 0.02), rng.uniform(0, 6.28), rng.uniform(0.5, 1.6))
for _ in range(40):
    x, y = rng.uniform(-12, 12), rng.uniform(-7, 11)
    if free(x, y, 0.4):
        place(bush, (x, y, ground_z(x, y)), rng.uniform(0, 6.28), rng.uniform(0.7, 1.3))

# sandbags + jersey barriers by the house door
for i in range(5):
    a = math.radians(-60 + i * 30)
    x, y = -3.2 + 1.3 * math.cos(a), 5.1 + 1.3 * math.sin(a)
    for z in (0.15, 0.42):
        box((x, y, ground_z(x, y) + z), (0.25, 0.16, 0.12), M["sandbag"], 0.08, rot=(0, 0, a + math.pi / 2))
for x, y in ((-5.6, 9.0), (-3.9, 8.6)):
    box((x, y, ground_z(x, y) + 0.4), (0.85, 0.25, 0.4), M["concrete"], 0.08, rot=(0, 0, 0.15))

# DNA canister by the house
CX, CY = -4.6, 2.2
can = limb((0, 0, 0), (0, 0, 0.7), 0.25, 0.25, M["canister"], verts=24)
can.location = (CX, CY, ground_z(CX, CY))
for z in (0.2, 0.5):
    ring = limb((0, 0, 0), (0, 0, 0.06), 0.265, 0.265, M["dna_glow"], verts=24)
    ring.location = (CX, CY, can.location.z + z)
blockers.append((CX, CY, 0.5))

# ---------- the swarm ----------
def facing(x, y):
    return math.atan2(-(MARINE_POS.x - x), MARINE_POS.y - y)


live = []


def try_bug(x, y, gap=0.6):
    if not (-12 < x < 12 and -7 < y < 11) or not free(x, y, 0.4):
        return False
    if any((x - a) ** 2 + (y - b) ** 2 < gap ** 2 for a, b in live):
        return False
    live.append((x, y))
    place(worker, (x, y, ground_z(x, y)), facing(x, y) + rng.uniform(-0.4, 0.4), rng.uniform(0.85, 1.15))
    return True


for _ in range(1500):  # main body from north-east, thicker further out
    if len(live) >= 105:
        break
    ang = rng.uniform(math.radians(-15), math.radians(105))
    dist = 4.2 + rng.random() ** 0.6 * 10
    try_bug(dist * math.sin(ang), dist * math.cos(ang))
for _ in range(600):  # flank from the west, past the house
    if len(live) >= 122:
        break
    try_bug(rng.uniform(-12, -5.5), rng.uniform(-0.8, 1.4), 0.65)

for x, y in ((4.6, 3.5), (2.2, 5.4), (-1.4, 5.2), (7.4, 5.4), (1.6, 8.0)):
    place(warrior, (x, y, ground_z(x, y)), facing(x, y))
place(tanker, (8.4, 2.4, ground_z(8.4, 2.4)), facing(8.4, 2.4) + 0.1)
for x, y in ((10.4, 0.2), (5.6, 9.2), (10.8, 4.8)):
    place(spitter, (x, y, ground_z(x, y)), facing(x, y))

# acid spit in flight + puddle near the marine
ellipsoid((4.6, 0.6, ground_z(4.6, 0.6) + 1.0), (0.14, 0.22, 0.12), M["sac"])
pass  # acid puddle removed: read as a flat disc

# dead bugs, splats and experience crystals around the marine
for _ in range(11):
    ang, dist = rng.uniform(-0.7, 2.3), rng.uniform(1.6, 4.0)
    x, y = dist * math.sin(ang), dist * math.cos(ang)
    if not free(x, y, 0.3):
        continue
    z = ground_z(x, y)
    s = make_splat(rng.uniform(0.32, 0.5))
    s.location = (x, y, z)
    s.rotation_euler = (0, 0, rng.uniform(0, 6.28))
    place(worker, (x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2), z + 0.45), rng.uniform(0, 6.28), 0.95, roll=math.pi)
    for _k in range(rng.randint(1, 2)):
        cx, cy = x + rng.uniform(-0.7, 0.7), y + rng.uniform(-0.7, 0.7)
        cr = place(crystal, (cx, cy, ground_z(cx, cy) + 0.22), rng.uniform(0, 6.28))
        cr.rotation_euler = (rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), 0)

for _ in range(7):  # fresh hits on live bugs in front
    x, y = rng.choice([p for p in live if (Vector((*p, 0)) - MARINE_POS).length < 7])
    s = make_splat(rng.uniform(0.25, 0.4), pieces=6)
    s.location = (x, y - 0.3, ground_z(x, y))

# tracers from the rifle towards the warrior ahead
muzzle = Vector((1.47, 0.98, 1.84 + ground_z(0, 0)))
for tx, ty, dash in ((4.6, 3.5, 0.1), (4.6, 3.5, 0.5), (4.3, 3.8, 0.8)):
    d = Vector((tx, ty, 0.6)) - muzzle
    a = muzzle + d * dash
    limb(a, a + d.normalized() * 1.1, 0.05, 0.03, M["tracer"], verts=6)
ellipsoid(muzzle + Vector((0.06, 0.14, 0)), (0.12, 0.24, 0.08), M["tracer"])

# orbital strike telegraph inside the thickest part of the swarm
OX, OY = 6.0, 7.0
oz = ground_z(OX, OY) + 0.35
bpy.ops.mesh.primitive_torus_add(major_radius=2.4, minor_radius=0.045, location=(OX, OY, oz))
bpy.context.object.data.materials.append(M["marker"])
bpy.ops.mesh.primitive_torus_add(major_radius=0.3, minor_radius=0.035, location=(OX, OY, oz))
bpy.context.object.data.materials.append(M["marker"])
for k in range(4):
    a = k * math.pi / 2
    p1 = Vector((OX + math.cos(a) * 0.6, OY + math.sin(a) * 0.6, oz))
    p2 = Vector((OX + math.cos(a) * 1.4, OY + math.sin(a) * 1.4, oz))
    limb(p1, p2, 0.035, 0.035, M["marker"], verts=6)

# ---------- light, camera, render ----------
sun_data = bpy.data.lights.new("sun", "SUN")
sun_data.energy = 5.0
sun_data.color = (1.0, 0.88, 0.72)
sun_data.angle = math.radians(2)
sun = bpy.data.objects.new("sun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_euler = (math.radians(48), math.radians(-22), math.radians(-35))

world = bpy.data.worlds.new("w")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.52, 0.72, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.55
scene.world = world

TARGET = Vector((0.4, 1.8, 0))
TILT = math.radians(35)
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 22
cam_data.clip_end = 200
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
direction = Vector((0, math.sin(TILT), -math.cos(TILT)))
cam.location = TARGET - direction * 60
cam.rotation_euler = (TILT, 0, 0)
scene.camera = cam

r = scene.render
try:
    r.engine = "BLENDER_EEVEE_NEXT"
except TypeError:
    r.engine = "BLENDER_EEVEE"
r.resolution_x, r.resolution_y = RES
r.resolution_percentage = 100
r.filepath = OUT
r.image_settings.file_format = "PNG"
ee = scene.eevee
ee.taa_render_samples = 64
for attr, val in (("use_raytracing", True), ("use_shadows", True)):
    if hasattr(ee, attr):
        setattr(ee, attr, val)
if hasattr(ee, "fast_gi_distance"):
    ee.fast_gi_distance = 2.5
scene.view_settings.view_transform = "AgX"
for look in ("AgX - Punchy", "AgX - Medium High Contrast"):
    try:
        scene.view_settings.look = look
        break
    except TypeError:
        continue

scene.use_nodes = True
nt = scene.node_tree
rl = nt.nodes.new("CompositorNodeRLayers")
glare = nt.nodes.new("CompositorNodeGlare")
for gt in ("BLOOM", "FOG_GLOW"):
    try:
        glare.glare_type = gt
        break
    except TypeError:
        continue
if hasattr(glare, "threshold"):
    glare.threshold = 1.2
if hasattr(glare, "mix"):
    glare.mix = -0.4
comp = nt.nodes.get("Composite") or nt.nodes.new("CompositorNodeComposite")
nt.links.new(rl.outputs["Image"], glare.inputs["Image"])
nt.links.new(glare.outputs["Image"], comp.inputs["Image"])

bpy.ops.render.render(write_still=True)
print("RENDERED", OUT)
