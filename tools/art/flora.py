# Деревья биомов — статичные меши с цветом вершин (без текстур: их сотни на экране, одним вызовом на вид):
# ель ярусами-юбками с зубчатым краем, пальма с кольчатым стволом и перистыми листьями, кактус-сагуаро с рёбрами
# (и куст бочковидных кактусов), болотное дерево на корнях-ходулях с плоскими кронами и свисающим мхом.
# По три варианта на вид. Цвет листвы — из data/biomes.toml (tree_color), затенение (AO) запекается в вершины.
# Высота 1,2–2,2 м: ниже настоящих, чтобы не закрывать бой с наклонной камеры (как и прежние процедурные).
# Формат assets/models/flora_<вид>_<n>.npz: positions (n,3) м, normals (n,3), colors (n,4) — sRGB 0..1,
# indices (m,) — треугольники. Запуск: blender.exe -b --python flora.py [-- pine palm cactus swamp] [--fast]
import sys, os, math, random, tomllib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Vector, noise
import style, preview
from style import Model, curve, taper, oval, SLAB

FAST = "--fast" in sys.argv
TAU = 2 * math.pi
with open(os.path.join(style.ROOT, "data", "biomes.toml"), "rb") as f:
    BIOMES = tomllib.load(f)
LEAF = {b["tree"]: tuple(v / 255 for v in b["tree_color"]) for b in BIOMES.values() if isinstance(b, dict) and "tree" in b}


def scaled(c, k):
    return tuple(min(1.0, v * k) for v in c)


class Flora(Model):
    """Модель из деталей с ключами цвета; colors — ключ → sRGB; GRADIENT — у листвы верх светлее низа."""

    MAX_TRIS = 1000              # деревьев на экране сотни: больше — упрощение сетки (цвета переносятся)

    def __init__(self, name, colors, gradient=()):
        super().__init__(name)
        self.colors = colors
        self.gradient = set(gradient)

    def blob(self, key, r, loc, squash=0.7, seed=0, subdiv=2, rough=1.0):
        """Неровный гладкий ком: крона, мох, бочка кактуса."""
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=1.0, location=(0, 0, 0))
        o = bpy.context.active_object
        off = Vector((seed * 3.1, seed * 1.7, seed * 2.3))
        for v in o.data.vertices:
            d = (noise.noise(v.co * 1.8 + off) * 0.22 + noise.noise(v.co * 4.0 + off) * 0.08) * rough
            v.co *= r * (1 + d)
            v.co.z *= squash
        o.location = loc
        return self._finish(o, "body", key, 0, 0)

    def bake_colors(self):
        """Цвет детали по ключу (у листвы — градиент по высоте детали), затем всё в один объект и AO в вершины."""
        for o in self.parts:
            me = o.data
            col = np.array(self.colors[o["key"]], np.float32)
            n = len(me.loops)
            loop_v = np.empty(n, np.int64)
            me.loops.foreach_get("vertex_index", loop_v)
            co = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            z = co.reshape(-1, 3)[:, 2] * o.matrix_world[2][2] + o.matrix_world[2][3]
            k = np.ones(n, np.float32)
            if o["key"] in self.gradient and z.max() - z.min() > 1e-4:
                t = (z[loop_v] - z.min()) / (z.max() - z.min())
                k = 0.72 + 0.5 * t
            rgba = np.ones((n, 4), np.float32)
            rgba[:, :3] = np.clip(col[None] * k[:, None], 0, 1)
            a = me.color_attributes.new("base", 'FLOAT_COLOR', 'CORNER')
            a.data.foreach_set("color", rgba.ravel())
        style._select(self.parts)
        bpy.ops.object.join()
        o = bpy.context.active_object
        o.name = self.name
        self.obj = o
        o.data.calc_loop_triangles()
        tris = len(o.data.loop_triangles)
        if tris > self.MAX_TRIS:
            mod = o.modifiers.new("dec", "DECIMATE")
            mod.ratio = self.MAX_TRIS / tris
            bpy.ops.object.modifier_apply(modifier="dec")
        # AO: земля под деревом затеняет низ ствола и нижние ярусы
        scn = bpy.context.scene
        scn.render.engine = 'CYCLES'
        scn.cycles.samples = 16 if FAST else 128
        scn.cycles.device = 'CPU'
        world = bpy.data.worlds.get("flora") or bpy.data.worlds.new("flora")
        scn.world = world
        world.light_settings.distance = 0.6
        bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
        ground = bpy.context.active_object
        ao = o.data.color_attributes.new("ao", 'FLOAT_COLOR', 'POINT')
        o.data.color_attributes.active_color = ao
        scn.render.bake.target = 'VERTEX_COLORS'
        style._select([o])
        bpy.ops.object.bake(type='AO')
        bpy.data.objects.remove(ground)
        # итог — в угловой атрибут col (им пользуется и превью)
        me = o.data
        n = len(me.loops)
        loop_v = np.empty(n, np.int64)
        me.loops.foreach_get("vertex_index", loop_v)
        base = np.empty(n * 4, np.float32)
        me.color_attributes["base"].data.foreach_get("color", base)
        occ = np.empty(len(me.vertices) * 4, np.float32)
        me.color_attributes["ao"].data.foreach_get("color", occ)
        occ = occ.reshape(-1, 4)[loop_v, 0]
        col = base.reshape(-1, 4)
        col[:, :3] *= (0.42 + 0.58 * occ)[:, None]
        c = me.color_attributes.new("col", 'FLOAT_COLOR', 'CORNER')
        c.data.foreach_set("color", col.ravel())

    def export(self, path):
        o = self.obj
        me = o.data
        me.calc_loop_triangles()
        tris = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("loops", tris)
        loop_v = np.empty(len(me.loops), np.int64)
        me.loops.foreach_get("vertex_index", loop_v)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.array(o.matrix_world, np.float32)
        co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        nrm = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get("vector", nrm)
        nrm = nrm.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        col = np.empty(len(me.loops) * 4, np.float32)
        me.color_attributes["col"].data.foreach_get("color", col)
        col = col.reshape(-1, 4)
        rows = np.concatenate([co[loop_v[tris]], nrm[tris], col[tris]], 1)
        uniq, inverse = np.unique(np.round(rows, 4), axis=0, return_inverse=True)
        np.savez_compressed(path, positions=uniq[:, :3].astype(np.float32), normals=uniq[:, 3:6].astype(np.float32),
                            colors=uniq[:, 6:10].astype(np.float32), indices=inverse.ravel().astype(np.uint32))
        print(f"[flora] {self.name}: {len(uniq)} вершин, {len(tris) // 3} треугольников → {path}", flush=True)


# ---------------------------------------------------------------------------------------------- ель

def skirt(m, top, under, z, R, h, rnd, n=14):
    """Ярус ели: выпуклый конус сверху с зубчатым свисающим краем (концы лап) и вогнутым низом."""
    rim = []
    for k in range(n):
        a = TAU * k / n + rnd.uniform(-0.1, 0.1)
        tip = k % 2 == 0
        r = R * (1.0 if tip else 0.74) * rnd.uniform(0.88, 1.1)
        rim.append((r * math.cos(a), r * math.sin(a), z - (0.08 if tip else 0.01) * rnd.uniform(0.7, 1.3)))
    mid = [(x * 0.55, y * 0.55, z + h * 0.5) for x, y, _ in rim]
    apex = (rnd.uniform(-0.02, 0.02), rnd.uniform(-0.02, 0.02), z + h)
    verts = [apex] + mid + rim
    faces = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    faces += [(1 + k, 1 + n + k, 1 + n + (k + 1) % n, 1 + (k + 1) % n) for k in range(n)]
    m.mesh(top, verts, faces, sharp=70)
    inner = [(x * 0.3, y * 0.3, z + 0.1) for x, y, _ in rim]
    verts = rim + inner + [(0, 0, z + 0.14)]
    faces = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    faces += [(n + k, n + (k + 1) % n, 2 * n) for k in range(n)]
    m.mesh(under, verts, faces, sharp=70)


def pine(v):
    rnd = random.Random(100 + v)
    leaf = LEAF["pine"]
    m = Flora(f"pine_{v}", {"leaf": scaled(leaf, 1.15), "leaf_lt": scaled(leaf, 1.45), "under": scaled(leaf, 0.55),
                            "bark": (0.36, 0.26, 0.17), "bark_dk": (0.24, 0.17, 0.11)}, gradient=("leaf", "leaf_lt"))
    tiers = (4, 5, 4)[v]
    height = (1.95, 2.25, 1.75)[v]
    m.sweep("bark", curve([(0, 0, 0), (0.01, 0, 0.5), (0, 0.01, 1.2)], 4), oval(7), taper(0.085, 0.045))
    z0 = 0.32
    step = (height - z0 - 0.45) / (tiers - 1)
    for i in range(tiers):
        f = i / (tiers - 1)
        R = (0.66 - 0.42 * f) * rnd.uniform(0.92, 1.08)
        skirt(m, "leaf" if i % 2 == 0 else "leaf_lt", "under", z0 + i * step, R, 0.5 - 0.12 * f, rnd)
    return m


# ---------------------------------------------------------------------------------------------- пальма

def frond(m, key, base, yaw, length, droop, rnd):
    """Перистый лист: средняя жилка дугой вниз, по бокам — пилообразные края, сложенные галочкой."""
    d = Vector((math.cos(yaw), math.sin(yaw), 0))
    side = Vector((-d.y, d.x, 0))
    spine = curve([base, base + d * length * 0.3 + Vector((0, 0, 0.14)), base + d * length * 0.65 + Vector((0, 0, 0.06)),
                   base + d * length - Vector((0, 0, droop))], 13)
    verts, faces = [], []
    for i, p in enumerate(spine):
        t = i / (len(spine) - 1)
        w = (0.18 * math.sin(math.pi * min(1.0, t * 1.1)) ** 0.7 + 0.015) * (1.0 if i % 2 else 0.62)
        fold = Vector((0, 0, -w * 0.35))
        verts += [tuple(p), tuple(p + side * w + fold), tuple(p - side * w + fold)]
    for i in range(len(spine) - 1):
        a, b = 3 * i, 3 * (i + 1)
        faces += [(a, a + 1, b + 1, b), (a, b, b + 2, a + 2)]
    m.mesh(key, verts, faces, sharp=80)


def palm(v):
    rnd = random.Random(200 + v)
    leaf = LEAF["palm"]
    m = Flora(f"palm_{v}", {"leaf": scaled(leaf, 0.9), "leaf_lt": scaled(leaf, 1.12), "bark": (0.47, 0.38, 0.26),
                            "bark_dk": (0.3, 0.23, 0.15), "nut": (0.38, 0.26, 0.12)}, gradient=("leaf", "leaf_lt"))
    lean = rnd.uniform(0, TAU)
    L = Vector((math.cos(lean), math.sin(lean), 0)) * (0.12, 0.22, 0.06)[v]
    top_z = (1.7, 1.9, 1.45)[v]
    path = curve([(0, 0, 0), L * 0.2 + Vector((0, 0, top_z * 0.35)), L * 0.65 + Vector((0, 0, top_z * 0.7)),
                  L + Vector((0, 0, top_z))], 14)
    m.sweep("bark", path, oval(10), taper(0.1, 0.065))
    for i in range(1, len(path) - 1, 2):                    # кольца ствола
        m.sweep("bark_dk", [path[i], path[i] + (path[i + 1] - path[i]) * 0.35], oval(10), (0.1 - 0.035 * i / len(path)) * 1.12)
    m.blob("bark", 0.15, (0, 0, 0.02), squash=0.45, seed=v)
    top = path[-1]
    n = (8, 9, 7)[v]
    for k in range(n):
        yaw = TAU * k / n + rnd.uniform(-0.2, 0.2)
        frond(m, "leaf" if k % 2 else "leaf_lt", top + Vector((0, 0, 0.02)), yaw, rnd.uniform(0.8, 1.0),
              rnd.uniform(0.25, 0.45), rnd)
    for k in range(3):
        a = TAU * k / 3 + 0.5
        m.sphere("nut", 0.055, tuple(top + Vector((math.cos(a) * 0.07, math.sin(a) * 0.07, -0.07))), seg=8, rings=5)
    return m


# ---------------------------------------------------------------------------------------------- кактус

RIBS = [(math.cos(a) * (1 + 0.13 * math.cos(10 * a)), math.sin(a) * (1 + 0.13 * math.cos(10 * a)))  # 10 рёбер
        for a in (TAU * k / 30 for k in range(30))]


def column(m, key, base, r, h, dome=0.9):
    """Ребристая колонна с куполом: ствол сагуаро, бочка."""
    rings = [[(base[0] + x * r * 1.06, base[1] + y * r * 1.06, base[2]) for x, y in RIBS]]
    rings += [[(base[0] + x * r, base[1] + y * r, base[2] + z) for x, y in RIBS] for z in (h * 0.3, h * 0.7)]
    for phi in (0.0, 0.45, 0.8, 1.15):
        k = math.cos(phi)
        rings.append([(base[0] + x * r * k, base[1] + y * r * k, base[2] + h + r * dome * math.sin(phi)) for x, y in RIBS])
    rings.append([(base[0], base[1], base[2] + h + r * dome)] * len(RIBS))
    m.loft(key, rings, sharp=80)


def cactus(v):
    rnd = random.Random(300 + v)
    body = LEAF["cactus"]
    flower = ((0.95, 0.55, 0.65), (0.98, 0.85, 0.3), (0.95, 0.45, 0.35))[v]
    m = Flora(f"cactus_{v}", {"body": scaled(body, 1.15), "flower": flower, "core": (0.98, 0.9, 0.5)}, gradient=("body",))
    if v < 2:                                    # сагуаро: колонна почти в клетку шириной и одна-две руки
        column(m, "body", (0, 0, 0), 0.25, 0.95)
        arms = ((1, 0.42, 0.42), (-1, 0.6, 0.3)) if v == 0 else ((1, 0.55, 0.36),)
        yaw = rnd.uniform(0, TAU)
        for s, za, rise in arms:
            d = Vector((math.cos(yaw) * s, math.sin(yaw) * s, 0))
            path = curve([d * 0.12 + Vector((0, 0, za)), d * 0.36 + Vector((0, 0, za + 0.02)), d * 0.42 + Vector((0, 0, za + 0.14)),
                          d * 0.42 + Vector((0, 0, za + rise + 0.1))], 16)
            # конец руки — купол: последние 15 % пути сечение сходится по окружности
            dome = lambda t: 0.105 * (1.0 if t < 0.85 else math.sqrt(max(0.0, 1 - ((t - 0.85) / 0.15) ** 2)))
            m.sweep("body", path, RIBS, dome)
        tops = [(0, 0, 0.95 + 0.25 * 0.9)]
    else:                                        # куст бочковидных: три ребристых бочки
        tops = []
        for (x, y, r) in ((0.0, 0.05, 0.22), (0.2, -0.14, 0.16), (-0.19, -0.13, 0.14)):
            column(m, "body", (x, y, 0), r, r * 0.9, dome=0.8)
            tops.append((x, y, r * 0.9 + r * 0.8))
    for x, y, z in tops:                          # цветы на макушках
        for k in range(rnd.randint(1, 3)):
            a = rnd.uniform(0, TAU)
            p = (x + math.cos(a) * 0.05, y + math.sin(a) * 0.05, z - 0.01)
            m.blob("flower", 0.045, p, squash=0.55, seed=k + v * 5, subdiv=1, rough=0.4)
            m.sphere("core", 0.015, (p[0], p[1], p[2] + 0.025), seg=6, rings=4)
    return m


# ---------------------------------------------------------------------------------------------- болотное дерево

def swamp(v):
    rnd = random.Random(400 + v)
    leaf = LEAF["swamp"]
    m = Flora(f"swamp_{v}", {"leaf": scaled(leaf, 1.1), "leaf_lt": scaled(leaf, 1.35), "bark": (0.31, 0.27, 0.2),
                             "bark_dk": (0.2, 0.17, 0.13), "moss": (0.45, 0.5, 0.36)}, gradient=("leaf", "leaf_lt", "moss"))
    t = [Vector((rnd.uniform(-0.08, 0.08), rnd.uniform(-0.08, 0.08), z)) for z in (0.3, 0.75, 1.1, 1.35)]
    t[0].x = t[0].y = 0.0
    trunk = curve(t, 12)
    m.sweep("bark", trunk, oval(10), taper(0.12, 0.06))
    for k in range(6):                                       # корни-ходули низкой аркой в воду
        a = TAU * k / 6 + rnd.uniform(-0.3, 0.3)
        d = Vector((math.cos(a), math.sin(a), 0))
        r = rnd.uniform(0.3, 0.42)
        m.sweep("bark_dk", curve([d * 0.05 + Vector((0, 0, 0.4)), d * r * 0.55 + Vector((0, 0, 0.24)), d * r + Vector((0, 0, -0.03))], 8),
                oval(7), taper(0.05, 0.03))
    crowns = [(trunk[-1] + Vector((0, 0, 0.1)), 0.42)]
    for k in range((2, 3, 2)[v]):
        a = TAU * k / 3 + rnd.uniform(0, 1)
        d = Vector((math.cos(a), math.sin(a), 0))
        src = trunk[8 + k]
        tip = src + d * rnd.uniform(0.3, 0.4) + Vector((0, 0, 0.1))
        m.sweep("bark", curve([src, (src + tip) / 2 + Vector((0, 0, 0.08)), tip], 6), oval(6), taper(0.045, 0.025))
        crowns.append((tip + Vector((0, 0, 0.04)), rnd.uniform(0.28, 0.32)))
    seed = v * 31
    for c, r in crowns:
        # крона — гроздь мелких комков, а не один: так читается листва, а не камень
        for j in range(6):
            seed += 1
            a = TAU * j / 6 + rnd.uniform(-0.3, 0.3)
            off = Vector((math.cos(a), math.sin(a), 0)) * r * rnd.uniform(0.35, 0.6) + Vector((0, 0, rnd.uniform(-0.05, 0.08)))
            m.blob("leaf" if j % 2 else "leaf_lt", r * rnd.uniform(0.45, 0.6), tuple(c + off), squash=0.7, seed=seed,
                   subdiv=3, rough=0.6)
        m.blob("leaf_lt", r * 0.55, tuple(c + Vector((0, 0, r * 0.25))), squash=0.7, seed=seed + 50, subdiv=3, rough=0.6)
        for k in range(4):                                  # мох свисает с нижней кромки кроны
            a = rnd.uniform(0, TAU)
            p = c + Vector((math.cos(a) * r * 0.75, math.sin(a) * r * 0.75, -r * 0.15))
            L = rnd.uniform(0.3, 0.55)
            m.sweep("moss", curve([p, p + Vector((0.02, 0.01, -L * 0.5)), p + Vector((0.03, -0.01, -L))], 5), SLAB,
                    taper(0.04, 0.004), 0.008, up=(math.cos(a), math.sin(a), 0))
    return m


KINDS = {"pine": (pine, (0, 0, 1.0), 4.2), "palm": (palm, (0, 0, 0.9), 4.2), "cactus": (cactus, (0, 0, 0.55), 3.0),
         "swamp": (swamp, (0, 0, 0.9), 4.0)}


def show_colors(o):
    """Материал превью: цвет — атрибут col (то же, что увидит игра)."""
    mat = bpy.data.materials.new("flora_col")
    mat.use_nodes = True
    nt = mat.node_tree
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "col"
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    nt.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    o.data.materials.clear()
    o.data.materials.append(mat)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    which = [a for a in args if not a.startswith("--")] or list(KINDS)
    for kind in which:
        make, target, dist = KINDS[kind]
        paths = []
        for v in range(3):
            style.reset()
            m = make(v)
            m.bake_colors()
            m.export(os.path.join(style.TEX, f"flora_{kind}_{v}.npz"))
            show_colors(m.obj)
            preview._scene((600, 600))
            for view in ("game", "hero"):
                pitch, yaw, k = preview.VIEWS[view]
                path = os.path.join(preview.OUT, f"_flora_{kind}_{v}_{view}.png")
                preview._shoot(path, target, pitch, yaw, dist * k)
                paths.append(path)
        preview.sheet(paths, os.path.join(preview.OUT, f"flora_{kind}.png"), cols=6)
    print("FLORA DONE", flush=True)
