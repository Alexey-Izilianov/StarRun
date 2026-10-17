# Стилевая библиотека арта StarRun (стилизация в духе StarCraft II, перенесена из RebirthOfThePlanet).
# Модель собирается из деталей-примитивов с фасками и плавных форм (лофт, протяжка, изогнутые пластины); детали
# объединяются в группы (подвижные части) или привязываются к костям (rig.py). Цвет детали — ключ палитры.
# Все детали получают общую развёртку и запекаются в одну текстуру: цвет палитры × затенение в углах (AO) ×
# градиент снизу вверх × пятнистость + светлые рёбра (рисованный блик) и износ. Отдельно — маска металл/гладкость.
# Перёд модели — +Y Blender, земля — Z=0, единица — метр. Экспорт в формат игры — export.py.
import bpy, bmesh, math, os, json
from mathutils import Vector, Matrix, Euler

ROOT = r"C:\Projects\StarRun"
ART = os.path.join(ROOT, "tools", "art")
TEX = os.path.join(ROOT, "assets", "models")          # текстуры рядом с моделями — их читает игра
for d in (os.path.join(ART, "blend"), os.path.join(ART, "previews"), TEX):
    os.makedirs(d, exist_ok=True)


def srgb_to_lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# Палитра: цвет в sRGB, metal (0/1), smooth (0..1), glow (сила свечения; > 0 — эмиссионный материал без запекания)
PAL = {
    # Мобильная пехота: оливковая броня, песочные вставки, оружейный металл, тёмный подкостюм
    "armor":    ((0.44, 0.50, 0.31), 0, 0.3, 0),
    "armor_dk": ((0.27, 0.31, 0.20), 0, 0.25, 0),
    "armor_lt": ((0.56, 0.61, 0.40), 0, 0.3, 0),
    "sand":     ((0.70, 0.62, 0.45), 0, 0.25, 0),
    "suit":     ((0.13, 0.14, 0.13), 0, 0.15, 0),
    "visor":    ((0.05, 0.10, 0.12), 1, 0.95, 0),     # тёмное стекло щитка
    "gunmetal": ((0.20, 0.21, 0.23), 1, 0.45, 0),
    "metal":    ((0.60, 0.61, 0.64), 1, 0.55, 0),
    "rubber":   ((0.10, 0.10, 0.11), 0, 0.15, 0),
    "strap":    ((0.30, 0.25, 0.18), 0, 0.2, 0),
    "team":     ((0.92, 0.72, 0.20), 0, 0.35, 0),     # жёлтые знаки отделения
    "white":    ((0.88, 0.88, 0.84), 0, 0.3, 0),
    "red":      ((0.72, 0.16, 0.12), 0, 0.3, 0),
    "wood":     ((0.50, 0.35, 0.21), 0, 0.15, 0),
    # Жуки: глянцевый хитин (рабочий — рыже-бурый, воин — красный), тёмные суставы, костяные когти
    "chitin":    ((0.50, 0.30, 0.15), 0, 0.6, 0),
    "chitin_dk": ((0.24, 0.13, 0.07), 0, 0.55, 0),
    "chitin_lt": ((0.70, 0.48, 0.26), 0, 0.65, 0),
    "red_chitin":    ((0.55, 0.15, 0.08), 0, 0.6, 0),
    "red_chitin_dk": ((0.26, 0.07, 0.05), 0, 0.55, 0),
    "red_chitin_lt": ((0.78, 0.36, 0.18), 0, 0.65, 0),
    "membrane":  ((0.36, 0.20, 0.17), 0, 0.35, 0),    # мягкие сочленения
    "belly":     ((0.62, 0.50, 0.36), 0, 0.4, 0),     # светлое брюхо
    "claw":      ((0.86, 0.78, 0.62), 0, 0.5, 0),     # костяные когти и жвалы
    "claw_dk":   ((0.14, 0.10, 0.08), 0, 0.6, 0),
    "eye":       ((0.04, 0.04, 0.05), 0, 0.95, 0),
    # стрелок — болотно-зелёный, мешок кислоты; летающий — песочный, крылья; танкер — серо-бурый камень;
    # мозговой — бледный хитин и розовый мозг; элитный воин — оранжевый; королева — тёмно-лиловая
    "green_chitin":    ((0.40, 0.43, 0.13), 0, 0.6, 0),
    "green_chitin_dk": ((0.16, 0.18, 0.06), 0, 0.55, 0),
    "green_chitin_lt": ((0.62, 0.63, 0.27), 0, 0.65, 0),
    "sac":       ((0.66, 0.74, 0.26), 0, 0.85, 0),
    "sand_chitin":    ((0.62, 0.46, 0.28), 0, 0.6, 0),
    "sand_chitin_dk": ((0.30, 0.21, 0.12), 0, 0.55, 0),
    "sand_chitin_lt": ((0.80, 0.66, 0.44), 0, 0.65, 0),
    "wing":      ((0.78, 0.70, 0.52), 0, 0.85, 0),
    "stone_chitin":    ((0.43, 0.37, 0.30), 0, 0.5, 0),
    "stone_chitin_dk": ((0.17, 0.14, 0.11), 0, 0.45, 0),
    "stone_chitin_lt": ((0.60, 0.54, 0.44), 0, 0.55, 0),
    "pale_chitin":    ((0.55, 0.42, 0.46), 0, 0.6, 0),
    "pale_chitin_dk": ((0.24, 0.16, 0.20), 0, 0.55, 0),
    "pale_chitin_lt": ((0.74, 0.62, 0.64), 0, 0.65, 0),
    "brain":     ((0.80, 0.52, 0.56), 0, 0.75, 0),
    "brain_dk":  ((0.42, 0.20, 0.26), 0, 0.6, 0),
    "elite_chitin":    ((0.70, 0.30, 0.08), 0, 0.6, 0),
    "elite_chitin_dk": ((0.30, 0.10, 0.04), 0, 0.55, 0),
    "elite_chitin_lt": ((0.92, 0.58, 0.24), 0, 0.65, 0),
    "queen_chitin":    ((0.36, 0.13, 0.30), 0, 0.65, 0),
    "queen_chitin_dk": ((0.14, 0.05, 0.12), 0, 0.6, 0),
    "queen_chitin_lt": ((0.60, 0.30, 0.50), 0, 0.7, 0),
    "egg_sac":   ((0.60, 0.40, 0.42), 0, 0.8, 0),
    # Светящиеся
    "optic":    ((0.25, 0.85, 1.00), 0, 0, 1.6),        # подсветка визора и прицела
    "light":    ((1.00, 0.84, 0.55), 0, 0, 4.0),
    "acid":     ((0.55, 1.00, 0.25), 0, 0, 3.0),        # кислота жуков
    "psi":      ((0.95, 0.45, 1.00), 0, 0, 2.5),        # узлы мозгового жука
}

# Детали без износа (сколов краски, потёков): у жуков живой хитин, а не крашеный металл
CHITINS = ("chitin", "red_chitin", "green_chitin", "sand_chitin", "stone_chitin", "pale_chitin", "elite_chitin",
           "queen_chitin")
CLEAN = {k + s for k in CHITINS for s in ("", "_dk", "_lt")} | {"membrane", "belly", "claw", "claw_dk", "eye", "sac",
                                                                "wing", "brain", "brain_dk", "egg_sac"}
# Угловатые пятна (канал A атрибута mat): у жуков — тёмный крап по хитину
CAMO = set(CHITINS)
CAMO_COLOR = {k: k + "_dk" for k in CHITINS}


def curve(points, n):
    """n точек на сплайне Катмулла — Рома через опорные точки (концы — первая и последняя опорные)."""
    P = [Vector(p) for p in points]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    segs = len(P) - 3
    for k in range(n):
        x = k / (n - 1) * segs
        i = min(int(x), segs - 1)
        t = x - i
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return out


def taper(a, b, power=1.0):
    """Размер сечения от a в начале пути до b в конце (для sweep)."""
    return lambda t: a + (b - a) * t ** power


def ramp(keys):
    """Плавная функция доли пути по опорам [(t, значение)]."""
    def f(t):
        for (t0, a), (t1, b) in zip(keys, keys[1:]):
            if t <= t1:
                k = (t - t0) / max(1e-6, t1 - t0)
                return a + (b - a) * k * k * (3 - 2 * k)
        return keys[-1][1]
    return f


def around(center, half, shrink=0.0):
    """Диапазон долготы center ± half (градусы) для plate; к верхней кромке сужается на долю shrink."""
    rad = math.radians
    return lambda t: (rad(center - half * (1 - shrink * t)), rad(center + half * (1 - shrink * t)))


# Профили сечений для sweep: x — вбок, y — вверх, полуразмер 1
def oval(n=12):
    return [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n)) for k in range(n)]


LENS = [(1, 0), (0.55, 0.24), (0, 0.32), (-0.55, 0.24), (-1, 0), (-0.55, -0.24), (0, -0.32), (0.55, -0.24)]   # лезвие
HEX = [(1, 0), (0.6, 0.8), (-0.6, 0.8), (-1, 0), (-0.6, -0.8), (0.6, -0.8)]                                     # гранёная трубка
SLAB = [(1, 0.55), (-1, 0.55), (-1, -0.55), (1, -0.55)]                                                          # полоса, полотнище


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.unit_settings.system = 'METRIC'


def _select(objs, active=None):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


class Model:
    """Модель: детали по группам. group — имя подвижной части; parent групп задаётся в group()."""

    def __init__(self, name, tex_size=1024):
        self.name = name
        self.tex_size = tex_size
        self.parts = []          # (obj, group, key)
        self.groups = {}         # name -> dict(parent, pivot)
        self.objects = {}        # name -> итоговый объект группы
        self.extra = []          # объекты, которые экспортируются как есть (гусеницы со своим материалом)
        self.camo_scale = 6.0    # частота крапа на хитине (на метр): у мелких моделей крап мельче
        self.group("body", None, (0, 0, 0))

    def group(self, name, parent="body", pivot=(0, 0, 0)):
        self.groups[name] = {"parent": parent, "pivot": Vector(pivot)}

    # ---------- примитивы ----------

    def _finish(self, o, group, key, bevel, seg, smooth_all=True):
        o.data.materials.clear()
        o["group"] = group
        o["key"] = key
        if bevel > 0:
            m = o.modifiers.new("bev", "BEVEL")
            m.width = bevel
            m.segments = seg
            m.limit_method = 'ANGLE'
            m.angle_limit = math.radians(35)
            m.harden_normals = True
            m.miter_outer = 'MITER_ARC'
        for poly in o.data.polygons:
            poly.use_smooth = True
        _select([o])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        if bevel > 0:
            bpy.ops.object.modifier_apply(modifier="bev")
        self.parts.append(o)
        return o

    def hexa(self, key, verts, group="body", bevel=0.05, seg=2, name="part"):
        """Произвольный шестигранник: 8 вершин — низ (-x-y, +x-y, +x+y, -x+y), затем верх в том же порядке."""
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        vs = [bm.verts.new(v) for v in verts]
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            bm.faces.new([vs[i] for i in f])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        return self._finish(o, group, key, bevel, seg)

    def box(self, key, size, loc, rot=(0, 0, 0), group="body", bevel=0.05, seg=2, top=(1, 1), shift=(0, 0), name="box"):
        """Коробка size (полные размеры) с центром loc; top — масштаб верхней грани (сужение), shift — её сдвиг."""
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        tx, ty = sx * top[0], sy * top[1]
        v = [(-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
             (-tx + shift[0], -ty + shift[1], sz), (tx + shift[0], -ty + shift[1], sz),
             (tx + shift[0], ty + shift[1], sz), (-tx + shift[0], ty + shift[1], sz)]
        R = Matrix.Rotation(rot[0], 4, 'X') if rot[0] else Matrix()
        if rot[1]: R = Matrix.Rotation(rot[1], 4, 'Y') @ R
        if rot[2]: R = Matrix.Rotation(rot[2], 4, 'Z') @ R
        L = Vector(loc)
        return self.hexa(key, [L + (R @ Vector(p)) for p in v], group, bevel, seg, name)

    def cyl(self, key, r, depth, loc, rot=(0, 0, 0), group="body", seg=16, bevel=0.03, r2=None, name="cyl", scale=(1, 1, 1)):
        """Цилиндр (r2 — радиус верхнего торца: конус/усечённый конус) вдоль локальной Z."""
        bpy.ops.mesh.primitive_cone_add(vertices=seg, radius1=r, radius2=r if r2 is None else r2, depth=depth,
                                        location=loc, rotation=rot)
        o = bpy.context.active_object
        o.name = name
        o.scale = scale
        return self._finish(o, group, key, bevel, 2)

    def sphere(self, key, r, loc, scale=(1, 1, 1), group="body", seg=16, rings=8, rot=(0, 0, 0), name="sph"):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings, radius=r, location=loc, rotation=rot)
        o = bpy.context.active_object
        o.name = name
        o.scale = scale
        return self._finish(o, group, key, 0, 0)

    def torus(self, key, r, thick, loc, rot=(0, 0, 0), group="body", seg=24, name="tor"):
        bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=thick, major_segments=seg, minor_segments=8,
                                         location=loc, rotation=rot)
        o = bpy.context.active_object
        o.name = name
        return self._finish(o, group, key, 0, 0)

    def prism(self, key, pts, z0, z1, group="body", center=(0, 0), rot_z=0.0, bevel=0.0, name="prism", matrix=None):
        """Призма из многоугольника pts (x, y) от z0 до z1: знаки, шевроны. Торцы триангулируются
        (невыпуклые многоугольники иначе в Unity бьются неверно)."""
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        c, s = math.cos(rot_z), math.sin(rot_z)
        P = [(x * c - y * s + center[0], x * s + y * c + center[1]) for x, y in pts]
        bottom = [bm.verts.new((x, y, z0)) for x, y in P]
        top = [bm.verts.new((x, y, z1)) for x, y in P]
        caps = [bm.faces.new(list(reversed(bottom))), bm.faces.new(top)]
        n = len(P)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((bottom[i], bottom[j], top[j], top[i]))
        bmesh.ops.triangulate(bm, faces=caps)
        if matrix is not None:
            bmesh.ops.transform(bm, matrix=matrix, verts=bm.verts)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        return self._finish(o, group, key, bevel, 1)

    def emblem(self, key, pts, depth, loc, rot=(0, 0, 0), group="body"):
        """Плоский знак (многоугольник pts в плоскости XY) толщиной depth, повёрнутый rot и поставленный в loc.
        Для фасада, смотрящего в +Y: rot=(-90°, 0, 0)."""
        M = Matrix.Translation(loc) @ Euler(rot).to_matrix().to_4x4()
        return self.prism(key, pts, -depth / 2, depth / 2, group=group, matrix=M)

    def rock(self, key, size, loc, seed=0, group="body", rot_z=0.0, name="rock", rough=1.0, subdiv=2):
        """Гранёный камень: икосфера с шумовым смещением вершин, приплюснутая, плоские грани.
        rough — сила смещения (больше — угловатее), subdiv 1 — крупные грани."""
        import random as _r
        from mathutils import noise as _n
        rnd = _r.Random(seed)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdiv, radius=0.5, location=(0, 0, 0))
        o = bpy.context.active_object
        off = Vector((rnd.uniform(0, 50), rnd.uniform(0, 50), rnd.uniform(0, 50)))
        for v in o.data.vertices:
            d = (_n.noise(v.co * 2.2 + off) * 0.22 + _n.noise(v.co * 5.0 + off) * 0.06) * rough
            v.co *= 1.0 + d
            if v.co.z < -0.15:
                v.co.z = -0.15 + (v.co.z + 0.15) * 0.3      # плоское дно — камень лежит, а не стоит на острие
        o.scale = size
        o.rotation_euler = (0, 0, rot_z)
        o.location = loc
        o.name = name
        o = self._finish(o, group, key, 0, 0)
        for p in o.data.polygons:
            p.use_smooth = False
        return o

    def crag(self, key, size, loc, seed=0, group="body", rot_z=0.0, bands=4, jag=1.0, name="crag"):
        """Скальная глыба: подразделённый куб, бока смещены шумом и сдвинуты по горизонтальным пластам (слоистая
        порода), верх почти ровный — глыба вкапывается в склон, а сверху продолжается плато."""
        import random as _r
        from mathutils import noise as _n
        rnd = _r.Random(seed)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4, use_grid_fill=True)
        off = Vector((rnd.uniform(0, 50), rnd.uniform(0, 50), rnd.uniform(0, 50)))
        shift = [(rnd.uniform(-0.1, 0.1), rnd.uniform(-0.14, 0.14)) for _ in range(bands)]
        for v in bm.verts:
            c = v.co.copy()
            band = shift[min(bands - 1, int((c.z + 1) * 0.5 * bands))]
            side = 1.0 - max(0.0, c.z - 0.6) * 1.5                  # у верхней кромки меньше рельефа
            out = Vector((c.x, c.y, 0))
            if out.length > 1e-6:
                out.normalize()
            d = (_n.noise(c * 1.8 + off) * 0.3 + _n.noise(c * 4.5 + off) * 0.11) * jag
            v.co = c + out * d * side + Vector((band[0], band[1], 0)) * side
            if c.z > 0.99:
                v.co.z = 1.0 + _n.noise(c * 2.0 + off) * 0.05
        # гладко внутри грани, жёстко на крупных изломах: без «шахматки» от плоского затенения мелких граней
        bmesh.ops.scale(bm, vec=(size[0] / 2, size[1] / 2, size[2] / 2), verts=bm.verts)
        bm.normal_update()
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > math.radians(48):
                e.smooth = False
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        o.rotation_euler = (0, 0, rot_z)
        o.location = loc
        return self._finish(o, group, key, 0, 0)

    def seam(self, p0, p1, normal, group="body", key="gunmetal", width=0.022, depth=0.012):
        """Шов панели: тонкая тёмная полоса от p0 до p1 поверх поверхности с нормалью normal."""
        a, b, n = Vector(p0), Vector(p1), Vector(normal).normalized()
        d = b - a
        side = n.cross(d).normalized() * width / 2
        lift = n * depth
        v = [a - side, b - side, b + side, a + side, a - side + lift, b - side + lift, b + side + lift, a + side + lift]
        return self.hexa(key, [tuple(x) for x in v], group, bevel=0.0, name="seam")

    def rivets(self, p0, p1, count, normal, group="body", key="metal", r=0.028):
        """Ряд заклёпок между p0 и p1; normal — ось наружу ('x', 'y', 'z' со знаком, например '-y')."""
        sign = -1 if normal.startswith("-") else 1
        ax = normal[-1]
        rot = {"x": (0, math.pi / 2, 0), "y": (math.pi / 2, 0, 0), "z": (0, 0, 0)}[ax]
        off = Vector({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]) * 0.01 * sign
        a, b = Vector(p0), Vector(p1)
        for i in range(count):
            t = (i + 0.5) / count
            self.cyl(key, r, 0.025, tuple(a + (b - a) * t + off), rot, group=group, seg=6, bevel=0.0, name="rivet")

    def vent(self, center, w, h, normal, slats=5, group="body", frame="gunmetal", slat="rubber"):
        """Решётка: рамка и наклонные ламели на поверхности с осью normal ('x'/'y'/'z' со знаком)."""
        sign = -1 if normal.startswith("-") else 1
        ax = normal[-1]
        c = Vector(center)
        if ax == "z":
            self.box(frame, (w, h, 0.05), tuple(c), group=group, bevel=0.01)
            for i in range(slats):
                self.box(slat, (w * 0.9, h * 0.8 / slats, 0.03), tuple(c + Vector((0, -h * 0.4 + (i + 0.5) * h * 0.8 / slats, 0.03 * sign))),
                         rot=(math.radians(25), 0, 0), group=group, bevel=0.0)
        else:
            rot = (0, 0, math.pi / 2) if ax == "x" else (0, 0, 0)
            self.box(frame, (w, 0.05, h), tuple(c), rot=rot, group=group, bevel=0.01)
            for i in range(slats):
                off = Vector((0, 0, -h * 0.4 + (i + 0.5) * h * 0.8 / slats))
                n = Vector((0.03 * sign, 0, 0)) if ax == "x" else Vector((0, 0.03 * sign, 0))
                self.box(slat, (w * 0.9, 0.03, h * 0.8 / slats), tuple(c + off + n), rot=(math.radians(-25) * sign, 0, rot[2]),
                         group=group, bevel=0.0)

    # ---------- плавные формы: лофт по сечениям, протяжка профиля, изогнутые пластины, кристаллы ----------

    def mesh(self, key, verts, faces, group="body", bevel=0.0, sharp=40.0, name="mesh"):
        """Меш из вершин и граней. Совпавшие вершины (острия) сливаются, многоугольники триангулируются;
        рёбра с углом между гранями больше sharp градусов — жёсткие, остальное сглажено."""
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        vs = [bm.verts.new(v) for v in verts]
        for f in faces:
            try:
                bm.faces.new([vs[i] for i in f])
            except ValueError:
                pass                      # вырожденная или повторная грань
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        lim = math.radians(sharp)
        for e in bm.edges:
            if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > lim:
                e.smooth = False
        bm.to_mesh(me)
        bm.free()
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        return self._finish(o, group, key, bevel, 2)

    def loft(self, key, rings, group="body", cap=(True, True), bevel=0.0, sharp=40.0, name="loft"):
        """Замкнутая поверхность через сечения rings (списки точек одинаковой длины); торцы закрываются,
        сечение, стянутое в точку, даёт острие."""
        n = len(rings[0])
        verts = [tuple(p) for r in rings for p in r]
        faces = []
        for i in range(len(rings) - 1):
            a, b = i * n, (i + 1) * n
            for j in range(n):
                k = (j + 1) % n
                faces.append((a + j, a + k, b + k, b + j))
        point = lambda r: max((Vector(p) - Vector(r[0])).length for p in r) < 1e-5
        if cap[0] and not point(rings[0]):
            faces.append(tuple(reversed(range(n))))
        if cap[1] and not point(rings[-1]):
            last = (len(rings) - 1) * n
            faces.append(tuple(range(last, last + n)))
        return self.mesh(key, verts, faces, group, bevel, sharp, name)

    def sweep(self, key, path, profile, width, height=None, up=(0, 0, 1), group="body", cap=(True, True),
              bevel=0.0, sharp=40.0, name="sweep"):
        """Протяжка профиля (x — вбок, y — вверх) вдоль ломаной path. width/height — полуразмеры сечения:
        число или функция доли пути t ∈ [0, 1]; ноль стягивает сечение в острие. up — куда смотрит «верх» профиля."""
        P = [Vector(p) for p in path]
        m = len(P)
        U = Vector(up)
        rings = []
        for i, p in enumerate(P):
            t = i / (m - 1)
            T = (P[min(i + 1, m - 1)] - P[max(i - 1, 0)]).normalized()
            S = T.cross(U)
            if S.length < 1e-6:
                S = T.orthogonal()
            S.normalize()
            Up = S.cross(T).normalized()
            w = width(t) if callable(width) else width
            h = w if height is None else (height(t) if callable(height) else height)
            rings.append([p + S * (x * w) + Up * (y * h) for x, y in profile])
        return self.loft(key, rings, group, cap, bevel, sharp, name)

    def plate(self, key, center, radii, u, v, thick=0.04, rot=(0, 0, 0), group="body", res=(16, 8),
              bevel=0.0, sharp=40.0, name="plate"):
        """Изогнутая пластина — участок эллипсоида толщиной thick внутрь. u — долгота (от +X к +Y), v — широта,
        в радианах; u может быть функцией доли широты s ∈ [0, 1] → (u0, u1): так края сходятся в острие."""
        cu, cv = res
        M = Euler(rot).to_matrix()
        c = Vector(center)
        rows_out, rows_in = [], []
        for i in range(cv + 1):
            s = i / cv
            lat = v[0] + (v[1] - v[0]) * s
            ua, ub = u(s) if callable(u) else u
            ro, ri = [], []
            for j in range(cu + 1):
                lon = ua + (ub - ua) * j / cu
                d = Vector((math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)))
                ro.append(c + M @ Vector((radii[0] * d.x, radii[1] * d.y, radii[2] * d.z)))
                ri.append(c + M @ Vector(((radii[0] - thick) * d.x, (radii[1] - thick) * d.y, (radii[2] - thick) * d.z)))
            rows_out.append(ro)
            rows_in.append(ri)
        W = cu + 1
        verts = [tuple(p) for r in rows_out for p in r] + [tuple(p) for r in rows_in for p in r]
        off = (cv + 1) * W
        idx = lambda i, j, inner=False: (off if inner else 0) + i * W + j
        faces = []
        for i in range(cv):
            for j in range(cu):
                faces.append((idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)))
                faces.append((idx(i, j, True), idx(i + 1, j, True), idx(i + 1, j + 1, True), idx(i, j + 1, True)))
        for j in range(cu):          # торцы по широте
            faces.append((idx(0, j), idx(0, j, True), idx(0, j + 1, True), idx(0, j + 1)))
            faces.append((idx(cv, j), idx(cv, j + 1), idx(cv, j + 1, True), idx(cv, j, True)))
        ring = not callable(u) and abs(u[1] - u[0] - 2 * math.pi) < 1e-6
        for i in range(cv if not ring else 0):          # торцы по долготе (у замкнутого пояса их нет)
            faces.append((idx(i, 0), idx(i + 1, 0), idx(i + 1, 0, True), idx(i, 0, True)))
            faces.append((idx(i, cu), idx(i, cu, True), idx(i + 1, cu, True), idx(i + 1, cu)))
        return self.mesh(key, verts, faces, group, bevel, sharp, name)

    def gem(self, key, r, h, loc, rot=(0, 0, 0), group="body", sides=6, low=0.6, name="gem"):
        """Гранёный кристалл: двойная пирамида вдоль локальной Z (верх h, низ h·low), плоские грани."""
        M = Matrix.Translation(loc) @ Euler(rot).to_matrix().to_4x4()
        ring = [M @ Vector((r * math.cos(2 * math.pi * k / sides), r * math.sin(2 * math.pi * k / sides), 0)) for k in range(sides)]
        verts = [tuple(p) for p in ring] + [tuple(M @ Vector((0, 0, h))), tuple(M @ Vector((0, 0, -h * low)))]
        faces = [(k, (k + 1) % sides, sides) for k in range(sides)] + [((k + 1) % sides, k, sides + 1) for k in range(sides)]
        return self.mesh(key, verts, faces, group, 0.0, 1.0, name)

    # семисегментные цифры: бортовые номера в военном трафаретном стиле
    SEG = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg"}

    @staticmethod
    def frame(loc, right, up):
        """Матрица плоскости знака: локальный X — right, Y — up, нормаль — их произведение."""
        r, u = Vector(right).normalized(), Vector(up).normalized()
        n = r.cross(u).normalized()
        u = n.cross(r).normalized()
        M = Matrix.Identity(4)
        for i in range(3):
            M[i][0], M[i][1], M[i][2], M[i][3] = r[i], u[i], n[i], loc[i]
        return M

    def number(self, text, loc, height, rot=(0, 0, 0), group="body", key="white", depth=0.02, axes=None):
        """Число из сегментов в плоскости знака: поворот rot или оси axes=(вправо, вверх), центр в loc."""
        w, t = height * 0.5, height * 0.13
        segs = {"a": (0, height / 2, w, t), "g": (0, 0, w, t), "d": (0, -height / 2, w, t),
                "f": (-w / 2, height / 4, t, height / 2), "b": (w / 2, height / 4, t, height / 2),
                "e": (-w / 2, -height / 4, t, height / 2), "c": (w / 2, -height / 4, t, height / 2)}
        M0 = self.frame(loc, *axes) if axes else Matrix.Translation(loc) @ Euler(rot).to_matrix().to_4x4()
        for k, ch in enumerate(text):
            x0 = (k - (len(text) - 1) / 2) * w * 1.6
            for sname in self.SEG.get(ch, ""):
                cx, cy, sw, sh = segs[sname]
                pts = [(-sw / 2, -sh / 2), (sw / 2, -sh / 2), (sw / 2, sh / 2), (-sw / 2, sh / 2)]
                M = M0 @ Matrix.Translation((x0 + cx, cy, 0))
                self.prism(key, pts, -depth / 2, depth / 2, group=group, matrix=M)

    @staticmethod
    def star_pts(r, inner=0.42, n=5):
        return [((r if i % 2 == 0 else r * inner) * math.cos(math.pi / 2 + math.pi * i / n),
                 (r if i % 2 == 0 else r * inner) * math.sin(math.pi / 2 + math.pi * i / n)) for i in range(2 * n)]

    # ---------- сборка, запекание, экспорт ----------

    TINT_FREE = ("team", "white", "hazard", "red")   # знаки и полосы — ровным цветом

    def _write_attrs(self, o, key):
        col, metal, smooth, glow = PAL[key]
        me = o.data
        lin = [srgb_to_lin(c) for c in col]
        if glow == 0 and key not in self.TINT_FREE:
            import random as _r, zlib
            self._tints = getattr(self, "_tints", 0) + 1     # воспроизводимо между запусками (hash() строк — нет)
            r = _r.Random(zlib.crc32(f"{self.name}|{o.name}|{self._tints}".encode()))
            v = 1.0 + r.uniform(-0.05, 0.05)
            warm = r.uniform(-0.015, 0.015)
            lin = [lin[0] * v * (1 + warm), lin[1] * v, lin[2] * v * (1 - warm)]
        a = me.color_attributes.new("pal", 'FLOAT_COLOR', 'CORNER')
        a.data.foreach_set("color", (lin + [1.0]) * len(me.loops))
        m = me.color_attributes.new("mat", 'FLOAT_COLOR', 'CORNER')
        m.data.foreach_set("color", [float(metal), float(smooth), 1.0 if key in CLEAN else 0.0,
                                     1.0 if key in CAMO else 0.0] * len(me.loops))

    def _join(self, objs, name, pivot):
        _select(objs)
        if len(objs) > 1:
            bpy.ops.object.join()
        o = bpy.context.active_object
        o.name = name
        o.data.name = name
        bpy.context.scene.cursor.location = pivot
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        return o

    def build(self):
        """Объединяет детали в объекты групп и светящиеся объекты, строит иерархию."""
        by_group = {}
        for o in self.parts:
            self._write_attrs(o, o["key"])
            by_group.setdefault(o["group"], []).append(o)
        self.glow_objects = []
        for g, info in self.groups.items():
            objs = by_group.get(g, [])
            solid = [o for o in objs if PAL[o["key"]][3] == 0]
            glows = {}
            for o in objs:
                if PAL[o["key"]][3] > 0:
                    glows.setdefault(o["key"], []).append(o)
            if solid:
                self.objects[g] = self._join(solid, g, info["pivot"])
            else:
                e = bpy.data.objects.new(g, None)
                bpy.context.scene.collection.objects.link(e)
                e.location = info["pivot"]
                self.objects[g] = e
            for key, lst in glows.items():
                go = self._join(lst, f"{g}_glow_{key}", info["pivot"])
                go["material"] = f"{self.name}_glow_{key}"
                self.glow_objects.append((go, g))
        # иерархия: сохраняем мировые трансформы
        self.root = bpy.data.objects.new(self.name, None)
        bpy.context.scene.collection.objects.link(self.root)
        for g, info in self.groups.items():
            parent = self.objects[info["parent"]] if info["parent"] else self.root
            self._parent(self.objects[g], parent)
        for go, g in self.glow_objects:
            self._parent(go, self.objects[g])
        for o in self.extra:
            self._parent(o, self.objects["body"])

    @staticmethod
    def _parent(child, parent):
        mw = child.matrix_world.copy()
        child.parent = parent
        child.matrix_world = mw

    def solid_meshes(self):
        return [o for o in self.objects.values() if o.type == 'MESH']

    def bake(self, samples=48):
        """Общая развёртка всех групп и запекание альбедо и маски металл/гладкость."""
        meshes = self.solid_meshes()
        _select(meshes)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.006, area_weight=0.0, scale_to_bounds=False)
        bpy.ops.uv.select_all(action='SELECT')   # иначе упаковка двигает только выделенные острова — наложения
        bpy.ops.uv.pack_islands(margin=0.005, rotate=True)
        bpy.ops.object.mode_set(mode='OBJECT')

        scn = bpy.context.scene
        scn.render.engine = 'CYCLES'
        scn.cycles.device = 'CPU'
        scn.cycles.samples = samples
        scn.cycles.use_denoising = False
        scn.render.bake.margin = 8
        scn.render.bake.margin_type = 'EXTEND'

        size = self.tex_size
        albedo = bpy.data.images.new(self.name + "_albedo", size, size, alpha=False)
        mask = bpy.data.images.new(self.name + "_mask_raw", size, size, alpha=False, is_data=True)
        zmin = min((o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box)
        zmax = max((o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box)

        mat = bpy.data.materials.new(self.name + "_body")
        mat.use_nodes = True
        nt = mat.node_tree
        img_node = self._albedo_graph(nt, albedo, zmin, zmax)
        for o in meshes:
            o.data.materials.clear()
            o.data.materials.append(mat)
        nt.nodes.active = img_node
        print(f"[style] {self.name}: запекание альбедо {size}px, {samples} сэмплов...", flush=True)
        _select(meshes)
        bpy.ops.object.bake(type='EMIT', use_clear=True)

        # маска: атрибут mat как есть, без цветового пространства
        out = nt.nodes["Material Output"]
        em = nt.nodes["bake_emit"]
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "mat"
        nt.links.new(attr.outputs["Color"], em.inputs["Color"])
        img_node.image = mask
        scn.cycles.samples = 1
        _select(meshes)
        bpy.ops.object.bake(type='EMIT', use_clear=True)

        self._save(albedo, self.name + "_albedo.png")
        self._save_mask(mask, self.name + "_mask.png")
        # материал для превью и FBX: Principled с запечёнными текстурами
        self._preview_material(mat, albedo)
        self.material = mat

    def _albedo_graph(self, nt, albedo, zmin, zmax):
        N = nt.nodes
        L = nt.links
        for n in list(N):
            if n.type != 'OUTPUT_MATERIAL':
                N.remove(n)
        out = N["Material Output"]
        def node(t, **kw):
            n = N.new(t)
            for k, v in kw.items():
                setattr(n, k, v)
            return n
        def vmath(op, a, b=None, scale=None):
            n = node("ShaderNodeVectorMath", operation=op)
            L.new(a, n.inputs[0])
            if b is not None:
                if isinstance(b, tuple): n.inputs[1].default_value = b
                else: L.new(b, n.inputs[1])
            if scale is not None:
                if isinstance(scale, float): n.inputs[3].default_value = scale
                else: L.new(scale, n.inputs[3])
            return n
        def math_(op, a, b):
            n = node("ShaderNodeMath", operation=op)
            for i, x in enumerate((a, b)):
                if isinstance(x, float): n.inputs[i].default_value = x
                else: L.new(x, n.inputs[i])
            return n.outputs[0]
        def remap(v, a, b, c, d):
            n = node("ShaderNodeMapRange", clamp=True)
            L.new(v, n.inputs["Value"])
            n.inputs["From Min"].default_value = a
            n.inputs["From Max"].default_value = b
            n.inputs["To Min"].default_value = c
            n.inputs["To Max"].default_value = d
            return n.outputs["Result"]

        pal = node("ShaderNodeAttribute", attribute_name="pal").outputs["Color"]
        mat_attr = node("ShaderNodeAttribute", attribute_name="mat")
        matc = node("ShaderNodeSeparateXYZ")          # X — металл, Y — гладкость, Z — деталь без износа
        L.new(mat_attr.outputs["Color"], matc.inputs[0])
        geo0 = node("ShaderNodeNewGeometry")
        # крап по хитину (канал A): мягкие тёмные пятна из шума, разных размеров, — живой панцирь, а не заливка
        spots = node("ShaderNodeTexNoise")
        spots.inputs["Scale"].default_value = self.camo_scale
        spots.inputs["Detail"].default_value = 3.0
        spots.inputs["Roughness"].default_value = 0.6
        L.new(geo0.outputs["Position"], spots.inputs["Vector"])
        patch = math_('MULTIPLY', remap(spots.outputs["Fac"], 0.56, 0.66, 0.0, 0.55), mat_attr.outputs["Alpha"])
        pal = vmath('SCALE', pal, scale=math_('SUBTRACT', 1.0, patch)).outputs[0]
        worn = remap(matc.outputs["Z"], 0.0, 1.0, 1.0, 0.0)
        ao = node("ShaderNodeAmbientOcclusion", samples=16)
        ao.inputs["Distance"].default_value = 0.6
        geo = node("ShaderNodeNewGeometry")
        sep = node("ShaderNodeSeparateXYZ")
        L.new(geo.outputs["Position"], sep.inputs[0])
        noise = node("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.2
        noise.inputs["Detail"].default_value = 4.0
        L.new(geo.outputs["Position"], noise.inputs["Vector"])
        bev = node("ShaderNodeBevel", samples=8)
        bev.inputs["Radius"].default_value = 0.035

        # рисованный объём: глубокие углы, тёмный низ и светлый верх, крупная пятнистость
        shade = math_('MULTIPLY', remap(ao.outputs["AO"], 0.0, 1.0, 0.3, 1.0),
                      remap(sep.outputs["Z"], zmin, zmax, 0.62, 1.15))
        shade = math_('MULTIPLY', shade, remap(noise.outputs["Fac"], 0.3, 0.7, 0.9, 1.08))
        base = vmath('SCALE', pal, scale=shade).outputs[0]
        # налёт грязи у земли пятнами
        grime = node("ShaderNodeTexNoise")
        grime.inputs["Scale"].default_value = 5.0
        grime.inputs["Detail"].default_value = 6.0
        L.new(geo.outputs["Position"], grime.inputs["Vector"])
        dirt_f = math_('MULTIPLY', remap(sep.outputs["Z"], zmin + 0.05, zmin + 1.1, 0.7, 0.0),
                       remap(grime.outputs["Fac"], 0.35, 0.65, 0.35, 1.0))
        dirt_f = math_('MULTIPLY', dirt_f, math_('ADD', math_('MULTIPLY', worn, 0.75), 0.25))
        dirt = node("ShaderNodeVectorMath", operation='SUBTRACT')
        dirt.inputs[0].default_value = (0.035, 0.028, 0.02)
        L.new(base, dirt.inputs[1])
        base = vmath('ADD', base, vmath('SCALE', dirt.outputs[0], scale=dirt_f).outputs[0]).outputs[0]
        # светлые тёплые рёбра (рисованный блик): нормаль со скруглением отличается от геометрической
        dot = vmath('DOT_PRODUCT', bev.outputs["Normal"], geo.outputs["Normal"]).outputs["Value"]
        edge = remap(dot, 0.86, 0.995, 0.7, 0.0)
        hi = node("ShaderNodeVectorMath", operation='MULTIPLY_ADD')
        L.new(pal, hi.inputs[0])
        hi.inputs[1].default_value = (1.5, 1.5, 1.45)
        hi.inputs[2].default_value = (0.09, 0.08, 0.05)
        diff = vmath('SUBTRACT', hi.outputs[0], base).outputs[0]
        col = vmath('ADD', base, vmath('SCALE', diff, scale=edge).outputs[0]).outputs[0]
        # сколы краски по рёбрам: пятнами проступает светлый металл (у металла и резины сколов нет)
        chipn = node("ShaderNodeTexNoise")
        chipn.inputs["Scale"].default_value = 16.0
        chipn.inputs["Detail"].default_value = 8.0
        L.new(geo.outputs["Position"], chipn.inputs["Vector"])
        chips = math_('MULTIPLY', math_('MULTIPLY', remap(dot, 0.8, 0.985, 1.0, 0.0), remap(chipn.outputs["Fac"], 0.56, 0.64, 0.0, 0.8)),
                      remap(matc.outputs["X"], 0.0, 1.0, 1.0, 0.0))
        chips = math_('MULTIPLY', chips, worn)
        bare = node("ShaderNodeVectorMath", operation='SUBTRACT')
        bare.inputs[0].default_value = (0.36, 0.36, 0.34)
        L.new(col, bare.inputs[1])
        col = vmath('ADD', col, vmath('SCALE', bare.outputs[0], scale=chips).outputs[0]).outputs[0]
        # потёки: шум, вытянутый по вертикали, темнее книзу
        mp = node("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (7.0, 7.0, 0.35)
        L.new(geo.outputs["Position"], mp.inputs["Vector"])
        streak = node("ShaderNodeTexNoise")
        streak.inputs["Scale"].default_value = 1.0
        streak.inputs["Detail"].default_value = 3.0
        L.new(mp.outputs["Vector"], streak.inputs["Vector"])
        sf = math_('MULTIPLY', remap(streak.outputs["Fac"], 0.45, 0.7, 1.0, 0.86), 1.0)
        sf = math_('ADD', math_('MULTIPLY', math_('MULTIPLY', math_('SUBTRACT', sf, 1.0), remap(sep.outputs["Z"], zmin, zmax, 1.0, 0.35)), worn), 1.0)
        col = vmath('SCALE', col, scale=sf).outputs[0]
        em = node("ShaderNodeEmission")
        em.name = "bake_emit"
        L.new(col, em.inputs["Color"])
        L.new(em.outputs[0], out.inputs["Surface"])
        img = node("ShaderNodeTexImage")
        img.image = albedo
        return img

    def _preview_material(self, mat, albedo):
        N = mat.node_tree.nodes
        L = mat.node_tree.links
        for n in list(N):
            if n.type != 'OUTPUT_MATERIAL':
                N.remove(n)
        bsdf = N.new("ShaderNodeBsdfPrincipled")
        tex = N.new("ShaderNodeTexImage")
        tex.image = albedo
        L.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.7
        L.new(bsdf.outputs[0], N["Material Output"].inputs["Surface"])

    @staticmethod
    def _save(img, filename):
        img.filepath_raw = os.path.join(TEX, filename)
        img.file_format = 'PNG'
        img.save()

    @staticmethod
    def _save_mask(raw, filename):
        """R — металл, A — гладкость (формат _MetallicGlossMap в URP Lit)."""
        import numpy as np
        w, h = raw.size
        px = np.empty(w * h * 4, np.float32)
        raw.pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        res = np.zeros_like(px)
        res[:, 0] = px[:, 0]
        res[:, 3] = px[:, 1]
        out = bpy.data.images.new(filename[:-4], w, h, alpha=True, is_data=True)
        out.pixels.foreach_set(res.ravel())
        out.filepath_raw = os.path.join(TEX, filename)
        out.file_format = 'PNG'
        out.save()

    def glow_materials(self):
        """Эмиссионные материалы светящихся деталей (для превью) и их описание для Unity."""
        mats = []
        for go, g in self.glow_objects:
            key = go.name.split("_glow_")[1]
            col, _, _, glow = PAL[key]
            name = go["material"]
            m = bpy.data.materials.get(name)
            if m is None:
                m = bpy.data.materials.new(name)
                m.use_nodes = True
                b = m.node_tree.nodes["Principled BSDF"]
                b.inputs["Base Color"].default_value = (*[srgb_to_lin(c) for c in col], 1)
                b.inputs["Emission Color"].default_value = (*[srgb_to_lin(c) for c in col], 1)
                b.inputs["Emission Strength"].default_value = glow
                mats.append({"name": name, "type": "glow", "color": list(col), "emission": glow})
            go.data.materials.clear()
            go.data.materials.append(m)
        return mats


# ---------- превью ----------

def preview(name, target=(0, 0, 1.0), dist=12.0, views=("rts", "hero"), ground=True, res=(960, 720)):
    """Рендер Eevee: вид RTS-камеры (наклон 55° с юга) и героический ракурс 3/4."""
    scn = bpy.context.scene
    try:
        scn.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scn.render.engine = 'BLENDER_EEVEE'
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.view_settings.view_transform = 'Standard'
    scn.view_settings.look = 'None'
    if hasattr(scn.eevee, "taa_render_samples"):
        scn.eevee.taa_render_samples = 32
    if hasattr(scn.eevee, "use_shadows"):
        scn.eevee.use_shadows = True
    if hasattr(scn.eevee, "use_gtao"):
        scn.eevee.use_gtao = True

    world = bpy.data.worlds.new("w")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.45, 0.55, 0.7, 1)
    bg.inputs["Strength"].default_value = 0.9

    tmp = []
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    sun.data.energy = 4.0
    sun.data.color = (1.0, 0.93, 0.82)
    sun.data.angle = math.radians(3)
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))
    scn.collection.objects.link(sun)
    tmp.append(sun)
    if ground:
        bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
        g = bpy.context.active_object
        gm = bpy.data.materials.new("ground")
        gm.use_nodes = True
        gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.07, 0.12, 0.03, 1)
        gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.95
        g.data.materials.append(gm)
        tmp.append(g)

    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scn.collection.objects.link(cam)
    scn.camera = cam
    tmp.append(cam)
    t = Vector(target)
    for v in views:
        if v == "rts":
            pitch, yaw, d, lens = 55, 0, dist * 1.25, 50
        elif v == "side":
            pitch, yaw, d, lens = 8, 90, dist, 50
        elif v == "front":
            pitch, yaw, d, lens = 18, 25, dist, 50
        else:
            pitch, yaw, d, lens = 24, 145, dist, 50
        cam.data.lens = lens
        p, y = math.radians(pitch), math.radians(yaw)
        cam.location = t + Vector((math.sin(y) * math.cos(p) * d, -math.cos(y) * math.cos(p) * d, math.sin(p) * d))
        cam.rotation_euler = (t - cam.location).to_track_quat('-Z', 'Y').to_euler()
        scn.render.filepath = os.path.join(ART, "previews", f"{name}_{v}.png")
        bpy.ops.render.render(write_still=True)
        print("[style] превью", scn.render.filepath, flush=True)
    for o in tmp:
        bpy.data.objects.remove(o)
