# Статичные детали карты — тем же конвейером, что деревья (flora.Flora: цвет вершин + AO, prop_<имя>.npz):
# сборные стены колонии (цоколь, панели, пояс, карниз), окно, дверь закрытая и открытая, забор, скалы, пни
# четырёх видов деревьев и находки (аптечка, магнит, образец ДНК, оружейный ящик, футляр подствольника).
# Клетка карты — 0,833 м (40 px), стена — 2,29 м (110 px). Стены — блоки во всю клетку: соседние сливаются в
# сплошную стену, детали на боковых гранях уходят внутрь соседа. Окно и дверь стоят вдоль X (игра их поворачивает).
# Запуск: blender.exe -b --python props.py [-- имена] [--fast]
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector
import style, preview
from style import curve, oval, taper
from flora import Flora, column, RIBS, show_colors, LEAF, scaled

TAU = 2 * math.pi
C = 0.8333 / 2                 # полклетки
H = 2.29                       # высота стены
WALL = {"wall": (0.58, 0.56, 0.52), "plinth": (0.34, 0.33, 0.31), "trim": (0.40, 0.42, 0.44), "cap": (0.46, 0.47, 0.48),
        "glass": (0.30, 0.46, 0.52), "frame": (0.22, 0.23, 0.24), "door": (0.38, 0.42, 0.32), "hazard": (0.85, 0.66, 0.15),
        "dark": (0.12, 0.12, 0.13)}


def shell(m, z0, z1, plinth=True, cap=True, xs=(-C, C)):
    """Кусок стены по X от xs[0] до xs[1] и по высоте z0..z1 во всю толщину клетки, с цоколем и карнизом."""
    w = xs[1] - xs[0]
    cx = (xs[0] + xs[1]) / 2
    lo, hi = max(z0, 0.3 if plinth else z0), min(z1, H - 0.12 if cap else z1)
    if hi > lo:
        m.box("wall", (w, 2 * C, hi - lo), (cx, 0, (lo + hi) / 2), bevel=0.01, seg=1)
        if lo < 1.2 < hi:
            m.box("trim", (w + 0.02, 2 * C + 0.02, 0.05), (cx, 0, 1.2), bevel=0.0)
    if plinth and z0 < 0.3:
        m.box("plinth", (w + 0.03, 2 * C + 0.03, 0.3), (cx, 0, 0.15), bevel=0.012, seg=1)
    if cap and z1 > H - 0.12:
        m.box("cap", (w + 0.04, 2 * C + 0.04, 0.12), (cx, 0, H - 0.06), bevel=0.015, seg=1)


def wall():
    m = Flora("prop_wall", WALL)
    shell(m, 0, H)
    return m


def window():
    m = Flora("prop_window", WALL)
    jamb = 0.11
    shell(m, 0, 0.95)
    shell(m, 1.62, H, plinth=False)
    for s in (-1, 1):
        shell(m, 0.95, 1.62, plinth=False, cap=False, xs=sorted((s * C, s * (C - jamb))))
    m.box("glass", (2 * (C - jamb), 0.03, 0.67), (0, 0, 1.285), bevel=0.0)
    for z in (0.97, 1.6):
        m.box("frame", (2 * (C - jamb), 0.07, 0.035), (0, 0, z), bevel=0.0)
    m.box("frame", (0.03, 0.07, 0.67), (0, 0, 1.285), bevel=0.0)
    m.box("cap", (2 * C, 2 * C + 0.1, 0.04), (0, 0, 0.94), bevel=0.008)      # подоконник наружу и внутрь
    return m


def door_frame(m):
    jamb = 0.07
    shell(m, 2.0, H, plinth=False)
    for s in (-1, 1):
        shell(m, 0, 2.0, cap=False, xs=sorted((s * C, s * (C - jamb))))


def door_leaf(m, loc, along_x=True):
    w, t, h = 2 * C - 0.16, 0.06, 1.96
    size = (w, t, h) if along_x else (t, w, h)
    m.box("door", size, loc, bevel=0.01, seg=1)
    off = (0, t / 2 + 0.002, 0) if along_x else (t / 2 + 0.002, 0, 0)
    for s in (-1, 1):                                       # жёлто-чёрная полоса внизу и окошко — с обеих сторон
        o = Vector(off) * s
        band = (w * 0.9, 0.006, 0.12) if along_x else (0.006, w * 0.9, 0.12)
        m.box("hazard", band, tuple(Vector(loc) + o + Vector((0, 0, -h / 2 + 0.2))), bevel=0.0)
        slot = (w * 0.4, 0.006, 0.16) if along_x else (0.006, w * 0.4, 0.16)
        m.box("dark", slot, tuple(Vector(loc) + o + Vector((0, 0, h * 0.25))), bevel=0.0)


def door():
    m = Flora("prop_door", WALL)
    door_frame(m)
    door_leaf(m, (0, 0, 0.98))
    return m


def door_open():
    m = Flora("prop_door_open", WALL)
    door_frame(m)
    door_leaf(m, (-C + 0.1, C - 0.05 + (C - 0.08) * 0.0, 0.98), along_x=False)     # распахнута внутрь на петлях
    return m


WOOD = {"wood": (0.5, 0.38, 0.25), "wood_dk": (0.32, 0.24, 0.16)}


def post():
    m = Flora("prop_post", WOOD, gradient=("wood",))
    m.box("wood", (0.15, 0.15, 0.82), (0, 0, 0.41), bevel=0.015, seg=1)
    m.mesh("wood", [(-0.075, -0.075, 0.82), (0.075, -0.075, 0.82), (0.075, 0.075, 0.82), (-0.075, 0.075, 0.82),
                    (0, 0, 0.94)], [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)], sharp=30)     # заострённый верх
    m.box("wood_dk", (0.17, 0.17, 0.08), (0, 0, 0.04), bevel=0.01)
    return m


def rail():
    """Две жерди от этой клетки к соседней (игра ставит их к соседу справа или сверху)."""
    m = Flora("prop_rail", WOOD, gradient=("wood",))
    rnd = random.Random(7)
    for z in (0.36, 0.7):
        tilt = rnd.uniform(-0.02, 0.02)
        m.box("wood", (2 * C, 0.07, 0.11), (C, 0, z + tilt), rot=(0, tilt, 0), bevel=0.012, seg=1)
    return m


def rock(v):
    rnd = random.Random(500 + v)
    m = Flora(f"prop_rock_{v}", {"rock": (0.4, 0.37, 0.33), "rock_dk": (0.28, 0.26, 0.24), "moss": (0.36, 0.42, 0.28)},
              gradient=("rock",))
    # слоистая глыба: пласты сдвинуты, верх почти ровный (style.crag), — не гладкий многогранник
    m.crag("rock", (1.05, 0.9, 0.75 + 0.15 * v), (0, 0, (0.75 + 0.15 * v) / 2 - 0.05), seed=v * 3,
           rot_z=rnd.uniform(0, TAU), bands=4, jag=1.7)
    for k in range(3):
        a = rnd.uniform(0, TAU)
        d = Vector((math.cos(a), math.sin(a), 0)) * rnd.uniform(0.45, 0.6)
        s = rnd.uniform(0.18, 0.3)
        m.rock("rock_dk", (s, s * 0.9, s * 0.7), tuple(d + Vector((0, 0, s * 0.2))), seed=v * 7 + k, rough=1.0, subdiv=1)
    return m


def stump(kind):
    leaf = LEAF[kind]
    bark = {"pine": (0.36, 0.26, 0.17), "palm": (0.47, 0.38, 0.26), "swamp": (0.31, 0.27, 0.2)}.get(kind)
    m = Flora(f"prop_stump_{kind}", {"bark": bark or scaled(leaf, 1.1), "cut": (0.78, 0.66, 0.46), "ring": (0.55, 0.42, 0.28),
                                      "pulp": (0.5, 0.56, 0.34)})
    if kind == "cactus":                       # срезанная ребристая колонна, мякоть светлая
        column(m, "bark", (0, 0, 0), 0.19, 0.14, dome=0.0)
        m.cyl("pulp", 0.165, 0.02, (0, 0, 0.145), seg=20, bevel=0.0)
        return m
    r = {"pine": 0.1, "palm": 0.11, "swamp": 0.13}[kind]
    m.cyl("bark", r * 1.15, 0.2, (0, 0, 0.1), seg=12, bevel=0.0, r2=r)
    m.cyl("cut", r * 0.93, 0.02, (0, 0, 0.2), seg=12, bevel=0.0)
    m.cyl("ring", r * 0.5, 0.022, (0, 0, 0.2), seg=12, bevel=0.0)
    for k in range(4 if kind != "swamp" else 5):
        a = TAU * k / 4 + 0.4
        d = Vector((math.cos(a), math.sin(a), 0))
        m.sweep("bark", curve([d * r * 0.6 + Vector((0, 0, 0.1)), d * r * 1.8 + Vector((0, 0, 0.02)), d * r * 2.6], 4),
                oval(6), taper(0.04, 0.012))
    return m


ITEM = {"white": (0.9, 0.89, 0.85), "red": (0.78, 0.18, 0.15), "steel": (0.7, 0.71, 0.74), "olive": (0.36, 0.4, 0.25),
        "olive_dk": (0.24, 0.27, 0.17), "yellow": (0.9, 0.74, 0.25), "glass": (0.55, 0.62, 0.66), "dna": (0.62, 0.42, 0.85),
        "dna2": (0.42, 0.85, 0.78), "dark": (0.12, 0.12, 0.13)}


def medkit():
    m = Flora("prop_medkit", ITEM)
    m.box("white", (0.42, 0.3, 0.2), (0, 0, 0.1), bevel=0.025)
    m.box("red", (0.22, 0.06, 0.012), (0, 0, 0.205), bevel=0.0)
    m.box("red", (0.06, 0.22, 0.012), (0, 0, 0.205), bevel=0.0)
    m.box("dark", (0.16, 0.04, 0.04), (0, 0, 0.225), bevel=0.01)                     # ручка
    m.box("red", (0.43, 0.31, 0.03), (0, 0, 0.1), bevel=0.0)
    return m


def magnet():
    m = Flora("prop_magnet", ITEM)
    path = [Vector((0.14 * math.cos(a), 0, 0.2 + 0.14 * math.sin(a))) for a in (math.pi * k / 12 for k in range(13))]
    path = [Vector((0.14, 0, 0.05))] + path + [Vector((-0.14, 0, 0.05))]
    m.sweep("red", curve(path, 20), oval(10), 0.05)
    for s in (-1, 1):
        m.cyl("steel", 0.052, 0.07, (s * 0.14, 0, 0.035), seg=12, bevel=0.005)
    return m


def dna():
    m = Flora("prop_dna", ITEM)
    m.cyl("steel", 0.09, 0.04, (0, 0, 0.02), seg=16, bevel=0.008)
    m.cyl("steel", 0.09, 0.04, (0, 0, 0.42), seg=16, bevel=0.008)
    for k in range(4):                                   # каркас из стоек: спираль внутри видна
        a = TAU * k / 4 + 0.4
        m.cyl("steel", 0.012, 0.38, (0.08 * math.cos(a), 0.08 * math.sin(a), 0.22), seg=6, bevel=0.0)
    for k in range(9):                                   # спираль из бусин двух цветов — видна сквозь «стекло» сверху
        z = 0.07 + k * 0.035
        a = k * 0.8
        for j, key in enumerate(("dna", "dna2")):
            b = a + math.pi * j
            m.sphere(key, 0.024, (0.05 * math.cos(b), 0.05 * math.sin(b), z), seg=8, rings=5)
    return m


def crate(name, long=False):
    m = Flora(name, ITEM)
    w, d, h = (0.62, 0.3, 0.2) if long else (0.5, 0.4, 0.3)
    m.box("olive", (w, d, h), (0, 0, h / 2), bevel=0.02)
    for s in (-1, 1):
        m.box("olive_dk", (0.04, d + 0.01, h + 0.01), (s * w * 0.35, 0, h / 2), bevel=0.005)
        m.box("dark", (0.012, 0.08, 0.03), (s * (w / 2 + 0.006), 0, h * 0.7), bevel=0.0)
    m.box("yellow", (w * 0.4, d * 0.8, 0.006), (0, 0, h + 0.003), bevel=0.0)
    m.box("yellow", (w * 0.5, 0.006, h * 0.25), (0, -d / 2 - 0.003, h * 0.55), bevel=0.0)
    return m


MAKERS = {"wall": wall, "window": window, "door": door, "door_open": door_open, "post": post, "rail": rail,
          **{f"rock_{v}": (lambda v=v: rock(v)) for v in range(3)},
          **{f"stump_{k}": (lambda k=k: stump(k)) for k in ("pine", "palm", "cactus", "swamp")},
          "medkit": medkit, "magnet": magnet, "dna": dna, "crate": lambda: crate("prop_crate"),
          "underbarrel": lambda: crate("prop_underbarrel", long=True)}

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    which = [a for a in args if not a.startswith("--")] or list(MAKERS)
    paths = []
    for name in which:
        style.reset()
        m = MAKERS[name]()
        Flora.MAX_TRIS = 1600
        m.bake_colors()
        m.export(os.path.join(style.TEX, f"prop_{name}.npz"))
        show_colors(m.obj)
        preview._scene((420, 420))
        big = name in ("wall", "window", "door", "door_open")
        path = os.path.join(preview.OUT, f"_prop_{name}.png")
        preview._shoot(path, (0, 0, 1.0 if big else 0.2), 30, 150, 5.0 if big else 1.8)
        paths.append(path)
    preview.sheet(paths, os.path.join(preview.OUT, "props.png"), cols=6)
    print("PROPS DONE", flush=True)
