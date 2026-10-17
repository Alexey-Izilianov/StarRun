# Стволы десантника — подмеши на кости weapon (игра включает тот, что выбран): рукоять у всех под правой
# кистью (y ≈ 0,12), цевьё — под левой (y ≈ 0,38), ось ствола — на высоте z. MAR — в marine.py.
# Мини-турель — отдельная модель (turret()): тренога, поворотная голова со спаренными стволами.
import math
from mathutils import Matrix, Vector
from style import curve, oval, taper
from rig import RigModel

R = math.radians
FWD = (-math.pi / 2, 0, 0)         # цилиндр вдоль +Y
UP_Y = (math.pi / 2, 0, 0)         # тор в плоскости XZ — кольцо вокруг оси Y
W = "weapon"


def grip(m, x, z, key="gunmetal"):
    m.box(key, (0.046, 0.06, 0.1), (x, 0.12, z - 0.08), rot=(R(12), 0, 0), group=W, bevel=0.015)


def shotgun(m, x, z):
    """Помповый дробовик: толстый ствол над трубчатым магазином, цевьё-помпа под левой рукой, патроны на боку."""
    m.sub = "shotgun"
    m.box("gunmetal", (0.072, 0.3, 0.1), (x, 0.22, z), group=W, bevel=0.016)
    m.box("armor", (0.074, 0.16, 0.05), (x, 0.2, z + 0.03), group=W, bevel=0.012)
    m.box("armor_dk", (0.06, 0.24, 0.09), (x, -0.04, z - 0.02), group=W, bevel=0.02, top=(1, 0.75))
    m.box("rubber", (0.064, 0.03, 0.105), (x, -0.17, z - 0.025), group=W, bevel=0.01)
    m.cyl("gunmetal", 0.024, 0.46, (x, 0.6, z + 0.015), FWD, group=W, seg=14, bevel=0.006)
    m.cyl("gunmetal", 0.03, 0.05, (x, 0.84, z + 0.015), FWD, group=W, seg=14, bevel=0.008)
    m.cyl("gunmetal", 0.02, 0.42, (x, 0.56, z - 0.035), FWD, group=W, seg=12, bevel=0.005)
    m.cyl("rubber", 0.034, 0.16, (x, 0.42, z - 0.028), FWD, group=W, seg=14, bevel=0.01)
    for y in (0.37, 0.41, 0.45):
        m.cyl("armor_dk", 0.036, 0.012, (x, y, z - 0.028), FWD, group=W, seg=14, bevel=0.0)
    m.box("gunmetal", (0.012, 0.02, 0.02), (x, 0.8, z + 0.045), group=W, bevel=0.0)
    for i in range(5):                                         # патронташ на правой щеке
        y = 0.13 + 0.032 * i
        m.cyl("red", 0.011, 0.045, (x + 0.043, y, z + 0.005), group=W, seg=8, bevel=0.0)
        m.cyl("team", 0.0115, 0.012, (x + 0.043, y, z - 0.022), group=W, seg=8, bevel=0.0)
    grip(m, x, z)
    m.sub = None


def sniper(m, x, z):
    """Снайперская винтовка: длинный ствол с дульным тормозом, большой прицел, скелетный приклад, сложенные сошки."""
    m.sub = "sniper"
    m.box("gunmetal", (0.07, 0.36, 0.1), (x, 0.2, z), group=W, bevel=0.016)
    m.box("armor", (0.072, 0.2, 0.045), (x, 0.22, z + 0.03), group=W, bevel=0.012)
    m.box("armor_dk", (0.05, 0.26, 0.035), (x, -0.09, z + 0.02), group=W, bevel=0.012)
    m.box("armor_dk", (0.05, 0.22, 0.03), (x, -0.1, z - 0.06), rot=(R(-8), 0, 0), group=W, bevel=0.01)
    m.box("armor", (0.052, 0.12, 0.04), (x, -0.05, z + 0.05), group=W, bevel=0.012)
    m.box("rubber", (0.058, 0.03, 0.14), (x, -0.23, z - 0.02), group=W, bevel=0.01)
    m.cyl("armor_dk", 0.032, 0.3, (x, 0.52, z + 0.005), FWD, group=W, seg=14, bevel=0.008)
    for y in (0.42, 0.5, 0.58):
        m.box("rubber", (0.066, 0.04, 0.012), (x, y, z + 0.03), group=W, bevel=0.0)
    m.cyl("gunmetal", 0.016, 0.42, (x, 0.86, z + 0.005), FWD, group=W, seg=10, bevel=0.0)
    m.box("gunmetal", (0.05, 0.07, 0.04), (x, 1.08, z + 0.005), group=W, bevel=0.01)
    for y in (1.06, 1.09):
        m.box("rubber", (0.052, 0.008, 0.026), (x, y, z + 0.005), group=W, bevel=0.0)
    m.cyl("gunmetal", 0.026, 0.3, (x, 0.24, z + 0.1), FWD, group=W, seg=16, bevel=0.006)        # прицел
    m.cyl("gunmetal", 0.026, 0.07, (x, 0.42, z + 0.1), FWD, group=W, seg=16, bevel=0.006, r2=0.036)
    m.cyl("gunmetal", 0.032, 0.05, (x, 0.07, z + 0.1), FWD, group=W, seg=16, bevel=0.006, r2=0.026)
    m.cyl("optic", 0.03, 0.004, (x, 0.457, z + 0.1), FWD, group=W, seg=16, bevel=0.0)
    m.cyl("gunmetal", 0.012, 0.05, (x + 0.03, 0.24, z + 0.1), (0, math.pi / 2, 0), group=W, seg=10, bevel=0.0)
    m.cyl("gunmetal", 0.012, 0.04, (x, 0.24, z + 0.13), group=W, seg=10, bevel=0.0)
    for y in (0.16, 0.32):
        m.box("gunmetal", (0.02, 0.03, 0.05), (x, y, z + 0.065), group=W, bevel=0.0)
    m.box("gunmetal", (0.045, 0.07, 0.09), (x, 0.3, z - 0.09), group=W, bevel=0.01)
    for s in (-1, 1):                                           # сошки сложены под стволом
        m.cyl("gunmetal", 0.007, 0.2, (x + s * 0.014, 0.74, z - 0.03), FWD, group=W, seg=8, bevel=0.0)
    m.box("team", (0.073, 0.03, 0.012), (x, 0.31, z + 0.051), group=W, bevel=0.0)
    grip(m, x, z)
    m.sub = None


def flame(m, x, z):
    """Огнемёт: короб с баком под ним (жёлтый, в чёрных полосах), сопло в теплозащитном кожухе, запальник."""
    m.sub = "flame"
    m.box("armor_dk", (0.08, 0.34, 0.11), (x, 0.2, z), group=W, bevel=0.02)
    m.box("armor", (0.082, 0.18, 0.05), (x, 0.24, z + 0.035), group=W, bevel=0.014)
    m.box("gunmetal", (0.06, 0.14, 0.08), (x, -0.02, z - 0.01), group=W, bevel=0.018, top=(1, 0.8))
    m.box("rubber", (0.064, 0.025, 0.09), (x, -0.1, z - 0.015), group=W, bevel=0.008)
    m.cyl("hazard", 0.046, 0.3, (x, 0.3, z - 0.1), FWD, group=W, seg=16, bevel=0.01)          # бак
    for y in (0.19, 0.27, 0.35, 0.43):
        m.cyl("rubber", 0.047, 0.02, (x, y, z - 0.1), FWD, group=W, seg=16, bevel=0.0)
    for y in (0.15, 0.45):
        m.cyl("gunmetal", 0.04, 0.02, (x, y, z - 0.1), FWD, group=W, seg=16, bevel=0.004)
    m.sweep("rubber", curve([(x, 0.14, z - 0.1), (x - 0.03, 0.06, z - 0.15), (x - 0.02, 0.0, z - 0.08),
                             (x, 0.05, z - 0.04)], 10), oval(8), 0.012, group=W)                   # шланг
    m.cyl("gunmetal", 0.028, 0.42, (x, 0.56, z + 0.01), FWD, group=W, seg=14, bevel=0.0)
    m.cyl("armor", 0.046, 0.24, (x, 0.53, z + 0.01), FWD, group=W, seg=16, bevel=0.01)          # кожух
    for y in (0.45, 0.51, 0.57, 0.63):
        m.cyl("armor_dk", 0.048, 0.014, (x, y, z + 0.01), FWD, group=W, seg=16, bevel=0.0)
    m.cyl("gunmetal", 0.032, 0.07, (x, 0.79, z + 0.01), FWD, group=W, seg=14, bevel=0.006, r2=0.044)
    m.cyl("gunmetal", 0.009, 0.12, (x, 0.74, z - 0.035), FWD, group=W, seg=8, bevel=0.0)          # запальник
    m.sphere("flame_glow", 0.012, (x, 0.805, z - 0.035), group=W, seg=8, rings=5)
    m.cyl("rubber", 0.034, 0.012, (x, 0.83, z + 0.01), FWD, group=W, seg=14, bevel=0.0)
    grip(m, x, z)
    m.sub = None


def rocket(m, x, z):
    """Ракетница: труба над рукоятью (на плече), раструб сзади, головка ракеты в жерле, коллиматор слева."""
    m.sub = "rocket"
    zt = z + 0.09
    m.cyl("armor", 0.06, 0.9, (x, 0.27, zt), FWD, group=W, seg=18, bevel=0.01)
    m.cyl("gunmetal", 0.06, 0.08, (x, -0.2, zt), FWD, group=W, seg=18, bevel=0.008, r2=0.074)     # раструб
    m.cyl("gunmetal", 0.067, 0.05, (x, 0.71, zt), FWD, group=W, seg=18, bevel=0.008)
    m.cyl("rubber", 0.05, 0.02, (x, 0.737, zt), FWD, group=W, seg=18, bevel=0.0)
    m.sphere("red", 0.044, (x, 0.72, zt), scale=(1, 1.6, 1), group=W, seg=12, rings=8)
    for y in (0.6, 0.64):
        m.cyl("team", 0.0615, 0.018, (x, y, zt), FWD, group=W, seg=18, bevel=0.0)
    for y in (0.0, 0.3):
        m.cyl("armor_dk", 0.063, 0.03, (x, y, zt), FWD, group=W, seg=18, bevel=0.0)
    m.box("gunmetal", (0.05, 0.42, 0.05), (x, 0.24, z + 0.01), group=W, bevel=0.012)
    m.box("rubber", (0.035, 0.045, 0.1), (x, 0.38, z - 0.04), group=W, bevel=0.01)
    m.box("gunmetal", (0.03, 0.07, 0.07), (x - 0.07, 0.3, zt + 0.04), group=W, bevel=0.008)
    m.box("optic", (0.004, 0.01, 0.03), (x - 0.07, 0.336, zt + 0.045), group=W, bevel=0.0)
    m.box("gunmetal", (0.03, 0.03, 0.03), (x - 0.04, 0.3, zt + 0.03), group=W, bevel=0.0)
    grip(m, x, z)
    m.sub = None


def shock(m, x, z):
    """Шокер: короб с батареей, сердечник в медных катушках, раздвоенный разрядник со светящимися остриями."""
    m.sub = "shock"
    m.box("armor_dk", (0.08, 0.34, 0.11), (x, 0.2, z), group=W, bevel=0.02)
    m.box("armor", (0.082, 0.2, 0.05), (x, 0.22, z + 0.035), group=W, bevel=0.014)
    m.box("gunmetal", (0.066, 0.12, 0.1), (x, -0.03, z - 0.005), group=W, bevel=0.016)
    m.box("optic", (0.068, 0.08, 0.01), (x, -0.03, z + 0.03), group=W, bevel=0.0)
    m.cyl("gunmetal", 0.022, 0.4, (x, 0.56, z + 0.005), FWD, group=W, seg=12, bevel=0.0)
    for y in (0.42, 0.5, 0.58, 0.66):
        m.torus("copper", 0.042, 0.013, (x, y, z + 0.005), rot=UP_Y, group=W, seg=18)
    for s in (-1, 1):
        m.box("gunmetal", (0.012, 0.16, 0.018), (x + s * 0.026, 0.8, z + 0.005), rot=(0, 0, s * R(-6)), group=W,
              bevel=0.0)
        m.sphere("optic", 0.011, (x + s * 0.034, 0.885, z + 0.005), group=W, seg=8, rings=5)
    m.sphere("optic", 0.016, (x, 0.76, z + 0.005), group=W, seg=10, rings=6)
    grip(m, x, z)
    m.sub = None


TURRET_JOINTS = {"root": ((0, 0, 0), None), "head": ((0, 0, 0.44), "root"), "muzzle": ((0, 0.5, 0.48), "head")}


def turret():
    """Мини-турель ~0,6 м: тренога, стойка, голова с бронещитком, спаренные стволы, ящик лент, глазок прицела."""
    m = RigModel("turret", TURRET_JOINTS, 512)
    m.camo_scale = 6.0
    for a in (90, 210, 330):
        d = Vector((math.cos(R(a)), math.sin(R(a)), 0))
        top = Vector((0, 0, 0.3)) + d * 0.04
        foot = d * 0.36
        m.sweep("gunmetal", [top, top.lerp(foot, 0.5) + Vector((0, 0, 0.03)), foot + Vector((0, 0, 0.02))], oval(8),
                taper(0.022, 0.016), group="root")
        m.box("rubber", (0.06, 0.06, 0.02), tuple(foot + Vector((0, 0, 0.01))), rot=(0, 0, R(a)), group="root", bevel=0.005)
    m.cyl("gunmetal", 0.035, 0.18, (0, 0, 0.36), group="root", seg=14, bevel=0.006)
    m.cyl("armor_dk", 0.06, 0.04, (0, 0, 0.43), group="head", seg=16, bevel=0.008)
    m.box("armor", (0.2, 0.24, 0.13), (0, 0.04, 0.5), group="head", bevel=0.025, top=(0.85, 0.85))
    m.box("armor_dk", (0.22, 0.04, 0.15), (0, 0.15, 0.5), rot=(R(-12), 0, 0), group="head", bevel=0.015)    # щиток
    m.box("team", (0.205, 0.03, 0.015), (0, 0.0, 0.567), group="head", bevel=0.0)
    for s in (-1, 1):
        m.cyl("gunmetal", 0.018, 0.34, (s * 0.04, 0.3, 0.49), FWD, group="head", seg=10, bevel=0.0)
        m.cyl("gunmetal", 0.026, 0.06, (s * 0.04, 0.46, 0.49), FWD, group="head", seg=10, bevel=0.006)
    m.box("hazard", (0.07, 0.12, 0.09), (0.14, -0.02, 0.47), group="head", bevel=0.01)
    m.sweep("gunmetal", curve([(0.12, 0.04, 0.5), (0.08, 0.1, 0.53), (0.04, 0.16, 0.5)], 6), oval(6), 0.012,
            group="head")
    m.sphere("optic", 0.02, (0, 0.17, 0.56), group="head", seg=10, rings=6)
    m.box("gunmetal", (0.06, 0.06, 0.04), (0, 0.12, 0.575), group="head", bevel=0.008)
    m.build()
    return m


def turret_actions(m):
    head = Vector(TURRET_JOINTS["head"][0])

    def idle(t):
        return {"head": Matrix.Translation(head) @ Matrix.Rotation(R(4) * math.sin(2 * math.pi * t), 4, 'Z')
                @ Matrix.Translation(-head)}

    def fire(t):
        k = math.exp(-8 * t) * math.sin(math.pi * min(1.0, t * 6))       # отдача: голова вздрагивает назад
        return {"head": Matrix.Translation((0, -0.03 * k, 0))}

    m.skin_action("idle", 60, idle)
    m.skin_action("fire", 6, fire, loop=False)
