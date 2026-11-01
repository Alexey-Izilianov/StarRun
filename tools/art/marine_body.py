# Тело десантника — детально: подкостюм гладкими трубками с мышечным утолщением, броня изогнутыми пластинами
# по форме тела (поножи, наручи, наплечник из трёх лам), ботинки с подошвой и ремнями, перчатки с костяшками и
# пальцами, пояс с подсумками и флягой, нагрудная разгрузка с магазинами, гранатами и ножом, ранец со скаткой
# и шлангами, шлем с рельсами, креплением ПНВ, назатыльником и нащёчниками. Суставы — marine.JOINTS.
import math
from mathutils import Euler, Vector
from style import around, curve, oval, taper

R = math.radians
FWD = (-math.pi / 2, 0, 0)         # цилиндр вдоль +Y
SIDE = (0, math.pi / 2, 0)         # цилиндр вдоль X
TAU = 2 * math.pi


def tube(m, key, a, b, r0, r1, group, bulge=0.0, seg=16, n=9):
    """Гладкая трубка от a к b: радиус r0 → r1 с утолщением bulge посередине (икра, бицепс, бедро)."""
    a, b = Vector(a), Vector(b)
    path = [a.lerp(b, k / (n - 1)) for k in range(n)]
    m.sweep(key, path, oval(seg), lambda t: r0 + (r1 - r0) * t + bulge * math.sin(math.pi * t) ** 1.5, group=group)


def band(m, key, center, radii, z_half, thick, group, rot=(0, 0, 0), u=(0, TAU), res=24):
    """Обруч вокруг вертикальной оси эллипсоида: ремень, манжета, кольцо."""
    lat = math.asin(min(0.99, z_half / radii[2]))
    m.plate(key, center, radii, u, (-lat, lat), thick, rot=rot, group=group, res=(res, 2))


def pouch(m, center, size, yaw, group, key="sand", flap="sand", stud="metal"):
    """Подсумок: скруглённый корпус, клапан чуть шире и с напуском, кнопка."""
    cx, cy, cz = center
    w, d, h = size
    m.box(key, (w, d, h), center, rot=(0, 0, yaw), group=group, bevel=0.012, seg=3)
    off = Vector((0, d * 0.08, 0))
    off.rotate(Euler((0, 0, yaw)))
    m.box(flap, (w * 1.06, d * 1.12, h * 0.32), (cx + off.x, cy + off.y, cz + h * 0.36), rot=(0, 0, yaw), group=group,
          bevel=0.008, seg=2)
    fwd = Vector((0, d * 0.62, 0))
    fwd.rotate(Euler((0, 0, yaw)))
    m.sphere(stud, 0.007, (cx + fwd.x, cy + fwd.y, cz + h * 0.28), group=group, seg=8, rings=4)


def on_ring(rx, ry, a):
    """Точка и курс на эллипсе пояса (угол a от +X): подсумок смотрит наружу."""
    return (rx * math.cos(a), ry * math.sin(a)), a - math.pi / 2


# ---------------------------------------------------------------------------------------------- ноги

def legs(m, P):
    for s, S in ((1, "R"), (-1, "L")):
        x = 0.11 * s
        F, SH, TH = "foot." + S, "shin." + S, "thigh." + S
        # ботинок: подошва с рантом, носок скруглённым сечением, голенище, бронированный носок, два ремня с пряжками
        boot = curve([(x, -0.09, 0.08), (x, 0.04, 0.072), (x, 0.17, 0.052)], 9)
        m.sweep("suit", boot, oval(16), taper(0.072, 0.06), taper(0.05, 0.022), group=F)
        sole = [Vector((p.x, p.y, 0.016)) for p in curve([(x, -0.1, 0), (x, 0.04, 0), (x, 0.185, 0)], 9)]
        m.sweep("rubber", sole, oval(16), lambda t: 0.068 - 0.012 * t ** 2, 0.016, group=F)          # подошва по контуру
        tube(m, "suit", (x, -0.02, 0.06), (x, -0.012, 0.21), 0.068, 0.062, F, seg=16, n=5)
        m.plate("armor_dk", (x, 0.12, 0.05), (0.068, 0.09, 0.05), around(90, 85), (R(0), R(80)), 0.012, group=F, res=(12, 5))
        for z, y in ((0.13, 0.0), (0.185, -0.01)):
            band(m, "strap", (x, y, z), (0.072, 0.074, 0.03), 0.012, 0.008, F, res=18)
            m.box("metal", (0.026, 0.008, 0.022), (x + 0.05 * s, y + 0.045, z), rot=(0, 0, R(40) * s), group=F, bevel=0.003)
        # голень: подкостюм с икрой, понож — выгнутая пластина спереди и с боков, три ребра, щиток на икре
        tube(m, "suit", P(SH), (x, -0.012, 0.19), 0.066, 0.056, SH, bulge=0.012)
        m.plate("armor", (x, 0.008, 0.33), (0.078, 0.084, 0.2), around(90, 108), (R(-58), R(62)), 0.022, group=SH,
                res=(20, 10))
        for k, lat in enumerate((R(-24), R(2), R(28))):
            m.plate("armor_dk", (x, 0.008, 0.33), (0.081, 0.088, 0.203), around(90, 34 - 4 * k), (lat - R(3), lat + R(3)),
                    0.006, group=SH, res=(8, 1))
        m.plate("armor_dk", (x, -0.005, 0.34), (0.074, 0.078, 0.16), around(270, 55), (R(-45), R(50)), 0.016, group=SH,
                res=(10, 6))
        # колено: наколенник выпуклой чашей с кантом и болтом
        m.plate("sand", (x, 0.025, 0.53), (0.085, 0.095, 0.085), around(90, 72), (R(-58), R(62)), 0.026, group=SH, res=(14, 8))
        m.plate("armor_dk", (x, 0.025, 0.53), (0.088, 0.098, 0.088), around(90, 74), (R(-62), R(-50)), 0.01, group=SH,
                res=(14, 1))
        m.sphere("metal", 0.011, (x, 0.118, 0.535), group=SH, seg=8, rings=5)
        # бедро: подкостюм с мышцей, пластина спереди-снаружи, щиток сбоку, два ремня
        tube(m, "suit", P(TH), P(SH), 0.086, 0.066, TH, bulge=0.014)
        m.plate("armor", (x, 0.0, 0.76), (0.114, 0.108, 0.22), around(90, 112), (R(-50), R(45)), 0.028, group=TH, res=(20, 9))
        m.plate("armor_dk", (x + 0.008 * s, 0.0, 0.8), (0.118, 0.104, 0.18), around(0 if s > 0 else 180, 52), (R(-45), R(40)),
                0.022, group=TH, res=(10, 7))
        for z in (0.68, 0.86):
            band(m, "strap", (x, 0.0, z), (0.104, 0.1, 0.03), 0.011, 0.008, TH, res=20)
    # кобура на правом бедре, подсумок-сброс на левом
    m.box("armor_dk", (0.055, 0.13, 0.18), (0.205, 0.0, 0.76), group="thigh.R", bevel=0.016, seg=3)
    m.box("gunmetal", (0.035, 0.05, 0.06), (0.205, 0.035, 0.87), rot=(R(-15), 0, 0), group="thigh.R", bevel=0.01, seg=2)
    pouch(m, (-0.2, 0.0, 0.74), (0.06, 0.1, 0.13), math.pi / 2, "thigh.L")


# ---------------------------------------------------------------------------------------------- пояс и живот

def waist(m):
    H, SP = "hips", "spine"
    m.box("suit", (0.3, 0.22, 0.18), (0, -0.01, 0.98), group=H, bevel=0.06, seg=4)
    band(m, "gunmetal", (0, -0.01, 1.04), (0.188, 0.134, 0.06), 0.03, 0.02, H, res=32)              # ремень
    m.box("metal", (0.065, 0.016, 0.045), (0, 0.127, 1.04), group=H, bevel=0.008, seg=2)              # пряжка
    m.box("gunmetal", (0.03, 0.018, 0.02), (0, 0.133, 1.04), group=H, bevel=0.004)
    # щиток паха и набедренные щитки — выгнутые пластины
    m.plate("armor", (0, 0.0, 0.95), (0.1, 0.115, 0.12), around(90, 38, 0.3), (R(-55), R(25)), 0.022, group=H, res=(10, 7))
    for s in (-1, 1):
        m.plate("armor_dk", (0, -0.01, 0.97), (0.2, 0.145, 0.16), around(90 - 90 * s, 34), (R(-50), R(10)), 0.02,
                group=H, res=(10, 6))
    # подсумки по поясу: магазины спереди, медпакет и утилитарный сбоку, фляга сзади справа
    for a, size in ((R(62), (0.07, 0.055, 0.1)), (R(118), (0.07, 0.055, 0.1)), (R(22), (0.075, 0.06, 0.09)),
                    (R(158), (0.075, 0.06, 0.09)), (R(-30), (0.07, 0.065, 0.09))):
        (px, py), yaw = on_ring(0.2, 0.145, a)
        pouch(m, (px, py - 0.01, 1.0), size, yaw, H)
    (px, py), yaw = on_ring(0.2, 0.145, R(-140))
    m.cyl("armor_dk", 0.04, 0.12, (px, py - 0.01, 1.0), group=H, seg=18, bevel=0.012)                    # фляга
    m.cyl("gunmetal", 0.016, 0.025, (px, py - 0.01, 1.07), group=H, seg=10, bevel=0.003)
    for x in (-0.07, -0.01):                                                                             # гранаты сзади
        m.sphere("armor_dk", 0.03, (x, -0.15, 1.0), scale=(1, 1, 1.25), group=H, seg=14, rings=8)
        m.cyl("metal", 0.011, 0.024, (x, -0.15, 1.045), group=H, seg=8, bevel=0.0)
        m.box("metal", (0.006, 0.012, 0.04), (x + 0.018, -0.15, 1.03), group=H, bevel=0.0)
    # живот: подкостюм и три гибкие полосы брони
    m.box("suit", (0.28, 0.2, 0.18), (0, 0.0, 1.15), group=SP, bevel=0.06, seg=4)
    for i, z in enumerate((1.09, 1.145, 1.2)):
        m.plate("armor_dk" if i % 2 == 0 else "armor", (0, 0.0, z), (0.15 - i * 0.004, 0.112, 0.06), around(90, 70),
                (R(-24), R(24)), 0.018, group=SP, res=(14, 2))


# ---------------------------------------------------------------------------------------------- корпус

def torso(m):
    C = "chest"

    def sect(z, w, d, fwd, k=16):
        pts = []
        for i in range(k * 4):
            a = TAU * i / (k * 4)
            c, s = math.cos(a), math.sin(a)
            x = math.copysign(abs(c) ** 0.85, c) * w              # органичная грудная клетка под разгрузкой
            y = math.copysign(abs(s) ** 0.85, s) * d + (fwd if s > 0 else 0) * s
            pts.append((x, y, z))
        return pts
    rings = [sect(1.22, 0.17, 0.12, 0.0), sect(1.26, 0.19, 0.127, 0.008), sect(1.31, 0.212, 0.134, 0.02),
             sect(1.37, 0.235, 0.14, 0.035), sect(1.43, 0.242, 0.142, 0.037), sect(1.48, 0.228, 0.137, 0.025),
             sect(1.52, 0.196, 0.128, 0.012), sect(1.555, 0.14, 0.1, 0.0)]
    m.loft("armor", rings, group=C, sharp=55)
    # нагрудник: плоская песочная пластина со скошенными углами, ребро посередине, нижний кант
    m.box("sand", (0.3, 0.04, 0.18), (0, 0.186, 1.41), rot=(R(-9), 0, 0), group=C, bevel=0.022, seg=4, top=(0.82, 1))
    m.box("armor_dk", (0.026, 0.024, 0.16), (0, 0.207, 1.41), rot=(R(-9), 0, 0), group=C, bevel=0.008, seg=2)
    m.box("armor_lt", (0.23, 0.035, 0.04), (0, 0.168, 1.29), group=C, bevel=0.012, seg=2, top=(0.9, 1))
    m.seam((-0.2, 0.135, 1.24), (0.2, 0.135, 1.24), (0, 1, 0), group=C, key="suit", width=0.015)
    for s in (-1, 1):
        m.seam((0.236 * s, 0.05, 1.29), (0.236 * s, 0.05, 1.49), (s, 0, 0), group=C, key="suit", width=0.014)
        m.rivets((0.237 * s, -0.06, 1.3), (0.237 * s, -0.06, 1.48), 3, "x", group=C, key="metal", r=0.008)
    # горжет и воротник из двух колец
    m.cyl("armor_dk", 0.1, 0.07, (0, 0.0, 1.56), group=C, seg=28, bevel=0.016, r2=0.086, scale=(1.15, 1, 1))
    m.cyl("armor", 0.088, 0.03, (0, 0.0, 1.6), group=C, seg=28, bevel=0.01, r2=0.08, scale=(1.12, 1, 1))
    m.cyl("suit", 0.06, 0.09, (0, 0.01, 1.62), group=C, seg=16, bevel=0.0)
    # разгрузка: ремни через грудь с D-кольцами, три магазина на животе, две гранаты слева, нож справа, рация
    for s in (-1, 1):
        m.box("strap", (0.038, 0.016, 0.3), (0.1 * s, 0.163, 1.4), rot=(R(-9), 0, R(-12) * s), group=C, bevel=0.004)
        m.torus("metal", 0.012, 0.003, (0.093 * s, 0.176, 1.46), rot=(R(90), 0, 0), group=C, seg=12)
    for k, x in enumerate((-0.075, 0.0, 0.075)):
        pouch(m, (x, 0.165, 1.255), (0.065, 0.045, 0.085), 0.0, C)
        m.box("gunmetal", (0.04, 0.02, 0.025), (x, 0.162, 1.31), group=C, bevel=0.005)                  # торец магазина
    for z in (1.42, 1.36):
        m.sphere("armor_dk", 0.026, (-0.12, 0.178, z), scale=(1, 1, 1.25), group=C, seg=14, rings=8)
        m.cyl("metal", 0.009, 0.02, (-0.12, 0.178, z + 0.036), group=C, seg=8, bevel=0.0)
    m.box("armor_dk", (0.03, 0.022, 0.16), (0.125, 0.176, 1.42), rot=(0, 0, R(-12)), group=C, bevel=0.008, seg=2)  # ножны
    m.box("rubber", (0.022, 0.024, 0.07), (0.13, 0.177, 1.53), rot=(0, 0, R(-12)), group=C, bevel=0.008, seg=2)
    m.box("metal", (0.04, 0.026, 0.008), (0.128, 0.177, 1.495), rot=(0, 0, R(-12)), group=C, bevel=0.002)
    m.box("sand", (0.075, 0.045, 0.07), (-0.115, 0.17, 1.3), group=C, bevel=0.014, seg=2)            # рация
    m.cyl("gunmetal", 0.008, 0.09, (-0.09, 0.17, 1.37), group=C, seg=6, bevel=0.0)
    m.box("white", (0.07, 0.006, 0.022), (0.075, 0.19, 1.5), rot=(R(-9), 0, 0), group=C, bevel=0.0)  # нашивка
    m.box("rubber", (0.05, 0.007, 0.01), (0.075, 0.194, 1.5), rot=(R(-9), 0, 0), group=C, bevel=0.0)


def backpack(m):
    C = "chest"
    m.box("armor_dk", (0.32, 0.16, 0.38), (0, -0.2, 1.34), group=C, bevel=0.06, seg=4, top=(0.92, 0.85))
    m.box("armor", (0.28, 0.05, 0.3), (0, -0.29, 1.35), group=C, bevel=0.03, seg=3)
    for x in (-0.08, 0.08):                                                       # баллоны с вентилями и шлангами
        m.cyl("metal", 0.05, 0.36, (x, -0.335, 1.33), group=C, seg=24, bevel=0.018)
        for z in (1.15, 1.51):
            m.cyl("gunmetal", 0.055, 0.035, (x, -0.335, z), group=C, seg=24, bevel=0.008)
        m.cyl("team", 0.052, 0.025, (x, -0.335, 1.44), group=C, seg=24, bevel=0.0)
        m.cyl("gunmetal", 0.014, 0.04, (x, -0.335, 1.55), group=C, seg=10, bevel=0.003)
        m.sweep("rubber", curve([(x, -0.335, 1.57), (x * 1.6, -0.3, 1.6), (x * 2.2, -0.2, 1.55), (x * 2.4, -0.12, 1.48)], 10),
                oval(8), 0.011, group=C)
    for s in (-1, 1):                                                             # боковые подсумки
        pouch(m, (0.18 * s, -0.2, 1.3), (0.06, 0.12, 0.16), math.pi / 2 * s, C)
    m.cyl("sand", 0.055, 0.34, (0, -0.21, 1.58), SIDE, group=C, seg=20, bevel=0.012)                 # скатка
    for x in (-0.11, 0.11):
        m.cyl("strap", 0.058, 0.02, (x, -0.21, 1.58), SIDE, group=C, seg=20, bevel=0.0)
    m.vent((0, -0.24, 1.5), 0.18, 0.08, "z", slats=4, group=C, frame="gunmetal", slat="rubber")
    m.cyl("gunmetal", 0.008, 0.55, (-0.15, -0.26, 1.82), rot=(R(-8), 0, 0), group=C, seg=6, bevel=0.0)  # антенна
    m.cyl("gunmetal", 0.017, 0.05, (-0.15, -0.27, 1.56), group=C, seg=10, bevel=0.0)
    m.sphere("rubber", 0.012, (-0.15, -0.3, 2.09), group=C, seg=8, rings=5)


# ---------------------------------------------------------------------------------------------- руки и плечи

def _rot_to(d):
    return tuple(Vector((0, 0, 1)).rotation_difference(Vector(d)).to_euler())


def arms(m, P):
    for s, S in ((1, "R"), (-1, "L")):
        UA, FA, HA = "upperarm." + S, "forearm." + S, "hand." + S
        sh, el, ha = Vector(P(UA)), Vector(P(FA)), Vector(P(HA))
        tube(m, "suit", sh, el, 0.06, 0.052, UA, bulge=0.012)
        # наплечник: три налегающие ламы по форме плеча, сверху — жёлтая полоса отделения и заклёпки
        c = (sh.x + 0.03 * s, -0.01, 1.43)
        tilt = (0, R(-20) * s, 0)
        side = 0 if s > 0 else 180
        m.plate("armor", c, (0.115, 0.13, 0.1), around(side, 120), (R(25), R(88)), 0.026, rot=tilt, group=UA, res=(20, 8))
        m.plate("armor_dk", c, (0.125, 0.138, 0.108), around(side, 105), (R(2), R(30)), 0.022, rot=tilt, group=UA,
                res=(18, 4))
        m.plate("armor", c, (0.133, 0.142, 0.115), around(side, 95), (R(-24), R(6)), 0.02, rot=tilt, group=UA, res=(18, 4))
        m.plate("team", c, (0.119, 0.134, 0.104), around(side, 116), (R(30), R(38)), 0.006, rot=tilt, group=UA, res=(20, 1))
        m.rivets((c[0] + 0.07 * s, -0.08, 1.475), (c[0] + 0.07 * s, 0.06, 1.475), 3, "z", group=UA, key="metal", r=0.008)
        # налокотник
        d = (ha - el).normalized()
        m.plate("sand", tuple(el), (0.058, 0.062, 0.058), around(270, 75), (R(-40), R(55)), 0.016, rot=_rot_to(d),
                group=UA, res=(12, 6))
        m.sphere("armor_dk", 0.05, tuple(el), group=UA, seg=16, rings=8)
        # предплечье: подкостюм, наруч-манжета по оси предплечья, накладка сверху, кольцо на запястье
        tube(m, "suit", el, ha, 0.052, 0.044, FA, bulge=0.01)
        L = (ha - el).length
        mid = el.lerp(ha, 0.48)
        m.plate("armor", tuple(mid), (0.062, 0.062, L * 0.36), (0, TAU), (R(-62), R(62)), 0.016, rot=_rot_to(d), group=FA,
                res=(20, 6))
        m.plate("armor_lt", tuple(mid), (0.065, 0.065, L * 0.33), around(90, 30), (R(-55), R(55)), 0.008, rot=_rot_to(d),
                group=FA, res=(6, 5))
        m.plate("armor_dk", tuple(el.lerp(ha, 0.86)), (0.05, 0.05, 0.03), (0, TAU), (R(-50), R(50)), 0.01, rot=_rot_to(d),
                group=FA, res=(18, 2))
        # перчатка: кисть, костяшки бронированной полосой, четыре пальца обхватывают рукоять, большой палец
        m.sphere("suit", 0.045, tuple(ha), scale=(0.95, 1.15, 0.85), group=HA, seg=18, rings=10)
        m.box("armor_dk", (0.07, 0.035, 0.026), (ha.x, ha.y + 0.012, ha.z + 0.03), group=HA, bevel=0.01, seg=2)
        for k in range(4):
            y = ha.y + 0.03 - k * 0.018
            m.sphere("suit", 0.014, (ha.x - 0.032 * s, y, ha.z - 0.012), scale=(1.1, 0.9, 1.2), group=HA, seg=10, rings=6)
        m.sweep("suit", curve([ha + Vector((0.02 * s, 0.02, 0.0)), ha + Vector((0.035 * s, 0.045, -0.012)),
                               ha + Vector((0.02 * s, 0.06, -0.02))], 6), oval(10), taper(0.014, 0.011), group=HA)


# ---------------------------------------------------------------------------------------------- шлем

def helmet(m):
    Hd = "head"
    m.sphere("suit", 0.09, (0, 0.03, 1.66), group=Hd, seg=24, rings=12)
    m.sphere("armor", 0.13, (0, 0.0, 1.72), scale=(1, 1.12, 0.97), group=Hd, seg=36, rings=18)
    m.plate("armor_dk", (0, 0.0, 1.67), (0.135, 0.15, 0.125), around(270, 115), (R(-20), R(8)), 0.025, group=Hd, res=(24, 4))
    m.plate("armor", (0, -0.01, 1.66), (0.14, 0.155, 0.12), around(270, 70), (R(-48), R(-14)), 0.02, group=Hd, res=(16, 4))
    m.box("strap", (0.026, 0.19, 0.008), (0, -0.01, 1.845), group=Hd, bevel=0.003, seg=2)  # крепление чехла
    # лицевой щиток в оправе с болтами, полоса подсветки
    m.plate("armor_dk", (0, 0.0, 1.7), (0.134, 0.147, 0.125), around(90, 62), (R(-22), R(16)), 0.025, group=Hd, res=(24, 6))
    m.plate("visor", (0, 0.003, 1.7), (0.134, 0.148, 0.126), around(90, 52), (R(-14), R(11)), 0.01, group=Hd, res=(20, 6))
    m.plate("metal", (0, 0.003, 1.7), (0.135, 0.149, 0.127), around(90, 46), (R(-13), R(-11)), 0.003, group=Hd, res=(16, 1))
    for a in (R(90 - 58), R(90 + 58)):
        m.sphere("metal", 0.009, (0.134 * math.cos(a), 0.15 * math.sin(a), 1.7), group=Hd, seg=8, rings=5)
    # дыхательная маска с фильтрами, нащёчники
    m.box("armor_dk", (0.11, 0.08, 0.08), (0, 0.105, 1.6), group=Hd, bevel=0.028, seg=4, top=(0.75, 0.8))
    for x in (-0.032, 0.032):
        m.cyl("rubber", 0.018, 0.028, (x, 0.146, 1.595), FWD, group=Hd, seg=14, bevel=0.003)
        m.cyl("gunmetal", 0.012, 0.006, (x, 0.162, 1.595), FWD, group=Hd, seg=12, bevel=0.0)
    for s in (-1, 1):
        m.plate("armor", (0, 0.02, 1.64), (0.128, 0.12, 0.1), around(90 - 55 * s, 22), (R(-45), R(5)), 0.018, group=Hd,
                res=(6, 5))
        # ушные модули и боковые рельсы с прорезями
        m.cyl("armor_dk", 0.042, 0.04, (0.13 * s, 0.0, 1.69), SIDE, group=Hd, seg=20, bevel=0.012)
        m.cyl("gunmetal", 0.024, 0.05, (0.145 * s, 0.0, 1.69), SIDE, group=Hd, seg=14, bevel=0.003)
        m.box("gunmetal", (0.014, 0.12, 0.018), (0.128 * s, -0.01, 1.75), group=Hd, bevel=0.003)
        for y in (-0.05, -0.01, 0.03):
            m.box("rubber", (0.016, 0.006, 0.02), (0.128 * s, y, 1.75), group=Hd, bevel=0.0)
    # крепление ПНВ на лбу, налобный фонарь, антенна
    m.box("gunmetal", (0.05, 0.03, 0.035), (0, 0.13, 1.79), rot=(R(-30), 0, 0), group=Hd, bevel=0.008, seg=2)
    m.cyl("metal", 0.009, 0.06, (0, 0.142, 1.775), SIDE, group=Hd, seg=10, bevel=0.0)
    m.box("armor_dk", (0.028, 0.055, 0.028), (0.135, 0.06, 1.74), group=Hd, bevel=0.008)
    m.cyl("light", 0.009, 0.005, (0.135, 0.09, 1.74), FWD, group=Hd, seg=10, bevel=0.0)
    m.cyl("gunmetal", 0.005, 0.2, (-0.125, -0.05, 1.8), rot=(R(-12), 0, 0), group=Hd, seg=6, bevel=0.0)


def field_details(m, P):
    """Ткань и экипировка связывают бронепластины в человеческий силуэт."""
    for side, sign in (("L", -1), ("R", 1)):
        # Мягкие складки у коленей: поперечные валики ткани под пластинами.
        for k in range(3):
            z = 0.59 + k * 0.024
            band(m, "suit", (0.11 * sign, 0.015, z), (0.076 - k * 0.003, 0.079, 0.018),
                 0.008, 0.004, "thigh." + side, res=20)
        for k in range(3):
            a = Vector(P("upperarm." + side)).lerp(Vector(P("forearm." + side)), 0.62 + k * 0.09)
            m.sphere("suit", 0.052, tuple(a), scale=(1.03, 1.05, 0.22), group="upperarm." + side, seg=20, rings=8)
        # Швы брюк и лямки на спине читаются крупными спокойными линиями.
        m.seam((sign * 0.168, 0.005, 0.65), (sign * 0.164, 0.005, 0.89), (sign, 0, 0),
               group="thigh." + side, key="strap", width=0.005)
        m.box("strap", (0.026, 0.02, 0.25), (sign * 0.145, -0.145, 1.39), rot=(0, R(sign * 8), 0),
              group="chest", bevel=0.006, seg=2)
    # Радиостанция на левом плече, гибкий кабель и неброская маркировка отделения.
    m.box("gunmetal", (0.048, 0.032, 0.07), (-0.15, 0.14, 1.49), group="chest", bevel=0.01, seg=3)
    m.sweep("rubber", curve([(-0.15, 0.14, 1.52), (-0.18, 0.11, 1.57), (-0.17, -0.04, 1.53)], 12),
            oval(10), 0.006, group="chest")
    for x in (-0.05, 0.0, 0.05):
        m.box("white", (0.027, 0.004, 0.011), (x, 0.211, 1.47), group="chest", bevel=0.001)
