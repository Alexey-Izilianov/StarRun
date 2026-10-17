# Десантник мобильной пехоты в человеческих пропорциях (рост 1,85 м, шлем 0,26 м, плечи с наплечниками 0,66 м):
# оливковая броня поверх тёмного подкостюма, закрытый шлем с тёмным стеклом и подсветкой, ранец с баллонами.
# Подмеши: mar — винтовка на кости weapon (кисти держат её через IK), axe — топор в правой кисти (только в рубке).
# Запуск: blender.exe -b --python marine.py [-- --fast]
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style, rig, export
from style import around
from rig import RigModel, keyed, add

R = math.radians
FWD = (-math.pi / 2, 0, 0)         # цилиндр вдоль +Y
SIDE = (0, math.pi / 2, 0)         # цилиндр вдоль X
FAST = "--fast" in sys.argv

# Суставы: длина плеча и предплечья ~0,29 м, бедра и голени ~0,45 м; кисти — на рукояти и цевье винтовки
JOINTS = {
    "root": ((0, 0, 0), None),
    "hips": ((0, 0, 0.98), "root"),
    "spine": ((0, 0, 1.1), "hips"),
    "chest": ((0, 0, 1.25), "spine"),
    "head": ((0, 0.01, 1.58), "chest"),
    "thigh.R": ((0.1, 0, 0.98), "hips"), "shin.R": ((0.11, 0.03, 0.53), "thigh.R"), "foot.R": ((0.11, -0.02, 0.09), "shin.R"),
    "thigh.L": ((-0.1, 0, 0.98), "hips"), "shin.L": ((-0.11, 0.03, 0.53), "thigh.L"), "foot.L": ((-0.11, -0.02, 0.09), "shin.L"),
    "upperarm.R": ((0.21, -0.01, 1.46), "chest"), "forearm.R": ((0.25, -0.05, 1.2), "upperarm.R"),
    "hand.R": ((0.1, 0.12, 1.08), "forearm.R"),
    "upperarm.L": ((-0.21, -0.01, 1.46), "chest"), "forearm.L": ((-0.17, 0.15, 1.2), "upperarm.L"),
    "hand.L": ((0.04, 0.38, 1.17), "forearm.L"),
    "weapon": ((0.09, 0.1, 1.16), "chest"), "muzzle": ((0.09, 0.9, 1.165), "weapon"),
}
P = lambda b: JOINTS[b][0]


# ---------------------------------------------------------------------------------------------- ноги и таз

def legs(m):
    for s, S in ((1, "R"), (-1, "L")):
        x = 0.11 * s
        F, SH, TH = "foot." + S, "shin." + S, "thigh." + S
        # ботинок: подошва, носок с бронеплитой, голенище
        m.box("rubber", (0.15, 0.3, 0.05), (x, 0.05, 0.025), group=F, bevel=0.02, top=(0.95, 0.92))
        m.box("armor_dk", (0.14, 0.26, 0.1), (x, 0.06, 0.095), group=F, bevel=0.035, seg=3, top=(0.85, 0.7), shift=(0, -0.02))
        m.box("sand", (0.12, 0.1, 0.05), (x, 0.15, 0.13), rot=(R(-28), 0, 0), group=F, bevel=0.02, seg=2)
        m.box("armor_dk", (0.13, 0.14, 0.11), (x, -0.02, 0.17), group=F, bevel=0.03)
        # голень: подкостюм, наголенник с рёбрами, наколенник
        m.limb("suit", P(SH), (x, -0.01, 0.15), 0.065, 0.058, SH)
        m.box("armor", (0.14, 0.09, 0.32), (x, 0.06, 0.33), group=SH, bevel=0.03, seg=3, top=(1.05, 0.75), shift=(0, 0.01))
        for z in (0.25, 0.32, 0.39):
            m.box("armor_dk", (0.1, 0.02, 0.018), (x, 0.105, z), group=SH, bevel=0.004)
        m.box("armor_lt", (0.11, 0.07, 0.1), (x, 0.02, 0.27), group=SH, bevel=0.025, top=(1, 0.8), shift=(0, -0.015))
        m.plate("sand", (x, 0.02, 0.53), (0.085, 0.092, 0.08), around(90, 70), (R(-55), R(60)), 0.028, group=SH, res=(12, 6))
        m.sphere("armor_dk", 0.048, (x, 0.0, 0.53), group=SH, seg=12, rings=6)
        # бедро: подкостюм и изогнутая пластина спереди и снаружи
        m.limb("suit", P(TH), P(SH), 0.085, 0.068, TH)
        m.plate("armor", (x, 0.0, 0.75), (0.112, 0.105, 0.22), around(90, 110), (R(-50), R(45)), 0.03,
                group=TH, res=(14, 7))
        m.plate("armor_dk", (x + 0.008 * s, 0.0, 0.8), (0.116, 0.1, 0.18), around(0 if s > 0 else 180, 50), (R(-45), R(40)),
                0.025, group=TH, res=(8, 6))
    # кобура на правом бедре
    m.box("armor_dk", (0.055, 0.12, 0.17), (0.2, 0.0, 0.78), group="thigh.R", bevel=0.02)
    m.box("strap", (0.2, 0.02, 0.02), (0.13, 0.09, 0.83), group="thigh.R", bevel=0.004)


def waist(m):
    H, SP = "hips", "spine"
    m.box("suit", (0.3, 0.22, 0.18), (0, -0.01, 0.98), group=H, bevel=0.045, seg=3)
    m.box("gunmetal", (0.36, 0.25, 0.06), (0, -0.01, 1.04), group=H, bevel=0.02, seg=2)        # пояс
    m.box("metal", (0.06, 0.02, 0.04), (0, 0.125, 1.04), group=H, bevel=0.008)                  # пряжка
    m.box("armor", (0.13, 0.06, 0.14), (0, 0.105, 0.93), group=H, bevel=0.03, seg=3, top=(0.75, 1))  # щиток паха
    for x in (-0.13, 0.13):                                                                     # подсумки
        m.box("sand", (0.08, 0.07, 0.1), (x, 0.11, 0.995), group=H, bevel=0.02, seg=2)
        m.box("sand", (0.085, 0.075, 0.03), (x, 0.11, 1.055), group=H, bevel=0.01)
    for x in (-0.19, 0.19):                                                                     # набедренные щитки
        m.box("armor_dk", (0.05, 0.17, 0.17), (x, -0.01, 0.93), rot=(0, R(10) * (1 if x > 0 else -1), 0), group=H,
              bevel=0.02, top=(1, 0.8))
    for x in (-0.09, -0.03, 0.03):                                                              # гранаты сзади
        m.cyl("armor_dk", 0.028, 0.075, (x, -0.135, 1.01), group=H, seg=10, bevel=0.008)
        m.cyl("metal", 0.01, 0.025, (x, -0.135, 1.06), group=H, seg=6, bevel=0.0)
    # живот: подкостюм и две бронированные полосы
    m.box("suit", (0.28, 0.2, 0.18), (0, 0.0, 1.15), group=SP, bevel=0.05, seg=3)
    for i, z in enumerate((1.1, 1.16)):
        m.box("armor_dk", (0.25 - i * 0.01, 0.06, 0.06), (0, 0.09, z), group=SP, bevel=0.02, seg=2, top=(1.05, 1))


# ---------------------------------------------------------------------------------------------- корпус

def torso(m):
    C = "chest"
    # кираса: лофт из скруглённых сечений — от талии к плечам расширяется, грудь выдаётся вперёд
    def sect(z, w, d, fwd, k=10):
        pts = []
        for i in range(k * 4):
            a = 2 * math.pi * i / (k * 4)
            c, s = math.cos(a), math.sin(a)
            # суперэллипс: прямые бока и скруглённые углы
            x = math.copysign(abs(c) ** 0.55, c) * w
            y = math.copysign(abs(s) ** 0.55, s) * d + (fwd if s > 0 else 0) * s
            pts.append((x, y, z))
        return pts
    rings = [sect(1.22, 0.17, 0.12, 0.0), sect(1.29, 0.2, 0.13, 0.015), sect(1.37, 0.235, 0.14, 0.035),
             sect(1.45, 0.24, 0.14, 0.035), sect(1.51, 0.2, 0.13, 0.015), sect(1.55, 0.14, 0.1, 0.0)]
    m.loft("armor", rings, group=C, sharp=50)
    # нагрудник поверх: песочная пластина с рёбром посередине
    m.box("sand", (0.3, 0.04, 0.18), (0, 0.185, 1.4), rot=(R(-9), 0, 0), group=C, bevel=0.025, seg=3, top=(0.82, 1))
    m.box("armor_dk", (0.028, 0.025, 0.16), (0, 0.207, 1.4), rot=(R(-9), 0, 0), group=C, bevel=0.008)
    m.box("armor_lt", (0.23, 0.035, 0.045), (0, 0.165, 1.29), group=C, bevel=0.012, top=(0.9, 1))
    m.seam((-0.2, 0.135, 1.24), (0.2, 0.135, 1.24), (0, 1, 0), group=C, key="suit", width=0.015)
    for s in (-1, 1):                                                      # швы боковых пластин
        m.seam((0.235 * s, 0.05, 1.29), (0.235 * s, 0.05, 1.49), (s, 0, 0), group=C, key="suit", width=0.014)
    # горжет и воротник
    m.cyl("armor_dk", 0.1, 0.07, (0, 0.0, 1.56), group=C, seg=18, bevel=0.016, r2=0.085, scale=(1.15, 1, 1))
    m.cyl("suit", 0.06, 0.09, (0, 0.01, 1.6), group=C, seg=12, bevel=0.0)
    # ремни разгрузки через грудь
    for s in (-1, 1):
        m.box("strap", (0.038, 0.018, 0.28), (0.1 * s, 0.155, 1.39), rot=(R(-8), 0, R(-12) * s), group=C, bevel=0.004)
    m.box("sand", (0.08, 0.05, 0.07), (-0.11, 0.165, 1.3), group=C, bevel=0.016)              # подсумок рации
    m.cyl("gunmetal", 0.01, 0.1, (-0.08, 0.165, 1.37), group=C, seg=6, bevel=0.0)


def backpack(m):
    C = "chest"
    m.box("armor_dk", (0.32, 0.16, 0.38), (0, -0.2, 1.34), group=C, bevel=0.05, seg=3, top=(0.92, 0.85))
    m.box("armor", (0.28, 0.05, 0.3), (0, -0.29, 1.35), group=C, bevel=0.03, seg=2)
    for x in (-0.08, 0.08):                                                       # баллоны
        m.cyl("metal", 0.05, 0.36, (x, -0.335, 1.33), group=C, seg=14, bevel=0.015)
        m.cyl("gunmetal", 0.055, 0.035, (x, -0.335, 1.15), group=C, seg=14, bevel=0.008)
        m.cyl("gunmetal", 0.055, 0.035, (x, -0.335, 1.51), group=C, seg=14, bevel=0.008)
        m.cyl("team", 0.052, 0.025, (x, -0.335, 1.44), group=C, seg=14, bevel=0.0)
    m.vent((0, -0.24, 1.54), 0.2, 0.1, "z", slats=4, group=C, frame="gunmetal", slat="rubber")
    m.cyl("gunmetal", 0.008, 0.55, (-0.13, -0.24, 1.78), rot=(R(-8), 0, 0), group=C, seg=6, bevel=0.0)  # антенна
    m.cyl("gunmetal", 0.017, 0.05, (-0.13, -0.25, 1.53), group=C, seg=8, bevel=0.0)
    m.box("gunmetal", (0.035, 0.025, 0.02), (0.13, -0.29, 1.53), group=C, bevel=0.004)


# ---------------------------------------------------------------------------------------------- руки и плечи

def arms(m):
    for s, S in ((1, "R"), (-1, "L")):
        UA, FA, HA = "upperarm." + S, "forearm." + S, "hand." + S
        sh, el, ha = P(UA), P(FA), P(HA)
        m.limb("suit", sh, el, 0.058, 0.052, UA)
        # наплечник: скруглённый блок, наклонённый наружу, под ним — второй слой, по наружной грани —
        # жёлтая полоса отделения
        cx = sh[0] + 0.035 * s
        tilt = (0, R(-22) * s, 0)
        m.box("armor", (0.17, 0.21, 0.1), (cx, -0.01, 1.47), rot=tilt, group=UA, bevel=0.04, seg=4, top=(0.78, 0.84))
        m.box("armor_dk", (0.18, 0.22, 0.05), (cx + 0.015 * s, -0.01, 1.41), rot=tilt, group=UA, bevel=0.02, seg=2,
              top=(0.95, 0.97))
        m.box("team", (0.035, 0.2, 0.025), (cx + 0.068 * s, -0.01, 1.485), rot=(0, R(-58) * s, 0), group=UA, bevel=0.008)
        m.box("sand", (0.07, 0.08, 0.018), (cx - 0.007 * s, 0.02, 1.52), rot=tilt, group=UA, bevel=0.008)
        m.rivets((cx - 0.04 * s, -0.08, 1.505), (cx - 0.04 * s, 0.07, 1.505), 3, "z", group=UA, key="metal", r=0.01)
        m.sphere("armor_dk", 0.052, el, group=UA, seg=12, rings=6)                    # локоть
        m.limb("suit", el, ha, 0.052, 0.045, FA)
        # наруч: усечённая коробка вдоль предплечья
        d = (ha[0] - el[0], ha[1] - el[1], ha[2] - el[2])
        L = math.sqrt(sum(c * c for c in d))
        mid = tuple(el[i] + d[i] * 0.45 for i in range(3))
        yaw = math.atan2(-d[0], d[1])
        pitch = math.atan2(d[2], math.hypot(d[0], d[1]))
        m.box("armor", (0.095, L * 0.6, 0.085), mid, rot=(pitch, 0, yaw), group=FA, bevel=0.025, seg=3, top=(0.85, 1))
        m.box("armor_lt", (0.045, L * 0.4, 0.02), tuple(mid[i] + (0, 0, 0.045)[i] for i in range(3)), rot=(pitch, 0, yaw),
              group=FA, bevel=0.008)
        # перчатка
        m.sphere("suit", 0.046, ha, scale=(1, 1.2, 0.9), group=HA, seg=12, rings=6)
        m.box("armor_dk", (0.06, 0.045, 0.028), (ha[0], ha[1], ha[2] + 0.03), group=HA, bevel=0.01)


# ---------------------------------------------------------------------------------------------- шлем

def helmet(m):
    Hd = "head"
    m.sphere("suit", 0.09, (0, 0.03, 1.66), group=Hd, seg=14, rings=8)
    # купол: шлем 0,26 × 0,29 × 0,25 м; по низу — тёмный обод, сверху — гребень
    m.sphere("armor", 0.13, (0, 0.0, 1.72), scale=(1, 1.12, 0.97), group=Hd, seg=24, rings=12)
    m.plate("armor_dk", (0, 0.0, 1.67), (0.135, 0.15, 0.125), around(270, 115), (R(-20), R(8)), 0.025, group=Hd, res=(16, 3))
    m.box("armor", (0.035, 0.27, 0.035), (0, -0.01, 1.84), group=Hd, bevel=0.014, top=(0.6, 0.9))            # гребень
    m.box("sand", (0.018, 0.15, 0.01), (0, 0.03, 1.858), group=Hd, bevel=0.003)
    # лицевой щиток: крупное тёмное стекло в оправе, под ним тонкая полоска подсветки, ниже — дыхательная маска
    m.plate("armor_dk", (0, 0.0, 1.7), (0.134, 0.147, 0.125), around(90, 62), (R(-22), R(16)), 0.025, group=Hd, res=(16, 4))
    m.plate("visor", (0, 0.003, 1.7), (0.134, 0.148, 0.126), around(90, 52), (R(-14), R(11)), 0.01, group=Hd, res=(14, 4))
    m.plate("optic", (0, 0.003, 1.7), (0.135, 0.149, 0.127), around(90, 46), (R(-13), R(-10)), 0.005, group=Hd, res=(12, 1))
    m.box("armor_dk", (0.11, 0.08, 0.08), (0, 0.105, 1.6), group=Hd, bevel=0.026, seg=3, top=(0.75, 0.8))
    for x in (-0.03, 0.03):
        m.cyl("rubber", 0.017, 0.025, (x, 0.145, 1.595), FWD, group=Hd, seg=8, bevel=0.0)
    for s in (-1, 1):                                                                                      # ушные модули
        m.cyl("armor_dk", 0.04, 0.04, (0.13 * s, 0.0, 1.69), SIDE, group=Hd, seg=14, bevel=0.012)
        m.cyl("gunmetal", 0.022, 0.05, (0.145 * s, 0.0, 1.69), SIDE, group=Hd, seg=10, bevel=0.0)
    m.box("armor_dk", (0.028, 0.055, 0.028), (0.135, 0.06, 1.74), group=Hd, bevel=0.008)                   # налобный фонарь
    m.cyl("light", 0.009, 0.005, (0.135, 0.09, 1.74), FWD, group=Hd, seg=8, bevel=0.0)
    m.cyl("gunmetal", 0.005, 0.2, (-0.125, -0.05, 1.8), rot=(R(-12), 0, 0), group=Hd, seg=5, bevel=0.0)    # антенна


# ---------------------------------------------------------------------------------------------- оружие

def mar(m):
    """MAR длиной ~1 м: ствольная коробка с оливковыми щеками, кожух ствола с прорезями, подствольный гранатомёт,
    изогнутый магазин, короткий приклад, коллиматор. Пистолетная рукоять — под правой кистью, цевьё — под левой."""
    m.sub, W = "mar", "weapon"
    x, z = 0.09, 1.16
    m.box("gunmetal", (0.075, 0.42, 0.12), (x, 0.24, z), group=W, bevel=0.018, seg=2)                    # коробка
    m.box("armor", (0.078, 0.26, 0.06), (x, 0.26, z + 0.03), group=W, bevel=0.015)                       # оливковые щёки
    m.box("gunmetal", (0.065, 0.2, 0.1), (x, -0.02, z - 0.015), group=W, bevel=0.02, top=(1, 0.7))      # приклад
    m.box("rubber", (0.068, 0.03, 0.115), (x, -0.12, z - 0.02), group=W, bevel=0.01)
    m.cyl("armor_dk", 0.038, 0.28, (x, 0.58, z + 0.005), FWD, group=W, seg=14, bevel=0.008)               # кожух
    for y in (0.5, 0.57, 0.64):
        m.box("rubber", (0.079, 0.028, 0.016), (x, y, z + 0.035), group=W, bevel=0.0)
    m.cyl("gunmetal", 0.017, 0.16, (x, 0.78, z + 0.005), FWD, group=W, seg=10, bevel=0.0)                # ствол
    m.cyl("gunmetal", 0.026, 0.06, (x, 0.86, z + 0.005), FWD, group=W, seg=10, bevel=0.008)              # дульный тормоз
    m.cyl("gunmetal", 0.03, 0.32, (x, 0.56, z - 0.065), FWD, group=W, seg=14, bevel=0.008)               # подствольник
    m.cyl("rubber", 0.031, 0.025, (x, 0.71, z - 0.065), FWD, group=W, seg=14, bevel=0.0)
    m.box("gunmetal", (0.05, 0.085, 0.16), (x, 0.3, z - 0.115), rot=(R(16), 0, 0), group=W, bevel=0.015, top=(1, 0.85))  # магазин
    m.box("gunmetal", (0.046, 0.06, 0.1), (x, 0.12, z - 0.08), rot=(R(12), 0, 0), group=W, bevel=0.015)  # рукоять
    m.box("gunmetal", (0.045, 0.12, 0.055), (x, 0.28, z + 0.09), group=W, bevel=0.012)                  # коллиматор
    m.box("optic", (0.035, 0.01, 0.032), (x, 0.345, z + 0.094), group=W, bevel=0.0)
    m.box("team", (0.08, 0.04, 0.015), (x, 0.4, z + 0.045), group=W, bevel=0.0)
    m.sub = None


def axe(m):
    """Топор в правой кисти: рукоять вперёд из кулака, топорище с лезвием поперёк — в рубке рука заносит его."""
    m.sub, Hn = "axe", "hand.R"
    hx, hy, hz = P(Hn)
    m.cyl("wood", 0.016, 0.5, (hx, hy + 0.18, hz), FWD, group=Hn, seg=8, bevel=0.0)
    m.cyl("rubber", 0.019, 0.11, (hx, hy + 0.0, hz), FWD, group=Hn, seg=8, bevel=0.0)
    m.box("gunmetal", (0.04, 0.065, 0.065), (hx, hy + 0.4, hz), group=Hn, bevel=0.008)
    blade = [(0.0, -0.036), (0.16, -0.072), (0.176, 0.0), (0.16, 0.072), (0.0, 0.036)]   # профиль лезвия (x — от обуха)
    m.prism("metal", blade, -0.01, 0.01, group=Hn, matrix=_axe_frame(hx, hy + 0.4, hz))
    m.sub = None


def _axe_frame(x, y, z):
    from mathutils import Matrix
    # лезвие в вертикальной плоскости, остриём вниз: локальный X профиля → мир −Z
    return Matrix(((0, 0, 1, x), (0, 1, 0, y), (-1, 0, 0, z - 0.016), (0, 0, 0, 1)))


# ---------------------------------------------------------------------------------------------- анимации

def run_f():
    return rig.run_pose(stride=36, lean=10)


def run_b():
    base = rig.run_pose(stride=28, lean=-3)
    return lambda t: base(1 - t)            # бег спиной: тот же цикл в обратную сторону, корпус чуть назад


def strafe(side):
    """Бег боком: таз и ноги развёрнуты по ходу, грудь — обратно к прицелу."""
    base = rig.run_pose(stride=32, lean=6)
    turn = -60 if side == "R" else 60

    def fn(t):
        p = base(t)
        add(p, "hips", (0, turn, 0))
        add(p, "spine", (0, -turn * 0.45, 0))
        add(p, "chest", (0, -turn * 0.55, 0))
        return p
    return fn


def chop_pose():
    """Рубка: винтовка уходит к левому бедру, правая рука заносит топор над плечом и бьёт вперёд-вниз."""
    base = {"thigh.L": (14, 0, -4), "shin.L": (-18, 0, 0), "thigh.R": (-12, 0, 6), "shin.R": (-14, 0, 0),
            "hips@loc": (0, 0, -0.05), "weapon": (-55, 35, 0), "weapon@loc": (-0.18, -0.1, -0.22),
            "upperarm.L": (-10, 0, -15), "forearm.L": (40, 0, 0)}
    wind = dict(base)
    wind.update({"spine": (6, 18, 0), "chest": (8, 22, 0), "upperarm.R": (150, 0, 25), "forearm.R": (70, 0, 0),
                 "hand.R": (30, 0, 0)})
    hit = dict(base)
    hit.update({"spine": (-10, -12, 0), "chest": (-14, -16, 0), "upperarm.R": (40, 0, 5), "forearm.R": (5, 0, 0),
                "hand.R": (-35, 0, 0), "head": (6, 0, 0)})
    return keyed([(0.0, wind), (0.5, wind), (0.72, hit), (0.85, hit), (1.0, wind)])


def vault_pose():
    """Перелезание через окно или забор: толчок, ноги поджаты в полёте, приземление с пружиной."""
    stand = rig.aim_stance()
    crouch = dict(stand)
    crouch.update({"hips@loc": (0, 0, -0.1), "thigh.L": (35, 0, 0), "shin.L": (-60, 0, 0), "thigh.R": (30, 0, 0),
                   "shin.R": (-55, 0, 0), "spine": (-12, 0, 0)})
    air = dict(stand)
    air.update({"hips@loc": (0, 0.15, 0.32), "thigh.L": (75, 0, -10), "shin.L": (-110, 0, 0), "thigh.R": (60, 0, 10),
                "shin.R": (-100, 0, 0), "spine": (-14, 0, 0), "chest": (-6, 0, 0)})
    land = dict(crouch)
    land["hips@loc"] = (0, 0.3, -0.12)
    return keyed([(0.0, stand), (0.2, crouch), (0.5, air), (0.8, land), (1.0, stand)])


def actions(m):
    m.ik_arms = ("L", "R")
    m.action("idle", 90, rig.aim_pose())
    m.action("run_f", 20, run_f())
    m.action("run_b", 22, run_b())
    m.action("run_l", 20, strafe("L"))
    m.action("run_r", 20, strafe("R"))
    m.action("shoot", 10, rig.shoot_pose(kick=6, slide=0.05), loop=False)
    m.action("chop", 12, chop_pose(), ik=False)
    m.action("vault", 18, vault_pose(), loop=False)
    m.action("hit", 14, rig.hit_pose(), loop=False)
    m.action("die", 44, rig.death_pose(0.98), loop=False, ik=False)


def build():
    m = RigModel("marine", JOINTS, tex_size=512 if FAST else 1024)
    m.camo_scale = 8.0
    legs(m)
    waist(m)
    torso(m)
    backpack(m)
    arms(m)
    helmet(m)
    mar(m)
    axe(m)
    m.build()
    m.bake(samples=8 if FAST else 64)
    actions(m)
    export.export(m)
    return m


if __name__ == "__main__":
    style.reset()
    m = build()
    import preview
    preview.model_views(m, "marine", target=(0, 0.1, 1.0), dist=3.8, hide=("axe",))
    preview.anim_strip(m, "marine", ("run_f", "run_r", "chop", "vault", "die"), dist=4.4, target=(0, 0.1, 0.95),
                       show={"chop": ("body", "mar", "axe")})
    print("MARINE DONE", flush=True)
