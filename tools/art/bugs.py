# Жуки: рабочий (шесть паучьих ног, крупное сегментированное брюшко, жвалы-клешни) и воин (четыре высоких
# ноги-копья, пара серповидных хватательных лап, вытянутая голова с гребнем, шипы по спине).
# Ноги анимируются через IK: в цикле шага стопа идёт по земле назад (опора) и по дуге вперёд (перенос), а
# поворот тазика, бедра и голени считается по точке стопы. Клипы — матрицами (RigModel.skin_action).
# Запуск: blender.exe -b --python bugs.py [-- worker warrior] [--fast]
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathutils import Vector, Matrix
import style, export, preview
from style import curve, around, taper, oval, LENS
from rig import RigModel, smooth

R = math.radians
FAST = "--fast" in sys.argv
Z = Vector((0, 0, 1))


# ---------------------------------------------------------------------------------------------- математика позы

def about(point, axis, angle):
    """Поворот на angle вокруг оси axis, проходящей через point (4×4)."""
    p = Vector(point)
    return Matrix.Translation(p) @ Matrix.Rotation(angle, 4, Vector(axis)) @ Matrix.Translation(-p)


def _horiz(v):
    h = Vector((v.x, v.y, 0))
    return h.normalized() if h.length > 1e-6 else Vector((1, 0, 0))


def leg_ik(H, K, F, T):
    """Нога из трёх костей (тазик — поворот вокруг вертикали в H, бедро H→K, голень K→F) ставит стопу в T.
    Всё в осях покоя. Колено остаётся сверху. Возвращает локальные повороты (C, Fm, Tb) для цепочки D."""
    H, K, F, T = (Vector(v) for v in (H, K, F, T))
    u0, ut = _horiz(F - H), _horiz(T - H)
    yaw = math.atan2(u0.cross(ut).z, u0.dot(ut))
    Tl = H + Matrix.Rotation(-yaw, 3, Z) @ (T - H)             # цель в плоскости ноги в покое
    plane = lambda v: (v.dot(u0), v.z)
    k, f, t = plane(K - H), plane(F - H), plane(Tl - H)
    l1 = math.hypot(*k)
    l2 = math.hypot(f[0] - k[0], f[1] - k[1])
    d = min(max(math.hypot(*t), abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3)
    at = math.atan2(t[1], t[0])
    alpha = math.acos(max(-1.0, min(1.0, (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d))))
    ka = at + alpha                                            # колено над линией бедро — стопа
    kn = (l1 * math.cos(ka), l1 * math.sin(ka))
    tn = (d * math.cos(at), d * math.sin(at))
    ang = lambda v: math.atan2(v[1], v[0])
    th1 = ka - ang(k)
    th2 = ang((tn[0] - kn[0], tn[1] - kn[1])) - ang((f[0] - k[0], f[1] - k[1])) - th1
    n = u0.cross(Z)                                            # поворот на + уводит u0 к Z (вверх)
    return about(H, Z, yaw), about(H, n, th1), about(K, n, th2)


class Bug(RigModel):
    """Жук: тело (body), голова, брюшко, жвалы, ноги «coxa/femur/tibia.<n><L|R>», у воина — лапы-серпы."""

    def __init__(self, name, joints, legs, tex_size):
        super().__init__(name, joints, tex_size)
        self.legs = legs                       # имя ноги → (H, K, F) в покое

    def pose(self, body=Matrix(), feet=None, extra=None):
        """D-матрицы позы: body — смещение тела (голова, брюшко и ноги следуют), feet — {нога: точка стопы в мире},
        extra — {кость: локальный поворот в осях покоя} для головы, брюшка, жвал, лап."""
        extra = extra or {}
        D = {"root": Matrix(), "body": body}
        inv = body.inverted()
        for name, (bone, parent) in self.joints.items():
            if name in ("root", "body") or name.split(".")[0] in ("coxa", "femur", "tibia"):
                continue
            D[name] = D[parent] @ extra.get(name, Matrix())
        for leg, (H, K, F) in self.legs.items():
            T = Vector((feet or {}).get(leg, F))
            c, fm, tb = leg_ik(H, K, F, inv @ T)
            D["coxa." + leg] = body @ c
            D["femur." + leg] = D["coxa." + leg] @ fm
            D["tibia." + leg] = D["femur." + leg] @ tb
        return D


def leg_joints(legs):
    j = {}
    for leg, (H, K, F) in legs.items():
        j["coxa." + leg] = (H, "body")
        j["femur." + leg] = (H, "coxa." + leg)
        j["tibia." + leg] = (K, "femur." + leg)
    return j


def mirror(legs_r):
    """Ноги правой стороны (x > 0) → обе стороны."""
    out = {}
    for n, (H, K, F) in legs_r.items():
        out[n + "R"] = (H, K, F)
        out[n + "L"] = tuple((-p[0], p[1], p[2]) for p in (H, K, F))
    return out


def build_leg(m, leg, H, K, F, key, key_dk, r_femur, r_tibia, spikes=True):
    """Бедро — изогнутая гранёная трубка, сустав — тёмная перепонка, голень сужается в костяной коготь."""
    H, K, F = Vector(H), Vector(K), Vector(F)
    mid1 = (H + K) / 2 + Vector((0, 0, 0.04))
    m.sweep(key, curve([H, mid1, K], 7), oval(8), taper(r_femur, r_femur * 0.75), group="femur." + leg, sharp=60)
    m.sphere("membrane", r_femur * 0.9, tuple(K), group="tibia." + leg, seg=10, rings=6)
    for a, b, r in ((H + (K - H) * 0.82, K, r_femur * 0.85), (K, K + (F - K) * 0.12, r_tibia * 1.05)):
        m.sweep(key_dk, [a, b], oval(8), r * 1.15, group=("femur." if a != K else "tibia.") + leg)   # тёмные кольца
    m.sweep(key_dk, curve([H + Vector((0, 0, r_femur * 0.8)), mid1 + Vector((0, 0, r_femur * 0.85)),
                           K + Vector((0, 0, r_femur * 0.6))], 6), LENS, taper(r_femur * 0.5, r_femur * 0.2),
            r_femur * 0.35, up=(1, 0, 0), group="femur." + leg)                          # гребень бедра
    m.sphere(key_dk, r_femur * 1.05, tuple(H), group="coxa." + leg, seg=10, rings=6)
    out = _horiz(F - H)
    mid2 = K + (F - K) * 0.45 + out * 0.04
    m.sweep(key, curve([K, mid2, F + Vector((0, 0, 0.03))], 9), oval(8),
            taper(r_tibia, r_tibia * 0.45), group="tibia." + leg, sharp=60)
    m.sweep("claw", curve([F + Vector((0, 0, 0.05)), F], 3), oval(6), taper(r_tibia * 0.5, 0.0), group="tibia." + leg)
    if spikes:                                     # шипы по наружной стороне голени
        for k in (0.25, 0.5):
            p = K + (F - K) * k
            tip = p + out * 0.05 + Vector((0, 0, 0.03))
            m.sweep("claw_dk", [p, tip], oval(5), taper(r_tibia * 0.35, 0.0), group="tibia." + leg)


def gait(m, t, groups, stride, lift, duty=0.6, bob=0.015, sway=3.0, body_z=0.0):
    """Цикл бега: groups — список групп ног, ходящих вместе (их фазы равномерно сдвинуты)."""
    feet = {}
    for gi, group in enumerate(groups):
        p = (t + gi / len(groups)) % 1.0
        for leg in group:
            F = Vector(m.legs[leg][2])
            if p < duty:                                   # опора: стопа скользит назад по земле
                s = p / duty
                dy, dz = stride * (0.5 - s), 0.0
            else:                                          # перенос: дугой вперёд
                s = (p - duty) / (1 - duty)
                e = smooth(s)
                dy, dz = stride * (-0.5 + e), lift * math.sin(math.pi * s)
            feet[leg] = F + Vector((0, dy, dz))
    ph = 2 * math.pi * t * len(groups)
    body = Matrix.Translation((0, 0, body_z + bob * math.sin(ph))) @ \
        Matrix.Rotation(R(sway) * math.sin(2 * math.pi * t), 4, 'Z')
    return body, feet


# ---------------------------------------------------------------------------------------------- рабочий

WORKER_LEGS = mirror({
    "1": ((0.1, 0.16, 0.3), (0.34, 0.36, 0.5), (0.47, 0.56, 0.0)),
    "2": ((0.12, 0.05, 0.3), (0.4, 0.07, 0.52), (0.56, 0.08, 0.0)),
    "3": ((0.1, -0.06, 0.3), (0.36, -0.26, 0.5), (0.5, -0.46, 0.0)),
})
WORKER_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0.05, 0.3), "root"), "head": ((0, 0.24, 0.31), "body"),
                 "abdomen": ((0, -0.12, 0.32), "body"),
                 "mand.R": ((0.07, 0.42, 0.25), "head"), "mand.L": ((-0.07, 0.42, 0.25), "head")}
WORKER_JOINTS.update(leg_joints(WORKER_LEGS))


def carapace(m, key, key_dk, key_lt, center, radii, group, ridge=True):
    """Хитиновый панцирь: тёмное основание, сверху — светлый щит с килем."""
    cx, cy, cz = center
    m.sphere(key_dk, 1.0, center, scale=radii, group=group, seg=20, rings=10)
    m.plate(key, (cx, cy, cz), tuple(r * 1.04 for r in radii), (0, 2 * math.pi), (R(-5), R(80)), 0.03,
            group=group, res=(20, 6))
    if ridge:
        m.sweep(key_lt, [(cx, cy - radii[1] * 0.85, cz + radii[2] * 0.6), (cx, cy, cz + radii[2] * 1.05),
                         (cx, cy + radii[1] * 0.85, cz + radii[2] * 0.6)], LENS, 0.025, 0.012, up=(1, 0, 0), group=group)


def worker():
    m = Bug("worker", WORKER_JOINTS, WORKER_LEGS, 512 if FAST else 1024)
    m.camo_scale = 9.0
    m.decimate = 0.25
    C, Cd, Cl = "chitin", "chitin_dk", "chitin_lt"
    # грудь
    carapace(m, C, Cd, Cl, (0, 0.05, 0.31), (0.14, 0.16, 0.1), "body")
    m.sphere("belly", 0.11, (0, 0.05, 0.25), scale=(1.1, 1.3, 0.6), group="body", seg=14, rings=6)
    # брюшко: четыре налегающих кольца, к хвосту уже, по бокам — короткие шипы
    for i in range(4):
        y = -0.15 - i * 0.12
        w = 0.21 - i * 0.03
        h = 0.15 - i * 0.02
        z = 0.34 - i * 0.015
        m.sphere(Cd, 1.0, (0, y, z), scale=(w, 0.11, h), group="abdomen", seg=18, rings=9)
        m.plate(C, (0, y + 0.02, z), (w * 1.05, 0.12, h * 1.05), (0, 2 * math.pi), (R(-10), R(85)), 0.025,
                group="abdomen", res=(20, 6))
        m.plate(Cl, (0, y + 0.02, z), (w * 1.07, 0.122, h * 1.07), around(90, 80), (R(30), R(55)), 0.01,
                group="abdomen", res=(10, 2))
        for s in (-1, 1):
            base = Vector((s * w * 0.9, y, z + h * 0.35))
            m.sweep("claw_dk", [base, base + Vector((s * 0.05, -0.03, 0.05))], oval(5), taper(0.018, 0.0),
                    group="abdomen")
    m.sphere("belly", 1.0, (0, -0.32, 0.26), scale=(0.15, 0.3, 0.08), group="abdomen", seg=14, rings=6)
    m.sweep(Cd, curve([(0, -0.58, 0.32), (0, -0.66, 0.28), (0, -0.7, 0.2)], 5), oval(6), taper(0.04, 0.0),
            group="abdomen")                                                         # жало
    # голова: клиновидный щит с килем, глаза, жвалы-клешни
    m.loft(C, [[(x * 0.1, 0.24, 0.31 + z * 0.06) for x, z in oval(12)],
                [(x * 0.15, 0.32, 0.31 + z * 0.075) for x, z in oval(12)],
                [(x * 0.13, 0.4, 0.28 + z * 0.06) for x, z in oval(12)],
                [(x * 0.08, 0.46, 0.25 + z * 0.035) for x, z in oval(12)]], group="head", sharp=70)
    m.plate(Cd, (0, 0.33, 0.3), (0.152, 0.128, 0.083), around(90, 70), (R(-35), R(-5)), 0.02, group="head", res=(12, 4))
    m.sweep(Cl, [(0, 0.24, 0.39), (0, 0.33, 0.395), (0, 0.42, 0.35)], LENS, 0.022, 0.01, up=(1, 0, 0), group="head")
    for s in (-1, 1):                                                                 # надбровья
        m.sweep(Cl, [(s * 0.05, 0.36, 0.37), (s * 0.12, 0.38, 0.34)], LENS, 0.02, 0.012, group="head")
    for s in (-1, 1):
        m.sphere("eye", 0.018, (s * 0.1, 0.39, 0.325), group="head", seg=10, rings=6)
        m.sphere("eye", 0.012, (s * 0.075, 0.425, 0.305), group="head", seg=8, rings=5)
        mj = Vector((s * 0.07, 0.42, 0.25))
        pts = curve([mj, mj + Vector((s * 0.09, 0.1, -0.01)), mj + Vector((s * 0.04, 0.22, -0.02)),
                     mj + Vector((-s * 0.04, 0.26, -0.03))], 9)
        m.sweep("claw", pts, LENS, taper(0.03, 0.002), 0.018, group="mand." + ("R" if s > 0 else "L"))
        m.sweep("claw_dk", [mj + Vector((s * 0.05, 0.08, 0.0)), mj + Vector((s * 0.02, 0.12, -0.01))], oval(5),
                taper(0.012, 0.0), group="mand." + ("R" if s > 0 else "L"))
    for leg, (H, K, F) in WORKER_LEGS.items():
        build_leg(m, leg, H, K, F, C, Cd, 0.032, 0.025)
    m.build()
    m.bake(samples=8 if FAST else 64)
    worker_actions(m)
    export.export(m)
    return m


def jaws(m, opening):
    """Жвалы раскрываются наружу (поворот вокруг вертикали у основания)."""
    return {"mand.R": about(m.joints["mand.R"][0], Z, -opening), "mand.L": about(m.joints["mand.L"][0], Z, opening)}


def worker_actions(m):
    tri = [("1L", "2R", "3L"), ("1R", "2L", "3R")]

    def run(t):
        body, feet = gait(m, t, tri, stride=0.3, lift=0.1)
        ph = 2 * math.pi * t
        extra = jaws(m, R(10 + 8 * math.sin(2 * ph)))
        extra["head"] = about(m.joints["head"][0], Z, R(4) * math.sin(ph))
        extra["abdomen"] = about(m.joints["abdomen"][0], (1, 0, 0), R(4) * math.sin(2 * ph))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        body = Matrix.Translation((0, 0, 0.008 * math.sin(ph)))
        extra = jaws(m, R(6 + 10 * max(0.0, math.sin(3 * ph)) ** 4))
        extra["head"] = about(m.joints["head"][0], Z, R(10) * math.sin(ph))
        extra["abdomen"] = about(m.joints["abdomen"][0], (1, 0, 0), R(3) * math.sin(ph))
        return m.pose(body, None, extra)

    def bite(t):
        # замах головой назад с раскрытыми жвалами, рывок вперёд, щелчок
        k = smooth(t / 0.4) if t < 0.4 else (1 - smooth((t - 0.4) / 0.2) * 1.6 if t < 0.6 else -0.6 + smooth((t - 0.6) / 0.4) * 0.6)
        body = Matrix.Translation((0, -0.1 * k, 0.05 * max(0.0, k))) @ about((0, 0.05, 0.3), (1, 0, 0), R(14) * k)
        extra = jaws(m, R(5 + 45 * max(0.0, k)))
        extra["head"] = about(m.joints["head"][0], (1, 0, 0), R(18) * k)
        return m.pose(body, None, extra)

    def die(t):
        # подламывается, заваливается на бок, ноги поджимаются к брюху
        k = smooth(min(1.0, t / 0.6))
        body = Matrix.Translation((0, 0, -0.17 * k)) @ about((0, 0.05, 0.3), (0, 1, 0), R(35) * k)
        feet = {}
        for leg, (H, K, F) in m.legs.items():
            curl = Vector(H) * 1.6 + Vector((0, 0, 0.12))
            feet[leg] = Vector(F).lerp(body @ curl, k)
        extra = jaws(m, R(30) * k)
        extra["head"] = about(m.joints["head"][0], (1, 0, 0), R(-15) * k)
        return m.pose(body, feet, extra)

    m.skin_action("run", 16, run)
    m.skin_action("idle", 60, idle)
    m.skin_action("attack", 14, bite, loop=False)
    m.skin_action("die", 24, die, loop=False)


# ---------------------------------------------------------------------------------------------- воин

WARRIOR_LEGS = mirror({
    "2": ((0.14, 0.05, 0.6), (0.52, 0.18, 0.98), (0.72, 0.38, 0.0)),
    "3": ((0.13, -0.12, 0.6), (0.5, -0.36, 0.95), (0.68, -0.66, 0.0)),
})
WARRIOR_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0.0, 0.62), "root"), "head": ((0, 0.3, 0.66), "body"),
                  "abdomen": ((0, -0.2, 0.66), "body"),
                  "mand.R": ((0.08, 0.66, 0.5), "head"), "mand.L": ((-0.08, 0.66, 0.5), "head"),
                  "arm.R": ((0.14, 0.24, 0.62), "body"), "blade.R": ((0.3, 0.46, 0.92), "arm.R"),
                  "arm.L": ((-0.14, 0.24, 0.62), "body"), "blade.L": ((-0.3, 0.46, 0.92), "arm.L")}
WARRIOR_JOINTS.update(leg_joints(WARRIOR_LEGS))


def warrior(name="warrior", keys=("red_chitin", "red_chitin_dk", "red_chitin_lt"), elite=False):
    """Воин; elite — полубосс «Элитный воин»: та же стать и клипы, но в броне — рога, наплечники, щитки на лапах."""
    m = Bug(name, WARRIOR_JOINTS, WARRIOR_LEGS, 512 if FAST else 1024)
    m.camo_scale = 7.0
    m.decimate = 0.3 if not elite else 0.6
    C, Cd, Cl = keys
    # грудь — вытянутый щит, по хребту — шипы назад
    carapace(m, C, Cd, Cl, (0, 0.05, 0.64), (0.17, 0.27, 0.13), "body")
    m.sphere("belly", 1.0, (0, 0.05, 0.56), scale=(0.13, 0.22, 0.07), group="body", seg=14, rings=6)
    for i, y in enumerate((0.22, 0.08, -0.06)):
        base = Vector((0, y, 0.75))
        m.sweep("claw", curve([base, base + Vector((0, -0.08, 0.12 - i * 0.02)), base + Vector((0, -0.16, 0.16 - i * 0.03))], 5),
                LENS, taper(0.035, 0.0), 0.015, group="body")
    # брюшко: поднятое, из трёх колец, с шипами
    for i in range(3):
        y = -0.3 - i * 0.15
        w, h = 0.19 - i * 0.035, 0.14 - i * 0.025
        z = 0.7 + i * 0.03
        m.sphere(Cd, 1.0, (0, y, z), scale=(w, 0.17, h), group="abdomen", seg=18, rings=9)
        m.plate(C, (0, y + 0.03, z), (w * 1.05, 0.175, h * 1.05), (0, 2 * math.pi), (R(-10), R(85)), 0.025,
                group="abdomen", res=(20, 6))
        base = Vector((0, y, z + h))
        m.sweep("claw", [base, base + Vector((0, -0.07, 0.1))], LENS, taper(0.03, 0.0), 0.014, group="abdomen")
    # голова: длинный череп вперёд-вниз, гребень назад над грудью, четыре жвала
    m.loft(C, [[(x * 0.11, 0.32, 0.62 + z * 0.08) for x, z in oval(10)],
               [(x * 0.13, 0.46, 0.6 + z * 0.1) for x, z in oval(10)],
               [(x * 0.09, 0.62, 0.52 + z * 0.07) for x, z in oval(10)],
               [(x * 0.05, 0.7, 0.48 + z * 0.04) for x, z in oval(10)]], group="head", sharp=70)
    m.sweep(C, curve([(0, 0.62, 0.6), (0, 0.42, 0.74), (0, 0.18, 0.84)], 7), LENS, taper(0.02, 0.05, 1.5),
            taper(0.03, 0.012), up=(1, 0, 0), group="head")                                     # гребень
    for s in (-1, 1):
        m.sphere("eye", 0.022, (s * 0.1, 0.52, 0.64), group="head", seg=10, rings=6)
        S = "R" if s > 0 else "L"
        mj = Vector((s * 0.08, 0.66, 0.5))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.1, 0.08, -0.04)), mj + Vector((s * 0.03, 0.2, -0.08)),
                               mj + Vector((-s * 0.05, 0.22, -0.1))], 9), LENS, taper(0.035, 0.002), 0.02, group="mand." + S)
        m.sweep("claw_dk", curve([mj + Vector((-s * 0.04, 0.02, -0.03)), mj + Vector((-s * 0.02, 0.1, -0.1)),
                                  mj + Vector((-s * 0.05, 0.14, -0.12))], 5), oval(5), taper(0.016, 0.0), group="head")
        # лапа-серп: плечо вверх-вперёд к локтю, от локтя — изогнутое лезвие вниз-вперёд
        sh, el = Vector((s * 0.14, 0.24, 0.62)), Vector((s * 0.3, 0.46, 0.92))
        m.sweep(C, curve([sh, (sh + el) / 2 + Vector((s * 0.03, 0, 0.04)), el], 7), oval(8), taper(0.045, 0.035),
                group="arm." + S, sharp=60)
        m.sphere("membrane", 0.04, tuple(el), group="blade." + S, seg=10, rings=6)
        blade = curve([el, el + Vector((s * 0.02, 0.18, 0.02)), el + Vector((-s * 0.02, 0.36, -0.14)),
                       el + Vector((-s * 0.05, 0.42, -0.38))], 11)
        m.sweep("claw", blade, LENS, taper(0.055, 0.004), 0.03, up=(s, 0, 0), group="blade." + S)
        m.sweep("claw_dk", [p + Vector((0, -0.01, 0.02)) for p in blade[:6]], LENS, taper(0.03, 0.0), 0.012,
                up=(s, 0, 0), group="blade." + S)
    for leg, (H, K, F) in WARRIOR_LEGS.items():
        build_leg(m, leg, H, K, F, C, Cd, 0.044, 0.035)
    if elite:
        elite_armor(m, C, Cd, Cl)
    m.build()
    m.bake(samples=8 if FAST else 64)
    warrior_actions(m)
    export.export(m)
    return m


def elite_armor(m, C, Cd, Cl):
    for s in (-1, 1):
        S = "R" if s > 0 else "L"
        # наплечник: щиток над основанием лапы с тремя шипами наружу
        m.plate(Cl, (s * 0.14, 0.18, 0.66), (0.14, 0.13, 0.1), (R(90 - 90 * s - 60), R(90 - 90 * s + 60)),
                (R(-5), R(70)), 0.025, group="body", res=(10, 5))
        for i in range(3):
            base = Vector((s * (0.2 + 0.03 * i), 0.24 - 0.07 * i, 0.72))
            m.sweep("claw", curve([base, base + Vector((s * 0.07, -0.03, 0.06)), base + Vector((s * 0.12, -0.08, 0.08))], 4),
                    oval(6), taper(0.024, 0.0), group="body")
        # рога: от затылка назад-вверх, по сторонам гребня
        m.sweep("claw", curve([(s * 0.07, 0.44, 0.7), (s * 0.13, 0.32, 0.86), (s * 0.12, 0.14, 0.98)], 8), oval(8),
                taper(0.035, 0.0), group="head")
        # щиток на плече лапы-серпа
        sh, el = Vector((s * 0.14, 0.24, 0.62)), Vector((s * 0.3, 0.46, 0.92))
        m.sweep(Cl, [sh.lerp(el, 0.2) + Vector((s * 0.03, 0, 0.03)), sh.lerp(el, 0.85) + Vector((s * 0.03, 0, 0.03))],
                LENS, 0.05, 0.03, up=(s, 0, 1), group="arm." + S)
    for i, y in enumerate((-0.3, -0.45, -0.6)):              # пары шипов по брюшку, по сторонам от гребня
        for s in (-1, 1):
            base = Vector((s * 0.08, y, 0.81))
            m.sweep("claw", curve([base, base + Vector((s * 0.04, -0.05, 0.09)), base + Vector((s * 0.06, -0.12, 0.13))], 4),
                    oval(6), taper(0.03, 0.0), group="abdomen")


def scythes(m, lift, spread=0.0):
    """Серпы: lift — подъём плеча (+ вверх-назад), spread — лезвия раскрыты (+) или сложены (−)."""
    out = {}
    for S, s in (("R", 1), ("L", -1)):
        out["arm." + S] = about(m.joints["arm." + S][0], (1, 0, 0), lift) @ about(m.joints["arm." + S][0], Z, -s * spread * 0.4)
        out["blade." + S] = about(m.joints["blade." + S][0], (1, 0, 0), spread)
    return out


def warrior_actions(m):
    diag = [("2L", "3R"), ("2R", "3L")]

    def run(t):
        body, feet = gait(m, t, diag, stride=0.46, lift=0.16, duty=0.55, bob=0.03, sway=4)
        ph = 2 * math.pi * t
        extra = scythes(m, R(6 * math.sin(2 * ph)), R(-10))
        extra.update(jaws(m, R(8 + 6 * math.sin(2 * ph))))
        extra["abdomen"] = about(m.joints["abdomen"][0], (1, 0, 0), R(5) * math.sin(2 * ph))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        body = Matrix.Translation((0, 0, 0.012 * math.sin(ph)))
        extra = scythes(m, R(4 * math.sin(ph)), R(-12))
        extra.update(jaws(m, R(5 + 12 * max(0.0, math.sin(2 * ph)) ** 6)))
        extra["head"] = about(m.joints["head"][0], Z, R(12) * math.sin(ph))
        return m.pose(body, None, extra)

    def windup(t):
        # встаёт на дыбы, серпы взлетают вверх-назад и раскрываются
        k = smooth(t)
        body = about((0, 0, 0.62), (1, 0, 0), R(14) * k) @ Matrix.Translation((0, -0.06 * k, 0.06 * k))
        extra = scythes(m, R(70) * k, R(-10 + 55 * k))
        extra.update(jaws(m, R(10 + 25 * k)))
        return m.pose(body, None, extra)

    def strike(t):
        # рушит серпы вниз-вперёд, тело бросается следом, затем возвращается
        k = smooth(min(1.0, t / 0.35))
        back = smooth(max(0.0, (t - 0.5) / 0.5))
        lean = R(14) * (1 - k) - R(16) * k * (1 - back)
        body = about((0, 0, 0.62), (1, 0, 0), lean) @ Matrix.Translation((0, 0.12 * k * (1 - back), -0.04 * k * (1 - back)))
        extra = scythes(m, R(70) * (1 - k) - R(50) * k * (1 - back) + R(0) * back, R(45) * (1 - k) - R(25) * k * (1 - back) - R(10) * back)
        extra.update(jaws(m, R(35) * (1 - k) + R(8)))
        return m.pose(body, None, extra)

    def die(t):
        k = smooth(min(1.0, t / 0.6))
        body = Matrix.Translation((0, 0, -0.4 * k)) @ about((0, 0, 0.62), (0, 1, 0), R(-30) * k)
        feet = {leg: Vector(F).lerp(body @ (Vector(H) * 1.7 + Vector((0, 0, 0.2))), k) for leg, (H, K, F) in m.legs.items()}
        extra = scythes(m, R(-30) * k, R(-40) * k)
        extra.update(jaws(m, R(30) * k))
        return m.pose(body, feet, extra)

    m.skin_action("run", 20, run)
    m.skin_action("idle", 60, idle)
    m.skin_action("windup", 24, windup, loop=False)
    m.skin_action("strike", 10, strike, loop=False)
    m.skin_action("die", 26, die, loop=False)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    which = [a for a in args if not a.startswith("--")] or ["worker", "warrior"]
    for name in which:
        style.reset()
        if name == "worker":
            m = worker()
            preview.model_views(m, "worker", target=(0, 0, 0.25), dist=2.6)
            preview.anim_strip(m, "worker", ("run", "attack", "die"), dist=2.8, target=(0, 0, 0.25), show={})
        else:
            m = warrior()
            preview.model_views(m, "warrior", target=(0, 0.1, 0.5), dist=3.6)
            preview.anim_strip(m, "warrior", ("run", "windup", "strike", "die"), dist=3.8, target=(0, 0.1, 0.5), show={})
    print("BUGS DONE", flush=True)
