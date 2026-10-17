# Выводок: стрелок (мешок кислоты над спиной, плюёт из трубки-хоботка), летающий (оса на четырёх крыльях с
# жалом), танкер (низкий бронированный жук с рогом-тараном и чешуёй брюшка) и мозговой (раздутый мозг на
# паучьих ножках, светящиеся узлы ауры). Скелеты и ноги — как у рабочего (bugs.py): IK по точке стопы.
# Запуск: blender.exe -b --python brood.py [-- spitter flyer tanker brain] [--fast]
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathutils import Vector, Matrix
import style, export, preview
from style import curve, taper, oval, LENS
from rig import smooth
from bugs import Bug, about, mirror, leg_joints, build_leg, gait, carapace, jaws, FAST, Z

R = math.radians
X = Vector((1, 0, 0))
TRIPOD = [("1L", "2R", "3L"), ("1R", "2L", "3R")]


def scaled(point, k):
    """Масштаб k вокруг точки: раздувание мешка, пульс мозга."""
    p = Vector(point)
    return Matrix.Translation(p) @ Matrix.Scale(k, 4) @ Matrix.Translation(-p)


def on_ellipsoid(c, r, along, side, k=1.0):
    """Точка на эллипсоиде (центр c, полуоси r): along ∈ [−1, 1] — от хвоста к голове (по Y), side — угол
    от макушки вбок, радианы (+ вправо). k > 1 — чуть над поверхностью."""
    ring = math.sqrt(max(0.0, 1 - along * along))
    return Vector((c[0] + r[0] * ring * math.sin(side) * k, c[1] + r[1] * along * k,
                   c[2] + r[2] * ring * math.cos(side) * k))


def band(m, key, c, r, along, half_width, thick, group, spread=80, res=14):
    """Хитиновый пояс поперёк эллипсоида (как обруч через спину): долгота пластины — вокруг продольной оси."""
    lat = math.asin(max(-0.99, min(0.99, along)))
    dl = half_width / r[1]
    # полюс пластины смотрит вдоль +Y: широта — положение пояса вдоль тела, долгота −90° — макушка
    m.plate(key, c, (r[0], r[2], r[1]), (R(-90 - spread), R(-90 + spread)), (lat - dl, lat + dl), thick,
            rot=(R(-90), 0, 0), group=group, res=(res, 2))


def dying_legs(m, body, k, lift=0.12, tuck=1.6):
    """Ноги поджимаются к брюху: стопа идёт из покоя к точке под тазиком."""
    return {leg: Vector(F).lerp(body @ (Vector(H) * tuck + Vector((0, 0, lift))), k) for leg, (H, K, F) in m.legs.items()}


# ---------------------------------------------------------------------------------------------- стрелок

SPITTER_LEGS = mirror({
    "1": ((0.09, 0.12, 0.28), (0.3, 0.32, 0.46), (0.42, 0.5, 0.0)),
    "2": ((0.11, 0.03, 0.28), (0.36, 0.05, 0.48), (0.52, 0.06, 0.0)),
    "3": ((0.09, -0.06, 0.28), (0.32, -0.24, 0.46), (0.46, -0.42, 0.0)),
})
SPITTER_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0.03, 0.28), "root"), "head": ((0, 0.17, 0.3), "body"),
                  "abdomen": ((0, -0.12, 0.34), "body"),
                  "mand.R": ((0.05, 0.32, 0.24), "head"), "mand.L": ((-0.05, 0.32, 0.24), "head")}
SPITTER_JOINTS.update(leg_joints(SPITTER_LEGS))
SAC_C, SAC_R = (0, -0.38, 0.47), (0.22, 0.26, 0.21)


def spitter():
    m = Bug("spitter", SPITTER_JOINTS, SPITTER_LEGS, 512 if FAST else 1024)
    m.camo_scale = 9.0
    m.decimate = 0.3
    G, Gd, Gl = "green_chitin", "green_chitin_dk", "green_chitin_lt"
    carapace(m, G, Gd, Gl, (0, 0.03, 0.29), (0.12, 0.14, 0.09), "body")
    m.sphere("belly", 0.1, (0, 0.03, 0.24), scale=(1.0, 1.2, 0.55), group="body", seg=14, rings=6)
    # мешок: стебелёк, тёмное основание, глянцевый пузырь, хитиновые обручи, светящиеся пятна кислоты
    m.sweep("membrane", [(0, -0.06, 0.3), (0, -0.16, 0.38)], oval(10), 0.05, group="abdomen")
    m.sphere(Gd, 1.0, (0, -0.36, 0.35), scale=(0.17, 0.22, 0.12), group="abdomen", seg=16, rings=8)
    m.sphere("sac", 1.0, SAC_C, scale=SAC_R, group="abdomen", seg=24, rings=14)
    for i, along in enumerate((0.62, 0.3, -0.05, -0.42)):
        r = tuple(v * 1.03 for v in SAC_R)
        band(m, G, SAC_C, r, along, 0.03 - i * 0.003, 0.022, "abdomen", spread=100 - i * 5)
        band(m, Gl, SAC_C, tuple(v * 1.045 for v in SAC_R), along + 0.06, 0.008, 0.008, "abdomen", spread=60)
    for along, side, rad in ((0.45, 0.35, 0.03), (0.45, -0.4, 0.025), (0.12, 0.05, 0.035), (0.14, 0.75, 0.022),
                             (0.15, -0.7, 0.026), (-0.22, 0.4, 0.03), (-0.25, -0.3, 0.024), (-0.6, 0.0, 0.022)):
        m.sphere("acid", rad, tuple(on_ellipsoid(SAC_C, SAC_R, along, side, 0.99)), scale=(1, 1, 0.7),
                 group="abdomen", seg=10, rings=6)
    # голова: короткий купол, трубка-хоботок со светящейся кромкой, малые жвалы
    m.loft(G, [[(x * 0.08, 0.17, 0.3 + z * 0.06) for x, z in oval(12)],
               [(x * 0.12, 0.24, 0.3 + z * 0.07) for x, z in oval(12)],
               [(x * 0.1, 0.31, 0.28 + z * 0.06) for x, z in oval(12)],
               [(x * 0.06, 0.35, 0.27 + z * 0.04) for x, z in oval(12)]], group="head", sharp=70)
    m.sweep(Gl, [(0, 0.18, 0.36), (0, 0.26, 0.375), (0, 0.32, 0.34)], LENS, 0.018, 0.01, up=(1, 0, 0), group="head")
    nozzle = curve([(0, 0.3, 0.29), (0, 0.38, 0.31), (0, 0.44, 0.345)], 6)
    m.sweep(Gd, nozzle, oval(10), taper(0.045, 0.036), group="head")
    for k in (0.3, 0.65):                                                           # кольца трубки
        p = nozzle[0].lerp(nozzle[-1], k)
        m.sweep(G, [p - Vector((0, 0.012, 0.004)), p + Vector((0, 0.012, 0.004))], oval(10), 0.047, group="head")
    m.torus("acid", 0.033, 0.009, (0, 0.444, 0.347), rot=(R(-63), 0, 0), group="head", seg=16)
    m.sphere("acid", 0.026, (0, 0.438, 0.344), scale=(1, 0.4, 1), group="head", seg=10, rings=6)
    for s in (-1, 1):
        S = "R" if s > 0 else "L"
        m.sphere("eye", 0.017, (s * 0.085, 0.27, 0.33), group="head", seg=10, rings=6)
        m.sphere("eye", 0.011, (s * 0.06, 0.305, 0.322), group="head", seg=8, rings=5)
        mj = Vector((s * 0.05, 0.32, 0.24))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.05, 0.06, -0.01)), mj + Vector((s * 0.01, 0.12, -0.03))], 7),
                LENS, taper(0.022, 0.002), 0.014, group="mand." + S)
    for leg, (H, K, F) in SPITTER_LEGS.items():
        build_leg(m, leg, H, K, F, G, Gd, 0.028, 0.022)
    m.build()
    m.bake(samples=8 if FAST else 64)
    spitter_actions(m)
    export.export(m)
    return m


def spitter_actions(m):
    A = m.joints["abdomen"][0]
    B = m.joints["body"][0]
    Hd = m.joints["head"][0]

    def sac(k, pitch=0.0):
        return {"abdomen": about(A, X, pitch) @ scaled(A, k)}

    def run(t):
        body, feet = gait(m, t, TRIPOD, stride=0.28, lift=0.09)
        ph = 2 * math.pi * t
        extra = sac(1.0 + 0.02 * math.sin(2 * ph), R(5) * math.sin(2 * ph))
        extra.update(jaws(m, R(8 + 6 * math.sin(2 * ph))))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        extra = sac(1.0 + 0.035 * math.sin(2 * ph))
        extra["head"] = about(Hd, Z, R(12) * math.sin(ph))
        extra.update(jaws(m, R(5 + 8 * max(0.0, math.sin(3 * ph)) ** 4)))
        return m.pose(Matrix.Translation((0, 0, 0.006 * math.sin(2 * ph))), None, extra)

    def windup(t):
        # встаёт на задние ноги, мешок раздувается и подрагивает, голова целится вперёд
        k = smooth(t)
        body = about(B, X, R(14) * k) @ Matrix.Translation((0, -0.05 * k, 0.03 * k))
        extra = sac(1.0 + 0.24 * k + 0.02 * k * math.sin(6 * math.pi * t), R(-10) * k)
        extra["head"] = about(Hd, X, R(-10) * k)
        extra.update(jaws(m, R(10 + 25 * k)))
        return m.pose(body, None, extra)

    def attack(t):
        # плевок: мешок резко сжимается, тело дёргается вперёд, голова — следом
        squeeze = 1.24 - 0.38 * smooth(min(1.0, t / 0.25)) + 0.14 * smooth(max(0.0, (t - 0.3) / 0.7))
        k = smooth(min(1.0, t / 0.2)) * (1 - smooth(max(0.0, (t - 0.35) / 0.65)))
        rear = R(14) * (1 - smooth(min(1.0, t / 0.2)))
        body = about(B, X, rear - R(8) * k) @ Matrix.Translation((0, 0.04 * k, 0))
        extra = sac(squeeze, R(-10) * (1 - k))
        extra["head"] = about(Hd, X, R(-16) * k)
        extra.update(jaws(m, R(35 - 20 * k)))
        return m.pose(body, None, extra)

    def die(t):
        k = smooth(min(1.0, t / 0.6))
        body = Matrix.Translation((0, 0, -0.17 * k)) @ about(B, (0, 1, 0), R(-30) * k)
        extra = sac(1.0 - 0.22 * k, R(12) * k)
        extra["head"] = about(Hd, X, R(-15) * k)
        extra.update(jaws(m, R(30) * k))
        return m.pose(body, dying_legs(m, body, k), extra)

    m.skin_action("run", 16, run)
    m.skin_action("idle", 60, idle)
    m.skin_action("windup", 14, windup, loop=False)
    m.skin_action("attack", 12, attack, loop=False)
    m.skin_action("die", 24, die, loop=False)


# ---------------------------------------------------------------------------------------------- летающий

FLYER_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0, 0.2), "root"), "head": ((0, 0.11, 0.21), "body"),
                "abdomen": ((0, -0.09, 0.2), "body"),
                "wing.FR": ((0.05, 0.05, 0.26), "body"), "wing.FL": ((-0.05, 0.05, 0.26), "body"),
                "wing.BR": ((0.045, -0.03, 0.25), "body"), "wing.BL": ((-0.045, -0.03, 0.25), "body"),
                "mand.R": ((0.03, 0.22, 0.17), "head"), "mand.L": ((-0.03, 0.22, 0.17), "head")}
WING = [(0.0, 0.025), (0.1, 0.065), (0.24, 0.085), (0.38, 0.07), (0.49, 0.03), (0.52, -0.005), (0.46, -0.035),
        (0.3, -0.05), (0.14, -0.04), (0.0, -0.02)]
WINGS = {"F": (1.0, R(-8)), "B": (0.72, R(-34))}           # длина и разворот назад


def _wing_frame(name):
    """Матрица крыла: из плоскости XY (корень в нуле, крыло вдоль +X) — к суставу, с разворотом назад."""
    pair, side = name[-2], name[-1]
    s = 1 if side == "R" else -1
    length, sweep = WINGS[pair]
    J = Vector(FLYER_JOINTS[name][0])
    return Matrix.Translation(J) @ Matrix.Rotation(s * sweep, 4, 'Z') @ Matrix.Diagonal((s * length, length, 1, 1))


def flyer():
    m = Bug("flyer", FLYER_JOINTS, {}, 512 if FAST else 1024)
    m.camo_scale = 10.0
    m.decimate = 0.35
    S, Sd, Sl = "sand_chitin", "sand_chitin_dk", "sand_chitin_lt"
    # грудь
    m.sphere(Sd, 1.0, (0, 0, 0.2), scale=(0.075, 0.1, 0.07), group="body", seg=16, rings=8)
    m.plate(S, (0, 0.0, 0.2), (0.079, 0.105, 0.074), (0, 2 * math.pi), (R(5), R(85)), 0.02, group="body", res=(16, 5))
    m.sweep(Sl, [(0, -0.07, 0.26), (0, 0.0, 0.278), (0, 0.08, 0.255)], LENS, 0.016, 0.008, up=(1, 0, 0), group="body")
    # голова: крупные фасеточные глаза, усики, короткие жвалы
    m.sphere(S, 1.0, (0, 0.15, 0.2), scale=(0.058, 0.06, 0.052), group="head", seg=14, rings=8)
    for s in (-1, 1):
        m.sphere("eye", 0.036, (s * 0.042, 0.165, 0.22), scale=(0.8, 1.0, 1.1), group="head", seg=12, rings=8)
        m.sweep(Sd, curve([(s * 0.018, 0.19, 0.245), (s * 0.05, 0.26, 0.3), (s * 0.09, 0.33, 0.3)], 6), oval(5),
                taper(0.007, 0.003), group="head")
        S_ = "R" if s > 0 else "L"
        mj = Vector((s * 0.03, 0.2, 0.17))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.03, 0.04, -0.01)), mj + Vector((0, 0.07, -0.02))], 5), LENS,
                taper(0.014, 0.002), 0.009, group="mand." + S_)
    # брюшко-хвост: сужается и загибается вниз к жалу, тёмные полосы
    tail = curve([(0, -0.07, 0.2), (0, -0.22, 0.2), (0, -0.37, 0.15), (0, -0.47, 0.07)], 12)
    rad = taper(0.068, 0.02, 1.3)
    m.sweep(S, tail, oval(12), rad, group="abdomen")
    for a, b in ((2, 4), (5, 7), (8, 10)):
        seg = tail[a:b + 1]
        m.sweep(Sd, seg, oval(12), lambda t, a=a, b=b: rad((a + (b - a) * t) / 11) * 1.06, group="abdomen")
    m.sweep("claw", [tail[-1], tail[-1] + Vector((0, 0.02, -0.07))], oval(6), taper(0.016, 0.0), group="abdomen")
    # ноги поджаты под грудь: на лету не ступают, IK не нужен
    for s in (-1, 1):
        for i in range(3):
            hip = Vector((s * 0.045, 0.04 - 0.045 * i, 0.16))
            knee = Vector((s * 0.11, 0.05 - 0.06 * i, 0.12))
            foot = Vector((s * 0.09, 0.0 - 0.09 * i, 0.03))
            m.sweep(Sd, curve([hip, knee, foot], 6), oval(6), taper(0.012, 0.005), group="body")
    # крылья: тонкие пластины с жилками, каждое на своей кости
    for name in ("wing.FR", "wing.FL", "wing.BR", "wing.BL"):
        M = _wing_frame(name)
        m.prism("wing", WING, -0.003, 0.003, group=name, matrix=M)
        for vein in ([(0.0, 0.02), (0.24, 0.078), (0.5, 0.02)], [(0.0, 0.0), (0.2, 0.01), (0.44, -0.01)],
                     [(0.02, -0.015), (0.16, -0.03), (0.3, -0.042)]):
            pts = [M @ Vector((x, y, 0.004)) for x, y in vein]
            m.sweep(Sd, curve(pts, 6), oval(5), taper(0.006, 0.003), group=name)
    m.build()
    m.bake(samples=8 if FAST else 64)
    flyer_actions(m)
    export.export(m)
    return m


def flyer_actions(m):
    A = m.joints["abdomen"][0]
    B = m.joints["body"][0]

    def wings(phase, amp=R(50), bias=R(8), fold=0.0):
        """Взмах: крыло поворачивается вокруг оси вдоль своего корня; fold — сложены назад вдоль тела."""
        out = {}
        for name in ("wing.FR", "wing.FL", "wing.BR", "wing.BL"):
            pair, side = name[-2], name[-1]
            s = 1 if side == "R" else -1
            J = Vector(m.joints[name][0])
            sweep = WINGS[pair][1]
            axis = Matrix.Rotation(s * sweep, 3, 'Z') @ Vector((0, 1, 0))
            ph = phase + (0.0 if pair == "F" else 0.2)
            flap = (bias + amp * math.sin(2 * math.pi * ph)) * (1 - fold)
            out[name] = about(J, Z, -s * R(70) * fold) @ about(J, axis, -s * flap)
        return out

    def fly(t):
        ph = 2 * math.pi * t
        body = Matrix.Translation((0, 0, 0.025 * math.sin(ph + math.pi / 2)))
        extra = wings(2 * t)
        extra["abdomen"] = about(A, X, R(7) * math.sin(ph))
        extra.update(jaws(m, R(10 + 8 * math.sin(2 * ph))))
        return m.pose(body, None, extra)

    def attack(t):
        # жалит: тело клюёт вперёд-вниз, хвост подгибается под брюхо и выбрасывает жало вперёд
        k = smooth(min(1.0, t / 0.4)) * (1 - smooth(max(0.0, (t - 0.6) / 0.4)))
        body = about(B, X, R(-20) * k) @ Matrix.Translation((0, 0.05 * k, -0.04 * k))
        extra = wings(3 * t, amp=R(40))
        extra["abdomen"] = about(A, X, R(70) * k)
        extra.update(jaws(m, R(30) * k))
        return m.pose(body, None, extra)

    def die(t):
        # крылья складываются назад, тело кувыркается на спину
        k = smooth(min(1.0, t / 0.5))
        body = about(B, (0, 1, 0), R(150) * k) @ Matrix.Translation((0, 0, -0.1 * k))
        extra = wings(4 * t * (1 - k), amp=R(40) * (1 - k), fold=k)
        extra["abdomen"] = about(A, X, R(40) * k)
        extra.update(jaws(m, R(25) * k))
        return m.pose(body, None, extra)

    m.skin_action("run", 10, fly)
    m.skin_action("idle", 10, fly)
    m.skin_action("attack", 14, attack, loop=False)
    m.skin_action("die", 20, die, loop=False)


# ---------------------------------------------------------------------------------------------- танкер

TANKER_LEGS = mirror({
    "1": ((0.22, 0.26, 0.34), (0.46, 0.42, 0.46), (0.56, 0.6, 0.0)),
    "2": ((0.26, 0.02, 0.34), (0.52, 0.04, 0.48), (0.66, 0.05, 0.0)),
    "3": ((0.22, -0.24, 0.34), (0.48, -0.42, 0.46), (0.58, -0.64, 0.0)),
})
TANKER_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0, 0.42), "root"), "head": ((0, 0.4, 0.38), "body"),
                 "abdomen": ((0, -0.3, 0.46), "body"),
                 "mand.R": ((0.13, 0.76, 0.2), "head"), "mand.L": ((-0.13, 0.76, 0.2), "head")}
TANKER_JOINTS.update(leg_joints(TANKER_LEGS))


def tanker():
    m = Bug("tanker", TANKER_JOINTS, TANKER_LEGS, 512 if FAST else 1024)
    m.camo_scale = 6.0
    m.decimate = 0.35
    S, Sd, Sl = "stone_chitin", "stone_chitin_dk", "stone_chitin_lt"
    carapace(m, S, Sd, Sl, (0, 0.05, 0.47), (0.34, 0.38, 0.24), "body")
    m.plate(S, (0, 0.05, 0.47), (0.36, 0.4, 0.255), (0, 2 * math.pi), (R(76), R(90)), 0.03, group="body",
            res=(16, 2))                                    # макушка щита: у крупного жука видна дыра у полюса
    m.sphere("belly", 1.0, (0, -0.1, 0.32), scale=(0.27, 0.5, 0.12), group="body", seg=16, rings=8)
    for s in (-1, 1):                                       # наплечники груди
        m.plate(Sd, (s * 0.12, 0.12, 0.5), (0.26, 0.24, 0.2), (R(90 - 90 * s - 50), R(90 - 90 * s + 50)),
                (R(-10), R(60)), 0.04, group="body", res=(10, 5))
    # брюшко: купол из налегающих щитков, по нижней кромке — шипы
    AC, AR = (0, -0.6, 0.44), (0.44, 0.48, 0.3)
    m.sphere(Sd, 1.0, AC, scale=AR, group="abdomen", seg=24, rings=12)
    for i, along in enumerate((0.72, 0.38, 0.02, -0.36)):
        grow = 1.08 - i * 0.012
        band(m, S, AC, tuple(v * grow for v in AR), along, 0.1, 0.05, "abdomen", spread=92 - i * 4)
        band(m, Sl, AC, tuple(v * (grow + 0.012) for v in AR), along - 0.16, 0.012, 0.012, "abdomen", spread=80)
    for s in (-1, 1):
        for along in (0.55, 0.25, -0.05, -0.35, -0.62):
            base = on_ellipsoid(AC, AR, along, s * R(98), 1.04)
            out = Vector((s, -0.35, -0.35)).normalized()
            m.sweep("claw", [base, base + out * 0.1], oval(6), taper(0.032, 0.0), group="abdomen")
    m.sweep("claw", [on_ellipsoid(AC, AR, -0.85, 0, 1.0), Vector((0, -1.1, 0.4))], oval(8), taper(0.05, 0.0),
            group="abdomen")
    # голова: широкий щит-отвал, большой рог вверх-вперёд, два малых по бокам
    m.loft(S, [[(x * 0.32, 0.36, 0.42 + z * 0.17) for x, z in oval(14)],
               [(x * 0.38, 0.54, 0.38 + z * 0.16) for x, z in oval(14)],
               [(x * 0.33, 0.72, 0.3 + z * 0.12) for x, z in oval(14)],
               [(x * 0.22, 0.84, 0.23 + z * 0.07) for x, z in oval(14)]], group="head", sharp=70)
    m.plate(Sl, (0, 0.56, 0.38), (0.39, 0.27, 0.17), (R(0), R(180)), (R(30), R(75)), 0.02, group="head", res=(14, 4))
    m.sweep("claw", curve([(0, 0.6, 0.5), (0, 0.8, 0.6), (0, 0.94, 0.86)], 9), oval(10), taper(0.11, 0.008),
            group="head")
    m.sweep("claw_dk", curve([(0, 0.6, 0.55), (0, 0.78, 0.64), (0, 0.9, 0.84)], 7), LENS, taper(0.04, 0.0), 0.025,
            up=(1, 0, 0), group="head")
    for s in (-1, 1):
        S_ = "R" if s > 0 else "L"
        m.sweep("claw", curve([(s * 0.17, 0.58, 0.44), (s * 0.25, 0.68, 0.52), (s * 0.3, 0.76, 0.6)], 6), oval(8),
                taper(0.04, 0.0), group="head")
        m.sphere("eye", 0.024, (s * 0.2, 0.68, 0.27), group="head", seg=10, rings=6)
        m.sphere("eye", 0.016, (s * 0.15, 0.73, 0.25), group="head", seg=8, rings=5)
        mj = Vector((s * 0.13, 0.76, 0.2))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.08, 0.06, -0.02)), mj + Vector((-s * 0.02, 0.14, -0.05))], 7),
                LENS, taper(0.04, 0.004), 0.026, group="mand." + S_)
    for leg, (H, K, F) in TANKER_LEGS.items():
        build_leg(m, leg, H, K, F, S, Sd, 0.085, 0.07)
    m.build()
    m.bake(samples=8 if FAST else 64)
    tanker_actions(m)
    export.export(m)
    return m


def tanker_actions(m):
    B = m.joints["body"][0]
    Hd = m.joints["head"][0]
    A = m.joints["abdomen"][0]

    def run(t):
        body, feet = gait(m, t, TRIPOD, stride=0.3, lift=0.09, duty=0.62, bob=0.025, sway=2.0)
        ph = 2 * math.pi * t
        extra = {"head": about(Hd, X, R(4) * math.sin(2 * ph)), "abdomen": about(A, X, R(3) * math.sin(2 * ph))}
        extra.update(jaws(m, R(6 + 4 * math.sin(2 * ph))))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        extra = {"head": about(Hd, Z, R(8) * math.sin(ph)), "abdomen": about(A, X, R(2) * math.sin(2 * ph))}
        extra.update(jaws(m, R(4 + 6 * max(0.0, math.sin(2 * ph)) ** 4)))
        return m.pose(Matrix.Translation((0, 0, 0.012 * math.sin(2 * ph))), None, extra)

    def attack(t):
        # таран: оседает назад, задирая рог, и бьёт головой вперёд-вверх
        back = smooth(min(1.0, t / 0.45))
        hit = smooth(max(0.0, min(1.0, (t - 0.45) / 0.15)))
        rest = smooth(max(0.0, (t - 0.65) / 0.35))
        k = back * (1 - hit)
        f = hit * (1 - rest)
        body = Matrix.Translation((0, -0.08 * k + 0.14 * f, -0.04 * k)) @ about(B, X, R(6) * k - R(8) * f)
        extra = {"head": about(Hd, X, R(-14) * k + R(22) * f)}
        extra.update(jaws(m, R(25) * k))
        return m.pose(body, None, extra)

    def die(t):
        k = smooth(min(1.0, t / 0.7))
        body = Matrix.Translation((0, 0, -0.24 * k)) @ about(B, (0, 1, 0), R(18) * k)
        extra = {"head": about(Hd, X, R(-18) * k)}
        extra.update(jaws(m, R(25) * k))
        return m.pose(body, dying_legs(m, body, k, lift=0.18, tuck=1.4), extra)

    m.skin_action("run", 24, run)
    m.skin_action("idle", 60, idle)
    m.skin_action("attack", 20, attack, loop=False)
    m.skin_action("die", 30, die, loop=False)


# ---------------------------------------------------------------------------------------------- мозговой

BRAIN_LEGS = mirror({
    "1": ((0.07, 0.1, 0.28), (0.26, 0.26, 0.44), (0.36, 0.4, 0.0)),
    "2": ((0.09, 0.0, 0.28), (0.3, 0.02, 0.46), (0.44, 0.02, 0.0)),
    "3": ((0.07, -0.1, 0.28), (0.26, -0.26, 0.44), (0.36, -0.4, 0.0)),
})
BRAIN_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0, 0.3), "root"), "brain": ((0, 0, 0.4), "body"),
                "head": ((0, 0.14, 0.3), "body"), "abdomen": ((0, -0.12, 0.3), "body"),
                "mand.R": ((0.03, 0.26, 0.25), "head"), "mand.L": ((-0.03, 0.26, 0.25), "head")}
BRAIN_JOINTS.update(leg_joints(BRAIN_LEGS))
LOBE_R = (0.15, 0.25, 0.17)


def lobe_center(s):
    return (s * 0.085, -0.02, 0.53)


def brain():
    m = Bug("brain", BRAIN_JOINTS, BRAIN_LEGS, 512 if FAST else 1024)
    m.camo_scale = 8.0
    m.decimate = 0.35
    P, Pd, Pl = "pale_chitin", "pale_chitin_dk", "pale_chitin_lt"
    carapace(m, P, Pd, Pl, (0, 0.0, 0.3), (0.11, 0.14, 0.08), "body")
    m.sphere(Pd, 1.0, (0, -0.02, 0.38), scale=(0.17, 0.22, 0.06), group="brain", seg=18, rings=8)     # воротник
    m.sphere(P, 1.0, (0, -0.2, 0.31), scale=(0.1, 0.12, 0.08), group="abdomen", seg=14, rings=8)
    # мозг: два полушария, извилины тёмными бороздами, светящиеся узлы
    for s in (-1, 1):
        c = lobe_center(s)
        m.sphere("brain", 1.0, c, scale=LOBE_R, group="brain", seg=22, rings=12)
        for a0 in (14, 38, 62, 86):
            pts = [on_ellipsoid(c, LOBE_R, y, s * R(a0 + 9 * math.sin(y * 13 + a0)), 1.005)
                   for y in [-0.85 + 1.7 * i / 23 for i in range(24)]]
            m.sweep("brain_dk", pts, oval(6), 0.009, group="brain")
        for along, a in ((0.5, 26), (0.0, 50), (-0.45, 30), (0.3, 72)):
            m.sphere("psi", 0.02, tuple(on_ellipsoid(c, LOBE_R, along, s * R(a), 0.995)), group="brain", seg=10, rings=6)
    m.sweep("brain_dk", [on_ellipsoid((0, -0.02, 0.53), (0.04, 0.25, 0.17), y, 0, 0.97) for y in
                         [-0.9 + 1.8 * i / 15 for i in range(16)]], oval(6), 0.02, group="brain")       # межполушарная щель
    # лицо: маленькая голова под мозгом, глаза, свисающие щупики и короткие жвалы
    m.sphere(P, 1.0, (0, 0.2, 0.3), scale=(0.07, 0.08, 0.055), group="head", seg=14, rings=8)
    for s in (-1, 1):
        S_ = "R" if s > 0 else "L"
        m.sphere("eye", 0.014, (s * 0.045, 0.255, 0.32), group="head", seg=8, rings=5)
        m.sweep(Pd, curve([(s * 0.03, 0.26, 0.27), (s * 0.06, 0.32, 0.2), (s * 0.05, 0.34, 0.1)], 7), oval(6),
                taper(0.012, 0.004), group="head")
        mj = Vector((s * 0.03, 0.26, 0.25))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.03, 0.04, -0.01)), mj + Vector((0, 0.07, -0.02))], 5), LENS,
                taper(0.014, 0.002), 0.009, group="mand." + S_)
    for leg, (H, K, F) in BRAIN_LEGS.items():
        build_leg(m, leg, H, K, F, P, Pd, 0.022, 0.017, spikes=False)
    m.build()
    m.bake(samples=8 if FAST else 64)
    brain_actions(m)
    export.export(m)
    return m


def brain_actions(m):
    Bc = m.joints["brain"][0]
    Hd = m.joints["head"][0]
    B = m.joints["body"][0]

    def pulse(k, pitch=0.0, roll=0.0):
        return {"brain": about(Bc, X, pitch) @ about(Bc, (0, 1, 0), roll) @ scaled(Bc, k)}

    def run(t):
        body, feet = gait(m, t, TRIPOD, stride=0.22, lift=0.07, sway=5.0)
        ph = 2 * math.pi * t
        extra = pulse(1.0 + 0.02 * math.sin(2 * ph), R(4) * math.sin(2 * ph), R(5) * math.sin(ph))
        extra.update(jaws(m, R(8 + 6 * math.sin(2 * ph))))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        extra = pulse(1.0 + 0.045 * math.sin(2 * ph) ** 2, 0.0, R(3) * math.sin(ph))
        extra["head"] = about(Hd, Z, R(10) * math.sin(ph))
        return m.pose(Matrix.Translation((0, 0, 0.008 * math.sin(ph))), None, extra)

    def die(t):
        k = smooth(min(1.0, t / 0.6))
        body = Matrix.Translation((0, 0, -0.17 * k)) @ about(B, (0, 1, 0), R(25) * k)
        extra = pulse(1.0 - 0.12 * k, R(-20) * k)
        extra.update(jaws(m, R(25) * k))
        return m.pose(body, dying_legs(m, body, k), extra)

    m.skin_action("run", 18, run)
    m.skin_action("idle", 40, idle)
    m.skin_action("die", 24, die, loop=False)


VIEW = {   # модель, цель камеры, дистанция, клипы для листа
    "spitter": (spitter, (0, -0.1, 0.3), 2.8, ("run", "windup", "attack", "die")),
    "flyer": (flyer, (0, -0.05, 0.15), 2.4, ("run", "attack", "die")),
    "tanker": (tanker, (0, -0.1, 0.45), 4.4, ("run", "attack", "die")),
    "brain": (brain, (0, 0, 0.35), 2.6, ("run", "idle", "die")),
}

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    which = [a for a in args if not a.startswith("--")] or list(VIEW)
    for name in which:
        style.reset()
        make, target, dist, clips = VIEW[name]
        m = make()
        preview.model_views(m, name, target=target, dist=dist)
        preview.anim_strip(m, name, clips, dist=dist * 1.05, target=target, show={})
    print("BROOD DONE", flush=True)
