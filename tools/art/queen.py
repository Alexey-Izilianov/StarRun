# Боссы: элитный воин (полубосс — воин в броне, bugs.warrior(elite=True)) и королева — высокая передняя часть
# с короной рогов и лапами-серпами, за ней волочится огромный яйцевой мешок в хитиновых обручах, со светящимися
# порами кислоты. Залп — встаёт на дыбы и плюёт; призыв — мешок пульсирует.
# Запуск: blender.exe -b --python queen.py [-- elite queen] [--fast]
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mathutils import Vector, Matrix
import style, export, preview
from style import curve, taper, oval, LENS
from rig import smooth
from bugs import Bug, about, mirror, leg_joints, build_leg, gait, carapace, jaws, scythes, warrior, FAST, Z
from brood import X, TRIPOD, scaled, on_ellipsoid, band, dying_legs

R = math.radians


def elite():
    return warrior("elite", ("elite_chitin", "elite_chitin_dk", "elite_chitin_lt"), elite=True)


QUEEN_LEGS = mirror({
    "1": ((0.2, 0.25, 0.85), (0.6, 0.55, 1.35), (0.85, 0.95, 0.0)),
    "2": ((0.24, 0.0, 0.85), (0.75, 0.05, 1.4), (1.1, 0.1, 0.0)),
    "3": ((0.2, -0.25, 0.85), (0.65, -0.5, 1.3), (0.95, -0.9, 0.0)),
})
QUEEN_JOINTS = {"root": ((0, 0, 0), None), "body": ((0, 0, 0.9), "root"), "head": ((0, 0.45, 1.05), "body"),
                "abdomen": ((0, -0.4, 0.85), "body"),
                "mand.R": ((0.11, 0.93, 0.82), "head"), "mand.L": ((-0.11, 0.93, 0.82), "head"),
                "arm.R": ((0.24, 0.35, 0.95), "body"), "blade.R": ((0.5, 0.7, 1.45), "arm.R"),
                "arm.L": ((-0.24, 0.35, 0.95), "body"), "blade.L": ((-0.5, 0.7, 1.45), "arm.L")}
QUEEN_JOINTS.update(leg_joints(QUEEN_LEGS))
EC, ER = (0, -1.42, 0.64), (0.64, 1.0, 0.56)


def queen():
    m = Bug("queen", QUEEN_JOINTS, QUEEN_LEGS, 512 if FAST else 2048)
    m.camo_scale = 5.0
    Q, Qd, Ql = "queen_chitin", "queen_chitin_dk", "queen_chitin_lt"
    # грудь, по хребту — шипы назад
    carapace(m, Q, Qd, Ql, (0, 0.05, 0.97), (0.37, 0.48, 0.29), "body")
    m.plate(Q, (0, 0.05, 0.97), (0.39, 0.5, 0.305), (0, 2 * math.pi), (R(76), R(90)), 0.03, group="body", res=(16, 2))
    for s in (-1, 1):                                     # наплечники над серпами
        m.plate(Ql, (s * 0.16, 0.25, 1.0), (0.26, 0.22, 0.18), (R(90 - 90 * s - 55), R(90 - 90 * s + 55)),
                (R(-5), R(65)), 0.035, group="body", res=(10, 5))
    m.sphere("belly", 1.0, (0, 0.05, 0.8), scale=(0.24, 0.36, 0.12), group="body", seg=16, rings=8)
    for i, y in enumerate((0.3, 0.12, -0.06, -0.24)):
        base = Vector((0, y, 1.17 - abs(y - 0.05) * 0.25))
        m.sweep("claw", curve([base, base + Vector((0, -0.1, 0.14 - i * 0.02)), base + Vector((0, -0.22, 0.2 - i * 0.03))], 5),
                LENS, taper(0.06, 0.0), 0.025, group="body")
    # голова: вытянутый череп вниз-вперёд, корона из пяти рогов назад, светящаяся пасть
    m.loft(Q, [[(x * 0.18, 0.42, 1.06 + z * 0.14) for x, z in oval(14)],
               [(x * 0.23, 0.6, 1.03 + z * 0.15) for x, z in oval(14)],
               [(x * 0.17, 0.82, 0.93 + z * 0.11) for x, z in oval(14)],
               [(x * 0.09, 0.97, 0.85 + z * 0.06) for x, z in oval(14)]], group="head", sharp=70)
    m.plate(Ql, (0, 0.62, 1.03), (0.235, 0.22, 0.155), (R(0), R(180)), (R(35), R(80)), 0.015, group="head", res=(14, 4))
    for a in (-44, -22, 0, 22, 44):
        sa = math.sin(R(a))
        length = 1.45 - abs(a) / 90
        base = Vector((sa * 0.15, 0.5, 1.15))
        m.sweep("claw", curve([base, base + Vector((sa * 0.22, -0.1, 0.3)) * length,
                               base + Vector((sa * 0.36, -0.36, 0.5)) * length], 9), oval(8), taper(0.085, 0.0),
                group="head")
    m.sphere("acid", 0.07, (0, 0.9, 0.8), scale=(1.2, 0.6, 0.7), group="head", seg=12, rings=8)
    for s in (-1, 1):
        S = "R" if s > 0 else "L"
        for k, (y, z, r) in enumerate(((0.74, 1.04, 0.034), (0.8, 1.0, 0.026), (0.68, 1.08, 0.02))):
            m.sphere("eye", r, (s * (0.15 - 0.02 * k), y, z), group="head", seg=10, rings=6)
        mj = Vector((s * 0.11, 0.93, 0.82))
        m.sweep("claw", curve([mj, mj + Vector((s * 0.15, 0.12, -0.06)), mj + Vector((s * 0.05, 0.3, -0.12)),
                               mj + Vector((-s * 0.07, 0.33, -0.15))], 9), LENS, taper(0.05, 0.003), 0.03, group="mand." + S)
        m.sweep("claw_dk", curve([mj + Vector((-s * 0.06, 0.03, -0.04)), mj + Vector((-s * 0.03, 0.15, -0.15)),
                                  mj + Vector((-s * 0.07, 0.2, -0.18))], 5), oval(5), taper(0.024, 0.0), group="head")
        # лапа-серп: как у воина, крупнее
        sh, el = Vector((s * 0.24, 0.35, 0.95)), Vector((s * 0.5, 0.7, 1.45))
        m.sweep(Q, curve([sh, (sh + el) / 2 + Vector((s * 0.05, 0, 0.06)), el], 7), oval(8), taper(0.075, 0.058),
                group="arm." + S, sharp=60)
        m.sweep(Ql, [sh.lerp(el, 0.2) + Vector((s * 0.05, 0, 0.05)), sh.lerp(el, 0.85) + Vector((s * 0.05, 0, 0.05))],
                LENS, 0.07, 0.04, up=(s, 0, 1), group="arm." + S)
        m.sphere("membrane", 0.065, tuple(el), group="blade." + S, seg=10, rings=6)
        blade = curve([el, el + Vector((s * 0.03, 0.3, 0.03)), el + Vector((-s * 0.03, 0.6, -0.24)),
                       el + Vector((-s * 0.08, 0.7, -0.64))], 11)
        m.sweep("claw", blade, LENS, taper(0.09, 0.006), 0.05, up=(s, 0, 0), group="blade." + S)
        m.sweep("claw_dk", [p + Vector((0, -0.015, 0.03)) for p in blade[:6]], LENS, taper(0.05, 0.0), 0.02,
                up=(s, 0, 0), group="blade." + S)
    # яйцевой мешок: стебель, тёмное днище, светлый мешок, обручи, яйца по бокам, поры кислоты, яйцеклад
    m.sweep("membrane", [(0, -0.28, 0.9), (0, -0.5, 0.82)], oval(12), 0.14, group="abdomen")
    m.sphere(Qd, 1.0, (EC[0], EC[1], EC[2] - 0.12), scale=(ER[0] * 0.9, ER[1] * 0.95, ER[2] * 0.7), group="abdomen",
             seg=24, rings=12)
    m.sphere("egg_sac", 1.0, EC, scale=ER, group="abdomen", seg=32, rings=18)
    for i, along in enumerate((0.78, 0.5, 0.2, -0.12, -0.44, -0.72)):
        band(m, Q, EC, tuple(v * 1.025 for v in ER), along, 0.11 - i * 0.005, 0.045, "abdomen", spread=82, res=18)
        band(m, Ql, EC, tuple(v * 1.04 for v in ER), along + 0.05, 0.012, 0.01, "abdomen", spread=60, res=18)
    for s in (-1, 1):
        for j, along in enumerate((0.62, 0.35, 0.05, -0.28, -0.58)):
            for side, r in ((112, 0.085), (128, 0.07)):
                m.sphere("egg_sac", r, tuple(on_ellipsoid(EC, ER, along + 0.06 * (side == 128), s * R(side), 0.97)),
                         scale=(1, 1.25, 1), group="abdomen", seg=12, rings=8)
    for along, side in ((0.64, 0), (0.35, 0.5), (0.35, -0.5), (0.05, 0), (-0.28, 0.45), (-0.28, -0.45), (-0.58, 0)):
        m.sphere("acid", 0.04, tuple(on_ellipsoid(EC, ER, along, side, 0.995)), scale=(1, 1, 0.6), group="abdomen",
                 seg=10, rings=6)
    tail = on_ellipsoid(EC, ER, -0.97, 0, 1.0)
    m.sweep("claw_dk", [tail, tail + Vector((0, -0.25, -0.12))], oval(10), taper(0.1, 0.0), group="abdomen")
    for leg, (H, K, F) in QUEEN_LEGS.items():
        build_leg(m, leg, H, K, F, Q, Qd, 0.09, 0.07)
    m.build()
    m.bake(samples=8 if FAST else 64)
    queen_actions(m)
    export.export(m)
    return m


def queen_actions(m):
    B = m.joints["body"][0]
    Hd = m.joints["head"][0]
    A = m.joints["abdomen"][0]

    def sac(k=1.0, pitch=0.0, yaw=0.0):
        return {"abdomen": about(A, Z, yaw) @ about(A, X, pitch) @ scaled(A, k)}

    def run(t):
        body, feet = gait(m, t, TRIPOD, stride=0.5, lift=0.18, duty=0.6, bob=0.03, sway=3.0)
        ph = 2 * math.pi * t
        extra = sac(1.0, R(2) * math.sin(2 * ph), R(4) * math.sin(ph))
        extra.update(scythes(m, R(5 * math.sin(2 * ph)), R(-12)))
        extra.update(jaws(m, R(8 + 5 * math.sin(2 * ph))))
        return m.pose(body, feet, extra)

    def idle(t):
        ph = 2 * math.pi * t
        extra = sac(1.0 + 0.02 * math.sin(2 * ph))
        extra.update(scythes(m, R(5 * math.sin(ph)), R(-14)))
        extra.update(jaws(m, R(5 + 14 * max(0.0, math.sin(2 * ph)) ** 6)))
        extra["head"] = about(Hd, Z, R(10) * math.sin(ph))
        return m.pose(Matrix.Translation((0, 0, 0.015 * math.sin(ph))), None, extra)

    def attack(t):
        # залп: встаёт на дыбы, раскрывает серпы и жвалы, клюёт головой вперёд-вниз — плевок
        up = smooth(min(1.0, t / 0.45))
        spit = smooth(max(0.0, min(1.0, (t - 0.45) / 0.15)))
        down = smooth(max(0.0, (t - 0.65) / 0.35))
        rear = up * (1 - down)
        body = about(B, X, R(16) * rear - R(6) * spit * (1 - down)) @ Matrix.Translation((0, -0.06 * rear, 0.05 * rear))
        extra = sac(1.0 - 0.06 * spit * (1 - down), R(-6) * rear)
        extra["head"] = about(Hd, X, R(10) * up * (1 - spit) - R(22) * spit * (1 - down))
        extra.update(scythes(m, R(40) * rear, R(-12 + 40 * rear)))
        extra.update(jaws(m, R(10 + 35 * rear)))
        return m.pose(body, None, extra)

    def summon(t):
        # призыв: мешок дважды вздувается, тело вздрагивает, серпы вскинуты
        k = smooth(min(1.0, t / 0.2)) * (1 - smooth(max(0.0, (t - 0.8) / 0.2)))
        swell = 0.1 * k * (0.5 + 0.5 * math.sin(4 * math.pi * t - math.pi / 2))
        body = Matrix.Translation((0.01 * k * math.sin(30 * t), 0, 0))
        extra = sac(1.0 + swell, R(-6) * k)
        extra.update(scythes(m, R(25) * k, R(10) * k))
        extra.update(jaws(m, R(10 + 20 * k)))
        return m.pose(body, None, extra)

    def die(t):
        k = smooth(min(1.0, t / 0.7))
        body = Matrix.Translation((0, 0, -0.6 * k)) @ about(B, (0, 1, 0), R(20) * k)
        extra = sac(1.0 - 0.1 * k, R(10) * k)
        extra["head"] = about(Hd, X, R(-20) * k)
        extra.update(scythes(m, R(-30) * k, R(-40) * k))
        extra.update(jaws(m, R(30) * k))
        return m.pose(body, dying_legs(m, body, k, lift=0.3, tuck=1.5), extra)

    m.skin_action("run", 26, run)
    m.skin_action("idle", 60, idle)
    m.skin_action("attack", 36, attack, loop=False)
    m.skin_action("summon", 30, summon, loop=False)
    m.skin_action("die", 36, die, loop=False)


VIEW = {
    "elite": (elite, (0, 0.1, 0.5), 3.8, ("run", "windup", "strike", "die")),
    "queen": (queen, (0, -0.4, 0.8), 7.5, ("run", "attack", "summon", "die")),
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
    print("QUEEN DONE", flush=True)
