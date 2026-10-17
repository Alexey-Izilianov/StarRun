# Десантник мобильной пехоты в человеческих пропорциях (рост 1,85 м, шлем 0,26 м, плечи с наплечниками 0,66 м):
# оливковая броня поверх тёмного подкостюма, закрытый шлем, ранец с баллонами; тело — marine_body.py.
# Подмеши: стволы на кости weapon (кисти держат их через IK; MAR — здесь, остальные — guns.py; игра включает
# выбранный), axe — топор в правой кисти (только в рубке). Мини-турель — отдельная модель turret.
# Запуск: blender.exe -b --python marine.py [-- --fast]
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style, rig, export, guns, marine_body as body
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
    turn = -45 if side == "R" else 45          # больше — корпус «перекручен» относительно ног

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
    m = RigModel("marine", JOINTS, tex_size=512 if FAST else 2048)    # тело и шесть стволов в одной текстуре
    m.camo_scale = 8.0
    body.legs(m, P)
    body.waist(m)
    body.torso(m)
    body.backpack(m)
    body.arms(m, P)
    body.helmet(m)
    mar(m)
    x, _, z = P("weapon")
    for gun in (guns.shotgun, guns.sniper, guns.flame, guns.rocket, guns.shock):
        gun(m, x, z)
    axe(m)
    m.build()
    m.bake(samples=8 if FAST else 64)
    actions(m)
    export.export(m)
    return m


def build_turret():
    m = guns.turret()
    m.bake(samples=8 if FAST else 64)
    guns.turret_actions(m)
    export.export(m)
    return m


GUNS = ("mar", "shotgun", "sniper", "flame", "rocket", "shock")

if __name__ == "__main__":
    style.reset()
    m = build()
    import preview
    for gun in GUNS:
        preview.model_views(m, "marine_" + gun, target=(0, 0.3, 1.1), dist=2.2, views=("hero", "side"),
                            hide=("axe",) + tuple(g for g in GUNS if g != gun))
    preview.model_views(m, "marine", target=(0, 0.1, 1.0), dist=3.8, hide=("axe",) + GUNS[1:])
    preview.anim_strip(m, "marine", ("run_f", "run_r", "chop", "vault", "die"), dist=4.4, target=(0, 0.1, 0.95),
                       show={"chop": ("body", "mar", "axe")})
    style.reset()
    t = build_turret()
    preview.model_views(t, "turret", target=(0, 0, 0.35), dist=1.8)
    print("MARINE DONE", flush=True)
