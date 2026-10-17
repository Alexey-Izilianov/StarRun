# Пехота со скелетом: бронированные детали жёстко привязаны к костям (без изломов в суставах),
# анимации задаются функциями позы от фазы t ∈ [0, 1]. Все кости смотрят вверх (+Z) с нулевым креном,
# поэтому оси поворота одинаковы у всех: X — наклон вперёд/назад (+ уводит низ кости вперёд, верх — назад),
# Y (локальная) — поворот вокруг вертикали, Z (локальная) — завал вбок.
import bpy, math
from mathutils import Vector, Matrix, Euler
import style
from style import Model, PAL, _select

FPS = 30


def world_to_bone(v):
    """Смещение в мире → локальные оси вертикальной кости (X=X, Y=Z, Z=−Y)."""
    return (v[0], v[2], -v[1])

# ---------- руки на оружии (двухзвенная IK) ----------
# Все кости вертикальные с нулевым креном: локальные оси кости в осях арматуры — X = X, Y = Z, Z = −Y.
_B = Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
_BI = _B.transposed()


def _arm_rot(euler_deg):
    """Поворот позы кости (эйлер в её осях) → поворот в осях арматуры."""
    return _B @ Euler([math.radians(v) for v in euler_deg], 'XYZ').to_matrix() @ _BI


def _bone_euler(rot):
    """Поворот в осях арматуры → эйлер позы кости, градусы."""
    return tuple(math.degrees(a) for a in (_BI @ rot @ _B).to_euler('XYZ'))


def solve_arms(joints, pose, sides):
    """Кисти держат оружие: по позе оружия (weapon, weapon@loc — относительно груди) считаются плечо, предплечье
    и кисть так, чтобы кисть пришла в свою точку хвата на оружии. Локоть сгибается в ту же сторону, что в покое.
    Руки и оружие — дети груди, поэтому всё считается в её системе (поза груди на расчёт не влияет)."""
    W = Vector(joints["weapon"][0])
    Rw = _arm_rot(pose.get("weapon", (0, 0, 0)))
    shift = Vector(pose.get("weapon@loc", (0, 0, 0)))
    for side in sides:
        S, E, H = (Vector(joints[b + side][0]) for b in ("upperarm.", "forearm.", "hand."))
        G = W + shift + Rw @ (H - W)
        a, b = (E - S).length, (H - E).length
        to = G - S
        d = min(max(to.length, abs(a - b) + 1e-3), a + b - 1e-3)
        u = to.normalized()
        h0 = (H - S).normalized()
        pole = h0.rotation_difference(u) @ ((E - S) - h0 * (E - S).dot(h0))
        pole -= u * pole.dot(u)
        if pole.length < 1e-6:
            pole = Vector((0, 0, -1)) - u * -u.z
        pole.normalize()
        ca = max(-1.0, min(1.0, (a * a + d * d - b * b) / (2 * a * d)))
        E2 = S + (u * ca + pole * math.sqrt(1 - ca * ca)) * a
        G2 = S + u * d
        R1 = (E - S).rotation_difference(E2 - S).to_matrix()
        R2w = (R1 @ (H - E)).rotation_difference(G2 - E2).to_matrix()
        pose["upperarm." + side] = _bone_euler(R1)
        pose["forearm." + side] = _bone_euler(R1.inverted() @ R2w @ R1)
        pose["hand." + side] = _bone_euler((R2w @ R1).inverted() @ Rw)
    return pose



class RigModel(Model):
    def __init__(self, name, joints, tex_size=1024):
        super().__init__(name, tex_size)
        self.joints = joints            # кость -> ((x, y, z) сустава, родитель); родители — раньше детей
        self.actions = []
        self.ik_arms = ()               # стороны ("L", "R"), чьи кисти держат оружие (solve_arms)
        self.sub = None                 # имя подмеша для новых деталей (оружие, топор): игра включает их отдельно
        self.decimate = 1.0             # доля треугольников после упрощения (жукам толпы — меньше)

    def _finish(self, o, group, key, bevel, seg, smooth_all=True):
        if self.sub:
            o["sub"] = self.sub
        return super()._finish(o, group, key, bevel, seg, smooth_all)

    def limb(self, key, a, b, r1, r2, bone, seg=10, bevel=0.02):
        """Усечённый конус от сустава a к суставу b (сегмент руки или ноги)."""
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = Vector((0, 0, 1)).rotation_difference(d).to_euler()
        return self.cyl(key, r1, d.length, tuple((a + b) / 2), tuple(rot), group=bone, seg=seg, bevel=bevel, r2=r2)

    def build(self):
        """Цельные детали — меш body и по мешу на подмеш (оружие, топор), светящиеся — по мешу на ключ; у каждой
        детали группа вершин своей кости с весом 1. Скелет — по суставам; меши — дети скелета с модификатором
        Armature. Подмеши запекаются в общую текстуру вместе с телом."""
        for o in self.parts:
            self._write_attrs(o, o["key"])
            vg = o.vertex_groups.new(name=o["group"])
            vg.add(list(range(len(o.data.vertices))), 1.0, 'REPLACE')
        by_sub = {}
        glows = {}
        for o in self.parts:
            if PAL[o["key"]][3] > 0:                  # свечение — по подмешу: подсветка ствола гаснет с ним
                glows.setdefault((o.get("sub", "body"), o["key"]), []).append(o)
            else:
                by_sub.setdefault(o.get("sub", "body"), []).append(o)
        self.objects = {sub: self._join(lst, sub, (0, 0, 0)) for sub, lst in by_sub.items()}
        if self.decimate < 1:
            # Толпа в сотни экземпляров: сетку упрощаем до запекания — отдельные детали не сливаются,
            # привязка к костям и цвета палитры переносятся на оставшиеся вершины
            for o in self.objects.values():
                mod = o.modifiers.new("dec", "DECIMATE")
                mod.ratio = self.decimate
                _select([o])
                bpy.ops.object.modifier_apply(modifier="dec")
        body = self.objects["body"]
        self.glow_objects = []
        for (sub, key), lst in glows.items():
            go = self._join(lst, f"{sub}_glow_{key}", (0, 0, 0))
            go["material"] = f"{self.name}_glow_{key}"
            self.glow_objects.append((go, "body"))

        arm_data = bpy.data.armatures.new("rig")
        arm = bpy.data.objects.new("rig", arm_data)
        bpy.context.scene.collection.objects.link(arm)
        _select([arm])
        bpy.ops.object.mode_set(mode='EDIT')
        for name, (head, parent) in self.joints.items():
            eb = arm_data.edit_bones.new(name)
            eb.head = head
            eb.tail = Vector(head) + Vector((0, 0, 0.12))
            eb.roll = 0.0
            eb.use_deform = True
            if parent:
                eb.parent = arm_data.edit_bones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')
        for pb in arm.pose.bones:
            pb.rotation_mode = 'XYZ'

        self.root = bpy.data.objects.new(self.name, None)
        bpy.context.scene.collection.objects.link(self.root)
        arm.parent = self.root
        for o in list(self.objects.values()) + [g for g, _ in self.glow_objects]:
            o.parent = arm
            mod = o.modifiers.new("arm", "ARMATURE")
            mod.object = arm
        self.arm = arm

    def skin_action(self, name, frames, pose_fn, loop=True):
        """Клип из матриц: pose_fn(t) → {кость: смещение D (4×4) в осях арматуры}, D переводит кость из покоя
        в позу (родитель уже учтён: D ребёнка включает D родителя). Так удобно задавать ноги через IK в мировых
        осях. Каналы позы восстанавливаются из D без пересчёта сцены: basis = (Rp⁻¹·Rb)⁻¹ · Pp⁻¹ · Pb, где
        R — матрица кости в покое, P = D·R — в позе."""
        arm = self.arm
        if arm.animation_data is None:
            arm.animation_data_create()
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
        arm.animation_data.action = act
        rest = {b.name: b.matrix_local.copy() for b in arm.data.bones}
        for pb in arm.pose.bones:
            pb.rotation_mode = 'QUATERNION'
        for f in range(frames + 1):
            D = pose_fn(f / frames)
            for pb in arm.pose.bones:
                b = pb.bone
                Pb = D.get(b.name, Matrix()) @ rest[b.name]
                if b.parent:
                    Pp = D.get(b.parent.name, Matrix()) @ rest[b.parent.name]
                    basis = (rest[b.parent.name].inverted() @ rest[b.name]).inverted() @ Pp.inverted() @ Pb
                else:
                    basis = rest[b.name].inverted() @ Pb
                loc, rot, scale = basis.decompose()        # масштаб — раздувание мешков (стрелок, королева)
                pb.location = loc
                pb.rotation_quaternion = rot
                pb.scale = scale
                pb.keyframe_insert("location", frame=f)
                pb.keyframe_insert("rotation_quaternion", frame=f)
                pb.keyframe_insert("scale", frame=f)
        act.use_frame_range = True
        act.frame_start, act.frame_end = 0, frames
        act.use_cyclic = loop
        self.actions.append(act)
        return act

    def action(self, name, frames, pose_fn, loop=True, ik=True):
        """pose_fn(t) → {кость: (rx, ry, rz) в градусах, 'кость@loc': смещение в мире}."""
        arm = self.arm
        if arm.animation_data is None:
            arm.animation_data_create()
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
        arm.animation_data.action = act
        for f in range(frames + 1):
            pose = pose_fn(f / frames)
            if self.ik_arms and ik:
                solve_arms(self.joints, pose, self.ik_arms)
            for pb in arm.pose.bones:
                r = pose.get(pb.name, (0, 0, 0))
                pb.rotation_euler = [math.radians(v) for v in r]
                pb.keyframe_insert("rotation_euler", frame=f)
                pb.location = world_to_bone(pose.get(pb.name + "@loc", (0, 0, 0)))
                pb.keyframe_insert("location", frame=f)
        act.use_frame_range = True
        act.frame_start, act.frame_end = 0, frames
        act.use_cyclic = loop
        self.actions.append(act)
        return act

    def rest(self):
        arm = self.arm
        if arm.animation_data:
            arm.animation_data.action = None
        for pb in arm.pose.bones:
            pb.rotation_euler = (0, 0, 0)
            pb.rotation_quaternion = (1, 0, 0, 0)
            pb.location = (0, 0, 0)
            pb.scale = (1, 1, 1)


# ---------- типовые анимации пехоты ----------
# Знаки поворотов (градусы): X у ноги «+» — колено/стопа вперёд, у голени «−» — сгиб колена, у стопы «+» — носок
# вверх; у корпуса и головы «+» — назад (вверх), «−» — наклон вперёд; Y — поворот против часовой (вид сверху),
# Z — завал верха влево. Смещение '<кость>@loc' — в мире (x вправо, y вперёд, z вверх).
# Оружие у бойцов Коалиции закреплено на груди вместе с руками — руки держат его и не машут; свободные руки
# (free_arms) у рабочего и у пришельца (левая) ходят в такт шагу.

def smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def lerp_pose(a, b, k):
    keys = set(a) | set(b)
    out = {}
    for key in keys:
        va = a.get(key, (0, 0, 0))
        vb = b.get(key, (0, 0, 0))
        out[key] = tuple(x + (y - x) * k for x, y in zip(va, vb))
    return out


def keyed(keys):
    """Поза по ключам [(t, pose)] с плавной интерполяцией."""
    def fn(t):
        for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
            if t <= t1:
                return lerp_pose(p0, p1, smooth((t - t0) / max(1e-6, t1 - t0)))
        return keys[-1][1]
    return fn


def add(p, key, v):
    a = p.get(key, (0, 0, 0))
    p[key] = tuple(x + y for x, y in zip(a, v))


def _gait(t, stride, swing_knee, stance_knee, toe_off, lean, bob, rise, twist, sway, arm_swing, free_arms, run):
    """Цикл шага: две ноги в противофазе. Перенос (бедро идёт вперёд) — колено сгибается, носок поднят;
    опора — колено пружинит в середине, в конце — толчок носком. Таз качается к опорной ноге, проворачивается
    за выносной и опускается на её стороне; плечи — навстречу тазу, голова держит взгляд."""
    ph = 2 * math.pi * t
    p = {}
    for side, off in (("L", 0.0), ("R", math.pi)):
        s, c = math.sin(ph + off), math.cos(ph + off)
        thigh = stride * (s + 0.12 * math.sin(2 * (ph + off)))           # вынос быстрее, чем отвод назад
        swing = smooth(max(0.0, c)) ** 0.8
        knee = -(swing_knee * swing + stance_knee * max(0.0, -c) ** 2 + 6)
        foot = -(thigh + knee) * 0.85 + 10 * swing - toe_off * max(0.0, -s) ** 3 * smooth((c + 0.4) / 0.6)
        p["thigh." + side] = (thigh, 0, 0)
        p["shin." + side] = (knee, 0, 0)
        p["foot." + side] = (foot, 0, 0)
        if side in free_arms:
            sgn = 1 if side == "L" else -1
            p["upperarm." + side] = (-arm_swing * s, 0, -sgn * 6)
            p["forearm." + side] = ((50 if run else 18) + 15 * max(0.0, -s), 0, 0)
    sl, cl = math.sin(ph), math.cos(ph)
    step = abs(sl) if run else 1 - abs(sl)                               # бег — выше в полёте, шаг — на переходе
    p["hips@loc"] = (sway * cl, 0, -bob + rise * step)
    p["hips"] = (0, -twist * sl, 3 * cl)
    p["spine"] = (-lean, twist * 0.7 * sl, -3 * cl)
    p["chest"] = (-lean * 0.25 + (2.5 * abs(sl) if run else 0), twist * 0.9 * sl, 0)
    p["head"] = (lean * 1.1, -twist * 0.5 * sl, 0)
    return p


def walk_pose(stride=22.0, free_arms=()):
    return lambda t: _gait(t, stride, 42, 10, 18, 3, 0.02, 0.025, 6, 0.025, 20, free_arms, False)


def run_pose(stride=38.0, lean=13.0, free_arms=()):
    return lambda t: _gait(t, stride, 92, 26, 32, lean, 0.075, 0.07, 9, 0.018, 38, free_arms, True)


def _stance(breath=0.0, sway=0.0):
    """Спокойная стойка: ноги чуть шире плеч, вес чуть на правой; дыхание поднимает грудь."""
    return {"thigh.L": (2, 0, -3), "shin.L": (-5, 0, 0), "foot.L": (3, 0, 3),
            "thigh.R": (-3, 0, 4), "shin.R": (-3, 0, 0), "foot.R": (6, 0, -4),
            "hips@loc": (0.012 * sway, 0, -0.01), "hips": (0, 0, 1.5 * sway),
            "spine": (0.6 * breath, 0, -1.2 * sway), "chest": (1.3 * breath, 0, -0.4 * sway),
            "upperarm.L": (0, 0, 0.8 * breath), "upperarm.R": (0, 0, -0.8 * breath)}


def idle_pose():
    """4 с: два вдоха, вес плавно переходит с ноги на ногу, голова неспешно поводит."""
    def fn(t):
        ph = 2 * math.pi * t
        p = _stance(math.sin(2 * ph), math.sin(ph))
        p["head"] = (-1.0 * math.sin(2 * ph), 6 * math.sin(ph + 1.0), 1.5 * math.sin(ph))
        return p
    return fn


def look_pose():
    """Оглядывается: голова и плечи налево, задержка, направо, обратно."""
    base = _stance()
    def at(yaw, pitch=0.0):
        p = dict(base)
        p["head"] = (pitch, yaw, 0)
        p["chest"] = (0, yaw * 0.25, 0)
        p["spine"] = (0, yaw * 0.12, 0)
        return p
    return keyed([(0.0, at(0)), (0.18, at(38, 4)), (0.42, at(36, -3)), (0.6, at(-30, 2)), (0.82, at(-28)), (1.0, at(0))])


def shift_pose():
    """Переминается: вес на левую, правая нога переставляется, плечи поводит."""
    base = _stance()
    lean = dict(base)
    lean.update({"hips@loc": (-0.035, 0, -0.01), "hips": (0, 0, -3), "spine": (0, 0, 3)})
    lift = dict(lean)
    lift.update({"thigh.R": (14, 0, 6), "shin.R": (-28, 0, 0), "foot.R": (12, 0, -4)})
    plant = dict(base)
    plant.update({"thigh.R": (-6, 0, 6), "foot.R": (8, 0, -6), "chest": (2, 0, 3), "head": (-3, 0, -2)})
    return keyed([(0.0, base), (0.25, lean), (0.45, lift), (0.62, plant), (0.8, plant), (1.0, base)])


def aim_stance(sway=0.0):
    """Стойка для стрельбы: левая нога впереди, колени присогнуты, корпус подан вперёд и чуть развёрнут
    (правое плечо назад), голова к прицелу; оружие удерживается грудью и стоит по линии огня."""
    return {"thigh.L": (12, 0, -3), "shin.L": (-16, 0, 0), "foot.L": (4, 0, 3),
            "thigh.R": (-14, 0, 5), "shin.R": (-12, 0, 0), "foot.R": (24, 0, -4),
            "hips@loc": (0.01 * sway, 0, -0.045), "hips": (0, 12, 0),
            "spine": (-5, -6, 0), "chest": (-2 + 0.6 * sway, -6, 0), "head": (-3, -2, 0)}


def aim_pose():
    return lambda t: aim_stance(math.sin(2 * math.pi * t))


def shoot_pose(kick=8.0, slide=0.07):
    """Выстрел из стойки: резкая отдача в плечо (корпус и голова назад, оружие на себя), плавный возврат."""
    def fn(t):
        k = 0.0 if t <= 0 else (t / 0.12 if t < 0.12 else max(0.0, 1 - (t - 0.12) / 0.7))
        k = smooth(k)
        p = aim_stance()
        add(p, "chest", (kick * k, 0, kick * 0.15 * k))
        add(p, "spine", (kick * 0.3 * k, 0, 0))
        add(p, "head", (kick * 0.5 * k, 0, 0))
        add(p, "hips@loc", (0, -0.015 * k, 0))
        p["weapon@loc"] = (0, -slide * k, 0.01 * k)
        return p
    return fn


def hit_pose():
    """Попадание: корпус дёргается назад с разворотом, голова мотается, быстрое восстановление."""
    hit = {"hips@loc": (0, -0.04, -0.02), "spine": (8, 6, 0), "chest": (10, 8, -4), "head": (14, -8, 6)}
    return keyed([(0.0, {}), (0.12, hit), (0.35, {"chest": (3, 2, 0), "head": (4, 0, 0)}), (1.0, {})])


def _limp(p, loose_weapon):
    if not loose_weapon:
        for k in [k for k in p if k.startswith("weapon")]:
            p.pop(k)
    return p


def death_pose(height=0.92, loose_weapon=True):
    """Падение навзничь: шаг назад, колени подламываются, тело заваливается на спину, руки раскинуты."""
    stand = aim_stance()
    stagger = {"hips@loc": (0, -0.12, -0.04), "hips": (-6, 0, 3), "spine": (14, 0, 0), "chest": (12, 0, 0), "head": (20, 0, 0),
               "thigh.L": (-10, 0, 0), "shin.L": (-10, 0, 0), "thigh.R": (22, 0, 0), "shin.R": (-20, 0, 0),
               "upperarm.L": (25, 0, -20), "upperarm.R": (20, 0, 20), "weapon": (-15, 10, 0)}
    buckle = {"hips@loc": (0, -0.12, -0.32), "hips": (10, 0, 4), "spine": (12, 0, 0), "chest": (10, 0, 0),
              "head": (16, 0, 0), "thigh.L": (52, 0, 0), "shin.L": (-80, 0, 0), "thigh.R": (38, 0, 0),
              "shin.R": (-70, 0, 0), "upperarm.L": (20, 0, -15), "upperarm.R": (25, 0, 15), "weapon": (-25, 10, 0)}
    fall = {"hips@loc": (0, -0.62, -(height - 0.17)), "hips": (84, 0, 10), "spine": (6, 0, 0), "chest": (4, 0, 0),
            "head": (22, 18, 0), "thigh.L": (25, 0, 0), "shin.L": (-45, 0, 0), "foot.L": (20, 0, 0),
            "thigh.R": (8, 0, -6), "shin.R": (-15, 0, 0), "foot.R": (10, 0, 0),
            "upperarm.L": (20, 0, -50), "forearm.L": (-20, 0, 0), "upperarm.R": (15, 0, 55), "forearm.R": (-30, 0, 0),
            "weapon": (-80, 30, 0), "weapon@loc": (0.25, 0.1, 0.0)}
    bounce = dict(fall)
    bounce["hips@loc"] = (0, -0.64, -(height - 0.24))
    bounce["head"] = (12, 14, 0)
    settle = dict(fall)
    settle["hips@loc"] = (0, -0.66, -(height - 0.13))
    settle["head"] = (26, 24, 0)
    return keyed([(0.0, stand), (0.14, _limp(stagger, loose_weapon)), (0.36, _limp(buckle, loose_weapon)),
                  (0.66, _limp(fall, loose_weapon)), (0.76, _limp(bounce, loose_weapon)), (1.0, _limp(settle, loose_weapon))])


def death_fwd_pose(height=0.92, loose_weapon=True):
    """Падение ничком: колени подгибаются, боец падает на колени и валится лицом вперёд."""
    stand = aim_stance()
    knees = {"hips@loc": (0, 0.12, -(height * 0.42)), "hips": (-10, 0, 0), "spine": (-15, 0, 0), "chest": (-10, 0, 0),
             "head": (-10, 0, 0), "thigh.L": (75, 0, 0), "shin.L": (-110, 0, 0), "foot.L": (-30, 0, 0),
             "thigh.R": (68, 0, 0), "shin.R": (-105, 0, 0), "foot.R": (-30, 0, 0),
             "upperarm.L": (20, 0, -8), "upperarm.R": (15, 0, 8), "weapon": (20, 0, 0)}
    fall = {"hips@loc": (0, 0.55, -(height - 0.22)), "hips": (-82, 0, -6), "spine": (-6, 0, 0), "chest": (-4, 0, 0),
            "head": (-8, -30, 0), "thigh.L": (-4, 0, 0), "shin.L": (-25, 0, 0), "foot.L": (-40, 0, 0),
            "thigh.R": (6, 0, 6), "shin.R": (-40, 0, 0), "foot.R": (-35, 0, 0),
            "upperarm.L": (70, 0, -25), "forearm.L": (30, 0, 0), "upperarm.R": (40, 0, 30), "forearm.R": (20, 0, 0),
            "weapon": (70, -25, 0), "weapon@loc": (-0.2, 0.25, 0.0)}
    settle = dict(fall)
    settle["hips@loc"] = (0, 0.6, -(height - 0.17))
    return keyed([(0.0, stand), (0.3, _limp(knees, loose_weapon)), (0.45, _limp(knees, loose_weapon)),
                  (0.78, _limp(fall, loose_weapon)), (1.0, _limp(settle, loose_weapon))])


def death_spin_pose(height=0.92, loose_weapon=True):
    """Падение с разворотом: удар в плечо разворачивает корпус, боец оседает и валится на бок."""
    stand = aim_stance()
    twist = {"hips@loc": (0.05, -0.05, -0.08), "hips": (0, 35, 0), "spine": (6, 25, -10), "chest": (8, 20, -12),
             "head": (10, 25, -10), "thigh.L": (8, 0, -4), "shin.L": (-25, 0, 0), "thigh.R": (-10, 0, 10),
             "shin.R": (-35, 0, 0), "upperarm.L": (30, 0, -35), "upperarm.R": (10, 0, 30), "weapon": (-20, 30, 0)}
    sag = {"hips@loc": (-0.1, -0.05, -(height * 0.5)), "hips": (0, 70, 25), "spine": (10, 10, 10), "chest": (8, 10, 8),
           "head": (12, 15, 15), "thigh.L": (50, 0, -8), "shin.L": (-90, 0, 0), "thigh.R": (35, 0, 15),
           "shin.R": (-80, 0, 0), "upperarm.L": (20, 0, -40), "upperarm.R": (30, 0, 40), "weapon": (-40, 40, 0)}
    fall = {"hips@loc": (-0.45, -0.1, -(height - 0.18)), "hips": (0, 85, 82), "spine": (4, 0, 6), "chest": (4, 0, 4),
            "head": (6, 10, 20), "thigh.L": (35, 0, -5), "shin.L": (-55, 0, 0), "thigh.R": (20, 0, 10),
            "shin.R": (-30, 0, 0), "upperarm.L": (40, 0, -70), "forearm.L": (-20, 0, 0), "upperarm.R": (60, 0, 20),
            "forearm.R": (30, 0, 0), "weapon": (-70, 70, 0), "weapon@loc": (0.3, -0.2, 0.0)}
    settle = dict(fall)
    settle["hips@loc"] = (-0.48, -0.1, -(height - 0.14))
    return keyed([(0.0, stand), (0.16, _limp(twist, loose_weapon)), (0.45, _limp(sag, loose_weapon)),
                  (0.8, _limp(fall, loose_weapon)), (1.0, _limp(settle, loose_weapon))])


def carried(fn, carry):
    """Клип с оружием «у пояса»: к позе добавляется поза переноски (оружие опущено и развёрнуто поперёк тела —
    руки за ним следуют через IK; у пришельца — опущенная рука с пушкой)."""
    def f(t):
        p = fn(t)
        for k, v in carry.items():
            add(p, k, v)
        return p
    return f


def raise_pose(carry):
    """Вскидка: из переноски оружие идёт к плечу, ноги переходят в стойку для стрельбы; в конце — лёгкий
    дожим к прицелу."""
    low = _stance()
    for k, v in carry.items():
        add(low, k, v)
    over = aim_stance()
    add(over, "chest", (-2, 0, 0))
    add(over, "head", (-2, 0, 0))
    return keyed([(0.0, low), (0.7, over), (1.0, aim_stance())])


def lower_pose(carry):
    low = _stance()
    for k, v in carry.items():
        add(low, k, v)
    return keyed([(0.0, aim_stance()), (1.0, low)])


def death_knees_pose(height=0.92, loose_weapon=True):
    """Ранен: оседает на колени, держится, затем заваливается на бок."""
    stand = aim_stance()
    knees = {"hips@loc": (0, 0.05, -(height * 0.45)), "hips": (-4, 0, 0), "spine": (-18, 0, 4), "chest": (-14, 0, 4),
             "head": (-20, 0, 0), "thigh.L": (85, 0, -6), "shin.L": (-120, 0, 0), "foot.L": (-35, 0, 0),
             "thigh.R": (80, 0, 6), "shin.R": (-118, 0, 0), "foot.R": (-35, 0, 0),
             "upperarm.L": (10, 0, -10), "upperarm.R": (12, 0, 12), "weapon": (35, 0, 0), "weapon@loc": (0, 0.05, -0.2)}
    sway = dict(knees)
    sway.update({"spine": (-24, 0, -6), "chest": (-18, 0, -8), "head": (-28, 10, 0)})
    fall = {"hips@loc": (0.42, 0.1, -(height - 0.2)), "hips": (0, -20, -80), "spine": (-10, 0, 0), "chest": (-6, 0, 0),
            "head": (-10, 0, -15), "thigh.L": (60, 0, 0), "shin.L": (-95, 0, 0), "thigh.R": (45, 0, 0), "shin.R": (-80, 0, 0),
            "upperarm.L": (40, 0, -30), "forearm.L": (40, 0, 0), "upperarm.R": (70, 0, 60), "weapon": (60, 0, 50),
            "weapon@loc": (0.3, 0.2, -0.3)}
    settle = dict(fall)
    settle["hips@loc"] = (0.45, 0.1, -(height - 0.15))
    return keyed([(0.0, stand), (0.25, _limp(knees, loose_weapon)), (0.5, _limp(sway, loose_weapon)),
                  (0.82, _limp(fall, loose_weapon)), (1.0, _limp(settle, loose_weapon))])


def death_blast_pose(height=0.92, loose_weapon=True):
    """Взрыв: тело отбрасывает назад и вверх, оно переворачивается в воздухе и падает на спину в паре метров."""
    stand = aim_stance()
    lift = {"hips@loc": (0, -0.9, 0.55), "hips": (40, 0, 8), "spine": (15, 0, 0), "chest": (12, 0, 0), "head": (25, 0, 0),
            "thigh.L": (40, 0, -15), "shin.L": (-50, 0, 0), "thigh.R": (60, 0, 15), "shin.R": (-30, 0, 0),
            "upperarm.L": (60, 0, -70), "upperarm.R": (50, 0, 80), "weapon": (-90, 60, 0), "weapon@loc": (0.4, 0.5, 0.6)}
    land = {"hips@loc": (0, -2.0, -(height - 0.2)), "hips": (88, 0, -6), "spine": (4, 0, 0), "chest": (2, 0, 0),
            "head": (20, -15, 0), "thigh.L": (20, 0, -10), "shin.L": (-30, 0, 0), "thigh.R": (35, 0, 10), "shin.R": (-50, 0, 0),
            "upperarm.L": (30, 0, -80), "forearm.L": (-15, 0, 0), "upperarm.R": (20, 0, 85), "weapon": (-120, 80, 0),
            "weapon@loc": (0.8, -0.4, 0.0)}
    bounce = dict(land)
    bounce["hips@loc"] = (0, -2.15, -(height - 0.35))
    settle = dict(land)
    settle["hips@loc"] = (0, -2.25, -(height - 0.13))
    return keyed([(0.0, stand), (0.3, _limp(lift, loose_weapon)), (0.62, _limp(land, loose_weapon)),
                  (0.74, _limp(bounce, loose_weapon)), (1.0, _limp(settle, loose_weapon))])


def infantry_actions(m, height=0.92, free_arms=(), kick=8.0, slide=0.07, loose_weapon=True, shoot=True, carry=None):
    """Полный набор клипов бойца (30 к/с; бег и шаг — один цикл из двух шагов). carry — поза переноски оружия:
    в покое, на ходу и на бегу оружие опущено, в бою — вскидка (raise) в стойку и опускание (lower) обратно.
    Клипы гибели — без IK: руки отпускают оружие."""
    c = carry or {}
    wrap = (lambda fn: carried(fn, c)) if c else (lambda fn: fn)
    m.action("idle", 120, wrap(idle_pose()))
    m.action("idle_look", 100, wrap(look_pose()), loop=False)
    m.action("idle_shift", 70, wrap(shift_pose()), loop=False)
    m.action("walk", 34, wrap(walk_pose(free_arms=free_arms)))
    m.action("run", 20, wrap(run_pose(free_arms=free_arms)))
    m.action("run_b", 18, wrap(run_pose(stride=44, lean=19, free_arms=free_arms)))      # бег пригнувшись
    if shoot:
        m.action("raise", 12, raise_pose(c), loop=False)
        m.action("lower", 16, lower_pose(c), loop=False)
        m.action("aim", 90, aim_pose())
        m.action("shoot", 14, shoot_pose(kick=kick, slide=slide), loop=False)
    m.action("hit", 16, hit_pose(), loop=False)
    m.action("die", 44, death_pose(height, loose_weapon), loop=False, ik=False)
    m.action("die_fwd", 44, death_fwd_pose(height, loose_weapon), loop=False, ik=False)
    m.action("die_spin", 46, death_spin_pose(height, loose_weapon), loop=False, ik=False)
    m.action("die_knees", 60, death_knees_pose(height, loose_weapon), loop=False, ik=False)
    m.action("die_blast", 40, death_blast_pose(height, loose_weapon), loop=False, ik=False)


def preview_frames(model, name, action, frames, dist=4.0, target=(0, 0, 0.9)):
    model.arm.animation_data.action = action
    for f in frames:
        bpy.context.scene.frame_set(f)
        style.preview(f"{name}_{action.name}_{f:02d}", target=target, dist=dist, views=("hero",), res=(480, 480))
