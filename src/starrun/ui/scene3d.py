"""Мир в 3D: десантник и жуки — моделями со скелетом, лес, стены, заборы и скалы — статичными мешами,
земля и эффекты — прежними 2D-слоями, лёгшими на плоскость через перспективную камеру. Тени — от солнца.

Порядок кадра: карта теней → земля (2D) → тени на землю → меши с проверкой глубины → полупрозрачное
(лес перед десантником, крыши) → эффекты на высоте груди.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path

import arcade
import numpy as np

from starrun.sim import terrain as T
from starrun.sim.swarm import BRAIN, FLYER, HIT_FLASH, QUEEN, SEMI_BOSS, SPITTER, TANKER, WARRIOR, WORKER
from starrun.sim.world import World
from starrun.ui import level_details
from starrun.ui import mesh3d as M
from starrun.ui.fx3d import GROUND_LIGHT_FS, Effects3D, Light, flame_puffs, splash_drops
from starrun.ui.gl3d import PX_PER_M, Camera3D, look_at, ortho
from starrun.ui.ground3d import Ground3D
from starrun.ui.gun_feedback import recoil_step
from starrun.ui.scene import ARC_LIFE, SceneRenderer
from starrun.ui.terrain_view import TerrainView, cell_hash
from starrun.ui.visibility import prop_opacity, shell_alpha

MODELS = Path(__file__).resolve().parents[3] / "assets" / "models"
SUN = np.array([0.42, -0.5, 0.76]) / np.linalg.norm([0.42, -0.5, 0.76])   # солнце с юго-востока: тени — на северо-запад
SHADOW_SIZE = 2048
SHADOW_HALF = 1000.0          # полуразмер области карты теней вокруг десантника: видимая земля — до ~±900 px
WALL_H = 110.0                # высота стен построек, px (2,3 м)
PROPS_CHUNK = 8               # окно статичных мешей сдвигается шагом в столько клеток — пересборка редкая
BLOB = (0.0, 0.0, 0.0, 0.3)   # пятно тени под жуком (толпа не рисуется в карту теней)
EFFECTS_LIFT = 52.0           # пули и пламя — на высоте груди
CORPSE_LIFE = 4.0
MARINE_RUN_SPEED = 230.0      # при этой скорости клип бега идёт с родной частотой
MARINE_TURN = 11.0            # рад/с: разворот к новой цели — плавный, а не прыжком
CLIP_FADE = 0.15              # с: смена клипа (бег вперёд → вбок, бег → рубка) — перетеканием

# Вид жука → модель, радиус вида (px), при котором модель в родном размере (охват ног ≈ 1,35 радиуса),
# подсветка и высота полёта
BUG_LOOK = {
    WORKER: ("worker", 20.0, (0, 0, 0, 0), 0.0),
    WARRIOR: ("warrior", 24.0, (0, 0, 0, 0), 0.0),
    SPITTER: ("spitter", 18.5, (0, 0, 0, 0), 0.0),
    FLYER: ("flyer", 13.0, (0, 0, 0, 0), 55.0),
    TANKER: ("tanker", 26.0, (0, 0, 0, 0), 0.0),
    BRAIN: ("brain", 16.0, (0, 0, 0, 0), 0.0),
    SEMI_BOSS: ("elite", 24.0, (0, 0, 0, 0), 0.0),
    QUEEN: ("queen", 60.0, (0, 0, 0, 0), 0.0),            # мешок волочится далеко за кругом столкновений
}
# Клип удара: играет, пока с удара (сброса отката атаки в симуляции) прошло меньше длины клипа
ATTACK_CLIP = {"worker": "attack", "warrior": "strike", "spitter": "attack", "flyer": "attack", "tanker": "attack",
               "elite": "strike"}
WINDUP = (WARRIOR, SPITTER)       # замах по сигналу симуляции (sw.windup)
# Боссы: клип по таймерам симуляции — (клип, таймер, период из настроек вида); с события прошло период − таймер
BOSS_CLIPS = {QUEEN: (("attack", "timer_a", "volley_every"), ("summon", "timer_b", "summon_every")),
              SEMI_BOSS: (("windup", "timer_b", "summon_every"),)}
FALL_TIME = 0.5                   # убитый летающий падает на землю
GUNS = ("mar", "shotgun", "sniper", "flame", "rocket", "shock")   # подмеши стволов в модели десантника
ROOF_COLORS = ((0.46, 0.22, 0.17), (0.3, 0.37, 0.41), (0.4, 0.39, 0.32), (0.5, 0.42, 0.3))   # ржавчина, сталь, …
WALL_COLOR = (0.58, 0.56, 0.52)
ITEM_MODELS = {T.MEDKIT: "medkit", T.MAGNET: "magnet", T.DNA: "dna", T.WEAPON_CRATE: "crate",
               T.UNDERBARREL: "underbarrel"}


def tree_mesh(kind: str, color: tuple[int, int, int]) -> list[M.Part]:
    """Деревья биомов, высота 1,6–3 м: ниже настоящих, чтобы не закрывать бой с наклонной камеры."""
    c = tuple(v / 255 for v in color)
    lt = tuple(min(1.0, v * 1.25) for v in c)
    bark = (0.36, 0.26, 0.17)
    if kind == "cactus":
        # ствол почти в клетку шириной: видимый кактус — там, где десантник в него упирается
        col = [(12.0, 0), (13.5, 6), (13.5, 46), (11.0, 56), (0.0, 60)]
        arm = [(6.5, 0), (7.0, 4), (7.0, 20), (5.5, 25), (0.0, 27)]
        return [M.lathe(col, c, 12), M.lathe(arm, c, 8, center=(17, 0, 22)), M.lathe(arm, c, 8, center=(-16, 3, 30)),
                M.box((18, 9, 9), (9, 0, 24), c), M.box((17, 9, 9), (-8, 2, 32), c)]
    if kind == "pine":
        return [M.lathe([(4.5, 0), (3.5, 30)], bark, 8),
                M.lathe([(25, 0), (22, 4), (0, 46)], c, 12, center=(0, 0, 22), jag=0.12, seed=1),
                M.lathe([(19, 0), (16, 4), (0, 38)], lt, 12, center=(0, 0, 48), jag=0.12, seed=2),
                M.lathe([(13, 0), (11, 3), (0, 30)], lt, 10, center=(0, 0, 72), jag=0.1, seed=3)]
    if kind == "palm":
        parts = [M.lathe([(5, 0), (4.5, 30), (4, 60), (3.5, 82)], bark, 8)]
        for k in range(7):
            yaw = k * 2 * math.pi / 7 + 0.3
            parts.append(M.leaf(38, 7, yaw, 22, (0, 0, 82), c if k % 2 else lt))
        parts.append(M.blob(7, (0, 0, 84), bark, seg=8, rings=4))
        return parts
    # болотное: кривой ствол, приплюснутые кроны, свисающий мох
    parts = [M.lathe([(6, 0), (5, 25), (4, 48)], bark, 8)]
    for i, (x, y, z, r) in enumerate(((0, 0, 60, 24), (12, 6, 52, 17), (-11, -5, 54, 16))):
        parts.append(M.blob(r, (x, y, z), c if i else lt, squash=0.7, jag=0.18, seed=i))
    for x, y in ((14, -8), (-10, 12), (4, 16)):
        parts.append(M.box((3, 3, 18), (x, y, 40), (0.22, 0.26, 0.16)))
    return parts


def bridge_cell() -> list[M.Part]:
    """Клетка моста, идущего вдоль Y: доски поперёк с зазорами на двух балках, по краям — столбики и перила."""
    plank, dark = (0.47, 0.35, 0.22), (0.3, 0.22, 0.14)
    parts = [M.box((40, 6.2, 4), (0, -20 + 4 + 8 * k, 6), plank if k % 2 else tuple(v * 0.92 for v in plank))
             for k in range(5)]
    parts += [M.box((5, 40, 5), (x, 0, 2), dark) for x in (-12, 12)]
    for s in (-1, 1):
        parts.append(M.box((4, 4, 26), (s * 18, 0, 13), dark))
        parts.append(M.box((4, 40, 4), (s * 18, 0, 24), plank))
    return parts


def shadow_proxy(kind: str) -> list[M.Part]:
    """Силуэт дерева для карты теней: тела вращения по 8 граней, 30–50 треугольников вместо сотен. Силуэт чуть
    меньше модели из Blender (flora.py) и целиком внутри неё — иначе он бросал бы тень на свою же крону. Пиксели."""
    c = (0.0, 0.0, 0.0)
    if kind == "pine":
        return [M.lathe([(26, 16), (17, 45), (9, 70), (0, 92)], c, 8)]
    if kind == "palm":
        return [M.lathe([(3.5, 0), (2.5, 76)], c, 6),
                M.lathe([(0, 82), (24, 80), (34, 66), (0, 72)], c, 8)]
    if kind == "cactus":
        return [M.lathe([(10, 0), (10, 46), (5, 55), (0, 57)], c, 8)]
    return [M.lathe([(4.5, 0), (3, 62)], c, 6),
            M.lathe([(0, 62), (18, 64), (26, 72), (18, 80), (0, 82)], c, 8)]


@dataclass
class Corpse:
    pos: np.ndarray
    yaw: float
    model: str
    scale: float
    tint: tuple[float, float, float, float]
    lift: float = 0.0
    age: float = 0.0


class Scene3D:
    def __init__(self, ctx: arcade.ArcadeContext) -> None:
        self.ctx = ctx
        self.ground = Ground3D(ctx)
        self.skinned = ctx.program(vertex_shader=M.SKINNED_VS, fragment_shader=M.SKINNED_FS)
        self.skinned_depth = ctx.program(vertex_shader=M.SKINNED_VS, fragment_shader=M.DEPTH_FS)
        self.static = ctx.program(vertex_shader=M.STATIC_VS, fragment_shader=M.STATIC_FS)
        self.static_depth = ctx.program(vertex_shader=M.STATIC_VS, fragment_shader=M.DEPTH_FS)
        self.details = {name: M.StaticMesh(ctx, parts, fadeable=False, wind=name in ("grass", "fern", "reed", "scrub"))
                        for name, parts in level_details.meshes().items()}
        self.receiver = ctx.program(vertex_shader=M.RECEIVER_VS, fragment_shader=M.RECEIVER_FS)
        names = {"marine", "turret"} | {look[0] for look in BUG_LOOK.values()}
        self.models = {n: M.SkinnedModel(ctx, MODELS / f"{n}.npz") for n in sorted(names)}
        self.trees: dict[tuple[str, tuple[int, int, int]], list[M.StaticMesh]] = {}
        self.shadow_lod: dict[M.StaticMesh, M.StaticMesh] = {}   # дерево → упрощённое для карты теней
        wall, glass, door = (0.36, 0.37, 0.39), (0.42, 0.62, 0.7), (0.42, 0.29, 0.18)
        self.props = {
            "wall": M.StaticMesh(ctx, [M.box((40, 14, WALL_H), (0, 0, WALL_H / 2), wall)]),
            "window": M.StaticMesh(ctx, [M.box((40, 14, 46), (0, 0, 23), wall),
                                         M.box((40, 14, 28), (0, 0, WALL_H - 14), wall),
                                         M.box((40, 6, 36), (0, 0, 64), glass)]),
            "door": M.StaticMesh(ctx, [M.box((40, 14, 24), (0, 0, WALL_H - 12), wall),
                                       M.box((36, 8, 86), (0, 0, 43), door)]),
            "door_open": M.StaticMesh(ctx, [M.box((40, 14, 24), (0, 0, WALL_H - 12), wall),
                                            M.box((8, 34, 86), (-17, 18, 43), door)]),
            "post": M.StaticMesh(ctx, [M.box((7, 7, 44), (0, 0, 22), (0.5, 0.4, 0.27))]),
            "rail": M.StaticMesh(ctx, [M.box((40, 4, 5), (20, 0, 18), (0.48, 0.38, 0.25)),
                                       M.box((40, 4, 5), (20, 0, 34), (0.48, 0.38, 0.25))]),
            "rock": M.StaticMesh(ctx, [M.blob(24, (0, 0, 18), (0.42, 0.4, 0.37), squash=0.75, jag=0.2, seed=5)]),
            "bridge": M.StaticMesh(ctx, bridge_cell(), fadeable=False),
        }
        # детали из Blender (tools/art/props.py) заменяют процедурные, если файлы есть
        for name in ("wall", "window", "door", "door_open", "post", "rail"):
            path = MODELS / f"prop_{name}.npz"
            if path.exists():
                self.props[name] = M.StaticMesh(ctx, [M.load_part(path, PX_PER_M)], fadeable=name != "rail")
        self.shell_names = {self.props[name]: name for name in ("wall", "window", "door", "door_open")}
        concrete, cap = (0.39, 0.41, 0.40), (0.56, 0.58, 0.55)
        sill = [M.box((40, 14, 10), (0, 0, 5), concrete), M.box((40, 16, 2), (0, 0, 11), cap)]
        jambs = [M.box((5, 14, 15), (x, 0, 7.5), cap) for x in (-18, 18)]
        threshold = [M.box((34, 12, 3), (0, 0, 1.5), (0.35, 0.38, 0.35))]
        self.cutaway = {
            "wall": M.StaticMesh(ctx, sill, fadeable=False),
            "window": M.StaticMesh(ctx, sill, fadeable=False),
            "door": M.StaticMesh(ctx, jambs + threshold +
                                  [M.box((31, 6, 28), (0, 0, 14), (0.38, 0.42, 0.32))], fadeable=False),
            "door_open": M.StaticMesh(ctx, jambs + threshold, fadeable=False),
        }
        self.rocks = [M.StaticMesh(ctx, [M.load_part(p, PX_PER_M)]) for p in sorted(MODELS.glob("prop_rock_*.npz"))]
        self.stumps = {p.stem.removeprefix("prop_stump_"): M.StaticMesh(ctx, [M.load_part(p, PX_PER_M)], fadeable=False)
                       for p in MODELS.glob("prop_stump_*.npz")}
        self.items = {kind: M.StaticMesh(ctx, [M.load_part(MODELS / f"prop_{name}.npz", PX_PER_M)], fadeable=False)
                      for kind, name in ITEM_MODELS.items() if (MODELS / f"prop_{name}.npz").exists()}
        self.shadow_tex = ctx.depth_texture((SHADOW_SIZE, SHADOW_SIZE))
        self.shadow_tex.compare_func = None                       # читаем глубину как число (см. LIGHT в mesh3d)
        self._shadow_color = ctx.texture((SHADOW_SIZE, SHADOW_SIZE), components=1)   # Arcade требует цветовой слой
        self.shadow_fbo = ctx.framebuffer(color_attachments=[self._shadow_color], depth_attachment=self.shadow_tex)
        self.quad = ctx.buffer(reserve=4 * 2 * 4)
        self.quad_geometry = ctx.geometry([arcade.gl.BufferDescription(self.quad, "2f", ["in_pos"])],
                                          mode=ctx.TRIANGLE_STRIP)
        self.light = np.identity(4)
        self.fx = Effects3D(ctx)
        self.case_mesh = M.StaticMesh(ctx, [M.box((6, 2.5, 2.5), (0, 0, 0), (0.72, 0.54, 0.25))], fadeable=False)
        self.mag_mesh = M.StaticMesh(ctx, [M.box((5, 10, 4), (0, 0, 0), (0.20, 0.23, 0.22))], fadeable=False)
        self.no_shadow = set(self.details.values()) | {self.case_mesh, self.mag_mesh}
        self.ground_light = ctx.program(vertex_shader=M.RECEIVER_VS, fragment_shader=GROUND_LIGHT_FS)
        self.point_lights: dict[str, tuple[float, ...]] = {}
        self._props_key: tuple | None = None  # type: ignore[type-arg]
        self._props_cache: list[tuple[M.StaticMesh, np.ndarray]] = []
        self.roof_meshes: dict[tuple[int, int], tuple[M.StaticMesh, np.ndarray, float]] = {}   # (карта, постройка)
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        if hasattr(self, "fx"):
            self.fx.reset()
        self.corpses: list[Corpse] = []
        self.prev_alive: np.ndarray | None = None
        self.marine_clip, self.marine_t = "idle", 0.0
        self.marine_yaw: float | None = None
        self.fade_from: tuple[str, float] | None = None       # прежний клип и его время, пока идёт перетекание
        self.fade_left = 0.0
        self.muzzle: np.ndarray | None = None                 # дуло в мире (x, y, z) — для вспышки выстрела
        self.prev_pos: np.ndarray | None = None
        self.death_t = 0.0
        self.shot_t = 999.0
        self.recoil = 0.0

    # ---------- состояние между кадрами ----------
    def update(self, dt: float, world: World) -> None:
        self.time += dt
        sw = world.swarm
        if self.prev_alive is not None:
            died = np.flatnonzero(self.prev_alive & ~sw.alive)
            for i in died[:60]:
                name, r0, tint, lift = BUG_LOOK[int(sw.kind[i])]
                to = world.player.pos - sw.pos[i]
                self.corpses.append(Corpse(sw.pos[i].copy(), math.atan2(to[1], to[0]) - math.pi / 2, name,
                                           float(sw.radius[i]) / r0, tint, lift))
        self.prev_alive = sw.alive.copy()
        self._marine_state(dt, world)
        self.fx.muzzle = self.muzzle
        self.fx.update(dt, world)
        for c in self.corpses:
            c.age += dt
        self.corpses = [c for c in self.corpses if c.age < CORPSE_LIFE][-300:]

    def _marine_state(self, dt: float, world: World) -> None:
        """Клип десантника: ноги — по ходу движения относительно взгляда (вперёд, назад, вбок)."""
        p = world.player
        self.recoil = recoil_step(self.recoil, dt, world.events)
        self.shot_t = 0.0 if any(e.startswith("shot_") for e in world.events) else self.shot_t + dt
        speed = 0.0 if self.prev_pos is None or dt <= 0 else float(np.linalg.norm(p.pos - self.prev_pos)) / dt
        self.prev_pos = p.pos.copy()
        if world.session.phase.name == "DEFEAT":
            self.death_t += dt
            clip, t = "die", self.death_t
        elif p.vaulting:
            clip, t = "vault", (self.marine_t + dt) if self.marine_clip == "vault" else 0.0
        elif p.chop >= 0:
            clip, t = "chop", p.chop * (self.models["marine"].clips["chop"].frames - 1) / 30.0
        elif speed > 20:
            rel = math.atan2(p.heading[1], p.heading[0]) - math.atan2(p.facing[1], p.facing[0])
            rel = (rel + math.pi) % (2 * math.pi) - math.pi
            a = abs(rel)
            clip = "run_f" if a < math.pi / 4 else "run_b" if a > 3 * math.pi / 4 else ("run_l" if rel > 0 else "run_r")
            t = self.marine_t + dt * speed / MARINE_RUN_SPEED
        elif self.shot_t < 0.18:
            clip, t = "shoot", self.shot_t
        else:
            clip, t = "idle", self.marine_t + dt
        if clip != self.marine_clip and clip != "die":
            self.fade_from, self.fade_left = (self.marine_clip, self.marine_t), CLIP_FADE
        elif self.fade_from is not None:
            self.fade_from = (self.fade_from[0], self.fade_from[1] + dt)
        self.fade_left = max(0.0, self.fade_left - dt)
        self.marine_clip, self.marine_t = clip, t
        # разворот к цели с ограниченной скоростью
        facing = p.heading if p.chop >= 0 else p.facing
        want = math.atan2(facing[1], facing[0]) - math.pi / 2
        if self.marine_yaw is None:
            self.marine_yaw = want
        diff = (want - self.marine_yaw + math.pi) % (2 * math.pi) - math.pi
        self.marine_yaw += max(-MARINE_TURN * dt, min(MARINE_TURN * dt, diff))
        a, b, f = self._marine_frames()
        local = self.models["marine"].bone_point("muzzle", a, b, f) * PX_PER_M
        k = float(np.clip((local[2] / PX_PER_M - 0.95) / 0.4, 0, 1))
        kick = self.recoil * k * k * (3 - 2 * k)
        y, z = local[1], local[2] - 1.1 * PX_PER_M
        local[1] = math.cos(kick) * y - math.sin(kick) * z
        local[2] = math.sin(kick) * y + math.cos(kick) * z + 1.1 * PX_PER_M
        c, s = math.cos(self.marine_yaw), math.sin(self.marine_yaw)
        self.muzzle = np.array([p.pos[0] + c * local[0] - s * local[1], p.pos[1] + s * local[0] + c * local[1],
                                local[2]], np.float32)

    def _marine_frames(self) -> tuple[float, float, float]:
        """Кадры позы: внутри клипа — соседние кадры, при смене клипа — кадр прежнего и кадр нового."""
        mar = self.models["marine"]
        a, b, f = (float(v) for v in mar.frames(self.marine_clip, self.marine_t))
        if self.fade_from is None or self.fade_left <= 0:
            return a, b, f
        oa, ob, of = (float(v) for v in mar.frames(*self.fade_from))
        return (oa if of < 0.5 else ob), (a if f < 0.5 else b), 1 - self.fade_left / CLIP_FADE

    # ---------- кадр ----------
    def draw(self, world: World, cam: Camera3D, scene: SceneRenderer, shapes) -> None:  # type: ignore[no-untyped-def]
        ctx = self.ctx
        rect = cam.ground_rect()
        view = (rect[0] - 80, rect[1] - 80, rect[2] + 80, rect[3] + 120)
        scene.map.prepare(world, view)
        props = self._visible(self._props(world, rect), rect) + self._items(world)
        scene.map.flat_stumps = scene.map.flat_items = not self.stumps    # пни и находки — моделями
        scene.map.flat_ground = False
        props = self._cutaway_props(world, props, scene.map.roof_alpha)
        props += [(self.case_mesh, self.fx.cases.instances()),
                  (self.mag_mesh, self.fx.cases.instances(magazine=True))]
        roofs = self._roofs(world, scene.map)
        units = self._units(world)

        arcs = [Light((float(a.points[len(a.points) // 2][0]), float(a.points[len(a.points) // 2][1]), 52.0), 220.0,
                      (0.6, 1.0, 1.6), ARC_LIFE, age=ARC_LIFE - a.left) for a in scene.arcs]
        self.point_lights = self.fx.light_uniforms(arcs)
        self._shadow_pass(cam, props, units)
        cam.apply(ctx)
        ctx.disable(ctx.DEPTH_TEST)
        self.ground.draw(world.terrain, cam, self.time, rect)
        scene.ground_layers(world, view, solid=False)
        scene.turret_rings(world)
        scene.blast_rings()
        self._blob_shadows(world, shapes)
        shapes.flush()
        self._receive_shadows(cam, rect)
        self._ground_lights(cam)

        ctx.enable(ctx.DEPTH_TEST)
        self._set_common(self.static, cam.view_proj, cam)
        self._set_common(self.skinned, cam.view_proj, cam)
        transparent = []
        for mesh, inst in props:
            alpha = (prop_opacity(inst[:, :2], world.player.pos, world.terrain, scene.map.roof_alpha)
                     if mesh.fadeable else 1.0)
            faded = inst.copy()
            faded[:, 7] *= alpha
            opaque = faded[:, 7] >= 0.995
            mesh.draw(self.static, faded[opaque])
            if (~opaque).any():
                transparent.append((mesh, faded[~opaque]))
        for name, inst, subs in units:
            self.skinned["u_recoil"] = self.recoil if name == "marine" else 0.0
            self.models[name].draw(self.skinned, inst, subs)
        transparent += roofs
        ctx.screen.depth_mask = False
        def distance(pair: tuple[M.StaticMesh, np.ndarray]) -> float:
            return float(np.linalg.norm(pair[1][:, :3] - cam.eye, axis=1).mean())
        for mesh, inst in sorted(transparent, key=distance, reverse=True):
            order = np.argsort(-np.linalg.norm(inst[:, :3] - cam.eye, axis=1))
            mesh.draw(self.static, inst[order])
        ctx.screen.depth_mask = True
        ctx.disable(ctx.DEPTH_TEST)
        self.fx.draw(cam.view_proj, cam.view, flame_puffs(world), splash_drops(world))

        shapes.lift = EFFECTS_LIFT
        scene.effects(world, volumetric=True)
        shapes.flush()
        shapes.lift = 0.0

    def _set_common(self, prog: arcade.gl.Program, vp: np.ndarray, cam: Camera3D | None) -> None:
        values: dict[str, object] = {"u_vp": tuple(vp.T.ravel()), "u_px": PX_PER_M, "u_anim": 1, "u_time": self.time}
        if cam is not None:
            values.update(u_albedo=2, u_mask=3, u_sun=tuple(SUN), u_eye=tuple(cam.eye), u_shadow=4,
                          u_light=tuple(self.light.T.ravel()), u_texel=1.0 / SHADOW_SIZE, **self.point_lights)
            self.shadow_tex.use(4)
        for name, value in values.items():
            with contextlib.suppress(KeyError):   # униформы нет в шейдере (или компилятор выкинул неиспользуемую)
                prog[name] = value

    def _cutaway_props(self, world: World, props: list[tuple[M.StaticMesh, np.ndarray]],
                       roof_alpha: dict[int, float]) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """У раскрытого дома верх стен исчезает полностью, остаётся тонкий низкий срез."""
        if world.terrain is None:
            return props
        out = []
        for mesh, inst in props:
            name = self.shell_names.get(mesh)
            if name is None:
                out.append((mesh, inst))
                continue
            upper = shell_alpha(inst[:, :2], world.terrain, roof_alpha)
            if (upper > 0.01).any():
                high = inst[upper > 0.01].copy()
                high[:, 7] *= upper[upper > 0.01]
                out.append((mesh, high))
            if (upper < 0.99).any():
                low = inst[upper < 0.99].copy()
                low[:, 7] *= 1 - upper[upper < 0.99]
                out.append((self.cutaway[name], low))
        groups: dict[M.StaticMesh, list[np.ndarray]] = {}
        for mesh, inst in out:
            groups.setdefault(mesh, []).append(inst)
        return [(mesh, chunks[0] if len(chunks) == 1 else np.concatenate(chunks)) for mesh, chunks in groups.items()]

    def _shadow_pass(self, cam: Camera3D, props: dict, units: list) -> None:  # type: ignore[type-arg]
        target = cam.target
        eye = target + SUN * 3000
        self.light = ortho(SHADOW_HALF, SHADOW_HALF, 100.0, 6000.0) @ look_at(eye, target, (0, 1, 0))
        ctx = self.ctx
        self.shadow_fbo.use()
        self.shadow_fbo.clear(depth=1.0)
        ctx.enable(ctx.DEPTH_TEST)
        self._set_common(self.static_depth, self.light, None)
        self._set_common(self.skinned_depth, self.light, None)
        for mesh, inst in props:
            if mesh in self.no_shadow:
                continue  # мелкий декор не перегружает карту теней
            if (inst[:, 7] < 0.5).any():
                inst = inst[inst[:, 7] >= 0.5]
            self.shadow_lod.get(mesh, mesh).draw(self.static_depth, inst)
        for name, inst, subs in units:
            if name in ("marine", "turret"):   # жуки — пятнами (_blob_shadows): толпа дорога для карты теней
                self.skinned_depth["u_recoil"] = self.recoil if name == "marine" else 0.0
                self.models[name].draw(self.skinned_depth, inst, subs, depth=True)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.screen.use()

    def _ground_lights(self, cam: Camera3D) -> None:
        """Свет взрывов и выстрелов на землю: тот же квад, что под тени, — сложением."""
        self._set_common(self.ground_light, cam.view_proj, cam)
        self.ctx.blend_func = self.ctx.BLEND_ADDITIVE
        self.quad_geometry.render(self.ground_light)
        self.ctx.blend_func = self.ctx.BLEND_DEFAULT

    def _receive_shadows(self, cam: Camera3D, rect: tuple[float, float, float, float]) -> None:
        l, b, r, t = rect  # noqa: E741
        self.quad.write(np.array([l, b, r, b, l, t, r, t], np.float32).tobytes())
        self._set_common(self.receiver, cam.view_proj, cam)
        self.ctx.enable(self.ctx.BLEND)
        self.quad_geometry.render(self.receiver)

    # ---------- экземпляры ----------
    def _tree(self, biome) -> list[M.StaticMesh]:  # type: ignore[no-untyped-def]
        """Варианты дерева биома: модели из Blender (flora_<вид>_<n>.npz), иначе — процедурное из кода."""
        key = (biome.tree, tuple(biome.tree_color))
        if key not in self.trees:
            files = sorted(MODELS.glob(f"flora_{biome.tree}_*.npz"))
            meshes = [M.StaticMesh(self.ctx, [M.load_part(f, PX_PER_M)], wind=biome.tree != "cactus") for f in files]
            self.trees[key] = meshes or [M.StaticMesh(self.ctx, tree_mesh(biome.tree, biome.tree_color))]
            # в карту теней — грубый силуэт: деревьев на экране сотни, а тень от лап и листьев не разглядеть
            proxy = M.StaticMesh(self.ctx, shadow_proxy(biome.tree))
            self.shadow_lod.update({mesh: proxy for mesh in self.trees[key]})
        return self.trees[key]

    @staticmethod
    def _visible(props: list[tuple[M.StaticMesh, np.ndarray]], rect: tuple[float, float, float, float],
                 margin: float = 140.0) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """Окно мешей шире экрана (сдвигается кусками) — рисуем только видимое и полосу под тени к северо-западу."""
        l, b, r, t = rect  # noqa: E741
        out = []
        for mesh, inst in props:
            x, y = inst[:, 0], inst[:, 1]
            keep = (x > l - margin) & (x < r + margin) & (y > b - margin) & (y < t + margin * 1.5)
            if keep.any():
                out.append((mesh, inst if keep.all() else inst[keep]))
        return out

    def _props(self, world: World, rect: tuple[float, float, float, float]) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """Лес, стены, двери, окна, заборы, скалы в окне вокруг видимой земли. Окно сдвигается кусками по
        PROPS_CHUNK клеток, а экземпляры пересобираются, только когда сдвинулось окно или изменилась карта
        (рубка, взрыв, дверь) — иначе это 2 мс Python каждый кадр."""
        t = world.terrain
        if t is None:
            return []
        cell, ch = t.cell, PROPS_CHUNK
        x0 = int(rect[0] // cell) // ch * ch - ch
        y0 = int(rect[1] // cell) // ch * ch - ch
        x1 = (int(rect[2] // cell) // ch + 2) * ch
        y1 = (int(rect[3] // cell) // ch + 2) * ch
        doors = tuple(int(t.kind[y, x]) for x, y in t.doors)
        key = (id(t), t.version, x0, y0, x1, y1, doors)
        if key != self._props_key:
            self._props_key = key
            self._props_cache = self._build_props(t, x0, y0, x1, y1)
        return self._props_cache

    def _build_props(self, t: T.Terrain, x0: int, y0: int, x1: int, y1: int) -> list[tuple[M.StaticMesh, np.ndarray]]:
        kind = np.full((y1 - y0, x1 - x0), T.ROCK, np.uint8)       # за краем карты — скалы
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(t.n, x1), min(t.n, y1)
        if sx0 < sx1 and sy0 < sy1:
            kind[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = t.kind[sy0:sy1, sx0:sx1]
        cell = t.cell
        out: list[tuple[M.StaticMesh, np.ndarray]] = []

        def cells(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            ys, xs = np.nonzero(mask)
            xs, ys = xs + x0, ys + y0
            return xs, ys, np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], 1).astype(np.float32)

        # лес
        xs, ys, pos = cells(kind == T.TREE)
        if len(xs):
            h = cell_hash(xs, ys, 7)
            spread = 6.0 if t.biome.tree == "cactus" else 30.0     # кроны шире клетки — их можно сдвинуть, кактус нет
            pos = pos + (np.stack([cell_hash(xs, ys, 9), cell_hash(xs, ys, 8)], 1) - 0.5) * spread   # без рядов
            hp = np.clip(t.hp[np.clip(ys, 0, t.n - 1), np.clip(xs, 0, t.n - 1)] / max(1, t.biome.tree_hp), 0, 1)
            size = 0.9 + 0.2 * h if t.biome.tree == "cactus" else 0.8 + 0.4 * h ** 1.5   # разнообразие — и вариантами
            s = size * (0.7 + 0.3 * hp)
            inst = M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s], 1))
            variants = self._tree(t.biome)
            pick = (cell_hash(xs, ys, 5) * len(variants)).astype(int) % len(variants)
            for v, mesh in enumerate(variants):
                if (pick == v).any():
                    out.append((mesh, inst[pick == v]))
        # стены, окна, двери
        for k, name in ((T.WALL, "wall"), (T.WINDOW, "window"), (T.DOOR, "door"), (T.OPEN_DOOR, "door_open")):
            xs, ys, pos = cells(kind == k)
            if not len(xs):
                continue
            along_x = TerrainView._along_x(kind, xs - x0, ys - y0)
            yaw = np.where(along_x, 0.0, math.pi / 2)
            out.append((self.props[name], M.instances(pos, yaw=yaw)))
            if k == T.WALL:
                # У тонкой угловой секции достраиваем половину перпендикулярной стены до соседней клетки.
                lx, ly = xs - x0, ys - y0
                wallish = np.isin(kind, (T.WALL, T.WINDOW, T.DOOR, T.OPEN_DOOR))
                for step in (-1, 1):
                    yy = np.clip(ly + step, 0, len(kind) - 1)
                    corner = along_x & wallish[yy, lx] & (ly + step >= 0) & (ly + step < len(kind))
                    if corner.any():
                        offset = pos[corner] + (0, step * cell / 4)
                        out.append((self.props[name], M.instances(offset, yaw=math.pi / 2, scale=(0.5, 1, 1))))
        # мосты: настил и перила (у клетки, идущей вдоль X, — поворот на 90°)
        xs, ys, pos = cells(kind == T.BRIDGE)
        if len(xs):
            lx, ly = xs - x0, ys - y0
            h_, w_ = kind.shape
            up = np.where(ly + 1 < h_, kind[np.minimum(ly + 1, h_ - 1), lx], T.GROUND)
            down = np.where(ly > 0, kind[np.maximum(ly - 1, 0), lx], T.GROUND)
            along_y = (up == T.BRIDGE) | (down == T.BRIDGE)
            out.append((self.props["bridge"], M.instances(pos, yaw=np.where(along_y, 0.0, math.pi / 2))))
        # заборы: столбы и перекладины к соседям справа и сверху
        xs, ys, pos = cells(kind == T.FENCE)
        if len(xs):
            out.append((self.props["post"], M.instances(pos)))
            lx, ly = xs - x0, ys - y0
            h_, w_ = kind.shape
            right = (lx + 1 < w_) & (kind[ly, np.minimum(lx + 1, w_ - 1)] == T.FENCE)
            up = (ly + 1 < h_) & (kind[np.minimum(ly + 1, h_ - 1), lx] == T.FENCE)
            out.append((self.props["rail"], np.concatenate([M.instances(pos[right]),
                                                            M.instances(pos[up], yaw=math.pi / 2)])))
        # скалы по краю: глыбы смыкаются в обрыв
        xs, ys, pos = cells(kind == T.ROCK)
        if len(xs):
            h = cell_hash(xs, ys, 5)
            pos = pos + (np.stack([h, cell_hash(xs, ys, 6)], 1) - 0.5) * 14
            if self.rocks:
                s = 0.95 + 0.45 * h
                inst = M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s * (0.8 + 0.4 * cell_hash(xs, ys, 3))], 1))
                pick = (cell_hash(xs, ys, 2) * len(self.rocks)).astype(int) % len(self.rocks)
                out += [(mesh, inst[pick == v]) for v, mesh in enumerate(self.rocks) if (pick == v).any()]
            else:
                s = 1.1 + 0.6 * h
                out.append((self.props["rock"], M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s * 0.9], 1))))
        # пни срубленных деревьев
        stump = self.stumps.get(t.biome.tree)
        xs, ys, pos = cells(kind == T.STUMP)
        if stump is not None and len(xs):
            h = cell_hash(xs, ys, 4)
            pos = pos + (h[:, None] - 0.5) * 10                        # там же, где был плоский пень
            s = 0.9 + 0.3 * h
            out.append((stump, M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s], 1))))
        for name, values in level_details.placements(t, x0, y0, x1, y1).items():
            out.append((self.details[name], M.instances(values[:, :2], yaw=values[:, 2])))
        return out

    def _items(self, world: World) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """Находки: медленно вращаются и покачиваются над землёй — видно издалека, что это не декор."""
        idx = np.flatnonzero(world.item_alive)
        out = []
        for kind, mesh in self.items.items():
            ii = idx[world.item_kind[idx] == kind]
            if len(ii):
                phase = self.time * 2.2 + ii * 1.3
                out.append((mesh, M.instances(world.item_pos[ii], yaw=self.time * 0.9 + ii, z=6 + 4 * np.sin(phase),
                                              scale=(1.3, 1.3, 1.3))))
        return out

    def _roofs(self, world: World, view: TerrainView) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """Двускатные крыши поверх всего: прячутся, когда десантник внутри (TerrainView.roof_alpha). Меш у каждой
        постройки свой (размеры разные), собирается один раз; конёк — вдоль длинной стороны."""
        t = world.terrain
        if t is None:
            return []
        cell = t.cell
        out = []
        for bid, bd in view._buildings(t):
            a = view.roof_alpha.get(bid, 1.0)
            if a <= 0.02:
                continue
            key = (id(t), bid)
            if key not in self.roof_meshes:
                w, d = (bd.x1 - bd.x0 + 1) * cell, (bd.y1 - bd.y0 + 1) * cell
                long, short = max(w, d), min(w, d)
                parts = M.gable_roof(long, short, min(70.0, short * 0.32), ROOF_COLORS[bid % len(ROOF_COLORS)],
                                     WALL_COLOR, seed=bid)
                center = np.array([(bd.x0 + bd.x1 + 1) / 2 * cell, (bd.y0 + bd.y1 + 1) / 2 * cell], np.float32)
                yaw = 0.0 if w >= d else math.pi / 2
                self.roof_meshes[key] = (M.StaticMesh(self.ctx, parts, capacity=1), center, yaw)
            mesh, center, yaw = self.roof_meshes[key]
            out.append((mesh, M.instances(center[None], yaw=yaw, z=WALL_H, alpha=a)))
        return out

    def _blob_shadows(self, world: World, shapes) -> None:  # type: ignore[no-untyped-def]
        """Пятна тени под жуками и трупами — сдвинуты от солнца, у летающих — дальше и бледнее."""
        sw = world.swarm
        idx = sw.alive_idx
        off = -SUN[:2] / SUN[2]
        if len(idx):
            fly = sw.kind[idx] == FLYER
            lift = np.where(fly, 55.0, 8.0)[:, None]
            size = sw.radius[idx] * np.where(fly, 1.6, 2.3)
            shapes.add(sw.pos[idx] + off * lift, size, np.array(BLOB, np.float32))
        if self.corpses:
            pos = np.array([c.pos for c in self.corpses], np.float32)
            size = np.array([c.scale * 30 for c in self.corpses], np.float32)
            shapes.add(pos + off * 6, size, np.array(BLOB, np.float32))

    def _marine_subs(self, world: World) -> tuple[str, ...]:
        """Тело и выбранный ствол (турель ставится сама — в руках MAR), топор — только в рубке. Свечение
        «<подмеш>_glow_<ключ>» включается вместе со своим подмешем, «glow_<ключ>» — с телом."""
        gun = world.weapons[world.active].id if world.weapons else "mar"
        shown = {"body", gun if gun in GUNS else "mar"} | ({"axe"} if self.marine_clip == "chop" else set())

        def owner(sub: str) -> str:
            return sub.split("_glow_")[0] if "_glow_" in sub else "body"
        return tuple(s for s in self.models["marine"].all_subs() if (s if "glow" not in s else owner(s)) in shown)

    def _turrets(self, world: World) -> list[tuple[str, np.ndarray, tuple[str, ...]]]:
        """Мини-турели: голова — к цели (поворот всей модели), отдача — пока откат свежий."""
        t = world.turrets
        idx = np.flatnonzero(t.alive)
        if not len(idx):
            return []
        model = self.models["turret"]
        yaw = np.arctan2(t.aim[idx, 1], t.aim[idx, 0]) - math.pi / 2
        a, b, f = model.frames("idle", self.time + idx * 0.7)
        period = 1 / (t.cfg.fire_rate * world.stats["turret_rate"])
        since = period - t.cooldown[idx]
        shot = (since >= 0) & (since < model.duration("fire"))
        if shot.any():
            a[shot], b[shot], f[shot] = model.frames("fire", since[shot])
        inst = np.zeros((len(idx), 12), np.float32)
        inst[:, 0:2] = t.pos[idx]
        inst[:, 3] = yaw
        inst[:, 4], inst[:, 5], inst[:, 6] = a, b, f
        inst[:, 7] = 1.0
        return [("turret", inst, model.all_subs())]

    def _units(self, world: World) -> list[tuple[str, np.ndarray, tuple[str, ...]]]:
        out: list[tuple[str, np.ndarray, tuple[str, ...]]] = []
        # десантник
        p = world.player
        a, b, f = self._marine_frames()
        yaw = self.marine_yaw if self.marine_yaw is not None else math.atan2(p.facing[1], p.facing[0]) - math.pi / 2
        hurt = (0.9, 0.2, 0.15, 0.35) if p.hurt > 0 else (0, 0, 0, 0)
        inst = np.array([[p.pos[0], p.pos[1], 0.0, yaw, a, b, f, 1.0, *hurt]], np.float32)
        out.append(("marine", inst, self._marine_subs(world)))
        out += self._turrets(world)
        # жуки
        sw = world.swarm
        idx = sw.alive_idx
        groups: dict[str, list[np.ndarray]] = {look[0]: [] for look in BUG_LOOK.values()}
        if len(idx):
            kind = sw.kind[idx]
            to = p.pos - sw.pos[idx]
            yaws = np.arctan2(to[:, 1], to[:, 0]) - math.pi / 2
            for k, (name, r0, tint, lift) in BUG_LOOK.items():
                sel = kind == k
                if not sel.any():
                    continue
                ii = idx[sel]
                model = self.models[name]
                time = self.time * sw.speed[ii] / 100.0 + ii * 0.37
                clip_a, clip_b, frac = model.frames("run", time)
                cfg = sw.cfg[k]
                hit = ATTACK_CLIP.get(name)
                if hit:
                    since = cfg.attack_cooldown - sw.cooldown[ii]
                    a = (since >= 0) & (since < model.duration(hit))
                    if a.any():
                        clip_a[a], clip_b[a], frac[a] = model.frames(hit, since[a])
                for clip, timer, period in BOSS_CLIPS.get(k, ()):
                    since = getattr(cfg, period) - getattr(sw, timer)[ii]
                    a = (since >= 0) & (since < model.duration(clip))
                    if a.any():
                        clip_a[a], clip_b[a], frac[a] = model.frames(clip, since[a])
                if k == SEMI_BOSS:                             # рывок — бег вдвое быстрее
                    d = sw.dash[ii] > 0
                    if d.any():
                        clip_a[d], clip_b[d], frac[d] = model.frames("run", time[d] * 2.5)
                if k in WINDUP:                                # замах: доля пройденного замаха → весь клип
                    w = sw.windup[ii] >= 0
                    if w.any():
                        done = (cfg.windup - sw.windup[ii[w]]) / cfg.windup
                        clip_a[w], clip_b[w], frac[w] = model.frames("windup", done * model.duration("windup"))
                stunned = sw.stun[ii] > 0
                if stunned.any():
                    sa, sb, sf = model.frames("idle", self.time + ii[stunned])
                    clip_a[stunned], clip_b[stunned], frac[stunned] = sa, sb, sf
                inst = np.zeros((len(ii), 12), np.float32)
                inst[:, 0:2] = sw.pos[ii]
                inst[:, 2] = lift
                inst[:, 3] = yaws[sel]
                inst[:, 4], inst[:, 5], inst[:, 6] = clip_a, clip_b, frac
                inst[:, 7] = sw.radius[ii] / r0
                inst[:, 8:12] = tint
                if k == TANKER:
                    cracked = sw.armor[ii] <= 0
                    inst[cracked, 8:12] = (0.36, 0.25, 0.16, 0.25)
                burn = sw.burn_left[ii] > 0
                inst[burn, 8:12] = (1.0, 0.5, 0.1, 0.4)
                flash = sw.flash[ii] > 0
                inst[flash, 8:12] = (1.0, 1.0, 1.0, 0.65)
                inst[flash, 11] *= np.clip(sw.flash[ii[flash]] / HIT_FLASH, 0, 1)
                groups[name].append(inst)
        for c in self.corpses:
            model = self.models[c.model]
            da, db, df = model.frames("die", c.age)
            sink = -max(0.0, c.age - 2.5) * 12 + c.lift * max(0.0, 1 - c.age / FALL_TIME) ** 2
            groups[c.model].append(np.array([[c.pos[0], c.pos[1], sink, c.yaw, da, db, df, c.scale, *c.tint]],
                                            np.float32))
        for name, parts in groups.items():
            if parts:
                out.append((name, np.concatenate(parts), self.models[name].all_subs()))
        return out
