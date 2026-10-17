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
from starrun.sim.swarm import BRAIN, FLYER, QUEEN, SEMI_BOSS, SPITTER, TANKER, WARRIOR, WORKER
from starrun.sim.world import World
from starrun.ui import mesh3d as M
from starrun.ui.gl3d import PX_PER_M, Camera3D, look_at, ortho
from starrun.ui.scene import SceneRenderer
from starrun.ui.terrain_view import TerrainView, cell_hash

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
        self.skinned = ctx.program(vertex_shader=M.SKINNED_VS, fragment_shader=M.SKINNED_FS)
        self.skinned_depth = ctx.program(vertex_shader=M.SKINNED_VS, fragment_shader=M.DEPTH_FS)
        self.static = ctx.program(vertex_shader=M.STATIC_VS, fragment_shader=M.STATIC_FS)
        self.static_depth = ctx.program(vertex_shader=M.STATIC_VS, fragment_shader=M.DEPTH_FS)
        self.receiver = ctx.program(vertex_shader=M.RECEIVER_VS, fragment_shader=M.RECEIVER_FS)
        names = {"marine"} | {look[0] for look in BUG_LOOK.values()}
        self.models = {n: M.SkinnedModel(ctx, MODELS / f"{n}.npz") for n in sorted(names)}
        self.trees: dict[tuple[str, tuple[int, int, int]], M.StaticMesh] = {}
        wall, glass, door = (0.36, 0.37, 0.39), (0.42, 0.62, 0.7), (0.42, 0.29, 0.18)
        self.props = {
            "wall": M.StaticMesh(ctx, [M.box((40, 40, WALL_H), (0, 0, WALL_H / 2), wall)]),
            "window": M.StaticMesh(ctx, [M.box((40, 40, 46), (0, 0, 23), wall),
                                         M.box((40, 40, 28), (0, 0, WALL_H - 14), wall),
                                         M.box((40, 6, 36), (0, 0, 64), glass)]),
            "door": M.StaticMesh(ctx, [M.box((40, 40, 24), (0, 0, WALL_H - 12), wall),
                                       M.box((36, 8, 86), (0, 0, 43), door)]),
            "door_open": M.StaticMesh(ctx, [M.box((40, 40, 24), (0, 0, WALL_H - 12), wall),
                                            M.box((8, 34, 86), (-17, 18, 43), door)]),
            "post": M.StaticMesh(ctx, [M.box((7, 7, 44), (0, 0, 22), (0.5, 0.4, 0.27))]),
            "rail": M.StaticMesh(ctx, [M.box((40, 4, 5), (20, 0, 18), (0.48, 0.38, 0.25)),
                                       M.box((40, 4, 5), (20, 0, 34), (0.48, 0.38, 0.25))]),
            "rock": M.StaticMesh(ctx, [M.blob(24, (0, 0, 18), (0.42, 0.4, 0.37), squash=0.75, jag=0.2, seed=5)]),
            "roof": M.StaticMesh(ctx, [M.box((1, 1, 1), (0, 0, 0.5), (0.33, 0.3, 0.27))], capacity=64),
            "ridge": M.StaticMesh(ctx, [M.box((1, 1, 1), (0, 0, 0.5), (0.2, 0.18, 0.16))], capacity=64),
        }
        self.shadow_tex = ctx.depth_texture((SHADOW_SIZE, SHADOW_SIZE))
        self.shadow_tex.compare_func = None                       # читаем глубину как число (см. LIGHT в mesh3d)
        self._shadow_color = ctx.texture((SHADOW_SIZE, SHADOW_SIZE), components=1)   # Arcade требует цветовой слой
        self.shadow_fbo = ctx.framebuffer(color_attachments=[self._shadow_color], depth_attachment=self.shadow_tex)
        self.quad = ctx.buffer(reserve=4 * 2 * 4)
        self.quad_geometry = ctx.geometry([arcade.gl.BufferDescription(self.quad, "2f", ["in_pos"])],
                                          mode=ctx.TRIANGLE_STRIP)
        self.light = np.identity(4)
        self._props_key: tuple | None = None  # type: ignore[type-arg]
        self._props_cache: list[tuple[M.StaticMesh, np.ndarray]] = []
        self.reset()

    def reset(self) -> None:
        self.time = 0.0
        self.corpses: list[Corpse] = []
        self.prev_alive: np.ndarray | None = None
        self.marine_clip, self.marine_t = "idle", 0.0
        self.prev_pos: np.ndarray | None = None
        self.death_t = 0.0

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
        for c in self.corpses:
            c.age += dt
        self.corpses = [c for c in self.corpses if c.age < CORPSE_LIFE][-300:]
        self._marine_state(dt, world)

    def _marine_state(self, dt: float, world: World) -> None:
        """Клип десантника: ноги — по ходу движения относительно взгляда (вперёд, назад, вбок)."""
        p = world.player
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
        else:
            clip, t = "idle", self.marine_t + dt
        self.marine_clip, self.marine_t = clip, t

    # ---------- кадр ----------
    def draw(self, world: World, cam: Camera3D, scene: SceneRenderer, shapes) -> None:  # type: ignore[no-untyped-def]
        ctx = self.ctx
        rect = cam.ground_rect()
        view = (rect[0] - 80, rect[1] - 80, rect[2] + 80, rect[3] + 120)
        scene.map.prepare(world, view)
        props = self._props(world, rect)
        roofs = self._roofs(world, scene.map)
        units = self._units(world)

        self._shadow_pass(cam, props, units)
        cam.apply(ctx)
        ctx.disable(ctx.DEPTH_TEST)
        scene.ground_layers(world, view, solid=False)
        self._blob_shadows(world, shapes)
        shapes.flush()
        self._receive_shadows(cam, rect)

        ctx.enable(ctx.DEPTH_TEST)
        self._set_common(self.static, cam.view_proj, cam)
        self._set_common(self.skinned, cam.view_proj, cam)
        self.static["u_player"] = tuple(world.player.pos)
        for mesh, inst in props:
            mesh.draw(self.static, inst)
        for name, inst, subs in units:
            self.models[name].draw(self.skinned, inst, subs)
        for mesh, inst in roofs:
            mesh.draw(self.static, inst)
        ctx.disable(ctx.DEPTH_TEST)

        shapes.lift = EFFECTS_LIFT
        scene.effects(world)
        shapes.flush()
        shapes.lift = 0.0

    def _set_common(self, prog: arcade.gl.Program, vp: np.ndarray, cam: Camera3D | None) -> None:
        values: dict[str, object] = {"u_vp": tuple(vp.T.ravel()), "u_px": PX_PER_M, "u_anim": 1}
        if cam is not None:
            values.update(u_albedo=2, u_mask=3, u_sun=tuple(SUN), u_eye=tuple(cam.eye), u_shadow=4,
                          u_light=tuple(self.light.T.ravel()), u_texel=1.0 / SHADOW_SIZE)
            self.shadow_tex.use(4)
        for name, value in values.items():
            with contextlib.suppress(KeyError):   # униформы нет в шейдере (или компилятор выкинул неиспользуемую)
                prog[name] = value

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
        self.static_depth["u_player"] = (1e9, 1e9)          # в тень — всё, растворение только для камеры
        for mesh, inst in props:
            mesh.draw(self.static_depth, inst)
        for name, inst, subs in units:
            if name == "marine":          # жуки — пятнами (_blob_shadows): толпа дорога для карты теней
                self.models[name].draw(self.skinned_depth, inst, subs, depth=True)
        ctx.disable(ctx.DEPTH_TEST)
        ctx.screen.use()

    def _receive_shadows(self, cam: Camera3D, rect: tuple[float, float, float, float]) -> None:
        l, b, r, t = rect  # noqa: E741
        self.quad.write(np.array([l, b, r, b, l, t, r, t], np.float32).tobytes())
        self._set_common(self.receiver, cam.view_proj, cam)
        self.ctx.enable(self.ctx.BLEND)
        self.quad_geometry.render(self.receiver)

    # ---------- экземпляры ----------
    def _tree(self, biome) -> M.StaticMesh:  # type: ignore[no-untyped-def]
        key = (biome.tree, tuple(biome.tree_color))
        if key not in self.trees:
            self.trees[key] = M.StaticMesh(self.ctx, tree_mesh(biome.tree, biome.tree_color))
        return self.trees[key]

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
            size = 0.9 + 0.2 * h if t.biome.tree == "cactus" else 0.72 + 0.55 * h ** 1.5
            s = size * (0.7 + 0.3 * hp)
            out.append((self._tree(t.biome), M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s], 1))))
        # стены, окна, двери
        for k, name in ((T.WALL, "wall"), (T.WINDOW, "window"), (T.DOOR, "door"), (T.OPEN_DOOR, "door_open")):
            xs, ys, pos = cells(kind == k)
            if not len(xs):
                continue
            yaw = np.where(TerrainView._along_x(kind, xs - x0, ys - y0), 0.0, math.pi / 2) if k != T.WALL else 0.0
            out.append((self.props[name], M.instances(pos, yaw=yaw)))
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
        # скалы по краю
        xs, ys, pos = cells(kind == T.ROCK)
        if len(xs):
            h = cell_hash(xs, ys, 5)
            s = 1.1 + 0.6 * h
            pos = pos + (np.stack([h, cell_hash(xs, ys, 6)], 1) - 0.5) * 14
            out.append((self.props["rock"], M.instances(pos, yaw=h * 6.28, scale=np.stack([s, s, s * 0.9], 1))))
        return out

    def _roofs(self, world: World, view: TerrainView) -> list[tuple[M.StaticMesh, np.ndarray]]:
        """Крыши поверх всего: прячутся, когда десантник внутри (TerrainView.roof_alpha)."""
        t = world.terrain
        if t is None:
            return []
        cell = t.cell
        roofs, ridges = [], []
        for bid, bd in view._buildings(t):
            a = view.roof_alpha.get(bid, 1.0)
            if a <= 0.02:
                continue
            w, d = (bd.x1 - bd.x0 + 1) * cell + 10, (bd.y1 - bd.y0 + 1) * cell + 10
            c = np.array([[(bd.x0 + bd.x1 + 1) / 2 * cell, (bd.y0 + bd.y1 + 1) / 2 * cell]], np.float32)
            roofs.append(M.instances(c, z=WALL_H, scale=(w, d, 14), alpha=a))
            ridge = (w, 10, 8) if w >= d else (10, d, 8)
            ridges.append(M.instances(c, z=WALL_H + 14, scale=ridge, alpha=a))
        if not roofs:
            return []
        return [(self.props["roof"], np.concatenate(roofs)), (self.props["ridge"], np.concatenate(ridges))]

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

    def _units(self, world: World) -> list[tuple[str, np.ndarray, tuple[str, ...]]]:
        out: list[tuple[str, np.ndarray, tuple[str, ...]]] = []
        # десантник
        p = world.player
        mar = self.models["marine"]
        a, b, f = mar.frames(self.marine_clip, self.marine_t)
        yaw = math.atan2(p.facing[1], p.facing[0]) - math.pi / 2
        hurt = (0.9, 0.2, 0.15, 0.35) if p.hurt > 0 else (0, 0, 0, 0)
        inst = np.array([[p.pos[0], p.pos[1], 0.0, yaw, a, b, f, 1.0, *hurt]], np.float32)
        skip = () if self.marine_clip == "chop" else ("axe",)
        out.append(("marine", inst, mar.all_subs(skip)))
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
                burn = sw.burn_left[ii] > 0
                inst[burn, 8:12] = (1.0, 0.5, 0.1, 0.4)
                flash = sw.flash[ii] > 0
                inst[flash, 8:12] = (1.0, 1.0, 1.0, 0.65)
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
