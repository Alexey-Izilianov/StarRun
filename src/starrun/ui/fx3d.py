"""Объёмные эффекты 3D: клубы огня и дыма, искры, вспышки выстрелов и столб орбитального удара — частицы-
billboard'ы, развёрнутые к камере; точечный свет от взрывов, выстрелов и огня подсвечивает модели и землю.

Симуляция ничего не знает о частицах: они рождаются здесь по событиям мира (взрывы, новые пули, пламя) и живут
только для картинки. Клубы огня и дыма перекрывают друг друга (обычная прозрачность — плотные, как в
стилизованных играх, и не выжигаются в белое); искры, вспышки, луч и пламя огнемёта светятся (складываются).
Все проверяют глубину моделей, но не пишут её — клубы не режут друг друга.
"""

from __future__ import annotations

from dataclasses import dataclass

import arcade
import numpy as np
from arcade.gl import BufferDescription
from pyglet import gl

from starrun.sim.combat import BULLET, GRENADE, PELLET, ROCKET, SNIPER
from starrun.sim.world import World
from starrun.ui.gun_feedback import SpentCases
from starrun.ui.mesh3d import POINT_LIGHTS

MAX_LIGHTS = 8
MUZZLE = (44.0, 56.0)         # дуло MAR от центра десантника: вперёд и вверх, px (кость muzzle модели)
TURRET_MUZZLE = (24.0, 23.0)
PLAYER_SHOTS = (BULLET, PELLET, SNIPER, ROCKET, GRENADE)

PUFF_VS = """
#version 330
uniform mat4 u_vp;
uniform vec3 u_right;
uniform vec3 u_up;
in vec2 in_corner;
in vec4 i_p;       // x, y, z, размер
in vec4 i_col;
out vec2 v_uv; out vec4 v_col; out float v_seed;
void main() {
    v_uv = in_corner * 2.0;
    v_col = i_col;
    v_seed = fract(i_p.x * 0.0131 + i_p.y * 0.0071 + i_p.w * 0.37);
    vec3 wp = i_p.xyz + (u_right * in_corner.x + u_up * in_corner.y) * i_p.w;
    gl_Position = u_vp * vec4(wp, 1.0);
}
"""

PUFF_FS = """
#version 330
in vec2 v_uv; in vec4 v_col; in float v_seed;
out vec4 f_color;
void main() {
    // клуб без текстуры: мягкий край и неровность из синусов со своей фазой у каждой частицы — не ровный диск
    float a = atan(v_uv.y, v_uv.x);
    float lumpy = 1.0 + 0.08 * sin(a * 5.0 + v_seed * 40.0) + 0.05 * sin(a * 9.0 - v_seed * 70.0);
    float r = length(v_uv) * lumpy;
    if (r > 1.0) discard;
    float soft = 1.0 - r * r;
    f_color = vec4(v_col.rgb, v_col.a * soft * soft * soft);
}
"""

GROUND_LIGHT_FS = """
#version 330
in vec3 v_wp;
out vec4 f_color;
""" + POINT_LIGHTS + """
void main() {
    vec3 c = vec3(0.34, 0.27, 0.2) * point_lights(vec3(0.0, 0.0, 1.0), v_wp);   // земля — как средняя почва
    if (max(c.r, max(c.g, c.b)) < 0.003) discard;
    f_color = vec4(c, 1.0);
}
"""


@dataclass
class Light:
    pos: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float]
    life: float
    flicker: bool = False
    age: float = 0.0

    def strength(self, time: float) -> float:
        k = max(0.0, 1 - self.age / self.life)
        wobble = 0.85 + 0.15 * np.sin(time * 37 + self.pos[0]) if self.flicker else 1.0
        return k * k * wobble


class Particles:
    """Пул частиц: цвет и размер плывут от начальных к конечным за жизнь; rise — подъёмная сила (дым вверх,
    искры — отрицательная, падают), drag — торможение."""

    def __init__(self, capacity: int) -> None:
        self.pos = np.zeros((capacity, 3), np.float32)
        self.vel = np.zeros((capacity, 3), np.float32)
        self.size = np.zeros((capacity, 2), np.float32)       # в начале и в конце
        self.col0 = np.zeros((capacity, 4), np.float32)
        self.col1 = np.zeros((capacity, 4), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.delay = np.zeros(capacity, np.float32)           # дым появляется после огня
        self.rise = np.zeros(capacity, np.float32)
        self.drag = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)
        self.cursor = 0

    def emit(self, pos: np.ndarray, vel: np.ndarray, size: np.ndarray, col0: tuple[float, ...],
             col1: tuple[float, ...], life: np.ndarray, rise: float = 0.0, drag: float = 0.0,
             delay: np.ndarray | float = 0.0) -> None:
        n = len(pos)
        cap = len(self.alive)
        free = np.flatnonzero(~self.alive)[:n]
        if len(free) < n:            # пул полон — затираются по кругу самые старые места
            extra = (self.cursor + np.arange(n - len(free))) % cap
            self.cursor = int((extra[-1] + 1) % cap)
            free = np.concatenate([free, extra])
        self.pos[free] = pos
        self.vel[free] = vel
        self.size[free] = size
        self.col0[free] = col0
        self.col1[free] = col1
        self.life[free] = self.max_life[free] = life
        self.delay[free] = delay
        self.rise[free] = rise
        self.drag[free] = drag
        self.alive[free] = True

    def step(self, dt: float) -> None:
        a = self.alive
        if not a.any():
            return
        waiting = a & (self.delay > 0)
        self.delay[waiting] -= dt
        live = a & ~waiting
        self.life[live] -= dt
        self.vel[live] *= (1 - np.minimum(1.0, self.drag[live] * dt))[:, None]
        self.vel[live, 2] += self.rise[live] * dt
        self.pos[live] += self.vel[live] * dt
        landed = live & (self.pos[:, 2] < 0)               # искры гаснут о землю
        self.alive[landed | (live & (self.life <= 0))] = False

    def instances(self) -> np.ndarray:
        idx = np.flatnonzero(self.alive & (self.delay <= 0))
        if not len(idx):
            return np.zeros((0, 8), np.float32)
        t = (1 - self.life[idx] / self.max_life[idx])[:, None]
        out = np.empty((len(idx), 8), np.float32)
        out[:, 0:3] = self.pos[idx]
        out[:, 3] = self.size[idx, 0] + (self.size[idx, 1] - self.size[idx, 0]) * t[:, 0]
        out[:, 4:8] = self.col0[idx] + (self.col1[idx] - self.col0[idx]) * t
        return out


class Effects3D:
    def __init__(self, ctx: arcade.ArcadeContext) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=PUFF_VS, fragment_shader=PUFF_FS)
        self.ground = None                       # программа света на земле — задаёт Scene3D (общий вершинный)
        corners = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5], np.float32)
        self.capacity = 4096
        self.inst = ctx.buffer(reserve=self.capacity * 8 * 4)
        self.geometry = ctx.geometry([BufferDescription(ctx.buffer(data=corners.tobytes()), "2f", ["in_corner"]),
                                      BufferDescription(self.inst, "4f 4f", ["i_p", "i_col"], instanced=True)],
                                     mode=ctx.TRIANGLE_STRIP)
        self.rng = np.random.default_rng(7)
        self.reset()

    def reset(self) -> None:
        self.fire = Particles(2048)              # светятся (складываются): искры, вспышки, луч
        self.smoke = Particles(2048)             # перекрывают: огненные клубы взрывов, дым, пыль
        self.lights: list[Light] = []
        self.time = 0.0
        self.prev_bullets: np.ndarray | None = None
        self.muzzle: np.ndarray | None = None      # дуло из позы модели (Scene3D); нет — оценка по MUZZLE
        self.cases = SpentCases()
        self.prev_reload: dict[str, float] = {}

    # ---------- события мира → частицы ----------
    def update(self, dt: float, world: World) -> None:
        self.time += dt
        for e in world.booms:
            self.explosion(e.pos, e.radius)
        for mark in world.blasts:
            self.orbital(mark.pos, mark.radius)
        if "chop" in world.events:
            p, g = world.player, self.rng
            pos = p.pos + p.heading * 34
            n = 9
            velocity = np.column_stack([g.normal(0, 45, (n, 2)), g.uniform(70, 150, n)])
            self.smoke.emit(np.tile([*pos, 32], (n, 1)), velocity, np.tile([5.0, 1.0], (n, 1)),
                            (0.56, 0.43, 0.25, 0.95), (0.38, 0.3, 0.18, 0.0), g.uniform(0.2, 0.45, n),
                            rise=-350, drag=1.0)
        self._shots(world)
        self._weapon_feedback(world)
        self._flames(dt, world)
        self._rockets(dt, world)
        self.fire.step(dt)
        self.smoke.step(dt)
        self.cases.step(dt)
        for light in self.lights:
            light.age += dt
        self.lights = [x for x in self.lights if x.age < x.life]

    def _weapon_feedback(self, world: World) -> None:
        p, gun = world.player, world.weapon
        for event in world.events:
            if event.startswith("shot_"):
                self.cases.emit(p.pos, p.facing, event[5:])
        for owned in world.weapons:
            if owned.id == gun.id and owned.reload_left > 0 and self.prev_reload.get(owned.id, 0.0) <= 0:
                self.cases.emit(p.pos, p.facing, gun.id, magazine=True)
            self.prev_reload[owned.id] = owned.reload_left
        hits = world.hit_feedback
        if not hits:
            return
        # Частицы ближайших попаданий: не более восьми всплесков за кадр при массовом уроне.
        nearest = sorted(hits, key=lambda hit: float(np.sum((hit.pos - p.pos) ** 2)))[:8]
        g = self.rng
        for hit in nearest:
            if np.linalg.norm(hit.pos - p.pos) > 900:
                continue
            n = 4 if hit.critical else 3
            pos = np.tile([*hit.pos, 30.0], (n, 1))
            velocity = np.column_stack([g.normal(0, 90 if hit.armored else 55, (n, 2)), g.uniform(30, 100, n)])
            if hit.armored:
                color = (0.68, 0.84, 1.0, 0.9)
            elif hit.critical:
                color = (1.0, 0.79, 0.28, 0.95)
            else:
                color = (0.42, 0.64, 0.18, 0.9)
            pool = self.fire if hit.armored or hit.critical else self.smoke
            pool.emit(pos, velocity, np.tile([5.0 if hit.critical else 3.5, 1.0], (n, 1)),
                      color, (*color[:3], 0.0), g.uniform(0.1, 0.22, n), rise=-350, drag=3.0)

    def explosion(self, pos: np.ndarray, radius: float, big: bool = False) -> None:
        r, g = radius, self.rng
        n = 14 if not big else 28
        around = g.normal(0, 1, (n, 2)) * r * 0.25
        out = around / (np.linalg.norm(around, axis=1, keepdims=True) + 1e-6)
        p = np.column_stack([pos + around, g.uniform(8, 30 + r * 0.2, n)])
        v = np.column_stack([out * g.uniform(60, 120 + r, (n, 1)), g.uniform(40, 120, n)])
        self.smoke.emit(p, v, np.column_stack([g.uniform(0.3, 0.5, n) * r, g.uniform(0.6, 0.85, n) * r]),
                        (1.0, 0.62, 0.22, 0.9), (0.35, 0.1, 0.04, 0.0), g.uniform(0.35, 0.6, n), rise=60, drag=3.0)
        self.fire.emit(np.array([[*pos, 30.0]]), np.zeros((1, 3)), np.array([[r * 0.9, r * 1.3]]),
                       (1.0, 0.75, 0.45, 0.7), (1.0, 0.5, 0.15, 0.0), np.array([0.12]))             # вспышка
        m = n - 4
        p = np.column_stack([pos + g.normal(0, 1, (m, 2)) * r * 0.3, g.uniform(20, 50, m)])
        v = np.column_stack([g.normal(0, 30, (m, 2)), g.uniform(30, 70, m)])
        self.smoke.emit(p, v, np.column_stack([g.uniform(0.4, 0.6, m) * r, g.uniform(1.0, 1.4, m) * r]),
                        (0.16, 0.14, 0.12, 0.6), (0.32, 0.29, 0.26, 0.0), g.uniform(1.2, 2.0, m), rise=25,
                        drag=1.2, delay=g.uniform(0.08, 0.2, m))
        k = 12 if not big else 24
        a = g.uniform(0, 2 * np.pi, k)
        sp = g.uniform(250, 520, k)
        v = np.column_stack([np.cos(a) * sp, np.sin(a) * sp, g.uniform(150, 420, k)])
        self.fire.emit(np.tile([*pos, 20.0], (k, 1)), v, np.full((k, 2), (7.0, 4.0)), (1.0, 0.9, 0.55, 1.0),
                       (1.0, 0.45, 0.1, 0.0), g.uniform(0.4, 0.8, k), rise=-900, drag=0.5)          # искры
        self.lights.append(Light((float(pos[0]), float(pos[1]), 60.0), r * 2.6, (1.1, 0.55, 0.22), 0.45))

    def orbital(self, pos: np.ndarray, radius: float) -> None:
        """Удар с орбиты: столб света сверху, затем большой взрыв."""
        n = 10                                   # выше ~450 px столб упирается в камеру и раздувается
        z = np.linspace(0, 450, n)
        self.fire.emit(np.column_stack([np.tile(pos, (n, 1)), z]), np.zeros((n, 3)), np.tile([60.0, 18.0], (n, 1)),
                       (0.5, 0.7, 1.0, 0.55), (0.4, 0.6, 1.0, 0.0), np.full(n, 0.3))
        self.explosion(pos, radius * 0.9, big=True)
        self.lights.append(Light((float(pos[0]), float(pos[1]), 120.0), radius * 2.4, (0.7, 0.8, 1.0), 0.6))

    def _shots(self, world: World) -> None:
        """Вспышка у дула на каждый новый выстрел: новые пули — те, что появились с прошлого кадра."""
        bl = world.bullets
        prev = self.prev_bullets if self.prev_bullets is not None else bl.alive
        new = np.flatnonzero(bl.alive & ~prev)
        self.prev_bullets = bl.alive.copy()
        if not len(new):
            return
        p = world.player
        kinds = bl.kind[new]
        if np.isin(kinds, PLAYER_SHOTS).any():
            m = self.muzzle if self.muzzle is not None else np.array([*(p.pos + p.facing * MUZZLE[0]), MUZZLE[1]])
            big = bool(np.isin(kinds, (SNIPER, ROCKET, PELLET)).any())
            size = 34.0 if big else 22.0
            self.fire.emit(m[None], np.zeros((1, 3)), np.array([[size, size * 0.6]]),
                           (1.0, 0.9, 0.6, 1.0), (1.0, 0.6, 0.2, 0.0), np.array([0.06]))
            self.lights.append(Light((float(m[0]), float(m[1]), float(m[2])), 200.0 if big else 130.0,
                                     (1.3, 0.95, 0.55), 0.07))
        t = world.turrets
        for i in np.flatnonzero(t.alive):
            if t.cooldown[i] > 1 / (t.cfg.fire_rate * world.stats["turret_rate"]) - 0.02:   # только что выстрелила
                m = t.pos[i] + t.aim[i] * TURRET_MUZZLE[0]
                self.fire.emit(np.array([[*m, TURRET_MUZZLE[1]]]), np.zeros((1, 3)), np.array([[16.0, 10.0]]),
                               (1.0, 0.9, 0.6, 1.0), (1.0, 0.6, 0.2, 0.0), np.array([0.05]))

    def _flames(self, dt: float, world: World) -> None:
        """Огнемёт: свет у сопла, пока летит пламя; немного дыма над клубами."""
        f = world.flames
        k = np.flatnonzero(f.alive)
        if not len(k):
            return
        p = world.player
        near = k[np.linalg.norm(f.pos[k] - p.pos, axis=1) < 260]
        if len(near):
            c = f.pos[near].mean(0)
            self.lights.append(Light((float(c[0]), float(c[1]), 50.0), 260.0, (1.2, 0.55, 0.18), 0.05, flicker=True))
        g = self.rng
        pick = k[g.random(len(k)) < dt * 3]
        if len(pick):
            n = len(pick)
            self.smoke.emit(np.column_stack([f.pos[pick], np.full(n, 70.0)]),
                            np.column_stack([g.normal(0, 15, (n, 2)), np.full(n, 50.0)]), np.tile([14.0, 40.0], (n, 1)),
                            (0.12, 0.1, 0.09, 0.35), (0.25, 0.23, 0.21, 0.0), g.uniform(0.8, 1.3, n), rise=20, drag=1.0)

    def _rockets(self, dt: float, world: World) -> None:
        """Дымный след за ракетами и гранатами."""
        bl = world.bullets
        idx = np.flatnonzero(bl.alive & np.isin(bl.kind, (ROCKET, GRENADE)))
        if not len(idx):
            return
        n = len(idx)
        g = self.rng
        self.smoke.emit(np.column_stack([bl.pos[idx], np.full(n, 52.0)]),
                        np.column_stack([g.normal(0, 8, (n, 2)), np.full(n, 10.0)]), np.tile([8.0, 26.0], (n, 1)),
                        (0.55, 0.53, 0.5, 0.45), (0.4, 0.38, 0.36, 0.0), g.uniform(0.5, 0.8, n), rise=10, drag=2.0)

    # ---------- свет и отрисовка ----------
    def light_uniforms(self, extra: list[Light]) -> dict[str, tuple[float, ...]]:
        """До восьми самых сильных источников — в униформы u_pl_pos / u_pl_col."""
        lights = sorted(self.lights + extra, key=lambda x: -x.strength(self.time) * x.radius)[:MAX_LIGHTS]
        pos = np.zeros((MAX_LIGHTS, 4), np.float32)
        col = np.zeros((MAX_LIGHTS, 4), np.float32)
        for i, x in enumerate(lights):
            pos[i] = (*x.pos, x.radius)
            col[i, :3] = np.array(x.color) * x.strength(self.time)
        return {"u_pl_pos": tuple(pos.ravel()), "u_pl_col": tuple(col.ravel())}

    def draw(self, vp: np.ndarray, view: np.ndarray, flames: np.ndarray, drops: np.ndarray) -> None:
        """flames — клубы пламени огнемёта, drops — брызги жуков (из симуляции) в формате экземпляров:
        x, y, z, размер, цвет."""
        ctx = self.ctx
        prog = self.program
        prog["u_vp"] = tuple(vp.T.ravel())
        prog["u_right"] = tuple(view[0, :3])
        prog["u_up"] = tuple(view[1, :3])
        ctx.enable(ctx.DEPTH_TEST, ctx.BLEND)
        gl.glDepthMask(gl.GL_FALSE)
        ctx.blend_func = ctx.BLEND_DEFAULT
        self._render(np.concatenate([drops, self.smoke.instances()]))
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE      # BLEND_ADDITIVE в Arcade — ONE, ONE: прозрачность не учлась бы
        self._render(np.concatenate([self.fire.instances(), flames]))
        ctx.blend_func = ctx.BLEND_DEFAULT
        gl.glDepthMask(gl.GL_TRUE)
        ctx.disable(ctx.DEPTH_TEST)

    def _render(self, inst: np.ndarray) -> None:
        n = min(len(inst), self.capacity)
        if not n:
            return
        self.inst.write(inst[:n].astype(np.float32).tobytes())
        self.geometry.render(self.program, instances=n)


def splash_drops(world: World) -> np.ndarray:
    """Брызги жуков: капли летят дугой с высоты корпуса и падают на землю к концу жизни (там уже лежит пятно)."""
    sp = world.splashes
    k = np.flatnonzero(sp.alive)
    if not len(k):
        return np.zeros((0, 8), np.float32)
    t = 1 - np.clip(sp.life[k] / sp.max_life[k], 0, 1)
    out = np.empty((len(k), 8), np.float32)
    out[:, 0:2] = sp.pos[k]
    out[:, 2] = np.maximum(2.0, 22 + 90 * t - 110 * t * t)
    out[:, 3] = sp.size[k] * 0.9
    out[:, 4:7] = (0.43, 0.85, 0.16)
    out[:, 7] = 0.95 * np.clip(1.6 - 1.6 * t, 0, 1)
    return out


def flame_puffs(world: World) -> np.ndarray:
    """Клубы огнемёта из симуляции: над землёй, поднимаются, остывая; ядро светлее."""
    f = world.flames
    k = np.flatnonzero(f.alive)
    if not len(k):
        return np.zeros((0, 8), np.float32)
    t = np.clip(f.life[k] / f.max_life[k], 0, 1)          # 1 — только вылетела
    out = np.empty((len(k) * 2, 8), np.float32)
    for j, (scale, core) in enumerate(((1.0, 0.0), (0.5, 1.0))):
        o = out[j::2]
        o[:, 0:2] = f.pos[k]
        o[:, 2] = 50 + (1 - t) * 40
        o[:, 3] = f.size[k] * (1.9 - t) * scale
        o[:, 4] = 1.0
        o[:, 5] = 0.25 + 0.4 * t + core * 0.3
        o[:, 6] = 0.05 + core * 0.4 * t
        o[:, 7] = (0.75 + 0.25 * core) * t ** (1 + core * 0.5)
    return out
