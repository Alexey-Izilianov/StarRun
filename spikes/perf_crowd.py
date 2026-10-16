"""Спайк этапа 0: 400 жуков, пули, брызги и кристаллы опыта в 1920×1080 — хватает ли Arcade + NumPy
на 60 FPS. Логика толпы векторная (NumPy, сетка плотности вместо попарных расстояний), отрисовка —
координаты из NumPy одним буфером в видеокарту, один вызов на вид объектов (без цикла по спрайтам).

Запуск: python spikes/perf_crowd.py [--bugs 400] [--seconds 20]
По завершении печатает среднее и 99-й процентиль времени кадра.
"""

from __future__ import annotations

import argparse
import statistics
import time

import arcade
import numpy as np
from arcade.gl import BufferDescription

WIDTH, HEIGHT = 1920, 1080
BUG_RADIUS = 14.0
BULLET_SPEED = 900.0
FIRE_RATE = 24.0  # выстр./с — с запасом против 8 у MAR
SPLASH_PER_KILL = 12
PICKUP_RADIUS = 120.0
CELL = 28.0  # клетка сетки плотности ≈ диаметр жука
GRID_W, GRID_H = 96, 72  # 2688×2016 px вокруг игрока

VERTEX = """
#version 330
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform float u_size;
in vec2 in_corner;  // угол квадрата, -1..1
in vec2 in_pos;     // центр — на экземпляр
out vec2 v_corner;
void main() {
    v_corner = in_corner;
    gl_Position = window.projection * window.view * vec4(in_pos + in_corner * u_size * 0.5, 0.0, 1.0);
}
"""
FRAGMENT = """
#version 330
uniform vec4 u_color;
in vec2 v_corner;
out vec4 f_color;
void main() {
    float d = dot(v_corner, v_corner);
    if (d > 1.0) discard;
    f_color = vec4(u_color.rgb, u_color.a * (1.0 - 0.35 * d * d));
}
"""
QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)


class Pool:
    """Сущности одного вида в NumPy; рисуются точками одного размера и цвета."""

    def __init__(self, ctx: arcade.ArcadeContext, capacity: int, size: float, color: tuple[int, int, int]) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)
        self.size = size
        self.color = tuple(c / 255 for c in color) + (1.0,)
        # Квадрат один на всех, центры — по экземпляру: один вызов отрисовки на вид объектов
        self.buffer = ctx.buffer(reserve=capacity * 8)
        corners = ctx.buffer(data=QUAD.tobytes())
        self.geometry = ctx.geometry([BufferDescription(corners, "2f", ["in_corner"]),
                                      BufferDescription(self.buffer, "2f", ["in_pos"], instanced=True)],
                                     mode=ctx.TRIANGLE_STRIP)

    def spawn(self, count: int, pos: np.ndarray, vel: np.ndarray | None = None, life: float = 0.0) -> None:
        free = np.flatnonzero(~self.alive)[:count]
        n = len(free)
        if n == 0:
            return
        self.pos[free] = pos[:n]
        self.vel[free] = 0 if vel is None else vel[:n]
        self.life[free] = life
        self.alive[free] = True

    def draw(self, program: arcade.gl.Program) -> None:
        live = self.pos[self.alive]
        if not len(live):
            return
        self.buffer.write(live.tobytes())
        program["u_size"] = self.size
        program["u_color"] = self.color
        self.geometry.render(program, vertices=4, instances=len(live))


class Spike(arcade.Window):
    def __init__(self, bugs: int, seconds: float) -> None:
        super().__init__(WIDTH, HEIGHT, "StarRun — спайк производительности", vsync=False,
                         update_rate=1 / 1000, draw_rate=1 / 1000)
        self.background_color = (38, 42, 34)
        self.program = self.ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.rng = np.random.default_rng(1)
        self.target_bugs = bugs
        self.seconds = seconds
        self.player = np.array([WIDTH / 2, HEIGHT / 2], np.float32)

        self.bugs = Pool(self.ctx, bugs, 2 * BUG_RADIUS, (150, 110, 70))
        self.bug_hp = np.zeros(bugs, np.float32)
        self.bullets = Pool(self.ctx, 512, 8, (255, 230, 140))
        self.splash = Pool(self.ctx, bugs * SPLASH_PER_KILL // 2, 9, (90, 220, 60))
        self.gems = Pool(self.ctx, 1024, 10, (90, 170, 255))
        self.hero = Pool(self.ctx, 1, 34, (200, 200, 210))
        self.hero.alive[0] = True
        self.fire_timer = 0.0
        self.angle = 0.0
        self.frame_ms: list[float] = []
        self.update_ms: list[float] = []
        self.started = time.perf_counter()
        self.kills = 0
        self.shot = ""
        self.hud = arcade.Text("", 16, HEIGHT - 32, arcade.color.WHITE, 16)

    def spawn_bugs(self, count: int) -> None:
        # За краем экрана, как в ТЗ
        angle = self.rng.uniform(0, 2 * np.pi, count)
        radius = self.rng.uniform(1100, 1300, count)
        pos = self.player + np.stack([np.cos(angle) * radius, np.sin(angle) * radius], 1).astype(np.float32)
        before = self.bugs.alive.copy()
        self.bugs.spawn(count, pos)
        self.bug_hp[self.bugs.alive & ~before] = 10

    def separation(self, pos: np.ndarray) -> np.ndarray:
        """Расталкивание через сетку плотности: жуки уходят из клеток, где их больше. O(N)."""
        origin = self.player - np.array([GRID_W * CELL / 2, GRID_H * CELL / 2], np.float32)
        cell = ((pos - origin) / CELL).astype(np.int32)
        inside = (cell[:, 0] >= 1) & (cell[:, 0] < GRID_W - 1) & (cell[:, 1] >= 1) & (cell[:, 1] < GRID_H - 1)
        flat = cell[:, 1] * GRID_W + cell[:, 0]
        density = np.bincount(flat[inside], minlength=GRID_W * GRID_H).reshape(GRID_H, GRID_W).astype(np.float32)
        gy, gx = np.gradient(density)
        push = np.zeros_like(pos)
        cx, cy = cell[inside, 0], cell[inside, 1]
        push[inside, 0] = -gx[cy, cx]
        push[inside, 1] = -gy[cy, cx]
        # В одной клетке градиента нет — небольшой случайный разброс, чтобы не слипались в точку
        crowded = inside & (density[np.clip(cell[:, 1], 0, GRID_H - 1), np.clip(cell[:, 0], 0, GRID_W - 1)] > 1)
        push[crowded] += self.rng.normal(0, 0.6, (int(crowded.sum()), 2)).astype(np.float32)
        return push * 90.0

    def on_update(self, dt: float) -> None:
        t0 = time.perf_counter()
        dt = min(dt, 1 / 20)
        # Игрок ходит по кругу — толпа всё время перестраивается
        self.angle += dt * 0.6
        self.player[:] = (WIDTH / 2 + np.cos(self.angle) * 250, HEIGHT / 2 + np.sin(self.angle) * 180)
        self.hero.pos[0] = self.player

        missing = self.target_bugs - int(self.bugs.alive.sum())
        if missing > 0:
            self.spawn_bugs(missing)

        b = self.bugs
        idx = np.flatnonzero(b.alive)
        pos = b.pos[idx]
        to_player = self.player - pos
        dist = np.linalg.norm(to_player, axis=1, keepdims=True) + 1e-3
        b.pos[idx] = pos + (to_player / dist * 70.0 + self.separation(pos)) * dt

        # Автострельба в ближайшего жука
        self.fire_timer += dt
        if len(idx) and self.fire_timer >= 1 / FIRE_RATE:
            self.fire_timer = 0.0
            nearest = b.pos[idx][np.argmin(dist[:, 0])]
            direction = nearest - self.player
            direction /= np.linalg.norm(direction) + 1e-3
            self.bullets.spawn(1, self.player[None], (direction * BULLET_SPEED)[None], life=1.2)

        # Пули: полёт, срок жизни, попадания (пуль в полёте десятки — попарно дёшево)
        u = self.bullets
        u.pos += u.vel * dt
        u.life -= dt
        u.alive &= u.life > 0
        live_bullets = np.flatnonzero(u.alive)
        if len(live_bullets) and len(idx):
            hit = ((u.pos[live_bullets][:, None, :] - b.pos[idx][None, :, :]) ** 2).sum(-1) < (BUG_RADIUS + 4) ** 2
            bullet_hit, bug_hit = np.nonzero(hit)
            if len(bullet_hit):
                first = np.unique(bullet_hit, return_index=True)[1]  # пуля бьёт одного жука
                u.alive[live_bullets[bullet_hit[first]]] = False
                np.subtract.at(self.bug_hp, idx[bug_hit[first]], 8)
        dead = np.flatnonzero(b.alive & (self.bug_hp <= 0))
        if len(dead):
            b.alive[dead] = False
            self.kills += len(dead)
            where = np.repeat(b.pos[dead], SPLASH_PER_KILL, 0)
            angle = self.rng.uniform(0, 2 * np.pi, len(where))
            speed = self.rng.uniform(40, 160, len(where))
            vel = np.stack([np.cos(angle) * speed, np.sin(angle) * speed], 1).astype(np.float32)
            self.splash.spawn(len(where), where, vel, life=0.6)
            self.gems.spawn(len(dead), b.pos[dead], life=30)

        s = self.splash
        s.pos += s.vel * dt
        s.vel *= 0.9
        s.life -= dt
        s.alive &= s.life > 0

        # Кристаллы опыта притягиваются в радиусе подбора и исчезают у игрока
        g = self.gems
        to_player = self.player - g.pos
        gd = np.linalg.norm(to_player, axis=1) + 1e-3
        pulled = g.alive & (gd < PICKUP_RADIUS)
        g.pos[pulled] += (to_player[pulled] / gd[pulled, None]) * 500 * dt
        g.alive &= gd > 20

        self.update_ms.append((time.perf_counter() - t0) * 1000)
        if time.perf_counter() - self.started > self.seconds:
            self.close()

    def on_draw(self) -> None:
        t0 = time.perf_counter()
        self.clear()
        for pool in (self.splash, self.gems, self.bugs, self.bullets, self.hero):
            pool.draw(self.program)
        recent = self.frame_ms[-120:] or [0.0]
        self.hud.text = (f"жуков: {int(self.bugs.alive.sum())}  пуль: {int(self.bullets.alive.sum())}  "
                         f"брызг: {int(self.splash.alive.sum())}  кристаллов: {int(self.gems.alive.sum())}  "
                         f"кадр: {statistics.fmean(recent):.2f} мс  FPS: {arcade.get_fps():.0f}")
        self.hud.draw()
        if self.shot and time.perf_counter() - self.started > 6:
            arcade.get_image().save(self.shot)
            self.shot = ""
        self.frame_ms.append((time.perf_counter() - t0) * 1000 + (self.update_ms[-1] if self.update_ms else 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bugs", type=int, default=400)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--shot", default="", help="снимок экрана на 6-й секунде")
    args = parser.parse_args()
    arcade.enable_timings()
    window = Spike(args.bugs, args.seconds)
    window.shot = args.shot
    arcade.run()
    frames = sorted(window.frame_ms[60:])  # без первой секунды прогрева
    if frames:
        p99 = frames[int(len(frames) * 0.99) - 1]
        print(f"bugs={args.bugs} frames={len(frames)} kills={window.kills}")
        print(f"frame_ms avg={statistics.fmean(frames):.2f} p99={p99:.2f} "
              f"logic_avg={statistics.fmean(window.update_ms[60:]):.2f} fps={arcade.get_fps():.0f}")


if __name__ == "__main__":
    main()
