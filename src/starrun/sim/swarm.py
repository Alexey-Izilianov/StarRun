"""Толпа жуков в массивах NumPy: движение, расталкивание, атаки, поведение полубосса и королевы.

Каждая характеристика — отдельный массив на всю ёмкость (structure of arrays); живые отмечены в alive.
Так 400 жуков обновляются за доли миллисекунды без цикла Python по каждому.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from starrun.balance import EnemyCfg

KINDS = ("worker", "warrior", "semi_boss", "queen")
WORKER, WARRIOR, SEMI_BOSS, QUEEN = range(4)

CELL = 32.0          # клетка сетки плотности ≈ диаметр рабочего
GRID_W, GRID_H = 100, 70  # 3200×2240 px вокруг игрока — экран с запасом
SEPARATION = 90.0    # сила расталкивания
HIT_FLASH = 0.1


@dataclass
class SwarmStep:
    hits: list[float] = field(default_factory=list)                         # удары по игроку
    summons: list[tuple[np.ndarray, int]] = field(default_factory=list)      # (где, сколько рабочих)
    volleys: list[np.ndarray] = field(default_factory=list)                  # откуда королева плюёт
    swings: int = 0                                                          # завершённых замахов воинов


class Swarm:
    def __init__(self, cfgs: dict[str, EnemyCfg], capacity: int, rng: np.random.Generator) -> None:
        self.cfg = [cfgs[k] for k in KINDS]
        self.rng = rng
        n = capacity
        self.pos = np.zeros((n, 2), np.float32)
        self.hp = np.zeros(n, np.float32)
        self.max_hp = np.ones(n, np.float32)
        self.damage = np.zeros(n, np.float32)
        self.speed = np.zeros(n, np.float32)
        self.radius = np.ones(n, np.float32)
        self.kind = np.zeros(n, np.int8)
        self.alive = np.zeros(n, bool)
        self.cooldown = np.zeros(n, np.float32)
        self.windup = np.full(n, -1.0, np.float32)   # ≥ 0 — воин замахнулся, осталось секунд
        self.flash = np.zeros(n, np.float32)         # подсветка попадания
        self.timer_a = np.zeros(n, np.float32)       # рывок полубосса / залп королевы
        self.timer_b = np.zeros(n, np.float32)       # призыв рабочих
        self.dash = np.zeros(n, np.float32)
        self.dash_dir = np.zeros((n, 2), np.float32)

    # ---------- состав ----------
    @property
    def alive_idx(self) -> np.ndarray:
        return np.flatnonzero(self.alive)

    def count(self, kind: int | None = None) -> int:
        if kind is None:
            return int(self.alive.sum())
        return int((self.alive & (self.kind == kind)).sum())

    def spawn(self, kind: int, pos: np.ndarray, hp_mult: float = 1.0, damage_mult: float = 1.0) -> np.ndarray:
        free = np.flatnonzero(~self.alive)[: len(pos)]
        if not len(free):
            return free
        c = self.cfg[kind]
        self.pos[free] = pos[: len(free)]
        self.hp[free] = self.max_hp[free] = c.hp * hp_mult
        self.damage[free] = c.damage * damage_mult
        self.speed[free] = c.speed * self.rng.uniform(0.9, 1.1, len(free))
        self.radius[free] = c.radius
        self.kind[free] = kind
        self.cooldown[free] = self.rng.uniform(0, c.attack_cooldown, len(free))
        self.windup[free] = -1
        self.flash[free] = 0
        self.timer_a[free] = c.dash_every or c.volley_every
        self.timer_b[free] = c.summon_every
        self.dash[free] = 0
        self.alive[free] = True
        return free

    # ---------- урон ----------
    def hit(self, idx: np.ndarray, amounts: np.ndarray | float) -> None:
        np.subtract.at(self.hp, idx, amounts)
        self.flash[idx] = HIT_FLASH

    def collect_dead(self) -> np.ndarray:
        """Снимает убитых с доски; их pos и kind остаются в массивах до следующего спавна в эту ячейку."""
        dead = np.flatnonzero(self.alive & (self.hp <= 0))
        self.alive[dead] = False
        return dead

    # ---------- запросы ----------
    def nearest(self, point: np.ndarray, max_range: float) -> int | None:
        idx = self.alive_idx
        if not len(idx):
            return None
        d2 = ((self.pos[idx] - point) ** 2).sum(1) - self.radius[idx] ** 2
        i = int(np.argmin(d2))
        return int(idx[i]) if d2[i] <= max_range**2 else None

    def in_radius(self, point: np.ndarray, radius: float) -> np.ndarray:
        idx = self.alive_idx
        d2 = ((self.pos[idx] - point) ** 2).sum(1)
        return idx[d2 <= (radius + self.radius[idx]) ** 2]

    def far_from(self, point: np.ndarray, distance: float) -> np.ndarray:
        """Жуки, отставшие дальше distance, — их переносят вперёд, чтобы давление не пропадало.
        Боссы тоже: медленную королеву иначе можно просто оставить позади на весь финал."""
        idx = self.alive_idx
        d2 = ((self.pos[idx] - point) ** 2).sum(1)
        return idx[d2 > distance**2]

    # ---------- шаг ----------
    def separation(self, pos: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Расталкивание через сетку плотности: жуки уходят из клеток, где их больше. O(N)."""
        origin = center - np.array([GRID_W * CELL / 2, GRID_H * CELL / 2], np.float32)
        cell = np.floor((pos - origin) / CELL).astype(np.int32)
        inside = (cell[:, 0] >= 1) & (cell[:, 0] < GRID_W - 1) & (cell[:, 1] >= 1) & (cell[:, 1] < GRID_H - 1)
        flat = cell[:, 1] * GRID_W + cell[:, 0]
        density = np.bincount(flat[inside], minlength=GRID_W * GRID_H).reshape(GRID_H, GRID_W).astype(np.float32)
        gy, gx = np.gradient(density)
        push = np.zeros_like(pos)
        cx, cy = cell[inside, 0], cell[inside, 1]
        push[inside, 0] = -gx[cy, cx]
        push[inside, 1] = -gy[cy, cx]
        # В одной клетке градиента нет — небольшой случайный разброс, чтобы не слипались в точку
        crowded = np.zeros(len(pos), bool)
        crowded[inside] = density[cy, cx] > 1
        push[crowded] += self.rng.normal(0, 0.6, (int(crowded.sum()), 2)).astype(np.float32)
        return push * SEPARATION

    def step(self, dt: float, player: np.ndarray, player_radius: float) -> SwarmStep:
        out = SwarmStep()
        idx = self.alive_idx
        if not len(idx):
            return out
        kind = self.kind[idx]
        pos = self.pos[idx]
        to = player - pos
        dist = np.linalg.norm(to, axis=1) + 1e-3
        direction = to / dist[:, None]

        speed = self.speed[idx].copy()
        speed[self.windup[idx] >= 0] = 0  # в замахе воин стоит
        move = direction * speed[:, None]
        dashing = self.dash[idx] > 0
        if dashing.any():
            move[dashing] = self.dash_dir[idx[dashing]] * self.cfg[SEMI_BOSS].dash_speed
        push = self.separation(pos, player)
        push[kind == QUEEN] = 0
        pos = pos + (move + push) * dt

        # Жуки не наползают на десантника: выталкиваются на край его круга
        reach = self.radius[idx] + player_radius
        to = player - pos
        dist = np.linalg.norm(to, axis=1) + 1e-3
        inside = dist < reach
        pos[inside] = player - to[inside] / dist[inside, None] * reach[inside, None]
        dist[inside] = reach[inside]
        self.pos[idx] = pos

        for arr in (self.cooldown, self.flash, self.dash):
            arr[idx] -= dt
        cooldown = self.cooldown[idx]

        # Рабочие, полубосс и королева бьют при касании
        touching = (dist <= reach + 4) & (cooldown <= 0) & (kind != WARRIOR)
        hitters = idx[touching]
        out.hits += self.damage[hitters].tolist()
        self.cooldown[hitters] = [self.cfg[k].attack_cooldown for k in self.kind[hitters]]

        # Воины: замах, затем удар, если десантник не успел отойти
        w = self.cfg[WARRIOR]
        warrior = kind == WARRIOR
        start = warrior & (self.windup[idx] < 0) & (cooldown <= 0) & (dist < w.reach + player_radius)
        self.windup[idx[start]] = w.windup
        winding = warrior & (self.windup[idx] >= 0)
        self.windup[idx[winding]] -= dt
        done = winding & (self.windup[idx] < 0)
        if done.any():
            out.swings += int(done.sum())
            landed = done & (dist < w.reach + player_radius + 16)
            out.hits += self.damage[idx[landed]].tolist()
            self.cooldown[idx[done]] = w.attack_cooldown
            self.windup[idx[done]] = -1

        # Боссы: их единицы — обычный цикл
        for i, k, d in zip(idx[kind >= SEMI_BOSS], kind[kind >= SEMI_BOSS], direction[kind >= SEMI_BOSS], strict=True):
            c = self.cfg[k]
            self.timer_a[i] -= dt
            self.timer_b[i] -= dt
            if self.timer_a[i] <= 0:
                self.timer_a[i] = c.dash_every or c.volley_every
                if k == SEMI_BOSS:
                    self.dash[i] = c.dash_time
                    self.dash_dir[i] = d
                else:
                    out.volleys.append(self.pos[i].copy())
            if self.timer_b[i] <= 0:
                self.timer_b[i] = c.summon_every
                out.summons.append((self.pos[i].copy(), c.summon_count))
        return out
