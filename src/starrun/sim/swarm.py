"""Толпа жуков в массивах NumPy: движение, расталкивание, атаки, особые виды, полубосс и королева.

Каждая характеристика — отдельный массив на всю ёмкость (structure of arrays); живые отмечены в alive.
Так 400 жуков обновляются за доли миллисекунды без цикла Python по каждому.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from starrun.balance import ENEMY_KINDS, EnemyCfg

KINDS = ENEMY_KINDS
WORKER, WARRIOR, SPITTER, FLYER, TANKER, BRAIN, SEMI_BOSS, QUEEN = range(len(KINDS))

CELL = 32.0          # клетка сетки плотности ≈ диаметр рабочего
GRID_W, GRID_H = 100, 70  # 3200×2240 px вокруг игрока — экран с запасом
SEPARATION = 90.0    # сила расталкивания
FLYER_PUSH = 0.3     # летающих толпа почти не толкает — они над ней
HIT_FLASH = 0.1


@dataclass
class SwarmStep:
    hits: list[float] = field(default_factory=list)                         # удары по игроку
    hit_kinds: list[int] = field(default_factory=list)                      # кто ударил (вид), параллельно hits
    summons: list[tuple[np.ndarray, int]] = field(default_factory=list)      # (где, сколько рабочих)
    volleys: list[np.ndarray] = field(default_factory=list)                  # откуда королева плюёт
    spits: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)  # стрелки: откуда, куда, урон
    swings: int = 0                                                          # завершённых замахов воинов


class Swarm:
    def __init__(self, cfgs: dict[str, EnemyCfg], capacity: int, rng: np.random.Generator) -> None:
        self.cfg = [cfgs[k] for k in KINDS]
        self.rng = rng
        self.time = 0.0
        n = capacity
        self.pos = np.zeros((n, 2), np.float32)
        self.hp = np.zeros(n, np.float32)
        self.max_hp = np.ones(n, np.float32)
        self.damage = np.zeros(n, np.float32)
        self.speed = np.zeros(n, np.float32)
        self.radius = np.ones(n, np.float32)
        self.armor = np.zeros(n, np.float32)
        self.keep = np.zeros(n, np.float32)          # держать дистанцию (стрелок, мозговой)
        self.immune = np.zeros(n, bool)              # не отбрасывается
        self.kind = np.zeros(n, np.int8)
        self.alive = np.zeros(n, bool)
        self.cooldown = np.zeros(n, np.float32)
        self.windup = np.full(n, -1.0, np.float32)   # ≥ 0 — замах воина или раздувание стрелка
        self.flash = np.zeros(n, np.float32)         # подсветка попадания
        self.stun = np.zeros(n, np.float32)
        self.burn_left = np.zeros(n, np.float32)
        self.burn_dps = np.zeros(n, np.float32)
        self.buffed = np.zeros(n, bool)              # в ауре мозгового
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
        self.armor[free] = c.armor
        self.keep[free] = c.keep_distance
        self.immune[free] = c.knockback_immune
        self.kind[free] = kind
        self.cooldown[free] = self.rng.uniform(0, c.attack_cooldown, len(free))
        self.windup[free] = -1
        for arr in (self.flash, self.stun, self.burn_left, self.burn_dps, self.dash):
            arr[free] = 0
        self.buffed[free] = False
        self.timer_a[free] = c.dash_every or c.volley_every
        self.timer_b[free] = c.summon_every
        self.alive[free] = True
        return free

    # ---------- урон и эффекты ----------
    def hit(self, idx: np.ndarray, amounts: np.ndarray | float, armor: bool = True) -> float:
        """Урон по жукам; броня снижает каждое попадание, но не ниже 1. Возвращает нанесённый урон."""
        idx = np.asarray(idx)
        if not len(idx):
            return 0.0
        dealt = np.broadcast_to(np.asarray(amounts, np.float32), idx.shape)
        if armor:
            dealt = np.maximum(dealt - self.armor[idx], np.minimum(dealt, 1.0))
        np.subtract.at(self.hp, idx, dealt)
        self.flash[idx] = HIT_FLASH
        return float(dealt.sum())

    def ignite(self, idx: np.ndarray, seconds: float, dps: float) -> None:
        self.burn_left[idx] = np.maximum(self.burn_left[idx], seconds)
        self.burn_dps[idx] = np.maximum(self.burn_dps[idx], dps)

    def burn_tick(self, dt: float) -> float:
        """Урон горения за кадр (броня его не держит). Возвращает нанесённый урон."""
        burning = np.flatnonzero(self.alive & (self.burn_left > 0))
        if not len(burning):
            return 0.0
        dealt = self.burn_dps[burning] * np.minimum(dt, self.burn_left[burning])
        self.hp[burning] -= dealt
        self.burn_left[burning] -= dt
        self.burn_dps[burning[self.burn_left[burning] <= 0]] = 0
        return float(dealt.sum())

    def stun_for(self, idx: np.ndarray, seconds: float) -> None:
        idx = np.asarray(idx)
        idx = idx[self.kind[idx] < SEMI_BOSS]  # боссов не оглушить
        self.stun[idx] = np.maximum(self.stun[idx], seconds)
        self.windup[idx] = -1

    def knock(self, idx: np.ndarray, push: np.ndarray) -> None:
        """Отбрасывание: сдвиг сразу, без инерции. Танкер и боссы стоят."""
        idx = np.asarray(idx)
        movable = ~self.immune[idx]
        self.pos[idx[movable]] += push[movable]

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

    def toughest(self, point: np.ndarray, max_range: float) -> int | None:
        """Самый живучий жук в радиусе — цель снайперки."""
        idx = self.in_radius(point, max_range)
        if not len(idx):
            return None
        return int(idx[np.argmax(self.hp[idx])])

    def in_radius(self, point: np.ndarray, radius: float) -> np.ndarray:
        idx = self.alive_idx
        d2 = ((self.pos[idx] - point) ** 2).sum(1)
        return idx[d2 <= (radius + self.radius[idx]) ** 2]

    def in_cone(self, origin: np.ndarray, direction: np.ndarray, reach: float, cone_deg: float) -> np.ndarray:
        idx = self.in_radius(origin, reach)
        rel = self.pos[idx] - origin
        dist = np.linalg.norm(rel, axis=1) + 1e-6
        cos = (rel @ direction) / dist
        # край тела в конусе — уже попал: угловой запас на радиус жука
        slack = np.arcsin(np.clip(self.radius[idx] / dist, 0, 1))
        return idx[np.arccos(np.clip(cos, -1, 1)) <= np.radians(cone_deg) / 2 + slack]

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

    def _auras(self, idx: np.ndarray, kind: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Множитель скорости и урона от ауры мозговых (1 — вне ауры)."""
        boost = np.ones(len(idx), np.float32)
        brains = kind == BRAIN
        self.buffed[idx] = False
        if brains.any():
            c = self.cfg[BRAIN]
            d2 = ((pos[:, None, :] - pos[brains][None, :, :]) ** 2).sum(-1)
            inside = (d2 < c.aura_radius**2).any(1) & ~brains
            boost[inside] += c.aura_bonus
            self.buffed[idx[inside]] = True
        return boost

    def step(self, dt: float, player: np.ndarray, player_radius: float) -> SwarmStep:
        out = SwarmStep()
        self.time += dt
        idx = self.alive_idx
        if not len(idx):
            return out
        kind = self.kind[idx]
        pos = self.pos[idx]
        to = player - pos
        dist = np.linalg.norm(to, axis=1) + 1e-3
        direction = to / dist[:, None]
        side = np.stack([-direction[:, 1], direction[:, 0]], 1)
        boost = self._auras(idx, kind, pos)
        stunned = self.stun[idx] > 0

        speed = self.speed[idx] * boost
        speed[(self.windup[idx] >= 0) | stunned] = 0  # в замахе и оглушённый — стоит
        heading = direction.copy()

        # Стрелок и мозговой держат дистанцию: ближе — пятятся, в своей полосе — ходят боком
        keep = self.keep[idx]
        holds = keep > 0
        if holds.any():
            too_close = dist < keep * 0.85
            too_far = dist > keep * 1.1
            radial = np.where(too_close, -1.0, np.where(too_far, 1.0, 0.0))
            strafe = np.where(too_close | too_far, 0.0, 0.6) * np.where(idx % 2, 1.0, -1.0)
            heading[holds] = (direction * radial[:, None] + side * strafe[:, None])[holds]

        # Летающие петляют — по ним труднее попасть
        flyers = kind == FLYER
        if flyers.any():
            weave = np.sin(self.time * 3.0 + idx[flyers] * 1.7)[:, None] * 0.7
            v = direction[flyers] + side[flyers] * weave
            heading[flyers] = v / np.linalg.norm(v, axis=1, keepdims=True)

        move = heading * speed[:, None]
        dashing = self.dash[idx] > 0
        if dashing.any():
            move[dashing] = self.dash_dir[idx[dashing]] * self.cfg[SEMI_BOSS].dash_speed
        push = self.separation(pos, player)
        push[kind == QUEEN] = 0
        push[flyers] *= FLYER_PUSH
        pos = pos + (move + push) * dt

        # Жуки не наползают на десантника: выталкиваются на край его круга
        reach = self.radius[idx] + player_radius
        to = player - pos
        dist = np.linalg.norm(to, axis=1) + 1e-3
        inside = dist < reach
        pos[inside] = player - to[inside] / dist[inside, None] * reach[inside, None]
        dist[inside] = reach[inside]
        self.pos[idx] = pos

        for arr in (self.cooldown, self.flash, self.dash, self.stun):
            arr[idx] -= dt
        cooldown = self.cooldown[idx]
        damage = self.damage[idx] * boost
        ready = (cooldown <= 0) & ~stunned

        # Рабочие, летающие, танкер, полубосс и королева бьют при касании
        contact = ~np.isin(kind, (WARRIOR, SPITTER, BRAIN))
        touching = contact & ready & (dist <= reach + 4)
        out.hits += damage[touching].tolist()
        out.hit_kinds += kind[touching].tolist()
        self.cooldown[idx[touching]] = [self.cfg[k].attack_cooldown for k in kind[touching]]

        self._windups(out, dt, idx, kind, pos, direction, dist, damage, ready, player_radius)
        self._bosses(out, dt, idx, kind, direction)
        return out

    def _windups(self, out: SwarmStep, dt: float, idx: np.ndarray, kind: np.ndarray, pos: np.ndarray,
                 direction: np.ndarray, dist: np.ndarray, damage: np.ndarray, ready: np.ndarray,
                 player_radius: float) -> None:
        """Воины замахиваются и бьют, стрелки раздувают мешок и плюют."""
        w, s = self.cfg[WARRIOR], self.cfg[SPITTER]
        idle = self.windup[idx] < 0
        start_w = (kind == WARRIOR) & idle & ready & (dist < w.reach + player_radius)
        start_s = (kind == SPITTER) & idle & ready & (dist < s.attack_range)
        self.windup[idx[start_w]] = w.windup
        self.windup[idx[start_s]] = s.windup
        winding = self.windup[idx] >= 0
        self.windup[idx[winding]] -= dt
        done = winding & (self.windup[idx] < 0)
        if not done.any():
            return
        self.windup[idx[done]] = -1
        swing = done & (kind == WARRIOR)
        if swing.any():
            out.swings += int(swing.sum())
            landed = swing & (dist < w.reach + player_radius + 16)
            out.hits += damage[landed].tolist()
            out.hit_kinds += kind[landed].tolist()
            self.cooldown[idx[swing]] = w.attack_cooldown
        spit = done & (kind == SPITTER)
        for p, d, dmg in zip(pos[spit], direction[spit], damage[spit], strict=True):
            out.spits.append((p.copy(), d.copy(), float(dmg)))
        self.cooldown[idx[spit]] = s.attack_cooldown

    def _bosses(self, out: SwarmStep, dt: float, idx: np.ndarray, kind: np.ndarray, direction: np.ndarray) -> None:
        """Боссов единицы — обычный цикл."""
        big = kind >= SEMI_BOSS
        for i, k, d in zip(idx[big], kind[big], direction[big], strict=True):
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
