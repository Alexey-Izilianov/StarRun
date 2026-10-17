"""Снаряды и удары по площади: пули всех стволов, взрывы, кислота жуков, удар с орбиты."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from starrun.balance import OrbitalCfg
from starrun.sim.progression import Stats
from starrun.sim.swarm import Swarm
from starrun.sim.terrain import Terrain

BULLET_RADIUS = 4.0
WALL_SAMPLES = np.array([0.25, 0.5, 0.75, 1.0], np.float32)

# Вид снаряда — для отрисовки
BULLET, PELLET, SNIPER, ROCKET, GRENADE, TURRET_SHOT = range(6)


@dataclass
class Explosion:
    pos: np.ndarray
    radius: float
    damage: float
    knock: float = 0.0
    cluster: int = 0            # сколько малых зарядов разлетится (кассета)
    cluster_damage: float = 0.0
    cluster_radius: float = 0.0


@dataclass
class Impacts:
    """Итог полёта пуль за кадр: прямые попадания и взрывы."""

    bug: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    damage: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    knock: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    burn_time: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    burn_dps: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    explosions: list[Explosion] = field(default_factory=list)


class Bullets:
    def __init__(self, capacity: int = 2048) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.damage = np.zeros(capacity, np.float32)
        self.pierce = np.zeros(capacity, np.int16)
        self.last_hit = np.full(capacity, -1, np.int32)
        self.kind = np.zeros(capacity, np.int8)
        self.blast = np.zeros(capacity, np.float32)       # радиус взрыва; 0 — обычная пуля
        self.knock = np.zeros(capacity, np.float32)
        self.burn_time = np.zeros(capacity, np.float32)
        self.burn_dps = np.zeros(capacity, np.float32)
        self.cluster = np.zeros(capacity, np.int8)
        self.cluster_damage = np.zeros(capacity, np.float32)
        self.cluster_radius = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)

    def fire(self, origin: np.ndarray, directions: np.ndarray, speed: float, damage: np.ndarray | float,
             life: float, *, pierce: int = 0, kind: int = BULLET, blast: float = 0.0, knock: float = 0.0,
             burn: tuple[float, float] = (0.0, 0.0), cluster: tuple[int, float, float] = (0, 0.0, 0.0)) -> None:
        free = np.flatnonzero(~self.alive)[: len(directions)]
        n = len(free)
        self.pos[free] = origin
        self.vel[free] = directions[:n] * speed
        self.life[free] = life
        self.damage[free] = np.broadcast_to(np.asarray(damage, np.float32), (len(directions),))[:n]
        self.pierce[free] = pierce
        self.last_hit[free] = -1
        self.kind[free] = kind
        self.blast[free] = blast
        self.knock[free] = knock
        self.burn_time[free], self.burn_dps[free] = burn
        self.cluster[free], self.cluster_damage[free], self.cluster_radius[free] = cluster
        self.alive[free] = True

    def _explode(self, b: np.ndarray, out: Impacts) -> None:
        for i in b:
            out.explosions.append(Explosion(self.pos[i].copy(), float(self.blast[i]), float(self.damage[i]),
                                            float(self.knock[i]), int(self.cluster[i]),
                                            float(self.cluster_damage[i]), float(self.cluster_radius[i])))
        self.alive[b] = False

    def step(self, dt: float, swarm: Swarm, terrain: Terrain | None = None) -> Impacts:
        """Полёт и попадания. Пуля за кадр бьёт одного — ближайшего; снаряд со взрывом рвётся при касании
        или в конце полёта (ракета долетает до точки цели). На карте лес, стены и закрытые двери
        останавливают снаряды (взрывные рвутся о них)."""
        out = Impacts()
        start = self.pos.copy()
        self.pos += self.vel * dt
        self.life -= dt
        expired = self.alive & (self.life <= 0)
        self._explode(np.flatnonzero(expired & (self.blast > 0)), out)
        self.alive &= self.life > 0
        b = np.flatnonzero(self.alive)
        if not len(b):
            return out
        wall = wall_t(terrain, start[b], self.pos[b])
        bugs = swarm.alive_idx
        any_hit = np.zeros(len(b), bool)
        if len(bugs):
            # Пуль в полёте десятки — попарная проверка с жуками дешевле любой структуры.
            # Проверяется отрезок полёта за кадр, а не точка: снайперская пуля проходит 40+ px за кадр
            # и иначе проскочила бы жука насквозь.
            p0 = start[b][:, None, :]
            seg = (self.pos[b] - start[b])[:, None, :]
            rel = swarm.pos[bugs][None, :, :] - p0
            t = np.clip((rel * seg).sum(-1) / ((seg**2).sum(-1) + 1e-9), 0, 1)
            d2 = ((rel - seg * t[..., None]) ** 2).sum(-1)
            hit = d2 < (swarm.radius[bugs][None, :] + BULLET_RADIUS) ** 2
            hit &= bugs[None, :] != self.last_hit[b][:, None]
            hit &= t < wall[:, None]  # жук за деревом не задет
            any_hit = hit.any(1)
        # В стену — без попадания в жука: снаряд гаснет там, взрывной рвётся
        stopped = ~any_hit & np.isfinite(wall)
        sb = b[stopped]
        self.pos[sb] = start[sb] + (self.pos[sb] - start[sb]) * wall[stopped, None]
        self._explode(sb[self.blast[sb] > 0], out)
        self.alive[sb] = False
        if not any_hit.any():
            return out
        first = np.argmin(np.where(hit, t, np.inf), 1)  # первый на пути, а не ближайший к концу отрезка
        hb = b[any_hit]
        bug = bugs[first[any_hit]]
        # Снаряд останавливается в точке попадания: взрыв там, следующий жук — со следующего кадра
        self.pos[hb] = start[hb] + (self.pos[hb] - start[hb]) * t[any_hit, first[any_hit], None]
        boom = self.blast[hb] > 0
        self._explode(hb[boom], out)
        hb, bug = hb[~boom], bug[~boom]
        speed = np.linalg.norm(self.vel[hb], axis=1, keepdims=True) + 1e-6
        out.bug = bug
        out.damage = self.damage[hb].copy()
        out.knock = self.vel[hb] / speed * self.knock[hb, None]
        out.burn_time = self.burn_time[hb].copy()
        out.burn_dps = self.burn_dps[hb].copy()
        self.last_hit[hb] = bug
        self.pierce[hb] -= 1
        self.alive[hb] &= self.pierce[hb] >= 0
        return out


class Acid:
    """Плевки королевы и стрелков: летят по прямой, бьют десантника при касании."""

    RADIUS = 10.0

    def __init__(self, capacity: int = 512) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.damage = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)

    def fire(self, origin: np.ndarray, directions: np.ndarray, speed: float, damage: float, life: float = 4.0) -> None:
        free = np.flatnonzero(~self.alive)[: len(directions)]
        self.pos[free] = origin
        self.vel[free] = directions[: len(free)] * speed
        self.life[free] = life
        self.damage[free] = damage
        self.alive[free] = True

    def step(self, dt: float, player: np.ndarray, player_radius: float, terrain: Terrain | None = None) -> list[float]:
        start = self.pos.copy()
        self.pos += self.vel * dt
        self.life -= dt
        self.alive &= self.life > 0
        if terrain is not None:  # кислота разбивается о лес и стены
            live = np.flatnonzero(self.alive)
            self.alive[live] = ~np.isfinite(wall_t(terrain, start[live], self.pos[live]))
        idx = np.flatnonzero(self.alive)
        d2 = ((self.pos[idx] - player) ** 2).sum(1)
        hit = idx[d2 < (player_radius + self.RADIUS) ** 2]
        self.alive[hit] = False
        return self.damage[hit].tolist()


def wall_t(terrain: Terrain | None, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Доля отрезка полёта до первой преграды для выстрела (inf — путь свободен). Отрезок проверяется
    точками через четверть: за кадр снаряд пролетает меньше двух клеток."""
    n = len(start)
    if terrain is None or not n:
        return np.full(n, np.inf, np.float32)
    seg = end - start
    pts = start[:, None, :] + seg[:, None, :] * WALL_SAMPLES[None, :, None]
    blocked = terrain.shot_blocked(pts.reshape(-1, 2)).reshape(n, len(WALL_SAMPLES))
    first = blocked.argmax(1)
    # Преграда — между предыдущей свободной точкой и первой занятой: берётся середина
    t = WALL_SAMPLES[first] - 0.125
    return np.where(blocked.any(1), np.maximum(t, 0.0), np.inf).astype(np.float32)


def fan(direction: np.ndarray, count: int, spread_deg: float) -> np.ndarray:
    """count направлений веером вокруг direction (единичные векторы)."""
    base = np.arctan2(direction[1], direction[0])
    half = np.radians(spread_deg) / 2
    angles = base + (np.linspace(-half, half, count) if count > 1 else np.zeros(1))
    return np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)


@dataclass
class Mark:
    pos: np.ndarray
    left: float
    radius: float
    damage: float


class Orbital:
    """Удар с орбиты: метка, задержка, взрыв по площади, перезарядка."""

    def __init__(self, cfg: OrbitalCfg) -> None:
        self.cfg = cfg
        self.cooldown_left = 0.0
        self.marks: list[Mark] = []

    @property
    def ready(self) -> bool:
        return self.cooldown_left <= 0

    def cooldown_total(self, stats: Stats) -> float:
        return self.cfg.cooldown / stats["orbital_rate"]

    def trigger(self, target: np.ndarray, player: np.ndarray, stats: Stats) -> bool:
        if not self.ready:
            return False
        offset = target - player
        dist = float(np.linalg.norm(offset))
        if dist > self.cfg.max_range:
            target = player + offset / dist * self.cfg.max_range
        self.marks.append(Mark(np.asarray(target, np.float32), self.cfg.delay,
                               self.cfg.radius * stats["orbital_radius"], self.cfg.damage * stats["damage_mult"]))
        self.cooldown_left = self.cooldown_total(stats)
        return True

    def update(self, dt: float) -> list[Mark]:
        self.cooldown_left = max(0.0, self.cooldown_left - dt)
        for m in self.marks:
            m.left -= dt
        fired = [m for m in self.marks if m.left <= 0]
        self.marks = [m for m in self.marks if m.left > 0]
        return fired
