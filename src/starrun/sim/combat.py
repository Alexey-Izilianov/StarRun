"""Стрельба: MAR с магазином, пули десантника, кислота жуков, удар с орбиты."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrun.balance import OrbitalCfg, WeaponCfg
from starrun.sim.progression import Stats
from starrun.sim.swarm import Swarm

BULLET_RADIUS = 4.0


class Gun:
    """Темп стрельбы и магазин. Сколько выстрелов в кадре — решает здесь, куда — мир."""

    def __init__(self, cfg: WeaponCfg) -> None:
        self.cfg = cfg
        self.cooldown = 0.0
        self.ammo = cfg.magazine
        self.reload_left = 0.0

    def magazine(self, stats: Stats) -> int:
        return max(1, round(self.cfg.magazine * stats["ammo_mult"]))

    @property
    def reloading(self) -> bool:
        return self.reload_left > 0

    def update(self, dt: float, stats: Stats, has_target: bool) -> int:
        if self.reloading:
            self.reload_left -= dt * stats["reload_mult"]
            if self.reload_left <= 0:
                self.reload_left = 0.0
                self.ammo = self.magazine(stats)
            return 0
        self.cooldown -= dt
        if not has_target:
            self.cooldown = max(self.cooldown, 0.0)  # без цели темп не копится
            return 0
        interval = 1 / (self.cfg.fire_rate * stats["mar_rate"])
        shots = 0
        while self.cooldown <= 0 and self.ammo > 0:
            shots += 1
            self.ammo -= 1
            self.cooldown += interval
        if self.ammo == 0:
            self.reload_left = self.cfg.reload
        return shots


class Bullets:
    def __init__(self, capacity: int = 1024) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.damage = np.zeros(capacity, np.float32)
        self.pierce = np.zeros(capacity, np.int16)
        self.last_hit = np.full(capacity, -1, np.int32)
        self.alive = np.zeros(capacity, bool)

    def fire(self, origin: np.ndarray, directions: np.ndarray, speed: float, damage: np.ndarray,
             pierce: int, life: float) -> None:
        free = np.flatnonzero(~self.alive)[: len(directions)]
        n = len(free)
        self.pos[free] = origin
        self.vel[free] = directions[:n] * speed
        self.life[free] = life
        self.damage[free] = damage[:n]
        self.pierce[free] = pierce
        self.last_hit[free] = -1
        self.alive[free] = True

    def step(self, dt: float, swarm: Swarm) -> tuple[np.ndarray, np.ndarray]:
        """Полёт и попадания; возвращает (номера жуков, урон). Пуля за кадр бьёт одного — ближайшего."""
        self.pos += self.vel * dt
        self.life -= dt
        self.alive &= self.life > 0
        b = np.flatnonzero(self.alive)
        bugs = swarm.alive_idx
        empty = np.empty(0, np.int64), np.empty(0, np.float32)
        if not len(b) or not len(bugs):
            return empty
        # Пуль в полёте десятки — попарная проверка с жуками дешевле любой структуры
        d2 = ((self.pos[b][:, None, :] - swarm.pos[bugs][None, :, :]) ** 2).sum(-1)
        hit = d2 < (swarm.radius[bugs][None, :] + BULLET_RADIUS) ** 2
        hit &= bugs[None, :] != self.last_hit[b][:, None]
        any_hit = hit.any(1)
        if not any_hit.any():
            return empty
        first = np.argmin(np.where(hit, d2, np.inf), 1)
        hb = b[any_hit]
        bug = bugs[first[any_hit]]
        damage = self.damage[hb].copy()
        self.last_hit[hb] = bug
        self.pierce[hb] -= 1
        self.alive[hb] &= self.pierce[hb] >= 0
        return bug, damage


class Acid:
    """Плевки королевы: летят по прямой, бьют десантника при касании."""

    RADIUS = 10.0

    def __init__(self, capacity: int = 256) -> None:
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

    def step(self, dt: float, player: np.ndarray, player_radius: float) -> list[float]:
        self.pos += self.vel * dt
        self.life -= dt
        self.alive &= self.life > 0
        idx = np.flatnonzero(self.alive)
        d2 = ((self.pos[idx] - player) ** 2).sum(1)
        hit = idx[d2 < (player_radius + self.RADIUS) ** 2]
        self.alive[hit] = False
        return self.damage[hit].tolist()


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
