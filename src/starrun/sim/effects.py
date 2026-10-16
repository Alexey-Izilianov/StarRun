"""Брызги, пятна крови на земле и кристаллы опыта."""

from __future__ import annotations

import numpy as np


class Particles:
    """Короткоживущие брызги: разлетаются, тормозят, гаснут."""

    def __init__(self, capacity: int = 6000) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.vel = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.size = np.zeros(capacity, np.float32)
        self.alive = np.zeros(capacity, bool)

    def burst(self, rng: np.random.Generator, centers: np.ndarray, per: int, speed: tuple[float, float],
              life: float, size: tuple[float, float]) -> None:
        if not len(centers):
            return
        pos = np.repeat(centers, per, 0)
        n = len(pos)
        angle = rng.uniform(0, 2 * np.pi, n)
        v = rng.uniform(*speed, n)
        vel = np.stack([np.cos(angle) * v, np.sin(angle) * v], 1)
        free = np.flatnonzero(~self.alive)[:n]
        k = len(free)
        self.pos[free] = pos[:k]
        self.vel[free] = vel[:k]
        self.life[free] = self.max_life[free] = life * rng.uniform(0.7, 1.2, k)
        self.size[free] = rng.uniform(*size, k)
        self.alive[free] = True

    def step(self, dt: float, drag: float = 6.0) -> None:
        idx = np.flatnonzero(self.alive)
        self.pos[idx] += self.vel[idx] * dt
        self.vel[idx] *= max(0.0, 1 - drag * dt)
        self.life[idx] -= dt
        self.alive[idx] &= self.life[idx] > 0


class Decals:
    """Пятна кислотной крови на земле: держатся, потом выцветают; старые уступают место новым."""

    def __init__(self, capacity: int = 700, lifetime: float = 25.0) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.size = np.zeros(capacity, np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.lifetime = lifetime
        self._next = 0

    def add(self, rng: np.random.Generator, pos: np.ndarray, size: np.ndarray) -> None:
        n = len(pos)
        cap = len(self.pos)
        slots = (self._next + np.arange(n)) % cap  # кольцевой буфер
        self._next = int((self._next + n) % cap)
        self.pos[slots] = pos + rng.normal(0, 4, (n, 2))
        self.size[slots] = size
        self.life[slots] = self.lifetime

    def step(self, dt: float) -> None:
        np.subtract(self.life, dt, out=self.life)
        np.maximum(self.life, 0, out=self.life)

    @property
    def alive(self) -> np.ndarray:
        return self.life > 0


class Gems:
    """Кристаллы опыта. В радиусе подбора притягиваются к десантнику и засчитываются при касании."""

    PULL_SPEED = 700.0
    COLLECT = 26.0

    def __init__(self, capacity: int = 1500) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.value = np.zeros(capacity, np.int32)
        self.pulled = np.zeros(capacity, bool)
        self.alive = np.zeros(capacity, bool)

    def drop(self, rng: np.random.Generator, pos: np.ndarray, values: np.ndarray) -> None:
        keep = values > 0
        pos, values = pos[keep], values[keep]
        free = np.flatnonzero(~self.alive)[: len(pos)]
        k = len(free)
        self.pos[free] = pos[:k]
        self.value[free] = values[:k]
        self.pulled[free] = False
        self.alive[free] = True
        if k < len(pos):
            # Мест нет: опыт не теряется — прибавляется к случайным лежащим кристаллам
            alive = np.flatnonzero(self.alive)
            np.add.at(self.value, rng.choice(alive, len(pos) - k), values[k:])

    def step(self, dt: float, player: np.ndarray, pickup_radius: float) -> int:
        """Возвращает собранный опыт."""
        idx = np.flatnonzero(self.alive)
        if not len(idx):
            return 0
        to = player - self.pos[idx]
        dist = np.linalg.norm(to, axis=1) + 1e-3
        self.pulled[idx] |= dist < pickup_radius  # раз потянулся — долетит, даже если десантник отошёл
        pulled = self.pulled[idx]
        speed = self.PULL_SPEED + 2 * np.maximum(0, pickup_radius - dist)
        step = np.minimum(speed[pulled] * dt, dist[pulled])  # не перелетать десантника
        self.pos[idx[pulled]] += to[pulled] / dist[pulled, None] * step[:, None]
        collected = idx[dist < self.COLLECT]
        self.alive[collected] = False
        return int(self.value[collected].sum())
