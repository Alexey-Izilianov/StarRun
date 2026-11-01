"""Ограниченный пул гильз и сброшенных магазинов; случайность не затрагивает симуляцию."""

from __future__ import annotations

import math

import numpy as np

RECOIL = {"shot_mar": 0.025, "shot_shotgun": 0.10, "shot_sniper": 0.075, "shot_rocket": 0.055,
          "shot_shock": 0.02}


def recoil_step(previous: float, dt: float, events: list[str]) -> float:
    return max(previous * math.exp(-24 * dt), max((RECOIL.get(e, 0.0) for e in events), default=0.0))


class SpentCases:
    def __init__(self, capacity: int = 128) -> None:
        self.pos = np.zeros((capacity, 3), np.float32)
        self.vel = np.zeros((capacity, 3), np.float32)
        self.yaw = np.zeros(capacity, np.float32)
        self.spin = np.zeros(capacity, np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.scale = np.ones(capacity, np.float32)
        self.magazine = np.zeros(capacity, bool)
        self.cursor = 0
        self.rng = np.random.default_rng(23)

    def emit(self, pos: np.ndarray, facing: np.ndarray, gun: str, *, magazine: bool = False) -> None:
        if gun not in ("mar", "shotgun", "sniper"):
            return
        i = self.cursor
        self.cursor = (i + 1) % len(self.life)
        right = np.array([facing[1], -facing[0]])
        direction = -right if magazine else right
        self.pos[i] = (*np.asarray(pos + direction * 13 + facing * 10), 53.0)
        self.vel[i] = (*(direction * self.rng.uniform(65, 105) + facing * self.rng.uniform(-25, 10)),
                       self.rng.uniform(40, 85) if magazine else self.rng.uniform(70, 120))
        self.yaw[i] = self.rng.uniform(0, 6.28)
        self.spin[i] = self.rng.uniform(-16, 16)
        self.scale[i] = 1.3 if gun == "shotgun" else 1.0
        self.life[i] = 3.0
        self.magazine[i] = magazine

    def step(self, dt: float) -> None:
        active = self.life > 0
        self.life[active] -= dt
        self.vel[active, 2] -= 440 * dt
        self.pos[active] += self.vel[active] * dt
        self.yaw[active] += self.spin[active] * dt
        landed = active & (self.pos[:, 2] < 1.3)
        self.pos[landed, 2] = 1.3
        self.vel[landed, 2] *= -0.22
        self.vel[landed, :2] *= np.exp(-dt * 15)
        self.spin[landed] *= np.exp(-dt * 15)

    def instances(self, *, magazine: bool = False) -> np.ndarray:
        idx = np.flatnonzero((self.life > 0) & (self.magazine == magazine))
        out = np.zeros((len(idx), 12), np.float32)
        out[:, :3] = self.pos[idx]
        out[:, 3] = self.yaw[idx]
        out[:, 4:7] = self.scale[idx, None]
        out[:, 7] = np.clip(self.life[idx] / 0.5, 0, 1)
        return out
