"""Сколько жуков держать на карте, какого вида и где их выпускать — всегда за краем экрана."""

from __future__ import annotations

import numpy as np

from starrun.balance import SpawnCfg

SIDES = ("left", "right", "bottom", "top")


def edge_positions(rng: np.random.Generator, count: int, center: np.ndarray, half: tuple[float, float],
                   margin: float, side: str | None = None) -> np.ndarray:
    """Точки на периметре прямоугольника экрана, расширенного на margin; side — только с одной стороны."""
    hw, hh = half[0] + margin, half[1] + margin
    if side is None:
        # Сторона выбирается пропорционально длине — плотность по периметру равномерная
        lengths = np.array([2 * hh, 2 * hh, 2 * hw, 2 * hw])
        sides = rng.choice(4, size=count, p=lengths / lengths.sum())
    else:
        sides = np.full(count, SIDES.index(side))
    t = rng.uniform(-1, 1, count)
    x = np.where(sides == 0, -hw, np.where(sides == 1, hw, t * hw))
    y = np.where(sides == 2, -hh, np.where(sides == 3, hh, t * hh))
    return (np.stack([x, y], 1) + center).astype(np.float32)


def front_side(rng: np.random.Generator, facing: np.ndarray) -> str:
    """Сторона экрана по ходу движения; при диагонали — случайно, с весом по составляющей."""
    fx, fy = abs(float(facing[0])), abs(float(facing[1]))
    horizontal = rng.random() * (fx + fy + 1e-9) < fx
    if horizontal:
        return "right" if facing[0] > 0 else "left"
    return "top" if facing[1] > 0 else "bottom"


class Spawner:
    def __init__(self, cfg: SpawnCfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.budget = 0.0  # накопленное право на спавн, чтобы не выпускать сотню за кадр
        self.rush_side = "top"
        self._rush_minute = -1

    def target_population(self, elapsed: float) -> int:
        c = self.cfg
        t = min(1.0, elapsed / (c.full_minute * 60))
        return min(c.cap, round(c.start_population + (c.full_population - c.start_population) * t))

    def warrior_share(self, elapsed: float) -> float:
        c = self.cfg
        minutes = elapsed / 60
        if minutes < c.warrior_from_minute:
            return 0.0
        t = (minutes - c.warrior_from_minute) / max(1e-6, c.warrior_share_minute - c.warrior_from_minute)
        return c.warrior_share_max * min(1.0, t)

    def in_rush(self, elapsed: float) -> bool:
        return (elapsed % 60) >= self.cfg.rush_start and (elapsed % 60) < self.cfg.rush_start + self.cfg.rush_length

    def rush_size(self, elapsed: float) -> int:
        return self.cfg.rush_base + self.cfg.rush_per_minute * int(elapsed // 60)

    def update(self, dt: float, elapsed: float, alive: int, center: np.ndarray,
               half: tuple[float, float], boost: float = 1.0) -> list[tuple[str, np.ndarray]]:
        """Возвращает пачки (вид жука, позиции) для спавна в этом кадре."""
        c = self.cfg
        batches: list[tuple[str, np.ndarray]] = []
        minute = int(elapsed // 60)
        if self.in_rush(elapsed) and self._rush_minute != minute:
            # Налёт: колонна с одной стороны, заметная заранее
            self._rush_minute = minute
            self.rush_side = SIDES[int(self.rng.integers(4))]
            n = min(self.rush_size(elapsed), c.cap - alive)
            if n > 0:
                batches += self._split(n, elapsed, edge_positions(self.rng, n, center, half, c.margin, self.rush_side))
                alive += n
        self.budget = min(self.budget + c.max_per_second * boost * dt, c.max_per_second)
        missing = min(round(self.target_population(elapsed) * boost), c.cap) - alive
        n = int(min(missing, self.budget))
        if n > 0:
            self.budget -= n
            batches += self._split(n, elapsed, edge_positions(self.rng, n, center, half, c.margin))
        return batches

    def _split(self, n: int, elapsed: float, pos: np.ndarray) -> list[tuple[str, np.ndarray]]:
        warriors = self.rng.random(n) < self.warrior_share(elapsed)
        out = [("worker", pos[~warriors])]
        if warriors.any():
            out.append(("warrior", pos[warriors]))
        return [(k, p) for k, p in out if len(p)]
