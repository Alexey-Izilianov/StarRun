"""Плавное раскрытие окружения: прозрачные экземпляры не пишут в буфер глубины."""

from __future__ import annotations

import numpy as np

from starrun.sim.terrain import Terrain


def shell_alpha(pos: np.ndarray, terrain: Terrain, roof_alpha: dict[int, float]) -> np.ndarray:
    """Доля верхней оболочки дома: полностью раскрытый дом не имеет высоких стен."""
    x, y = terrain.cells(pos)
    ids = terrain.building[y, x]
    lookup = np.ones(len(terrain.buildings) + 1, np.float32)
    for bid, alpha in roof_alpha.items():
        if 0 <= bid < len(terrain.buildings):
            lookup[bid] = np.clip(alpha, 0.0, 1.0) ** 2
    return lookup[ids]  # -1 — последняя единица, вне постройки оболочка остаётся


def prop_opacity(pos: np.ndarray, player: np.ndarray, terrain: Terrain | None,
                 roof_alpha: dict[int, float]) -> np.ndarray:
    rel = pos - player
    corridor = (rel[:, 1] < 40) & (rel[:, 1] > -230) & (np.abs(rel[:, 0]) < 70 + np.maximum(0, -rel[:, 1]) * 0.15)
    opacity = np.where(corridor, 0.18, 1.0).astype(np.float32)
    if terrain is not None:
        x, y = terrain.cells(pos)
        ids = terrain.building[y, x]
        for bid, alpha in roof_alpha.items():
            opacity[ids == bid] = np.minimum(opacity[ids == bid], 0.18 + 0.82 * alpha)
    return opacity
