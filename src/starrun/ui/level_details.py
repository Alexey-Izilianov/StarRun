"""Детали колонии и биомов: небольшие меши, многократно используемые инстансингом."""

from __future__ import annotations

import math

import numpy as np

from starrun.sim import terrain as T
from starrun.ui import mesh3d as M
from starrun.ui.terrain_view import cell_hash


def meshes() -> dict[str, list[M.Part]]:
    steel, dark, canvas = (0.42, 0.46, 0.45), (0.17, 0.21, 0.22), (0.36, 0.40, 0.30)
    console = [M.box((30, 18, 29), (0, 0, 14.5), dark),
               M.box((25, 14, 3), (0, 0, 30), steel),
               M.box((20, 3, 14), (0, 6, 38), dark),
               M.box((16, 1, 10), (0, 4, 38), (0.25, 0.54, 0.57))]
    bed = [M.box((29, 62, 5), (0, 0, 15), steel), M.box((26, 59, 5), (0, 0, 20), canvas),
           M.box((24, 12, 4), (0, 19, 24), (0.64, 0.63, 0.54))]
    bed += [M.box((3, 3, 15), (x, y, 7.5), dark) for x in (-12, 12) for y in (-26, 26)]
    rack = [M.box((32, 22, 3), (0, 0, z), steel) for z in (5, 24, 44)]
    rack += [M.box((3, 3, 45), (x, y, 22.5), dark) for x in (-14, 14) for y in (-9, 9)]
    rack += [M.box((10, 13, 13), (x, 0, z), canvas) for x in (-7, 7) for z in (13, 32)]
    barrel = [M.lathe([(9, 0), (10, 2), (10, 27), (9, 29)], canvas, 16)]
    barrel += [M.lathe([(10.4, z), (10.4, z + 2)], steel, 16) for z in (5, 22)]
    stone = [M.blob(8, (0, 0, 3), (0.46, 0.43, 0.36), squash=0.45, seg=9, rings=5, jag=0.14, seed=4)]
    debris = [M.box((19, 9, 2), (0, 0, 1), dark), M.box((7, 8, 5), (6, 5, 2), steel)]
    result = dict(console=console, bed=bed, rack=rack, barrel=barrel, stone=stone, debris=debris)
    for name, color, height in (("grass", (0.32, 0.37, 0.23), 13), ("fern", (0.25, 0.41, 0.25), 19),
                                ("reed", (0.37, 0.39, 0.24), 24), ("scrub", (0.48, 0.42, 0.25), 10)):
        parts = []
        for k in range(5):
            a = k * math.tau / 5
            parts.append(M.leaf(height * 0.7, 2.4, a, height * 0.5, (0, 0, height * 0.4), color))
        parts.append(M.lathe([(1.0, 0), (0.5, height)], color, 5))
        result[name] = parts
    return result


def placements(t: T.Terrain, x0: int, y0: int, x1: int, y1: int) -> dict[str, np.ndarray]:
    """Детерминированный декор; входы, пути и находки остаются читаемыми."""
    out: dict[str, list[tuple[float, float, float]]] = {}
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(t.n, x1), min(t.n, y1)
    area = np.s_[y0:y1, x0:x1]
    valid = (t.kind[area] == T.GROUND) & (t.building[area] < 0) & ~t.paths[area]
    ys, xs = np.nonzero(valid)
    xs, ys = xs + x0, ys + y0
    h = cell_hash(xs, ys, 43)
    plant = {"cactus": "scrub", "pine": "grass", "palm": "fern", "swamp": "reed"}.get(t.biome.tree, "grass")
    for name, mask in ((plant, h < 0.11), ("stone", (h > 0.94) & (t.soil[ys, xs] == T.PLAIN)),
                       ("debris", (h > 0.91) & (h < 0.92))):
        xx, yy = xs[mask], ys[mask]
        yaw = cell_hash(xx, yy, 44) * math.tau
        out[name] = list(zip((xx + 0.25 + cell_hash(xx, yy, 45) * 0.5) * t.cell,
                            (yy + 0.25 + cell_hash(xx, yy, 46) * 0.5) * t.cell, yaw, strict=True))
    for bid, bd in enumerate(t.buildings):
        if bd.x1 < x0 or bd.x0 >= x1 or bd.y1 < y0 or bd.y0 >= y1:
            continue
        doors = [(x, y) for x, y in t.doors if int(t.building[y, x]) == bid]
        for ix in range(bd.x0 + 1, bd.x1, 2):
            for iy in (bd.y0 + 1, bd.y1 - 1):
                if t.kind[iy, ix] != T.FLOOR or any(abs(ix - x) + abs(iy - y) < 3 for x, y in doors):
                    continue
                name = (("bed", "rack"), ("console", "rack"), ("console", "barrel"))[bid % 3][iy == bd.y1 - 1]
                out.setdefault(name, []).append(((ix + 0.5) * t.cell, (iy + 0.5) * t.cell,
                                                 0 if iy == bd.y0 + 1 else math.pi))
    return {name: np.asarray(values, np.float32).reshape(-1, 3) for name, values in out.items() if values}
