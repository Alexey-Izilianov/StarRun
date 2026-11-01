"""Небольшие пиктограммы полевого интерфейса: входят в общий пакет Shapes."""

from __future__ import annotations

import numpy as np

from starrun.ui.render import Shapes, rgba


def icon(s: Shapes, name: str, x: float, y: float, size: float, color: tuple[int, ...]) -> None:
    """Центр и квадратный размер; все детали — доли условной сетки 32 × 32."""
    k = size / 32

    def rect(l: float, b: float, r: float, t: float) -> None:  # noqa: E741
        s.rect(x + l * k, y + b * k, x + r * k, y + t * k, color)

    def ring(cx: float, cy: float, diameter: float, width: float = 0.15) -> None:
        s.add_one(x + cx * k, y + cy * k, diameter * k, rgba(color), ring=width)

    if name in ("mar", "shotgun", "sniper", "flame", "rocket", "shock", "turret"):
        if name == "rocket":
            rect(-14, -4, 13, 4)
            rect(-15, -6, -12, 6)
            rect(12, -6, 15, 6)
            rect(-5, -10, -1, -4)
            rect(1, 4, 7, 7)
        elif name == "turret":
            rect(-10, 2, 8, 8)
            rect(8, 4, 16, 7)
            rect(-3, -6, 2, 2)
            rect(-10, -12, -6, -6)
            rect(-6, -8, 7, -5)
            rect(7, -12, 11, -6)
        elif name == "shock":
            rect(-10, -2, 7, 5)
            rect(7, 2, 13, 6)
            rect(8, -4, 13, 0)
            rect(-6, -9, -2, -2)
            for cx in (-6, 0, 6):
                rect(cx, 7, cx + 2, 10)
        else:
            rect(-9, -2, 7, 5)
            rect(7, 1, 15, 4)
            rect(-15, 0, -9, 4)
            rect(-12, -3, -8, 1)
            rect(-4, -8, 0, -2)
            rect(2, -7, 6, -2)
            if name == "sniper":
                rect(-1, 7, 8, 10)
                rect(2, 5, 4, 7)
                rect(14, 1, 18, 3)
            elif name == "shotgun":
                rect(3, -4, 12, -1)
                rect(8, 5, 15, 7)
            elif name == "flame":
                rect(-9, -8, -5, -2)
                rect(13, -1, 15, 6)
    elif name in ("health", "max_hp", "regen", "armor"):
        rect(-3, -12, 3, 12)
        rect(-12, -3, 12, 3)
        if name == "armor":
            s.outline(x - size * 0.45, y - size * 0.45, x + size * 0.45, y + size * 0.45, color, k)
    elif name in ("speed", "sprint", "sprint_mult", "stamina"):
        rect(-10, 5, 5, 9)
        rect(-5, 1, 10, 5)
        rect(-10, -3, 5, 1)
        rect(-5, -7, 10, -3)
    elif name in ("dna", "kills"):
        for i in range(5):
            yy = -12 + i * 6
            off = float(np.sin(i * 1.7) * 7)
            rect(-abs(off), yy - 0.8, abs(off), yy + 0.8)
            s.add_one(x + off * k, y + yy * k, 4 * k, rgba(color))
            s.add_one(x - off * k, y + yy * k, 4 * k, rgba(color))
    elif name in ("pickup", "pickup_radius"):
        ring(0, 0, 23)
        rect(-2, -6, 2, 6)
        rect(-6, -2, 6, 2)
        rect(-15, -1, -11, 1)
        rect(11, -1, 15, 1)
    elif name in ("orbital", "orbital_rate", "orbital_radius", "range_mult"):
        ring(0, 0, 20)
        ring(0, 0, 8, 0.3)
        rect(-16, -1, -8, 1)
        rect(8, -1, 16, 1)
        rect(-1, -16, 1, -8)
        rect(-1, 8, 1, 16)
    else:
        ring(0, 0, 24)
        rect(-5, -3, 5, 3)
        rect(-2, -7, 2, 7)


def corners(s: Shapes, box: tuple[float, float, float, float], color: tuple[int, ...],
            length: float = 12, width: float = 2) -> None:
    l, b, r, t = box  # noqa: E741
    for x, direction in ((l, 1), (r, -1)):
        for y, vertical in ((b, 1), (t, -1)):
            s.rect(min(x, x + direction * length), min(y, y + vertical * width),
                   max(x, x + direction * length), max(y, y + vertical * width), color)
            s.rect(min(x, x + direction * width), min(y, y + vertical * length),
                   max(x, x + direction * width), max(y, y + vertical * length), color)
