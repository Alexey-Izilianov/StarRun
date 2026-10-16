"""Отрисовка массовых объектов: круги, кольца и прямоугольники с цветом на экземпляр, один вызов на пакет.

Данные (центр, размер, цвет, кольцо, форма) собираются в NumPy и целиком уходят в видеокарту —
никаких спрайтов на каждого жука (спайк этапа 0: SpriteList на 400 объектов давал 44 FPS, так — 300+).
Прямоугольники интерфейса идут тем же путём: два десятка плашек HUD — один вызов, а не двадцать.
"""

from __future__ import annotations

import arcade
import numpy as np
from arcade.gl import BufferDescription

VERTEX = """
#version 330
uniform WindowBlock { mat4 projection; mat4 view; } window;
in vec2 in_corner;
in vec2 in_pos;
in vec2 in_size;
in vec4 in_color;
in float in_ring;
in float in_round;
out vec2 v_corner;
out vec4 v_color;
out float v_ring;
out float v_round;
void main() {
    v_corner = in_corner;
    v_color = in_color;
    v_ring = in_ring;
    v_round = in_round;
    gl_Position = window.projection * window.view * vec4(in_pos + in_corner * in_size * 0.5, 0.0, 1.0);
}
"""
FRAGMENT = """
#version 330
in vec2 v_corner;
in vec4 v_color;
in float v_ring;
in float v_round;
out vec4 f_color;
void main() {
    float a = 1.0;
    if (v_round > 0.5) {
        float r = length(v_corner);
        float aa = fwidth(r);                   // сглаживание края в один пиксель экрана
        a = 1.0 - smoothstep(1.0 - aa, 1.0, r);
        if (v_ring > 0.0) a *= smoothstep(1.0 - v_ring - aa, 1.0 - v_ring, r);
        if (a <= 0.0) discard;
    }
    f_color = vec4(v_color.rgb, v_color.a * a);
}
"""
QUAD = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
FLOATS = 10  # pos 2 + size 2 + color 4 + ring 1 + round 1


def rgba(color: tuple[int, ...], alpha: float | None = None) -> np.ndarray:
    a = alpha if alpha is not None else (color[3] / 255 if len(color) > 3 else 1.0)
    return np.array([color[0] / 255, color[1] / 255, color[2] / 255, a], np.float32)


class Shapes:
    """Пакет фигур: копит экземпляры через add*(), рисует всё одним вызовом во flush()."""

    def __init__(self, ctx: arcade.ArcadeContext, capacity: int = 20_000) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.capacity = capacity
        self.data = np.zeros((capacity, FLOATS), np.float32)
        self.count = 0
        self.buffer = ctx.buffer(reserve=capacity * FLOATS * 4)
        corners = ctx.buffer(data=QUAD.tobytes())
        self.geometry = ctx.geometry(
            [BufferDescription(corners, "2f", ["in_corner"]),
             BufferDescription(self.buffer, "2f 2f 4f 1f 1f",
                               ["in_pos", "in_size", "in_color", "in_ring", "in_round"], instanced=True)],
            mode=ctx.TRIANGLE_STRIP)

    def _reserve(self, n: int) -> np.ndarray:
        if self.count + n > self.capacity:
            self.flush()
        n = min(n, self.capacity)
        chunk = self.data[self.count:self.count + n]
        self.count += n
        return chunk

    def add(self, pos: np.ndarray, size: np.ndarray | float, color: np.ndarray, ring: np.ndarray | float = 0.0) -> None:
        """Круги: pos (n,2); size — диаметр; color (4,) или (n,4) в 0..1; ring — доля радиуса под кольцо."""
        n = len(pos)
        if not n:
            return
        chunk = self._reserve(n)
        n = len(chunk)
        chunk[:, 0:2] = pos[:n]
        diameter = size if np.isscalar(size) else np.asarray(size)[:n]
        chunk[:, 2] = diameter
        chunk[:, 3] = diameter
        chunk[:, 4:8] = color if np.ndim(color) == 1 else np.asarray(color)[:n]
        chunk[:, 8] = ring if np.isscalar(ring) else np.asarray(ring)[:n]
        chunk[:, 9] = 1.0

    def add_one(self, x: float, y: float, size: float, color: np.ndarray, ring: float = 0.0) -> None:
        self.add(np.array([[x, y]], np.float32), size, color, ring)

    def rect(self, left: float, bottom: float, right: float, top: float, color: tuple[int, ...]) -> None:
        chunk = self._reserve(1)
        chunk[0] = ((left + right) / 2, (bottom + top) / 2, right - left, top - bottom, *rgba(color), 0.0, 0.0)

    def outline(self, left: float, bottom: float, right: float, top: float, color: tuple[int, ...],
                width: float = 1.0) -> None:
        w = width
        self.rect(left, bottom, right, bottom + w, color)
        self.rect(left, top - w, right, top, color)
        self.rect(left, bottom + w, left + w, top - w, color)
        self.rect(right - w, bottom + w, right, top - w, color)

    def flush(self) -> None:
        if not self.count:
            return
        self.buffer.write(self.data[:self.count].tobytes())
        self.ctx.enable(self.ctx.BLEND)
        self.geometry.render(self.program, vertices=4, instances=self.count)
        self.count = 0
