"""Камера 3D: сверху под наклоном с перспективой, как в Deep Rock Galactic: Survivor.

Мир остаётся в пикселях симуляции (x вправо, y «вверх по карте», z — высота), модели в метрах масштабируются
на PX_PER_M. Матрицы — NumPy (row-major, вектор-столбец справа); в Arcade уходят транспонированными.
"""

from __future__ import annotations

import math

import numpy as np
from pyglet.math import Mat4

PX_PER_M = 48.0      # десантник радиусом 22 px ≈ 0,46 м


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], np.float64)


def ortho(half_w: float, half_h: float, near: float, far: float) -> np.ndarray:
    return np.array([[1 / half_w, 0, 0, 0],
                     [0, 1 / half_h, 0, 0],
                     [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                     [0, 0, 0, 1]], np.float64)


def look_at(eye: np.ndarray, target: np.ndarray, up: tuple[float, float, float] = (0, 0, 1)) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def to_mat4(m: np.ndarray) -> Mat4:
    return Mat4(*m.T.ravel().tolist())


class Camera3D:
    """Следует за целью с юга под наклоном pitch. Дистанция и угол обзора подобраны так, чтобы видимый кусок
    земли был не больше зоны, за которой симуляция выпускает жуков (view_half), — иначе они появлялись бы
    на глазах."""

    def __init__(self, pitch: float = 55.0, fov: float = 30.0, distance: float = 1400.0) -> None:
        self.pitch, self.fov, self.distance = pitch, fov, distance
        self.aspect = 16 / 9
        self.target = np.zeros(3)
        self.shake = np.zeros(2)
        self.view = np.identity(4)
        self.proj = np.identity(4)
        self.update(np.zeros(2))

    @property
    def eye(self) -> np.ndarray:
        p = math.radians(self.pitch)
        return self.target + np.array([0, -math.cos(p), math.sin(p)]) * self.distance

    def update(self, target: np.ndarray, aspect: float | None = None) -> None:
        if aspect:
            self.aspect = aspect
        self.target = np.array([target[0] + self.shake[0], target[1] + self.shake[1], 0.0])
        self.view = look_at(self.eye, self.target)
        self.proj = perspective(self.fov, self.aspect, 40.0, self.distance * 4)

    @property
    def view_proj(self) -> np.ndarray:
        return self.proj @ self.view

    def apply(self, ctx) -> None:  # type: ignore[no-untyped-def]
        """Матрицы — в общий блок окна Arcade: им пользуются и фигуры 2D-слоёв, они ложатся на землю."""
        ctx.projection_matrix = to_mat4(self.proj)
        ctx.view_matrix = to_mat4(self.view)

    # ---------- экран ↔ мир ----------
    def ground_point(self, ndc_x: float, ndc_y: float) -> np.ndarray:
        """Точка земли (z = 0) под точкой экрана в нормализованных координатах −1..1."""
        inv = np.linalg.inv(self.view_proj)
        near = inv @ np.array([ndc_x, ndc_y, -1, 1])
        far = inv @ np.array([ndc_x, ndc_y, 1, 1])
        near, far = near[:3] / near[3], far[:3] / far[3]
        d = far - near
        t = -near[2] / d[2] if abs(d[2]) > 1e-9 else 0.0
        return (near + d * t)[:2]

    def to_ndc(self, point: np.ndarray, z: float = 0.0) -> tuple[float, float]:
        c = self.view_proj @ np.array([point[0], point[1], z, 1.0])
        w = c[3] if abs(c[3]) > 1e-9 else 1e-9
        return float(c[0] / w), float(c[1] / w)

    def view_half(self) -> tuple[float, float]:
        """Полуразмеры прямоугольника вокруг цели, в который целиком влезает видимая земля."""
        corners = [self.ground_point(x, y) - self.target[:2] for x in (-1, 1) for y in (-1, 1)]
        xs = [abs(c[0]) for c in corners]
        ys = [abs(c[1]) for c in corners]
        return max(xs), max(ys)

    def ground_rect(self) -> tuple[float, float, float, float]:
        """Ограничивающий прямоугольник видимой земли — для отсечения 2D-слоёв."""
        pts = np.array([self.ground_point(x, y) for x in (-1, 1) for y in (-1, 1)])
        return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())
