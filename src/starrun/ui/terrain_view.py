"""Карта на простых фигурах: почва, вода, мосты, лес, постройки с крышами, заборы, находки.

Рисуются только видимые клетки, все однотипные — одним пакетом. Чтобы сетка не читалась «мозаикой», круглое
(лес, вода, топь, песок, скалы) рисуется перекрывающимися кругами со сдвигом и разбросом размера по хэшу клетки,
а квадратными остаются только рукотворные стены, полы и мосты.
"""

from __future__ import annotations

import numpy as np

from starrun.sim import terrain as T
from starrun.sim.world import World
from starrun.ui.render import Shapes, rgba

SHADOW = rgba((0, 0, 0), 0.28)
ROCK = (64, 60, 56)
ROCK_DARK = (44, 41, 38)
WATER = (38, 74, 98)
WATER_LIGHT = (84, 132, 156)
SAND = (168, 142, 96)
SAND_DARK = (128, 104, 68)
MUD = (40, 46, 34)
PLANK = (112, 82, 52)
PLANK_DARK = (70, 50, 32)
FLOOR = (92, 92, 86)
WALL = (52, 54, 58)
GLASS = (130, 180, 200)
DOOR = (100, 68, 42)
FENCE = (126, 104, 72)
STUMP = (96, 72, 46)
STUMP_RING = (62, 46, 30)
ROOF_LIGHT = (92, 84, 74)
ROOF_DARK = (70, 64, 58)
ROOF_EDGE = (40, 36, 32)

ALT_CELL = 360.0   # сетка пятен почвы другого оттенка
ROOF_FADE = 6.0    # скорость, с которой крыша прячется и возвращается
MARGIN = 2         # клеток за краем экрана — чтобы крупные кроны не обрезались

# Находки: основной цвет и цвет детали
ITEM_COLORS = {
    T.MEDKIT: ((236, 232, 222), (200, 50, 44)),
    T.MAGNET: ((190, 54, 48), (200, 200, 205)),
    T.DNA: ((160, 110, 220), (110, 220, 200)),
    T.WEAPON_CRATE: ((92, 100, 64), (220, 190, 70)),
    T.UNDERBARREL: ((92, 100, 64), (240, 200, 60)),
}


def cell_hash(x: np.ndarray, y: np.ndarray, salt: int = 0) -> np.ndarray:
    """Псевдослучайное 0..1 на клетку: одно и то же при каждом кадре."""
    h = (x.astype(np.int64) * 73856093) ^ (y.astype(np.int64) * 19349663) ^ (salt * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def along(size: np.ndarray, kx: np.ndarray | float, ky: np.ndarray | float) -> np.ndarray:
    """Смещения (n,2) в долях размера каждого объекта: блик, ветка, рука кактуса."""
    return np.stack([size * kx, size * ky * np.ones_like(size)], 1).astype(np.float32)


def shade(color: tuple[int, int, int], k: float, alpha: float = 1.0) -> np.ndarray:
    return rgba(tuple(int(min(255, c * k)) for c in color), alpha)


class TerrainView:
    def __init__(self, shapes: Shapes) -> None:
        self.flat_stumps = True         # в 3D пни и находки — моделями (Scene3D), здесь только кольцо находки
        self.flat_items = True
        self.c = shapes
        self.time = 0.0
        self.roof_alpha: dict[int, float] = {}
        self.view = (0.0, 0.0, 0.0, 0.0)
        self.win: tuple[np.ndarray, np.ndarray, int, int] | None = None

    def update(self, dt: float, world: World) -> None:
        self.time += dt
        t = world.terrain
        if t is None:
            return
        ix, iy = (int(v) for v in world.player.pos // t.cell)
        inside = int(t.building[iy, ix]) if 0 <= ix < t.n and 0 <= iy < t.n else -1
        for bid in range(len(t.buildings)):
            target = 0.0 if bid == inside else 1.0
            a = self.roof_alpha.get(bid, 1.0)
            step = ROOF_FADE * dt
            self.roof_alpha[bid] = min(target, a + step) if target > a else max(target, a - step)

    # ---------- видимая часть ----------
    def prepare(self, world: World, view: tuple[float, float, float, float]) -> None:
        """В начале кадра: вырезает видимые клетки (за краем карты — скалы) для всех слоёв этого кадра."""
        t = world.terrain
        self.view = view
        if t is None:
            self.win = None
            return
        c = t.cell
        left, bottom, right, top = view
        x0, y0 = int(left // c) - MARGIN, int(bottom // c) - MARGIN
        x1, y1 = int(right // c) + MARGIN + 1, int(top // c) + MARGIN + 1
        kind = np.full((y1 - y0, x1 - x0), T.ROCK, np.uint8)
        soil = np.zeros_like(kind)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(t.n, x1), min(t.n, y1)
        if sx0 < sx1 and sy0 < sy1:
            kind[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = t.kind[sy0:sy1, sx0:sx1]
            soil[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = t.soil[sy0:sy1, sx0:sx1]
        self.win = kind, soil, x0, y0

    def _seen(self, left: float, bottom: float, right: float, top: float) -> bool:
        vl, vb, vr, vt = self.view
        return right > vl and left < vr and top > vb and bottom < vt

    def _buildings(self, t: T.Terrain) -> list[tuple[int, T.Building]]:
        """Постройки, хоть краем попавшие на экран."""
        c = t.cell
        return [(bid, bd) for bid, bd in enumerate(t.buildings)
                if self._seen(bd.x0 * c - c, bd.y0 * c - c, (bd.x1 + 2) * c, (bd.y1 + 2) * c)]

    @staticmethod
    def _cells(mask: np.ndarray, x0: int, y0: int) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(mask)
        return xs + x0, ys + y0

    def _centers(self, xs: np.ndarray, ys: np.ndarray, cell: float) -> np.ndarray:
        return np.stack([(xs + 0.5) * cell, (ys + 0.5) * cell], 1).astype(np.float32)

    def _sprinkle(self, xs: np.ndarray, ys: np.ndarray, cell: float, salt: int, share: float
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Мелкие детали (блики, лужи, воронки) — лишь в доле клеток и со сдвигом до полклетки,
        иначе они выстраиваются по сетке. Возвращает (центры, случайное 0..1 на деталь)."""
        k = cell_hash(xs, ys, salt)
        pick = k < share
        xs, ys, k = xs[pick], ys[pick], k[pick] / share
        shift = np.stack([cell_hash(xs, ys, salt + 1), cell_hash(xs, ys, salt + 2)], 1) - 0.5
        return self._centers(xs, ys, cell) + shift * cell, k

    # ---------- слои ----------
    def ground(self, world: World) -> None:
        """Под всем остальным: пятна почвы, топь, песок, вода, мосты, полы, пни."""
        t = world.terrain
        if t is None or self.win is None:
            return
        c, cell = self.c, t.cell
        b = t.biome
        self._alt_patches(self.view, b.ground_alt)
        kind, soil, x0, y0 = self.win

        mud = soil == T.MUD
        if mud.any():
            xs, ys = self._cells(mud, x0, y0)
            h = cell_hash(xs, ys, 1)
            c.add(self._centers(xs, ys, cell) + (h[:, None] - 0.5) * 10, cell * (1.45 + 0.3 * h), rgba(MUD, 0.55))
            pos, k = self._sprinkle(xs, ys, cell, 21, 0.12)  # лужи стоячей воды — редко и вразброс
            c.add(pos, cell * (0.5 + 0.6 * k), rgba(WATER, 0.45))

        sand = (soil == T.SAND) & (kind == T.GROUND)
        if sand.any():
            xs, ys = self._cells(sand, x0, y0)
            h = cell_hash(xs, ys, 2)
            pos = self._centers(xs, ys, cell)
            c.add(pos, cell * (1.75 + 0.2 * h), rgba(SAND_DARK, 0.6))  # тёмная кромка — граница опасного места
            c.add(pos, cell * (1.5 + 0.2 * h), rgba(SAND, 0.95))
            pos, k = self._sprinkle(xs, ys, cell, 22, 0.15)  # воронки медленно «дышат» — это не простая земля
            ring = np.tile(rgba(SAND_DARK), (len(pos), 1))
            ring[:, 3] = 0.2 + 0.25 * (0.5 + 0.5 * np.sin(self.time * 1.5 + k * 6.28))
            c.add(pos, cell * (1.0 + 1.0 * k), ring, ring=0.07)
            c.add(pos, cell * (0.4 + 0.4 * k), ring, ring=0.12)

        water = kind == T.WATER
        if water.any():
            xs, ys = self._cells(water, x0, y0)
            c.add(self._centers(xs, ys, cell), cell * 1.35, rgba(WATER))  # почти без захода на берег: суша не тонет
            pos, k = self._sprinkle(xs, ys, cell, 23, 0.2)
            glint = np.tile(rgba(WATER_LIGHT), (len(pos), 1))
            glint[:, 3] = 0.1 + 0.15 * (0.5 + 0.5 * np.sin(self.time * 3.0 + k * 6.28))
            c.add(pos, cell * (0.2 + 0.3 * k), glint)

        self._bridges(kind, x0, y0, cell)
        self._floors(world)

        stumps = kind == T.STUMP
        if stumps.any() and self.flat_stumps:
            xs, ys = self._cells(stumps, x0, y0)
            h = cell_hash(xs, ys, 4)
            pos = self._centers(xs, ys, cell) + (h[:, None] - 0.5) * 10
            c.add(pos, cell * 0.42, rgba(STUMP))
            c.add(pos, cell * 0.42, rgba(STUMP_RING), ring=0.3)

    def pebble_mask(self, world: World, pos: np.ndarray) -> np.ndarray:
        """Камушки — только на открытой земле, не на воде, полах и в песке."""
        t = world.terrain
        if t is None:
            return np.ones(len(pos), bool)
        k = t.kinds_at(pos)
        return ((k == T.GROUND) | (k == T.STUMP)) & (t.soil_at(pos) == T.PLAIN)

    def objects(self, world: World, solid: bool = True) -> None:
        """Над землёй, под жуками: стены, двери, окна, заборы, скалы, лес (solid) и находки."""
        t = world.terrain
        if t is None or self.win is None:
            return
        kind, _, x0, y0 = self.win
        if solid:
            self._walls(kind, x0, y0, t)
            self._fences(kind, x0, y0, t.cell)
            self._rocks(kind, x0, y0, t.cell)
            self._trees(world, kind, x0, y0)
        self._items(world)

    def roofs(self, world: World) -> None:
        """Поверх десантника и жуков: крыша прячется, когда десантник внутри."""
        t = world.terrain
        if t is None:
            return
        c, cell = self.c, t.cell
        for bid, bd in self._buildings(t):
            a = self.roof_alpha.get(bid, 1.0)
            l, b = bd.x0 * cell - 4, bd.y0 * cell - 4  # noqa: E741
            r, top = (bd.x1 + 1) * cell + 4, (bd.y1 + 1) * cell + 4
            if a <= 0.01:
                c.outline(l, b, r, top, (*ROOF_EDGE, 110), 2)
                continue
            wide = (r - l) >= (top - b)
            alpha = int(255 * a)
            if wide:  # конёк вдоль длинной стороны: скат к свету светлее
                mid = (b + top) / 2
                c.rect(l, mid, r, top, (*ROOF_LIGHT, alpha))
                c.rect(l, b, r, mid, (*ROOF_DARK, alpha))
                c.rect(l, mid - 3, r, mid + 3, (*ROOF_EDGE, alpha))
            else:
                mid = (l + r) / 2
                c.rect(l, b, mid, top, (*ROOF_LIGHT, alpha))
                c.rect(mid, b, r, top, (*ROOF_DARK, alpha))
                c.rect(mid - 3, b, mid + 3, top, (*ROOF_EDGE, alpha))
            c.outline(l, b, r, top, (*ROOF_EDGE, alpha), 3)

    # ---------- части ----------
    def _alt_patches(self, view: tuple[float, float, float, float], color: tuple[int, int, int]) -> None:
        """Крупные мягкие пятна почвы другого оттенка — земля не выглядит заливкой."""
        left, bottom, right, top = view
        xs = np.arange(np.floor(left / ALT_CELL) - 1, np.ceil(right / ALT_CELL) + 2)
        ys = np.arange(np.floor(bottom / ALT_CELL) - 1, np.ceil(top / ALT_CELL) + 2)
        gx, gy = (a.ravel() for a in np.meshgrid(xs, ys))
        h1, h2, h3 = cell_hash(gx, gy, 11), cell_hash(gx, gy, 12), cell_hash(gx, gy, 13)
        pos = np.stack([(gx + h1) * ALT_CELL, (gy + h2) * ALT_CELL], 1).astype(np.float32)
        self.c.add(pos, ALT_CELL * (0.7 + 0.8 * h3), rgba(color, 0.45))

    def _bridges(self, kind: np.ndarray, x0: int, y0: int, cell: float) -> None:
        bridge = kind == T.BRIDGE
        if not bridge.any():
            return
        xs, ys = self._cells(bridge, x0, y0)
        pos = self._centers(xs, ys, cell)
        ly, lx = ys - y0, xs - x0
        h_, w_ = kind.shape
        up = np.where(ly + 1 < h_, kind[np.minimum(ly + 1, h_ - 1), lx], T.GROUND)
        down = np.where(ly > 0, kind[np.maximum(ly - 1, 0), lx], T.GROUND)
        vertical = (up == T.BRIDGE) | (down == T.BRIDGE)  # мост идёт вдоль y — доски поперёк
        c = self.c
        c.boxes(pos, (cell, cell), rgba(PLANK))
        for k in (-1, 1):
            plank_v = pos[vertical] + [0, k * cell / 6]
            plank_h = pos[~vertical] + [k * cell / 6, 0]
            c.boxes(plank_v, (cell, 2), rgba(PLANK_DARK, 0.7))
            c.boxes(plank_h, (2, cell), rgba(PLANK_DARK, 0.7))
            c.boxes(pos[vertical] + [k * (cell / 2 - 3), 0], (6, cell), rgba(PLANK_DARK))   # перила
            c.boxes(pos[~vertical] + [0, k * (cell / 2 - 3)], (cell, 6), rgba(PLANK_DARK))

    def _floors(self, world: World) -> None:
        t = world.terrain
        assert t is not None
        cell = t.cell
        for _, bd in self._buildings(t):
            self.c.rect(bd.x0 * cell, bd.y0 * cell, (bd.x1 + 1) * cell, (bd.y1 + 1) * cell, (*FLOOR, 255))

    def _walls(self, kind: np.ndarray, x0: int, y0: int, t: T.Terrain) -> None:
        c, cell = self.c, t.cell
        for k, color in ((T.WALL, WALL), (T.WINDOW, WALL), (T.DOOR, WALL)):
            xs, ys = self._cells(kind == k, x0, y0)
            if len(xs):
                c.boxes(self._centers(xs, ys, cell), (cell, cell), rgba(color))
        for k in (T.WINDOW, T.DOOR, T.OPEN_DOOR):
            xs, ys = self._cells(kind == k, x0, y0)
            if not len(xs):
                continue
            pos = self._centers(xs, ys, cell)
            ax = self._along_x(kind, xs - x0, ys - y0)
            if k == T.WINDOW:
                c.boxes(pos[ax], (cell * 0.8, 9), rgba(GLASS, 0.85))
                c.boxes(pos[~ax], (9, cell * 0.8), rgba(GLASS, 0.85))
            elif k == T.DOOR:
                c.boxes(pos[ax], (cell * 0.9, 12), rgba(DOOR))
                c.boxes(pos[~ax], (12, cell * 0.9), rgba(DOOR))
            else:  # распахнута: створка повёрнута поперёк проёма, у петли
                c.boxes(pos[ax] + [-cell * 0.45, cell * 0.4], (8, cell * 0.85), rgba(DOOR))
                c.boxes(pos[~ax] + [cell * 0.4, -cell * 0.45], (cell * 0.85, 8), rgba(DOOR))

    @staticmethod
    def _along_x(kind: np.ndarray, lx: np.ndarray, ly: np.ndarray) -> np.ndarray:
        """Окно или дверь в стене, идущей по x: сосед слева или справа — тоже стена."""
        wallish = np.isin(kind, (T.WALL, T.WINDOW, T.DOOR, T.OPEN_DOOR))
        w = kind.shape[1]
        left = wallish[ly, np.maximum(lx - 1, 0)] & (lx > 0)
        right = wallish[ly, np.minimum(lx + 1, w - 1)] & (lx + 1 < w)
        return left | right

    def _fences(self, kind: np.ndarray, x0: int, y0: int, cell: float) -> None:
        fence = kind == T.FENCE
        if not fence.any():
            return
        xs, ys = self._cells(fence, x0, y0)
        pos = self._centers(xs, ys, cell)
        lx, ly = xs - x0, ys - y0
        h, w = kind.shape
        right = (lx + 1 < w) & (kind[ly, np.minimum(lx + 1, w - 1)] == T.FENCE)
        up = (ly + 1 < h) & (kind[np.minimum(ly + 1, h - 1), lx] == T.FENCE)
        c = self.c
        c.boxes(pos[right] + [cell / 2, 0], (cell, 4), rgba(FENCE))
        c.boxes(pos[up] + [0, cell / 2], (4, cell), rgba(FENCE))
        c.boxes(pos, (9, 9), shade(FENCE, 0.7))

    def _rocks(self, kind: np.ndarray, x0: int, y0: int, cell: float) -> None:
        rock = kind == T.ROCK
        if not rock.any():
            return
        xs, ys = self._cells(rock, x0, y0)
        h = cell_hash(xs, ys, 5)
        pos = self._centers(xs, ys, cell) + (np.stack([h, cell_hash(xs, ys, 6)], 1) - 0.5) * 14
        size = cell * (1.35 + 0.35 * h)
        self.c.add(pos, size + 6, rgba(ROCK_DARK))
        self.c.add(pos, size, rgba(ROCK))
        self.c.add(pos + along(size, -0.12, 0.12), size * 0.45, shade(ROCK, 1.2))

    def _trees(self, world: World, kind: np.ndarray, x0: int, y0: int) -> None:
        t = world.terrain
        assert t is not None
        tree = kind == T.TREE
        if not tree.any():
            return
        cell, b, c = t.cell, t.biome, self.c
        xs, ys = self._cells(tree, x0, y0)
        h = cell_hash(xs, ys, 7)
        jitter = (np.stack([h, cell_hash(xs, ys, 8)], 1) - 0.5) * 12
        pos = self._centers(xs, ys, cell) + jitter
        # Подрубленное дерево ужимается — видно, сколько осталось
        hp = t.hp[np.clip(ys, 0, t.n - 1), np.clip(xs, 0, t.n - 1)] / max(1, b.tree_hp)
        shrink = 0.75 + 0.25 * np.clip(hp, 0, 1)
        color = b.tree_color
        if b.tree == "cactus":
            # Кактус-бочка сверху: рёбра и венец светлых колючек по краю — не спутать с круглым жуком
            size = cell * (0.7 + 0.25 * h) * shrink
            c.add(pos + [5, -6], size * 1.1, SHADOW)
            c.add(pos, size + 4, shade(color, 0.5))
            c.add(pos, size, rgba(color))
            c.add(pos, size * 0.62, shade(color, 0.8), ring=0.18)
            spines = rgba((226, 216, 170))
            for k in range(8):
                a = h * 6.28 + k * 0.785
                c.add(pos + along(size, 0.46 * np.cos(a), 0.46 * np.sin(a)), 4, spines)
            bloom = h > 0.8  # изредка цветок на макушке
            c.add(pos[bloom], size[bloom] * 0.22, rgba((214, 96, 120)))
            return
        size = cell * (1.2 + 0.35 * h) * shrink
        c.add(pos + [7, -9], size * 1.05, SHADOW)
        c.add(pos, size + 5, shade(color, 0.6))
        if b.tree == "palm":  # веер листьев вокруг ствола
            for k in range(5):
                a = h * 6.28 + k * 1.2566
                leaf = pos + np.stack([np.cos(a), np.sin(a)], 1) * size[:, None] * 0.26
                c.add(leaf, size * 0.55, rgba(color) if k % 2 else shade(color, 1.15))
            c.add(pos, size * 0.28, shade(color, 0.75))
        else:
            c.add(pos, size, rgba(color))
            c.add(pos + along(size, -0.12, 0.14), size * 0.55, shade(color, 1.22))
            if b.tree == "swamp":  # свисающий мох — тёмные пятна по кроне
                c.add(pos + along(size, 0.2, -0.15), size * 0.3, shade(color, 0.7, 0.8))

    def _items(self, world: World) -> None:
        vl, vb, vr, vt = self.view
        p = world.item_pos
        idx = np.flatnonzero(world.item_alive & (p[:, 0] > vl - 40) & (p[:, 0] < vr + 40)
                             & (p[:, 1] > vb - 40) & (p[:, 1] < vt + 40))
        if not len(idx):
            return
        c = self.c
        pulse = 0.5 + 0.5 * np.sin(self.time * 3)
        pos_all = world.item_pos[idx]
        c.add(pos_all, 46 + 6 * pulse, rgba((255, 255, 255), 0.06 + 0.06 * pulse))
        if not self.flat_items:
            return
        for kind, (main, detail) in ITEM_COLORS.items():
            pos = pos_all[world.item_kind[idx] == kind]
            if not len(pos):
                continue
            if kind == T.MEDKIT:
                c.boxes(pos, (24, 18), rgba(main))
                c.boxes(pos, (12, 4), rgba(detail))
                c.boxes(pos, (4, 12), rgba(detail))
            elif kind == T.MAGNET:
                c.add(pos, 22, rgba(main), ring=0.42)
                c.boxes(pos + [-8, -8], (6, 6), rgba(detail))
                c.boxes(pos + [8, -8], (6, 6), rgba(detail))
            elif kind == T.DNA:
                for k in range(3):
                    off = np.array([(k - 1) * 7.0, (k - 1) * 7.0])
                    c.add(pos + off + [-4, 3], 7, rgba(main))
                    c.add(pos + off + [4, -3], 7, rgba(detail))
            else:
                c.boxes(pos, (30, 20), shade(main, 0.6))
                c.boxes(pos, (26, 16), rgba(main))
                c.boxes(pos, (26, 4), rgba(detail))
