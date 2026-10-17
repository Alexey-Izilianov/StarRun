"""Карта забега: сетка клеток, генерация биома, проходимость, лес под топор, двери, поле путей для жуков.

Сетка — массивы NumPy [y, x]: вид клетки (kind), почва (soil: песок, топь), прочность (hp) и номер постройки.
Мир — в пикселях от (0, 0) в левом нижнем углу; клетка (ix, iy) занимает [ix·c, (ix+1)·c) × [iy·c, (iy+1)·c).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from starrun.balance import BiomeCfg, ItemsCfg, MapCfg

GROUND, TREE, WALL, WATER, FLOOR, DOOR, WINDOW, FENCE, BRIDGE, ROCK, STUMP, OPEN_DOOR = range(12)
PLAIN, SAND, MUD = range(3)

KIND_COUNT = 12


def _table(*kinds: int) -> np.ndarray:
    t = np.zeros(KIND_COUNT, bool)
    t[list(kinds)] = True
    return t


BUG_BLOCK = _table(TREE, WALL, WATER, DOOR, FENCE, ROCK)        # дверь держит, пока не выломана
FLOW_BLOCK = _table(TREE, WALL, WATER, FENCE, ROCK)             # двери и окна — см. ENTRANCE в _flow_steps
PLAYER_BLOCK = _table(TREE, WALL, WATER, ROCK)                  # дверь открывается, окно и забор — перелезть
SHOT_BLOCK = _table(TREE, WALL, ROCK, DOOR)
SMASHABLE = _table(TREE, FENCE)                                 # танкер проламывает
OPEN = _table(GROUND, STUMP, BRIDGE)                            # сюда можно выпустить жука или поставить зону
ENTRANCE = _table(DOOR, OPEN_DOOR, WINDOW)                      # проёмы в стенах построек

# Восемь соседей: смещения (dx, dy) и единичные направления к ним
NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
NEIGHBOUR_DIRS = np.array([(dx, dy) for dx, dy in NEIGHBOURS], np.float32)
NEIGHBOUR_DIRS /= np.linalg.norm(NEIGHBOUR_DIRS, axis=1, keepdims=True)


def _search_order(r: int) -> tuple[np.ndarray, np.ndarray]:
    """Смещения клеток квадрата радиуса r от ближних к дальним — порядок поиска свободного места."""
    d = np.arange(-r, r + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(d, d))
    order = np.argsort(dx * dx + dy * dy, kind="stable")
    return dx[order], dy[order]


SEARCH_DX, SEARCH_DY = _search_order(12)
SEARCH_RING = np.maximum(np.abs(SEARCH_DX), np.abs(SEARCH_DY))

DOOR_REACH = 0.7  # дверь открывается, когда десантник ближе этой доли клетки к её краю
FLOW_STEPS = 20   # шагов обхода в ширину поля путей за кадр

# Находки на карте
MEDKIT, MAGNET, DNA, WEAPON_CRATE, UNDERBARREL = range(5)


@dataclass
class Building:
    x0: int
    y0: int
    x1: int  # включительно
    y1: int


def smooth_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Гладкий шум 0..1: случайная решётка с шагом scale, билинейно растянутая на всю сетку."""
    h, w = shape
    gh, gw = int(h / scale) + 2, int(w / scale) + 2
    grid = rng.random((gh, gw))
    ys = np.arange(h) / scale
    xs = np.arange(w) / scale
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)  # плавный, без «ромбиков» билинейки
    a = grid[y0][:, x0]
    b = grid[y0][:, x0 + 1]
    c = grid[y0 + 1][:, x0]
    d = grid[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def dilate8(b: np.ndarray) -> np.ndarray:
    """Расширение маски на 8 соседей (квадрат 3×3)."""
    h = b.copy()
    h[:, 1:] |= b[:, :-1]
    h[:, :-1] |= b[:, 1:]
    out = h.copy()
    out[1:, :] |= h[:-1, :]
    out[:-1, :] |= h[1:, :]
    return out


class Terrain:
    def __init__(self, cfg: MapCfg, biome: BiomeCfg, items: ItemsCfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.biome = biome
        self.items_cfg = items
        self.rng = rng
        n = cfg.size
        self.n = n
        self.cell = cfg.cell
        self.size_px = n * cfg.cell
        self.kind = np.zeros((n, n), np.uint8)
        self.soil = np.zeros((n, n), np.uint8)
        self.hp = np.zeros((n, n), np.float32)
        self.building = np.full((n, n), -1, np.int16)
        self.buildings: list[Building] = []
        self.doors: list[tuple[int, int]] = []  # клетки дверей (x, y), в том числе уже выломанных
        self.items: list[tuple[np.ndarray, int]] = []
        self.version = 0  # растёт при каждом изменении — по нему пересчитываются поле путей и кэши отрисовки
        self._flow_version = -1
        self._flow_timer = 0.0
        self._flow_cell = (-1, -1)
        self._flow_job: Iterator[bool] | None = None
        self.flow_origin = (0, 0)
        self.flow_dist = np.zeros((1, 1), np.float32)
        self.flow_vec = np.zeros((1, 1, 2), np.float32)
        self.generate()

    @property
    def center(self) -> np.ndarray:
        return np.array([self.size_px / 2, self.size_px / 2], np.float32)

    # ---------- генерация ----------
    def generate(self) -> None:
        b, n, rng = self.biome, self.n, self.rng
        noise = smooth_noise((n, n), b.forest_scale, rng) * 0.75 + smooth_noise((n, n), b.forest_scale / 3, rng) * 0.25
        forest = noise > np.quantile(noise, 1 - b.forest)
        self.kind[forest] = TREE
        self.hp[forest] = b.tree_hp
        if b.mud > 0:
            mud = smooth_noise((n, n), 14, rng)
            self.soil[mud > np.quantile(mud, 1 - b.mud)] = MUD
        for i in range(b.rivers):
            self._river(horizontal=i % 2 == 0)
        for _ in range(b.quicksand):
            self._patch(SAND, int(rng.integers(3, 6)))
        for _ in range(b.buildings):
            self._building()
        self._clear_circle(n // 2, n // 2, self.cfg.start_clear, water=True)
        border = self.cfg.border
        self.kind[:border, :] = self.kind[-border:, :] = ROCK
        self.kind[:, :border] = self.kind[:, -border:] = ROCK
        self._scatter_items()

    def _free_spot(self, margin: int, avoid_center: int) -> tuple[int, int]:
        n, c = self.n, self.n // 2
        for _ in range(200):
            x, y = (int(v) for v in self.rng.integers(margin, n - margin, 2))
            if (x - c) ** 2 + (y - c) ** 2 > avoid_center**2:
                return x, y
        return margin, margin

    def _clear_circle(self, cx: int, cy: int, r: int, water: bool = False) -> None:
        ys, xs = np.ogrid[: self.n, : self.n]
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
        removable = (self.kind == TREE) | ((self.kind == WATER) if water else False)
        self.kind[inside & removable] = GROUND
        self.soil[inside & (self.soil == SAND)] = PLAIN

    def _patch(self, soil: int, r: int) -> None:
        cx, cy = self._free_spot(r + 4, self.cfg.start_clear + r + 4)
        ys, xs = np.ogrid[: self.n, : self.n]
        wobble = r + self.rng.uniform(-1, 1)
        inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= wobble * wobble
        self.soil[inside] = soil
        self.kind[inside & (self.kind == TREE)] = GROUND

    def _river(self, horizontal: bool) -> None:
        """Река поперёк карты блуждающей полосой шириной 3–4 клетки, мосты через каждые ~35 клеток."""
        n, rng = self.n, self.rng
        width = int(rng.integers(3, 5))
        across = float(rng.uniform(n * 0.2, n * 0.8))
        drift = 0.0
        bridge_at = int(rng.integers(15, 30))
        for along in range(n):
            drift = np.clip(drift + rng.normal(0, 0.25), -0.8, 0.8)
            across = float(np.clip(across + drift, 10, n - 10))
            lo = int(across) - width // 2
            cells = range(lo, lo + width)
            bridge = along == bridge_at
            if bridge:
                bridge_at += int(rng.integers(28, 42))
            for a in cells:
                y, x = (a, along) if horizontal else (along, a)
                self.kind[y, x] = BRIDGE if bridge else WATER
                self.soil[y, x] = PLAIN
            if bridge:  # подходы к мосту свободны от леса
                for a in range(lo - 3, lo + width + 3):
                    for d in (-1, 0, 1):
                        y, x = (a, along + d) if horizontal else (along + d, a)
                        if 0 <= x < n and 0 <= y < n and self.kind[y, x] == TREE:
                            self.kind[y, x] = GROUND

    def _building(self) -> None:
        """Модуль колонистов: стены, пол, 1–2 двери, окна; иногда забор вокруг с воротами."""
        rng = self.rng
        w, h = int(rng.integers(7, 11)), int(rng.integers(6, 9))
        for _ in range(60):
            x0, y0 = self._free_spot(8, self.cfg.start_clear + 10)
            x1, y1 = x0 + w - 1, y0 + h - 1
            if x1 >= self.n - 8 or y1 >= self.n - 8:
                continue
            area = (slice(y0 - 3, y1 + 4), slice(x0 - 3, x1 + 4))
            if (self.building[area] >= 0).any() or np.isin(self.kind[area], (WATER, BRIDGE)).any():
                continue
            break
        else:
            return
        bid = len(self.buildings)
        self.buildings.append(Building(x0, y0, x1, y1))
        yard = (slice(y0 - 3, y1 + 4), slice(x0 - 3, x1 + 4))
        self.kind[yard] = np.where(self.kind[yard] == TREE, GROUND, self.kind[yard])
        self.soil[yard] = PLAIN
        self.kind[y0 : y1 + 1, x0 : x1 + 1] = WALL
        self.kind[y0 + 1 : y1, x0 + 1 : x1] = FLOOR
        self.building[y0 : y1 + 1, x0 : x1 + 1] = bid
        # Стена по периметру без углов — места для дверей и окон
        sides = [[(x, y0) for x in range(x0 + 1, x1)], [(x, y1) for x in range(x0 + 1, x1)],
                 [(x0, y) for y in range(y0 + 1, y1)], [(x1, y) for y in range(y0 + 1, y1)]]
        order = rng.permutation(4)
        for k, s in enumerate(order):
            side = sides[s]
            mid = len(side) // 2
            if k < int(rng.integers(1, 3)):          # двери — посередине стороны
                x, y = side[mid]
                self.kind[y, x] = DOOR
                self.hp[y, x] = self.cfg.door_hp
                self.doors.append((x, y))
            for x, y in (side[1], side[-2]):        # окна — ближе к углам
                if self.kind[y, x] == WALL and rng.random() < 0.7:
                    self.kind[y, x] = WINDOW
        if rng.random() < 0.5:  # забор вокруг двора с воротами
            fx0, fy0, fx1, fy1 = x0 - 2, y0 - 2, x1 + 2, y1 + 2
            ring = ([(x, fy0) for x in range(fx0, fx1 + 1)] + [(x, fy1) for x in range(fx0, fx1 + 1)]
                    + [(fx0, y) for y in range(fy0 + 1, fy1)] + [(fx1, y) for y in range(fy0 + 1, fy1)])
            gate = int(rng.integers(len(ring)))
            for i, (x, y) in enumerate(ring):
                if min(abs(i - gate), len(ring) - abs(i - gate)) > 1 and self.kind[y, x] == GROUND:
                    self.kind[y, x] = FENCE
                    self.hp[y, x] = 1
        # Ящик с оружием внутри
        ix, iy = int(rng.integers(x0 + 1, x1)), int(rng.integers(y0 + 1, y1))
        kind = UNDERBARREL if bid == 0 else WEAPON_CRATE
        self.items.append((self.cell_center(ix, iy), kind))

    def _scatter_items(self) -> None:
        free = np.argwhere((self.kind == GROUND) & (self.soil == PLAIN) & (self.building < 0))
        c = self.items_cfg
        for kind, count in ((MEDKIT, c.medkits), (MAGNET, c.magnets), (DNA, c.dna)):
            for y, x in free[self.rng.choice(len(free), count, replace=False)]:
                self.items.append((self.cell_center(int(x), int(y)), kind))

    # ---------- адреса ----------
    def cell_center(self, ix: int, iy: int) -> np.ndarray:
        return np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell], np.float32)

    # Запросы к сетке идут десятки раз за кадр на массивах в сотни точек: здесь важна каждая операция NumPy,
    # поэтому обе координаты считаются одним массивом, а np.clip заменён на minimum/maximum (обёртка clip
    # на малых массивах дороже самой работы).
    def _flat(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Плоский номер клетки (y·n + x, с прижатием к краю) и маска точек за пределами карты."""
        p = np.atleast_2d(pos)
        q = np.minimum(np.maximum(p // self.cell, 0), self.n - 1).astype(np.intp)
        outside = ((p < 0) | (p >= self.size_px)).any(1)
        return q[:, 1] * self.n + q[:, 0], outside

    def cells(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat, _ = self._flat(pos)
        return flat % self.n, flat // self.n

    def kinds_at(self, pos: np.ndarray) -> np.ndarray:
        flat, outside = self._flat(pos)
        k = self.kind.ravel()[flat]
        k[outside] = ROCK
        return k

    def soil_at(self, pos: np.ndarray) -> np.ndarray:
        return self.soil.ravel()[self._flat(pos)[0]]

    # ---------- запросы для толпы и снарядов ----------
    def bug_blocked(self, pos: np.ndarray) -> np.ndarray:
        return BUG_BLOCK[self.kinds_at(pos)]

    def shot_blocked(self, pos: np.ndarray) -> np.ndarray:
        return SHOT_BLOCK[self.kinds_at(pos)]

    def bug_speed(self, pos: np.ndarray) -> np.ndarray:
        """Множитель скорости наземных жуков: окно, топь, зыбучий песок."""
        c = self.cfg
        flat, _ = self._flat(pos)
        k = self.kind.ravel()[flat]
        s = self.soil.ravel()[flat]
        f = np.ones(len(k), np.float32)
        f[k == WINDOW] *= c.window_bug_speed
        f[s == MUD] *= c.mud_speed
        f[s == SAND] *= c.sand_bug_speed
        return f

    def spawnable(self, pos: np.ndarray) -> np.ndarray:
        return ~self.bug_blocked(pos) & (self.building[self.cells(pos)[::-1]] < 0)

    # ---------- изменения карты ----------
    def _touch(self) -> None:
        self.version += 1

    def damage_tree(self, ix: int, iy: int, amount: float = 1.0) -> bool:
        """Удар топором; True — дерево упало."""
        if self.kind[iy, ix] != TREE:
            return False
        self.hp[iy, ix] -= amount
        if self.hp[iy, ix] <= 0:
            self.kind[iy, ix] = STUMP
            self._touch()
            return True
        return False

    def clear_circle_px(self, center: np.ndarray, radius: float, kinds: tuple[int, ...] = (TREE,)) -> int:
        """Взрыв или огонь выжигает лес (и заборы, если переданы) в круге. Возвращает число клеток."""
        c = self.cell
        r = int(radius // c) + 1
        cx, cy = int(center[0] // c), int(center[1] // c)
        x0, x1 = max(0, cx - r), min(self.n, cx + r + 1)
        y0, y1 = max(0, cy - r), min(self.n, cy + r + 1)
        sub = self.kind[y0:y1, x0:x1]
        target = np.zeros(KIND_COUNT, bool)
        target[list(kinds)] = True
        burnable = target[sub]
        if not burnable.any():  # босс под собой проверяет это каждый кадр — обычно леса там уже нет
            return 0
        ys, xs = np.ogrid[y0:y1, x0:x1]
        inside = ((xs + 0.5) * c - center[0]) ** 2 + ((ys + 0.5) * c - center[1]) ** 2 <= radius * radius
        hit = inside & burnable
        if hit.any():
            sub[hit] = np.where(sub[hit] == TREE, STUMP, GROUND)
            self._touch()
        return int(hit.sum())

    def smash(self, pos: np.ndarray) -> int:
        """Танкер проламывает лес и заборы, в которые упёрся."""
        p = np.atleast_2d(pos)
        k = self.kinds_at(p)
        hit = SMASHABLE[k]
        if not hit.any():
            return 0
        ix, iy = self.cells(p[hit])
        self.kind[iy, ix] = np.where(self.kind[iy, ix] == TREE, STUMP, GROUND)
        self._touch()
        return int(hit.sum())

    def press_doors(self, pos: np.ndarray, dt: float) -> int:
        """Жуки, упёршиеся в закрытую дверь, выламывают её. Возвращает число выломанных."""
        p = np.atleast_2d(pos)
        doors = self.kinds_at(p) == DOOR
        if not doors.any():
            return 0
        ix, iy = self.cells(p[doors])
        np.subtract.at(self.hp, (iy, ix), dt)
        broken = (self.kind == DOOR) & (self.hp <= 0)
        count = int(broken.sum())
        if count:
            self.kind[broken] = FLOOR
            self._touch()
        return count

    def carve_evac(self, center: np.ndarray) -> None:
        self.clear_circle_px(center, self.cfg.evac_clear * self.cell, (TREE, FENCE))

    def place(self, pos: np.ndarray, max_ring: int = 4, reachable: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Сдвигает точки на ближайшие свободные клетки вне построек (точка на свободной клетке остаётся,
        где была). reachable — только клетки, откуда жуку есть путь до десантника по полю путей.
        Возвращает (точки, нашлось ли место)."""
        p = np.atleast_2d(np.asarray(pos, np.float32))
        near = max_ring >= SEARCH_RING
        cx = (p[:, 0] // self.cell).astype(np.int64)[:, None] + SEARCH_DX[near][None, :]
        cy = (p[:, 1] // self.cell).astype(np.int64)[:, None] + SEARCH_DY[near][None, :]
        inside = (cx >= 0) & (cy >= 0) & (cx < self.n) & (cy < self.n)
        x, y = np.clip(cx, 0, self.n - 1), np.clip(cy, 0, self.n - 1)
        ok = inside & OPEN[self.kind[y, x]] & (self.building[y, x] < 0)
        if reachable:
            ok &= self.reachable(x, y)
        found = ok.any(1)
        first = ok.argmax(1)
        moved = np.flatnonzero(found & (first > 0))
        out = p.copy()
        if len(moved):  # в центр найденной клетки с разбросом — чтобы не вставали стопкой
            jitter = self.rng.uniform(-0.3, 0.3, (len(moved), 2))
            out[moved, 0] = (x[moved, first[moved]] + 0.5 + jitter[:, 0]) * self.cell
            out[moved, 1] = (y[moved, first[moved]] + 0.5 + jitter[:, 1]) * self.cell
        return out, found

    def nearest_open(self, pos: np.ndarray, max_ring: int = 12) -> np.ndarray:
        """Ближайшая свободная точка вне построек; не нашлось — та же точка."""
        return self.place(pos, max_ring)[0][0]

    def reachable(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        """Клетки, откуда по полю путей можно дойти до десантника."""
        lx, ly = ix - self.flow_origin[0], iy - self.flow_origin[1]
        h, w = self.flow_dist.shape
        inside = (lx >= 0) & (ly >= 0) & (lx < w) & (ly < h)
        return inside & np.isfinite(self.flow_dist[np.clip(ly, 0, h - 1), np.clip(lx, 0, w - 1)])

    def update_doors(self, player: np.ndarray, radius: float) -> int:
        """Двери распахиваются перед десантником и закрываются за ним; открытая не держит ни жуков, ни пуль.
        Возвращает число только что открытых — для звука."""
        reach = radius + self.cell * DOOR_REACH
        opened = 0
        for x, y in self.doors:
            k = self.kind[y, x]
            if k not in (DOOR, OPEN_DOOR):  # выломана
                continue
            near = self.distance_to_cell(player, x, y) <= reach
            if near and k == DOOR:
                self.kind[y, x] = OPEN_DOOR
                opened += 1
            elif not near and k == OPEN_DOOR:
                self.kind[y, x] = DOOR
        return opened

    def distance_to_cell(self, p: np.ndarray, x: int, y: int) -> float:
        """От точки до ближайшей точки клетки (0 — внутри)."""
        c = self.cell
        px, py = float(p[0]), float(p[1])
        qx = min(max(px, x * c), (x + 1) * c)
        qy = min(max(py, y * c), (y + 1) * c)
        return math.hypot(px - qx, py - qy)

    # ---------- десантник ----------
    def resolve_player(self, pos: np.ndarray, radius: float) -> np.ndarray:
        """Выталкивает круг десантника из непроходимых клеток вокруг. Считается на числах Python:
        клеток девять, NumPy здесь только замедлял бы."""
        c = self.cell
        px = min(max(float(pos[0]), radius), self.size_px - radius)
        py = min(max(float(pos[1]), radius), self.size_px - radius)
        for _ in range(2):
            cx, cy = int(px // c), int(py // c)
            for y in range(cy - 1, cy + 2):
                for x in range(cx - 1, cx + 2):
                    if not (0 <= x < self.n and 0 <= y < self.n) or not PLAYER_BLOCK[self.kind[y, x]]:
                        continue
                    dx = px - min(max(px, x * c), (x + 1) * c)
                    dy = py - min(max(py, y * c), (y + 1) * c)
                    d = math.hypot(dx, dy)
                    if d >= radius:
                        continue
                    if d < 1e-6:  # центр внутри клетки — наружу по ближайшей стороне
                        out = [(px - x * c, -1, 0), ((x + 1) * c - px, 1, 0),
                               (py - y * c, 0, -1), ((y + 1) * c - py, 0, 1)]
                        depth, ox, oy = min(out)
                        px += ox * (depth + radius)
                        py += oy * (depth + radius)
                    else:
                        px += dx / d * (radius - d)
                        py += dy / d * (radius - d)
        return np.array([px, py], np.float32)

    def tree_ahead(self, pos: np.ndarray, direction: np.ndarray, radius: float) -> tuple[int, int] | None:
        """Клетка-дерево, в которую упирается десантник, — её рубит топор."""
        probe = pos + direction * (radius + self.cell * 0.35)
        if self.kinds_at(probe)[0] != TREE:
            return None
        ix, iy = self.cells(probe)
        return int(ix[0]), int(iy[0])

    def touching(self, pos: np.ndarray, radius: float, kind: int) -> bool:
        c = self.cell
        cx, cy = int(pos[0] // c), int(pos[1] // c)
        for y in range(max(0, cy - 1), min(self.n, cy + 2)):
            for x in range(max(0, cx - 1), min(self.n, cx + 2)):
                if self.kind[y, x] == kind and self.distance_to_cell(pos, x, y) <= radius + 3:
                    return True
        return False

    # ---------- поле путей ----------
    def update_flow(self, player: np.ndarray, dt: float, force: bool = False) -> None:
        """Поле путей жуков: расстояние по проходимым клеткам от десантника в окне вокруг него и направление
        к соседу ближе к цели. Пересчёт — по таймеру, при смене клетки десантника и при изменении карты.
        Обход в ширину стоит ~2 мс, поэтому он идёт по FLOW_STEPS шагов за кадр, а жуки пока ходят по
        прежнему полю; force — посчитать сразу целиком."""
        self._flow_timer -= dt
        pc = (int(player[0] // self.cell), int(player[1] // self.cell))
        stale = self._flow_timer <= 0 or self._flow_version != self.version or pc != self._flow_cell
        if force or (stale and self._flow_job is None):
            self._flow_timer = self.cfg.flow_every
            self._flow_version = self.version
            self._flow_cell = pc
            self._flow_job = self._flow_steps(pc)
        job = self._flow_job
        if job is None:
            return
        if force:
            for _ in job:
                pass
        else:
            next(job, None)

    def _flow_steps(self, pc: tuple[int, int]) -> Iterator[bool]:
        """Обход в ширину порциями: каждый yield — FLOW_STEPS шагов; в конце поле публикуется целиком."""
        rx, ry = self.cfg.flow_radius
        x0 = int(np.clip(pc[0] - rx, 0, self.n - 2 * rx - 1))
        y0 = int(np.clip(pc[1] - ry, 0, self.n - 2 * ry - 1))
        area = (slice(y0, y0 + 2 * ry + 1), slice(x0, x0 + 2 * rx + 1))
        kinds = self.kind[area]
        free = ~FLOW_BLOCK[kinds]
        # Двери и окна — путь внутрь, только если десантник в этой постройке; иначе жуки обходят её,
        # а не срезают дорогу насквозь, выламывая двери
        inside = self.building[pc[1], pc[0]] if 0 <= pc[0] < self.n and 0 <= pc[1] < self.n else -1
        free &= ~(ENTRANCE[kinds] & (self.building[area] != inside))
        h, w = free.shape
        dist = np.full((h, w), np.inf, np.float32)
        lx, ly = pc[0] - x0, pc[1] - y0
        if 0 <= lx < w and 0 <= ly < h:
            frontier = np.zeros((h, w), bool)
            frontier[ly, lx] = True
            visited = frontier.copy()
            dist[ly, lx] = 0
            step = 0
            while frontier.any():
                step += 1
                frontier = dilate8(frontier) & free & ~visited
                visited |= frontier
                dist[frontier] = step
                if step % FLOW_STEPS == 0:
                    yield True
        # Направление: к соседу с наименьшим расстоянием
        padded = np.pad(dist, 1, constant_values=np.inf)
        stack = np.stack([padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dx, dy in NEIGHBOURS])
        best = np.argmin(stack, axis=0)
        better = np.take_along_axis(stack, best[None], 0)[0] < dist
        vec = NEIGHBOUR_DIRS[best]
        vec[~better] = 0
        self.flow_origin = (x0, y0)
        self.flow_dist = dist
        self.flow_vec = vec
        self._flow_job = None

    def flow(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Направление по полю путей и маска, где оно есть (в окне, достижимо, не у самой цели)."""
        ix, iy = self.cells(pos)
        lx, ly = ix - self.flow_origin[0], iy - self.flow_origin[1]
        h, w = self.flow_dist.shape
        inside = (lx >= 0) & (ly >= 0) & (lx < w) & (ly < h)
        lx, ly = np.minimum(np.maximum(lx, 0), w - 1), np.minimum(np.maximum(ly, 0), h - 1)
        d = self.flow_dist[ly, lx]
        vec = self.flow_vec[ly, lx]
        ok = inside & np.isfinite(d) & (d > 1) & (np.abs(vec).sum(1) > 0)
        return vec, ok
