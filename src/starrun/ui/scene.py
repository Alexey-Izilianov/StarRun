"""Мир на простых фигурах (этап 1): земля, пятна, жуки, десантник, пули, брызги, метки и зона эвакуации."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrun.sim.swarm import QUEEN, SEMI_BOSS, WARRIOR
from starrun.sim.world import World
from starrun.ui.render import Shapes, rgba

GROUND = (86, 70, 50)
PEBBLE_DARK = rgba((66, 53, 38), 0.9)
PEBBLE_LIGHT = rgba((112, 94, 68), 0.8)
OUTLINE = rgba((28, 20, 14))
HEAD = rgba((40, 26, 18))
BLOOD = rgba((110, 235, 40))
BLOOD_DECAL = (40, 120, 20)
GEM = rgba((90, 220, 240))
GEM_CORE = rgba((220, 250, 255))
BULLET = rgba((255, 214, 120))
ACID = rgba((170, 230, 40))
MARINE = rgba((112, 124, 84))
MARINE_PAD = rgba((150, 152, 132))
HURT = rgba((230, 70, 50))
STRIKE = (235, 50, 35)
EVAC = (90, 200, 240)
WINDUP = (255, 120, 40)

CELL = 256.0     # клетка «камушков» на земле — чтобы движение было заметно
PEBBLES = 7


@dataclass
class Blast:
    pos: np.ndarray
    radius: float
    left: float = 0.45


def pebbles(left: float, bottom: float, right: float, top: float) -> tuple[np.ndarray, np.ndarray]:
    """Детерминированные камушки в видимых клетках: одни и те же на одном месте при любом проходе."""
    xs = np.arange(np.floor(left / CELL), np.ceil(right / CELL) + 1)
    ys = np.arange(np.floor(bottom / CELL), np.ceil(top / CELL) + 1)
    cx, cy = (a.ravel() for a in np.meshgrid(xs, ys))
    k = np.arange(PEBBLES)
    seed = (cx[:, None] * 73856093 + cy[:, None] * 19349663 + k[None, :] * 83492791) % 1_000_003
    fx = (np.sin(seed * 12.9898) * 43758.5453) % 1.0
    fy = (np.sin(seed * 78.233) * 12543.1234) % 1.0
    fs = (np.sin(seed * 39.425) * 24634.6345) % 1.0
    pos = np.stack([(cx[:, None] + fx) * CELL, (cy[:, None] + fy) * CELL], -1).reshape(-1, 2).astype(np.float32)
    return pos, (4 + fs.ravel() * 9).astype(np.float32)


class SceneRenderer:
    def __init__(self, shapes: Shapes) -> None:
        self.c = shapes
        self.blasts: list[Blast] = []
        self.time = 0.0

    def update(self, dt: float, world: World) -> None:
        self.time += dt
        for mark in world.blasts:
            self.blasts.append(Blast(mark.pos.copy(), mark.radius))
        for b in self.blasts:
            b.left -= dt
        self.blasts = [b for b in self.blasts if b.left > 0]

    def draw(self, world: World, view: tuple[float, float, float, float]) -> None:
        c, p = self.c, world.player
        left, bottom, right, top = view

        pos, size = pebbles(left - 20, bottom - 20, right + 20, top + 20)
        light = size > 9
        c.add(pos[~light], size[~light], PEBBLE_DARK)
        c.add(pos[light], size[light], PEBBLE_LIGHT)

        # Пятна крови: клякса из трёх смещённых кругов, к концу жизни выцветают
        d = world.decals
        idx = np.flatnonzero(d.alive)
        if len(idx):
            color = np.tile(rgba(BLOOD_DECAL), (len(idx), 1))
            color[:, 3] = np.clip(d.life[idx] / 6, 0, 1) * 0.5
            for k, scale in enumerate((0.8, 0.62, 0.5)):
                angle = idx * 2.4 + k * 2.1
                offset = np.stack([np.cos(angle), np.sin(angle)], 1) * (d.size[idx, None] * 0.45 * (k > 0))
                c.add(d.pos[idx] + offset, d.size[idx] * 2 * scale, color)

        if world.evac_pos is not None:
            r = world.balance.evac.radius
            c.add_one(*world.evac_pos, r * 2, rgba(EVAC, 0.14))
            c.add_one(*world.evac_pos, r * 2, rgba(EVAC, 0.9), ring=0.03)
            done = world.session.evac_progress / world.balance.evac.seconds
            c.add_one(*world.evac_pos, r * 2 * max(0.05, done), rgba(EVAC, 0.25))

        for m in world.orbital.marks:
            pulse = 0.5 + 0.5 * np.sin(self.time * 18)
            c.add_one(*m.pos, m.radius * 2, rgba(STRIKE, 0.10 + 0.08 * pulse))
            c.add_one(*m.pos, m.radius * 2, rgba(STRIKE, 0.95), ring=0.025)
            c.add_one(*m.pos, 26, rgba(STRIKE, 0.95), ring=0.3)

        g = world.gems
        if g.alive.any():
            c.add(g.pos[g.alive], 12, GEM)
            c.add(g.pos[g.alive], 5, GEM_CORE)

        a = world.acid
        if a.alive.any():
            c.add(a.pos[a.alive], 18, ACID)

        self._bugs(world)

        # Десантник: тёмный контур, корпус, наплечники, ствол в сторону прицела
        r = world.stats["radius"]
        body = HURT if p.hurt > 0 else MARINE
        c.add_one(*p.pos, r * 2 + 6, OUTLINE)
        c.add_one(*p.pos, r * 2, body)
        side = np.array([-p.facing[1], p.facing[0]])
        for s in (-1, 1):
            c.add_one(*(p.pos + side * s * r * 0.62), r * 0.8, MARINE_PAD)
        c.add_one(*(p.pos + p.facing * r * 0.95), 12, OUTLINE)
        c.add_one(*(p.pos + p.facing * r * 1.45), 8, OUTLINE)

        bl = world.bullets
        if bl.alive.any():
            idx = np.flatnonzero(bl.alive)
            v = bl.vel[idx] / (np.linalg.norm(bl.vel[idx], axis=1, keepdims=True) + 1e-6)
            c.add(bl.pos[idx] - v * 10, 5, rgba((255, 190, 90), 0.6))
            c.add(bl.pos[idx], 7, BULLET)

        sp = world.splashes
        if sp.alive.any():
            k = sp.alive
            color = np.tile(BLOOD, (int(k.sum()), 1))
            color[:, 3] = np.clip(sp.life[k] / sp.max_life[k], 0, 1)
            c.add(sp.pos[k], sp.size[k], color)

        for b in self.blasts:
            t = b.left / 0.45
            c.add_one(*b.pos, b.radius * 2 * (1.1 - 0.3 * t), rgba((255, 200, 120), 0.55 * t))
            c.add_one(*b.pos, b.radius * 2, rgba(STRIKE, 0.8 * t), ring=0.06)
        self._boss_bars(world)
        c.flush()

    def _bugs(self, world: World) -> None:
        c, sw, p = self.c, world.swarm, world.player
        idx = sw.alive_idx
        if not len(idx):
            return
        pos, radius, kind = sw.pos[idx], sw.radius[idx], sw.kind[idx]
        base = np.array([rgba(cfg.color) for cfg in sw.cfg])[kind]
        flash = np.clip(sw.flash[idx] / 0.1, 0, 1)[:, None]
        base[:, :3] = base[:, :3] * (1 - flash) + flash  # попадание — вспышка белым
        to = p.pos - pos
        facing = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-6)

        big = kind >= SEMI_BOSS
        c.add(pos[big], radius[big] * 2 + 10, rgba((0, 0, 0), 0.35))  # тень боссов
        c.add(pos, radius * 2 + 4, OUTLINE)
        c.add(pos, radius * 2, base)
        c.add(pos + facing * radius[:, None] * 0.75, radius * 0.9, HEAD)

        # Замах воина: кольцо досягаемости наливается, пока не грянет удар
        w = (kind == WARRIOR) & (sw.windup[idx] >= 0)
        if w.any():
            cfg = sw.cfg[WARRIOR]
            t = 1 - np.clip(sw.windup[idx][w] / cfg.windup, 0, 1)
            color = np.tile(rgba(WINDUP), (int(w.sum()), 1))
            color[:, 3] = 0.25 + 0.6 * t
            c.add(pos[w] + facing[w] * cfg.reach * 0.5, cfg.reach * 2, color, ring=0.08 + 0.1 * t)

        q = kind == QUEEN
        if q.any():
            c.add(pos[q], radius[q] * 1.3, rgba((200, 230, 60), 0.5 + 0.2 * np.sin(self.time * 4)))

    def _boss_bars(self, world: World) -> None:
        sw = world.swarm
        idx = sw.alive_idx
        idx = idx[sw.kind[idx] >= SEMI_BOSS]
        for i in idx:
            x, y = sw.pos[i]
            w = sw.radius[i] * 2
            top = y + sw.radius[i] + 18
            share = max(0.0, float(sw.hp[i] / sw.max_hp[i]))
            self.c.rect(x - w / 2, top, x + w / 2, top + 7, (20, 16, 12, 200))
            self.c.rect(x - w / 2, top, x - w / 2 + w * share, top + 7, (215, 70, 50, 255))
