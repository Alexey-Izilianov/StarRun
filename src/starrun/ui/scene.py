"""Мир на простых фигурах: карта, пятна, жуки, десантник, снаряды, пламя, взрывы, метки и зона эвакуации."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starrun.sim.combat import BULLET, GRENADE, PELLET, ROCKET, SNIPER, TURRET_SHOT
from starrun.sim.swarm import BRAIN, FLYER, QUEEN, SEMI_BOSS, SPITTER, TANKER, WARRIOR
from starrun.sim.world import World
from starrun.ui.render import Shapes, rgba
from starrun.ui.terrain_view import TerrainView, shade

GROUND = (86, 70, 50)
PEBBLE_DARK = rgba((66, 53, 38), 0.9)
PEBBLE_LIGHT = rgba((112, 94, 68), 0.8)
OUTLINE = rgba((28, 20, 14))
HEAD = rgba((40, 26, 18))
SHADOW = rgba((0, 0, 0), 0.3)
BLOOD = rgba((110, 235, 40))
BLOOD_DECAL = (40, 120, 20)
GEM = rgba((90, 220, 240))
GEM_CORE = rgba((220, 250, 255))
ACID = rgba((170, 230, 40))
MARINE = rgba((112, 124, 84))
MARINE_PAD = rgba((150, 152, 132))
HURT = rgba((230, 70, 50))
STRIKE = (235, 50, 35)
EVAC = (90, 200, 240)
WINDUP = (255, 120, 40)
SAC = (150, 220, 40)
PLATE = rgba((150, 132, 108))
AURA = (230, 120, 190)
FIRE = np.array([1.0, 0.55, 0.15])
STUN = rgba((140, 220, 255), 0.9)
ARC = rgba((150, 220, 255), 0.95)
ARC_CORE = rgba((240, 250, 255))
TURRET_BASE = rgba((70, 76, 64))
AXE = rgba((120, 124, 130))
AXE_EDGE = rgba((225, 228, 232))
AXE_HANDLE = rgba((150, 108, 64))
SAND_RING = (150, 124, 82)

# Снаряды: (размер, цвет, длина хвоста в точках)
PROJECTILES = {
    BULLET: (7, rgba((255, 214, 120)), 1),      # MAR
    PELLET: (5, rgba((255, 170, 90)), 1),
    SNIPER: (6, rgba((245, 245, 255)), 4),
    ROCKET: (11, rgba((230, 70, 40)), 3),
    GRENADE: (10, rgba((90, 110, 50)), 1),
    TURRET_SHOT: (5, rgba((255, 230, 150)), 1),
}

CELL = 256.0     # клетка «камушков» на земле — чтобы движение было заметно
PEBBLES = 7
ARC_LIFE = 0.2


@dataclass
class Blast:
    pos: np.ndarray
    radius: float
    color: tuple[int, int, int]
    left: float = 0.45
    total: float = 0.45


@dataclass
class Arc:
    points: np.ndarray
    left: float = ARC_LIFE


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


def tinted(colors: np.ndarray, mask: np.ndarray, rgb: np.ndarray, amount: np.ndarray | float) -> None:
    """Смешивает цвет части жуков с rgb (на месте)."""
    a = np.broadcast_to(np.asarray(amount, np.float32), (int(mask.sum()),))[:, None]
    colors[mask, :3] = colors[mask, :3] * (1 - a) + rgb * a


class SceneRenderer:
    def __init__(self, shapes: Shapes) -> None:
        self.c = shapes
        self.map = TerrainView(shapes)
        self.blasts: list[Blast] = []
        self.arcs: list[Arc] = []
        self.time = 0.0

    def update(self, dt: float, world: World) -> None:
        self.time += dt
        self.map.update(dt, world)
        for mark in world.blasts:
            self.blasts.append(Blast(mark.pos.copy(), mark.radius, STRIKE))
        for e in world.booms:
            self.blasts.append(Blast(e.pos.copy(), e.radius, (255, 150, 60), 0.3, 0.3))
        self.arcs += [Arc(points) for points in world.arcs]
        for b in self.blasts:
            b.left -= dt
        for a in self.arcs:
            a.left -= dt
        self.blasts = [b for b in self.blasts if b.left > 0]
        self.arcs = [a for a in self.arcs if a.left > 0]

    def draw(self, world: World, view: tuple[float, float, float, float]) -> None:
        c = self.c
        left, bottom, right, top = view
        self.map.prepare(world, view)
        self.map.ground(world)

        pos, size = pebbles(left - 20, bottom - 20, right + 20, top + 20)
        keep = self.map.pebble_mask(world, pos)
        pos, size = pos[keep], size[keep]
        light = size > 9
        dark_color, light_color = PEBBLE_DARK, PEBBLE_LIGHT
        if world.terrain is not None:  # камушки в тон почвы биома
            dark_color = shade(world.terrain.biome.ground, 0.78, 0.9)
            light_color = shade(world.terrain.biome.ground_alt, 1.12, 0.8)
        c.add(pos[~light], size[~light], dark_color)
        c.add(pos[light], size[light], light_color)

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

        self._napalm(world)
        self.map.objects(world)

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
            c.add(a.pos[a.alive], 16, ACID)

        self._bugs(world, flying=False)
        self._turrets(world)
        self._marine(world)
        self.map.roofs(world)

        self._bugs(world, flying=True)
        self._projectiles(world)
        self._flames(world)

        sp = world.splashes
        if sp.alive.any():
            k = sp.alive
            color = np.tile(BLOOD, (int(k.sum()), 1))
            color[:, 3] = np.clip(sp.life[k] / sp.max_life[k], 0, 1)
            c.add(sp.pos[k], sp.size[k], color)

        self._arcs()
        for b in self.blasts:
            t = b.left / b.total
            c.add_one(*b.pos, b.radius * 2 * (1.1 - 0.3 * t), rgba((255, 200, 120), 0.55 * t))
            c.add_one(*b.pos, b.radius * 2, rgba(b.color, 0.8 * t), ring=0.06)
        self._boss_bars(world)
        c.flush()

    # ---------- слои ----------
    def _marine(self, world: World) -> None:
        """Десантник: тёмный контур, корпус, наплечники, ствол в сторону прицела. Перелезая окно или забор —
        чуть крупнее (в прыжке) с тенью; в зыбучем песке — уходит вниз, вокруг — кольцо песка;
        у дерева — топор описывает дугу по ходу движения."""
        c, p = self.c, world.player
        r = world.stats["radius"]
        scale = 1.0
        if p.vaulting:
            scale = 1.15
            c.add_one(*(p.pos + [6, -10]), r * 2, SHADOW)
        if p.sink > 0:
            scale *= 1 - 0.3 * p.sink
            c.add_one(*p.pos, r * 2.6, rgba(SAND_RING, 0.4 + 0.5 * p.sink), ring=0.3)
        rs = r * scale
        body = HURT if p.hurt > 0 else MARINE
        c.add_one(*p.pos, rs * 2 + 6, OUTLINE)
        c.add_one(*p.pos, rs * 2, body)
        side = np.array([-p.facing[1], p.facing[0]])
        for s in (-1, 1):
            c.add_one(*(p.pos + side * s * rs * 0.62), rs * 0.8, MARINE_PAD)
        c.add_one(*(p.pos + p.facing * rs * 0.95), 12, OUTLINE)
        c.add_one(*(p.pos + p.facing * rs * 1.45), 8, OUTLINE)
        if p.chop >= 0:
            # Замах от плеча через фронт: -80° → +30° от направления хода, удар — в конце дуги
            base = np.arctan2(p.heading[1], p.heading[0])
            a = base + np.radians(-80 + 110 * p.chop ** 2)
            d = np.array([np.cos(a), np.sin(a)], np.float32)
            for k in np.linspace(0.0, 1.0, 5):  # рукоять
                c.add_one(*(p.pos + d * rs * (0.6 + 0.9 * k)), 9, OUTLINE)
                c.add_one(*(p.pos + d * rs * (0.6 + 0.9 * k)), 6, AXE_HANDLE)
            head = p.pos + d * rs * 1.65
            c.add_one(*head, 22, OUTLINE)
            c.add_one(*head, 18, AXE)
            c.add_one(*(head + d * 6), 9, AXE_EDGE)

    def _napalm(self, world: World) -> None:
        f = world.fires
        idx = np.flatnonzero(f.alive)
        if not len(idx):
            return
        fade = np.clip(f.life[idx] / f.max_life[idx] * 3, 0, 1)
        flicker = 0.75 + 0.25 * np.sin(self.time * 20 + idx * 1.3)
        color = np.tile(rgba((235, 110, 30)), (len(idx), 1))
        color[:, 3] = 0.35 * fade * flicker
        self.c.add(f.pos[idx], f.radius[idx] * 2, color)
        color[:, 3] = 0.5 * fade
        self.c.add(f.pos[idx], f.radius[idx] * 1.1, color)

    def _bugs(self, world: World, flying: bool) -> None:
        c, sw, p = self.c, world.swarm, world.player
        idx = sw.alive_idx
        idx = idx[(sw.kind[idx] == FLYER) == flying]
        if not len(idx):
            return
        pos, radius, kind = sw.pos[idx], sw.radius[idx], sw.kind[idx]
        base = np.array([rgba(cfg.color) for cfg in sw.cfg])[kind]
        burning = sw.burn_left[idx] > 0
        if burning.any():
            tinted(base, burning, FIRE, 0.45 + 0.2 * np.sin(self.time * 25 + idx[burning]))
        flash = np.clip(sw.flash[idx] / 0.1, 0, 1)[:, None]
        base[:, :3] = base[:, :3] * (1 - flash) + flash  # попадание — вспышка белым
        to = p.pos - pos
        facing = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-6)
        side = np.stack([-facing[:, 1], facing[:, 0]], 1)

        if flying:
            # Тень на земле ниже тела и крылья, бьющие часто
            c.add(pos + [0, -16], radius * 1.6, SHADOW)
            flap = 0.75 + 0.35 * np.abs(np.sin(self.time * 30 + idx))
            wing = rgba((230, 225, 210), 0.45)
            for s in (-1, 1):
                c.add(pos + side * s * radius[:, None] * 0.9, radius * 1.3 * flap, wing)
        else:
            brains = kind == BRAIN
            if brains.any():
                cfg = sw.cfg[BRAIN]
                pulse = 0.05 + 0.02 * np.sin(self.time * 3)
                c.add(pos[brains], cfg.aura_radius * 2, rgba(AURA, pulse))
                c.add(pos[brains], cfg.aura_radius * 2, rgba(AURA, 0.3), ring=0.015)
            big = kind >= SEMI_BOSS
            c.add(pos[big], radius[big] * 2 + 10, SHADOW)
            spit = kind == SPITTER
            if spit.any():  # мешок с кислотой позади, раздувается перед плевком
                cfg = sw.cfg[SPITTER]
                swell = np.where(sw.windup[idx][spit] >= 0, 1 - sw.windup[idx][spit] / cfg.windup, 0)
                glow = np.tile(rgba(SAC, 0.9), (int(spit.sum()), 1))
                c.add(pos[spit] - facing[spit] * radius[spit, None] * 0.8, radius[spit] * (1.3 + 0.6 * swell), glow)

        c.add(pos, radius * 2 + 4, OUTLINE)
        c.add(pos, radius * 2, base)
        tanks = kind == TANKER
        if tanks.any():  # бронеплиты — светлое кольцо по краю
            c.add(pos[tanks], radius[tanks] * 2, PLATE, ring=0.28)
        c.add(pos + facing * radius[:, None] * 0.75, radius * 0.9, HEAD)
        if not flying:
            buffed = sw.buffed[idx]
            if buffed.any():
                c.add(pos[buffed], radius[buffed] * 2 + 8, rgba(AURA, 0.8), ring=0.12)

        stunned = sw.stun[idx] > 0
        if stunned.any():
            c.add(pos[stunned], radius[stunned] * 2 + 10, STUN, ring=0.12)

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

    def _turrets(self, world: World) -> None:
        t = world.turrets
        idx = np.flatnonzero(t.alive)
        if not len(idx):
            return
        c = self.c
        c.add(t.pos[idx], 38, OUTLINE)
        c.add(t.pos[idx], 32, TURRET_BASE)
        c.add(t.pos[idx], 32, rgba((200, 210, 150), 0.6), ring=0.12 * t.life[idx] / t.max_life[idx] + 0.02)
        c.add(t.pos[idx] + t.aim[idx] * 14, 12, OUTLINE)
        c.add(t.pos[idx], 14, MARINE_PAD)

    def _projectiles(self, world: World) -> None:
        bl = world.bullets
        idx = np.flatnonzero(bl.alive)
        if not len(idx):
            return
        v = bl.vel[idx] / (np.linalg.norm(bl.vel[idx], axis=1, keepdims=True) + 1e-6)
        kinds = bl.kind[idx]
        for kind, (size, color, tail) in PROJECTILES.items():
            k = kinds == kind
            if not k.any():
                continue
            for j in range(tail, 0, -1):
                fade = color.copy()
                fade[3] = 0.5 * (1 - j / (tail + 1))
                trail_size = size * (0.9 if kind == ROCKET else 0.75)
                trail = (rgba((150, 140, 130), 0.4) if kind == ROCKET else fade)
                self.c.add(bl.pos[idx[k]] - v[k] * (8 + 9 * j), trail_size, trail)
            self.c.add(bl.pos[idx[k]], size, color)

    def _flames(self, world: World) -> None:
        f = world.flames
        k = np.flatnonzero(f.alive)
        if not len(k):
            return
        t = np.clip(f.life[k] / f.max_life[k], 0, 1)  # 1 — только вылетела
        size = f.size[k] * (1.9 - t)                   # клуб огня растёт, остывая
        glow = np.empty((len(k), 4), np.float32)
        glow[:, 0] = 1.0
        glow[:, 1] = 0.2 + 0.45 * t
        glow[:, 2] = 0.05
        glow[:, 3] = 0.55 * t
        self.c.add(f.pos[k], size, glow)
        core = glow.copy()
        core[:, 1] = 0.55 + 0.4 * t
        core[:, 2] = 0.15 + 0.6 * t**2
        core[:, 3] = 0.9 * t**1.5
        self.c.add(f.pos[k], size * 0.5, core)

    def _arcs(self) -> None:
        for arc in self.arcs:
            pts = arc.points
            for a, b in zip(pts[:-1], pts[1:], strict=True):
                n = max(2, int(np.linalg.norm(b - a) / 9))
                t = np.linspace(0, 1, n)[:, None]
                line = a + (b - a) * t
                normal = np.array([-(b - a)[1], (b - a)[0]]) / (np.linalg.norm(b - a) + 1e-6)
                jitter = np.sin(t[:, 0] * 23 + self.time * 90) * 6 * np.sin(np.pi * t[:, 0])
                line = (line + normal * jitter[:, None]).astype(np.float32)
                fade = arc.left / ARC_LIFE
                glow = ARC.copy()
                glow[3] *= fade
                core = ARC_CORE.copy()
                core[3] *= fade
                self.c.add(line, 11, glow)
                self.c.add(line, 4, core)

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
