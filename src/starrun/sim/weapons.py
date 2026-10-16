"""Оружие десантника: общий темп, магазин и перезарядка — в Weapon, своё поведение — в наследниках.

Стреляет только выбранный ствол; перезарядка остальных идёт в фоне. Уровень ствола растёт его улучшениями
(до 5); на 5-м уровне открывается модификация (mod = True) — у каждого ствола своя.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from starrun.balance import WeaponCfg
from starrun.sim.combat import GRENADE, PELLET, ROCKET, SNIPER, TURRET_SHOT

if TYPE_CHECKING:
    from starrun.sim.world import World

MAX_LEVEL = 5
NAPALM_EVERY = 0.5      # огнемёт с напалмом оставляет лужу раз в полсекунды
RAIL_PIERCE = 99        # «рельса» пробивает всех на линии
RAIL_SPEED = 1.5


def unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class Weapon:
    def __init__(self, wid: str, cfg: WeaponCfg) -> None:
        self.id = wid
        self.cfg = cfg
        self.level = 1
        self.mod = False
        self.ammo = cfg.magazine
        self.cooldown = 0.0
        self.reload_left = 0.0

    # ---------- характеристики с учётом улучшений ----------
    def stat(self, world: World, name: str) -> float:
        return world.stats[f"{self.id}_{name}"]

    def magazine(self, world: World) -> int:
        if not self.cfg.magazine:
            return 0
        return max(1, round(self.cfg.magazine * world.stats["ammo_mult"]))

    def reach(self, world: World) -> float:
        return self.cfg.range * world.stats["range_mult"]

    def damage(self, world: World) -> float:
        return self.cfg.damage * self.stat(world, "damage") * world.stats["damage_mult"]

    def rate(self, world: World) -> float:
        return self.cfg.fire_rate * self.stat(world, "rate")

    @property
    def reloading(self) -> bool:
        return self.reload_left > 0

    # ---------- цикл ----------
    def tick(self, dt: float, world: World) -> None:
        """Каждый кадр, выбран ствол или нет: перезарядка идёт в фоне."""
        if self.reloading:
            self.reload_left -= dt * world.stats["reload_mult"]
            if self.reload_left <= 0:
                self.reload_left = 0.0
                self.ammo = self.magazine(world)

    def update(self, dt: float, world: World) -> None:
        """Только у выбранного ствола: темп, магазин, выстрелы."""
        if self.reloading:
            return
        self.cooldown -= dt
        target = self.aim(world)
        if target is None:
            self.cooldown = max(self.cooldown, 0.0)  # без цели темп не копится
            return
        mag = self.magazine(world)
        interval = 1 / self.rate(world)
        shots = 0
        while self.cooldown <= 0 and (not mag or self.ammo > 0):
            shots += 1
            self.cooldown += interval
            if mag:
                self.ammo -= 1
        if mag and self.ammo <= 0:
            self.reload_left = self.cfg.reload
        if shots:
            self.fire(world, target, shots)
            world.player.facing = unit(target - world.player.pos)
            world.events.append(f"shot_{self.id}")

    def aim(self, world: World) -> np.ndarray | None:
        i = world.pick_target(self.reach(world))
        return None if i is None else world.swarm.pos[i].copy()

    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        raise NotImplementedError

    def muzzle(self, world: World, direction: np.ndarray) -> np.ndarray:
        return world.player.pos + direction * (world.stats["radius"] + 6)


class Mar(Weapon):
    def __init__(self, wid: str, cfg: WeaponCfg) -> None:
        super().__init__(wid, cfg)
        self.grenade_left = cfg.mod_every

    def update(self, dt: float, world: World) -> None:
        super().update(dt, world)
        if not self.mod:
            return
        self.grenade_left -= dt
        if self.grenade_left <= 0:
            target = self.aim(world)
            if target is not None:
                self.grenade_left = self.cfg.mod_every
                self._grenade(world, target)

    def _grenade(self, world: World, target: np.ndarray) -> None:
        c = self.cfg
        d = unit(target - world.player.pos)
        speed = 700.0
        dist = float(np.linalg.norm(target - world.player.pos))
        world.bullets.fire(self.muzzle(world, d), d[None], speed, c.mod_damage * world.stats["damage_mult"],
                           dist / speed, kind=GRENADE, blast=c.mod_radius, knock=60)

    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        stats = world.stats
        aim = target - world.player.pos
        angles = np.arctan2(aim[1], aim[0]) + np.radians(world.rng.normal(0, self.cfg.spread_deg / 2, shots))
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
        damage = np.full(shots, self.damage(world), np.float32)
        damage[world.rng.random(shots) < stats["mar_crit"]] *= 2
        world.bullets.fire(self.muzzle(world, dirs[0]), dirs, self.cfg.bullet_speed, damage,
                           self.reach(world) / self.cfg.bullet_speed, pierce=int(stats["mar_pierce"]))


class Shotgun(Weapon):
    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        c = self.cfg
        pellets = c.pellets + int(self.stat(world, "pellets"))
        cone = np.radians(c.cone_deg * self.stat(world, "spread"))
        base = np.arctan2(*(target - world.player.pos)[::-1])
        burn = (c.mod_time, c.mod_damage * world.stats["damage_mult"]) if self.mod else (0.0, 0.0)
        for _ in range(shots):
            angles = base + world.rng.uniform(-cone / 2, cone / 2, pellets)
            dirs = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
            speed = c.bullet_speed * world.rng.uniform(0.85, 1.1, pellets)[:, None]
            world.bullets.fire(self.muzzle(world, dirs[0]), dirs * speed / c.bullet_speed, c.bullet_speed,
                               self.damage(world), self.reach(world) / c.bullet_speed,
                               kind=PELLET, knock=c.knockback, burn=burn)


class Sniper(Weapon):
    def aim(self, world: World) -> np.ndarray | None:
        i = world.swarm.toughest(world.player.pos, self.reach(world))
        return None if i is None else world.swarm.pos[i].copy()

    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        c = self.cfg
        d = unit(target - world.player.pos)
        speed = c.bullet_speed * (RAIL_SPEED if self.mod else 1.0)
        pierce = RAIL_PIERCE if self.mod else c.pierce + int(self.stat(world, "pierce"))
        dirs = np.repeat(d[None], shots, 0)
        world.bullets.fire(self.muzzle(world, d), dirs, speed, self.damage(world), self.reach(world) / speed,
                           pierce=pierce, kind=SNIPER)


class Flame(Weapon):
    def __init__(self, wid: str, cfg: WeaponCfg) -> None:
        super().__init__(wid, cfg)
        self.napalm_left = 0.0

    def reach(self, world: World) -> float:
        return super().reach(world) * self.stat(world, "range")

    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        c = self.cfg
        p = world.player.pos
        d = unit(target - p)
        reach = self.reach(world)
        caught = world.swarm.in_cone(p, d, reach, c.cone_deg)
        if len(caught):
            world.deal(caught, np.full(len(caught), self.damage(world) * shots, np.float32))
            world.swarm.ignite(caught, c.burn_time, c.burn_dps * self.stat(world, "burn") * world.stats["damage_mult"])
        # Язык пламени — частицы веером по конусу
        n = 14 * shots
        angles = np.arctan2(d[1], d[0]) + world.rng.uniform(-1, 1, n) * np.radians(c.cone_deg) / 2
        speed = world.rng.uniform(0.7, 1.0, n) * reach / 0.35
        vel = np.stack([np.cos(angles), np.sin(angles)], 1) * speed[:, None]
        world.flames.emit(self.muzzle(world, d), vel.astype(np.float32), 0.35, world.rng, size=(16, 30))
        if self.mod:
            self.napalm_left -= shots / self.rate(world)
            if self.napalm_left <= 0:
                self.napalm_left = NAPALM_EVERY
                dps = c.mod_damage * world.stats["damage_mult"]
                world.fires.add(p + d * reach * 0.8, c.mod_radius, c.mod_time, dps)


class Rocket(Weapon):
    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        c = self.cfg
        p = world.player.pos
        d = unit(target - p)
        dist = min(float(np.linalg.norm(target - p)), self.reach(world))
        cluster = (c.mod_count, c.mod_damage * world.stats["damage_mult"], c.mod_radius) if self.mod else (0, 0.0, 0.0)
        dirs = np.repeat(d[None], shots, 0)
        world.bullets.fire(self.muzzle(world, d), dirs, c.bullet_speed, self.damage(world), dist / c.bullet_speed,
                           kind=ROCKET, blast=c.radius * self.stat(world, "radius"), knock=c.knockback,
                           cluster=cluster)


class Turret(Weapon):
    """Пока выбрана — ставит турели раз в deploy_every секунд; поставленные работают до конца срока сами."""

    def update(self, dt: float, world: World) -> None:
        self.cooldown -= dt
        c = self.cfg
        if self.cooldown > 0 or world.swarm.nearest(world.player.pos, self.reach(world)) is None:
            return
        self.cooldown = c.deploy_every
        count = c.mod_count if self.mod else 1
        for k in range(count):
            angle = world.rng.uniform(0, 2 * np.pi) if count > 1 else 0.0
            offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * 40 * (k > 0)
            world.turrets.deploy(world.player.pos + offset, c.life * self.stat(world, "life"),
                                 c.max_active * count)
        world.events.append("turret")

    def fire(self, world: World, target: np.ndarray, shots: int) -> None:  # стреляют сами турели
        pass


class Shock(Weapon):
    def fire(self, world: World, target: np.ndarray, shots: int) -> None:
        c = self.cfg
        sw = world.swarm
        links = (c.mod_count if self.mod else c.chain) + int(self.stat(world, "chain"))
        for _ in range(shots):
            first = sw.nearest(world.player.pos, self.reach(world))
            if first is None:
                return
            chain = [first]
            while len(chain) < links:
                near = sw.in_radius(sw.pos[chain[-1]], c.jump)
                near = near[~np.isin(near, chain)]
                if not len(near):
                    break
                d2 = ((sw.pos[near] - sw.pos[chain[-1]]) ** 2).sum(1)
                chain.append(int(near[np.argmin(d2)]))
            hit = np.array(chain)
            world.deal(hit, np.full(len(hit), self.damage(world), np.float32))
            if self.mod:
                sw.stun_for(hit, c.mod_time)
            world.arcs.append(np.vstack([world.player.pos[None], sw.pos[hit]]).astype(np.float32))


WEAPON_CLASSES: dict[str, type[Weapon]] = {
    "mar": Mar, "shotgun": Shotgun, "sniper": Sniper, "flame": Flame,
    "rocket": Rocket, "turret": Turret, "shock": Shock,
}


def make_weapon(wid: str, cfg: WeaponCfg) -> Weapon:
    return WEAPON_CLASSES[wid](wid, cfg)


class Turrets:
    """Поставленные мини-турели: стреляют сами по ближайшему жуку, пока не кончится срок."""

    def __init__(self, cfg: WeaponCfg, capacity: int = 16) -> None:
        self.cfg = cfg
        self.pos = np.zeros((capacity, 2), np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.cooldown = np.zeros(capacity, np.float32)
        self.aim = np.zeros((capacity, 2), np.float32)
        self.aim[:, 1] = 1
        self.alive = np.zeros(capacity, bool)

    def deploy(self, pos: np.ndarray, life: float, max_active: int) -> None:
        alive = np.flatnonzero(self.alive)
        if len(alive) >= max_active:  # лимит — убирается самая старая
            self.alive[alive[np.argmin(self.life[alive])]] = False
        free = np.flatnonzero(~self.alive)
        if not len(free):
            return
        i = free[0]
        self.pos[i] = pos
        self.life[i] = self.max_life[i] = life
        self.cooldown[i] = 0.3
        self.alive[i] = True

    def step(self, dt: float, world: World) -> None:
        c = self.cfg
        stats = world.stats
        rate = c.fire_rate * stats["turret_rate"]
        damage = c.damage * stats["turret_damage"] * stats["damage_mult"]
        reach = c.range * stats["range_mult"]
        for i in np.flatnonzero(self.alive):
            self.life[i] -= dt
            if self.life[i] <= 0:
                self.alive[i] = False
                continue
            self.cooldown[i] -= dt
            if self.cooldown[i] > 0:
                continue
            j = world.swarm.nearest(self.pos[i], reach)
            if j is None:
                self.cooldown[i] = 0
                continue
            d = unit(world.swarm.pos[j] - self.pos[i])
            self.aim[i] = d
            self.cooldown[i] += 1 / rate
            world.bullets.fire(self.pos[i] + d * 16, d[None], c.bullet_speed, damage, reach / c.bullet_speed,
                               kind=TURRET_SHOT)


class Fires:
    """Горящие лужи напалма: жгут всех жуков внутри, пока не погаснут."""

    def __init__(self, capacity: int = 32) -> None:
        self.pos = np.zeros((capacity, 2), np.float32)
        self.radius = np.zeros(capacity, np.float32)
        self.life = np.zeros(capacity, np.float32)
        self.max_life = np.ones(capacity, np.float32)
        self.dps = np.zeros(capacity, np.float32)
        self._next = 0

    def add(self, pos: np.ndarray, radius: float, life: float, dps: float) -> None:
        i = self._next
        self._next = (i + 1) % len(self.pos)  # кольцевой буфер: самая старая лужа уступает место
        self.pos[i], self.radius[i], self.life[i], self.max_life[i], self.dps[i] = pos, radius, life, life, dps

    @property
    def alive(self) -> np.ndarray:
        return self.life > 0

    def step(self, dt: float, world: World) -> None:
        for i in np.flatnonzero(self.alive):
            self.life[i] -= dt
            caught = world.swarm.in_radius(self.pos[i], self.radius[i])
            if len(caught):
                world.deal(caught, np.full(len(caught), self.dps[i] * dt, np.float32), armor=False)

