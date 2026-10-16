"""Мир забега: связывает десантника, толпу, оружие, опыт и ход сессии. Без графики — тестируется целиком."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from starrun.balance import Balance, Stamina
from starrun.sim.combat import Acid, Bullets, Gun, Mark, Orbital, fan
from starrun.sim.effects import Decals, Gems, Particles
from starrun.sim.progression import Offer, Progress, Stats, Upgrades
from starrun.sim.session import Phase, Session
from starrun.sim.spawner import Spawner, edge_positions
from starrun.sim.swarm import KINDS, QUEEN, SEMI_BOSS, WORKER, Swarm

VIEW_HALF = (960.0, 540.0)  # полэкрана 1080p — мир всегда видится в этом масштабе
MAX_DT = 1 / 30             # при подвисании кадра симуляция не перескакивает сквозь стены жуков
EVAC_SPAWN_BOOST = 1.6      # пока ждём челнок, рой давит сильнее
TARGET_SWITCH = 0.6         # MAR бросает цель ради жука, который вдвое ближе (с запасом)


@dataclass
class Controls:
    move: tuple[float, float] = (0.0, 0.0)
    sprint: bool = False
    ability_target: tuple[float, float] | None = None  # точка в мире, если в этом кадре нажат пробел


@dataclass
class Player:
    stamina_cfg: Stamina
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    hp: float = 0.0
    stamina: float = 0.0
    facing: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], np.float32))
    sprinting: bool = False
    hurt: float = 0.0  # подсветка после попадания
    _rest: float = 0.0

    def move(self, dt: float, direction: tuple[float, float], sprint: bool, stats: Stats) -> None:
        d = np.array(direction, np.float32)
        n = float(np.linalg.norm(d))
        moving = n > 1e-6
        self.sprinting = sprint and moving and self.stamina > 0
        if moving:
            d /= n
            self.facing = d
            speed = stats["speed"] * (stats["sprint_mult"] if self.sprinting else 1.0)
            self.pos = self.pos + d * speed * dt
        c = self.stamina_cfg
        if self.sprinting:
            self.stamina = max(0.0, self.stamina - c.drain * dt)
            self._rest = c.regen_delay
        else:
            self._rest -= dt
            if self._rest <= 0:
                self.stamina = min(c.max, self.stamina + c.regen * dt)
        self.hp = min(stats["max_hp"], self.hp + stats["regen"] * dt)
        self.hurt = max(0.0, self.hurt - dt)

    def take_hit(self, amount: float, rng: np.random.Generator, dodge: float) -> bool:
        """True — удар прошёл, False — уклонился."""
        if rng.random() < dodge:
            return False
        self.hp -= amount
        self.hurt = 0.2
        return True


class World:
    def __init__(self, balance: Balance, seed: int | None = None) -> None:
        self.balance = b = balance
        self.rng = np.random.default_rng(seed)
        self.stats = Stats(b.player.stats, b.player.caps)
        self.player = Player(b.player.stamina, hp=self.stats["max_hp"], stamina=b.player.stamina.max)
        self.swarm = Swarm(b.enemies, b.spawn.cap + 120, self.rng)
        self.bullets = Bullets()
        self.acid = Acid()
        self.gun = Gun(b.mar)
        self.orbital = Orbital(b.orbital)
        self.splashes = Particles()
        self.decals = Decals()
        self.gems = Gems()
        self.progress = Progress(b.xp)
        self.upgrades = Upgrades(b.upgrades, b.rarities)
        self.session = Session(b.alert, b.evac, b.scaling, b.semi_boss_minutes)
        self.spawner = Spawner(b.spawn, self.rng)
        self.kills: Counter[str] = Counter()
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.events: list[str] = []      # что случилось за последний шаг — для звука, баннеров, тряски
        self.blasts: list[Mark] = []     # взрывы орбитального удара за последний шаг
        self.evac_pos: np.ndarray | None = None
        self.target: int | None = None   # номер жука, в которого целится MAR
        self._offer: list[Offer] = []

    # ---------- выбор улучшений ----------
    @property
    def awaiting_choice(self) -> bool:
        return self.progress.pending > 0 and not self.session.over

    def offer(self) -> list[Offer]:
        if self.awaiting_choice and not self._offer:
            self._offer = self.upgrades.offer(self.rng)
        return self._offer

    def choose(self, i: int) -> None:
        offer = self.offer()[i]
        old_max = self.stats["max_hp"]
        self.upgrades.apply(offer, self.stats)
        self.player.hp += max(0.0, self.stats["max_hp"] - old_max)  # новая броня приходит целой
        self.progress.pending -= 1
        self._offer = []

    # ---------- шаг ----------
    def step(self, dt: float, controls: Controls) -> None:
        self.events = []
        self.blasts = []
        if self.session.over or self.awaiting_choice:
            return
        dt = min(dt, MAX_DT)
        b, s, p = self.balance, self.session, self.player

        p.move(dt, controls.move, controls.sprint, self.stats)
        if controls.ability_target is not None:
            target = np.array(controls.ability_target, np.float32)
            self.events.append("orbital" if self.orbital.trigger(target, p.pos, self.stats) else "orbital_not_ready")

        for event in s.update(dt):
            kind = SEMI_BOSS if event == "semi_boss" else QUEEN
            self.swarm.spawn(kind, edge_positions(self.rng, 1, p.pos, VIEW_HALF, 120), s.boss_hp_mult, s.damage_mult)
            self.events.append(event)

        boost = EVAC_SPAWN_BOOST if s.phase is Phase.EVAC else 1.0
        for name, pos in self.spawner.update(dt, s.elapsed, self.swarm.count(), p.pos, VIEW_HALF, boost):
            self.swarm.spawn(KINDS.index(name), pos, s.hp_mult, s.damage_mult)
        far = self.swarm.far_from(p.pos, b.spawn.despawn_distance)
        if len(far):
            self.swarm.pos[far] = edge_positions(self.rng, len(far), p.pos, VIEW_HALF, b.spawn.margin)

        step = self.swarm.step(dt, p.pos, self.stats["radius"])
        hits = step.hits + self.acid.step(dt, p.pos, self.stats["radius"])
        for amount in hits:
            if p.take_hit(amount, self.rng, self.stats["dodge"]):
                self.damage_taken += amount
                self.events.append("player_hit")
            else:
                self.events.append("dodge")
        for where, count in step.summons:
            angle = self.rng.uniform(0, 2 * np.pi, count)
            ring = where + np.stack([np.cos(angle), np.sin(angle)], 1) * self.rng.uniform(70, 140, (count, 1))
            self.swarm.spawn(WORKER, ring.astype(np.float32), s.hp_mult, s.damage_mult)
        queen = b.enemies["queen"]
        for where in step.volleys:
            aim = p.pos - where
            aim = aim / (np.linalg.norm(aim) + 1e-6)
            self.acid.fire(where, fan(aim, queen.volley_count, queen.volley_spread_deg),
                           queen.acid_speed, queen.acid_damage * s.damage_mult)
            self.events.append("volley")
        if step.swings:
            self.events.append("swing")

        self._shoot(dt)
        bug, damage = self.bullets.step(dt, self.swarm)
        self._damage(bug, damage)
        for mark in self.orbital.update(dt):
            caught = self.swarm.in_radius(mark.pos, mark.radius)
            self._damage(caught, np.full(len(caught), mark.damage, np.float32))
            self.splashes.burst(self.rng, mark.pos[None], 60, (100, 700), 0.8, (5, 12))
            self.blasts.append(mark)
            self.events.append("blast")
        self._collect_dead()

        gained = self.progress.add(self.gems.step(dt, p.pos, self.stats["pickup_radius"]))
        if gained:
            self.events.append("level_up")
        self.splashes.step(dt)
        self.decals.step(dt)

        if self.evac_pos is not None:
            inside = float(np.linalg.norm(p.pos - self.evac_pos)) < b.evac.radius
            if s.evac_tick(dt, inside):
                self.events.append("victory")
        if p.hp <= 0 and not s.over:
            s.player_died()
            self.events.append("defeat")

    def _shoot(self, dt: float) -> None:
        mar, stats, p = self.balance.mar, self.stats, self.player
        reach = mar.range * stats["range_mult"]
        self.target = self._pick_target(reach)
        shots = self.gun.update(dt, stats, self.target is not None)
        if not shots or self.target is None:
            return
        aim = self.swarm.pos[self.target] - p.pos
        base = np.arctan2(aim[1], aim[0])
        angles = base + np.radians(self.rng.normal(0, mar.spread_deg / 2, shots))
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1).astype(np.float32)
        damage = np.full(shots, mar.damage * stats["mar_damage"] * stats["damage_mult"], np.float32)
        damage[self.rng.random(shots) < stats["mar_crit"]] *= 2
        muzzle = p.pos + dirs[0] * (stats["radius"] + 6)
        self.bullets.fire(muzzle, dirs, mar.bullet_speed, damage, int(stats["mar_pierce"]), reach / mar.bullet_speed)
        p.facing = dirs[0]
        self.events.append("shot")

    def _pick_target(self, reach: float) -> int | None:
        """Ближайший жук, но с «залипанием»: пока цель жива и в радиусе, MAR добивает её, а не размазывает
        урон по всей подходящей толпе. Переключение — только на жука заметно ближе (вцепившегося в десантника)."""
        sw, pos = self.swarm, self.player.pos
        nearest = sw.nearest(pos, reach)
        current = self.target
        if current is None or nearest is None or not sw.alive[current]:
            return nearest
        d_current = float(np.linalg.norm(sw.pos[current] - pos))
        if d_current > reach:
            return nearest
        d_nearest = float(np.linalg.norm(sw.pos[nearest] - pos))
        return nearest if d_nearest < d_current * TARGET_SWITCH else current

    def _damage(self, idx: np.ndarray, amounts: np.ndarray) -> None:
        if len(idx):
            self.swarm.hit(idx, amounts)
            self.damage_dealt += float(np.sum(amounts))

    def _collect_dead(self) -> None:
        sw = self.swarm
        dead = sw.collect_dead()
        if not len(dead):
            return
        kinds = sw.kind[dead]
        pos = sw.pos[dead]
        for k in np.unique(kinds):
            self.kills[KINDS[k]] += int((kinds == k).sum())
        big = sw.radius[dead]
        self.splashes.burst(self.rng, pos, 10, (60, 260), 0.55, (4, 9))
        self.decals.add(self.rng, pos, big * self.rng.uniform(1.1, 1.7, len(dead)))
        self.gems.drop(self.rng, pos, np.array([sw.cfg[k].xp for k in kinds], np.int32))
        self.events.append("kill")
        if (kinds == SEMI_BOSS).any():
            self.session.semi_boss_killed()
            self.events.append("semi_boss_killed")
        if (kinds == QUEEN).any():
            self.session.queen_killed()
            angle = self.rng.uniform(0, 2 * np.pi)
            offset = np.array([np.cos(angle), np.sin(angle)], np.float32) * self.balance.evac.distance
            self.evac_pos = self.player.pos + offset
            self.events.append("evac")
