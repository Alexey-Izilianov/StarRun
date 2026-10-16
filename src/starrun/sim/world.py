"""Мир забега: связывает десантника, толпу, оружие, опыт и ход сессии. Без графики — тестируется целиком."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from starrun.balance import Balance, Stamina
from starrun.sim.combat import Acid, Bullets, Explosion, Impacts, Mark, Orbital, fan
from starrun.sim.effects import Decals, Gems, Particles
from starrun.sim.progression import Offer, Progress, Stats, Upgrades
from starrun.sim.session import Phase, Session
from starrun.sim.spawner import Spawner, edge_positions, front_side
from starrun.sim.swarm import KINDS, QUEEN, SEMI_BOSS, WORKER, Swarm
from starrun.sim.weapons import Fires, Turrets, Weapon, make_weapon

VIEW_HALF = (960.0, 540.0)  # полэкрана 1080p — мир всегда видится в этом масштабе
MAX_DT = 1 / 30             # при подвисании кадра симуляция не перескакивает сквозь стены жуков
EVAC_SPAWN_BOOST = 1.3      # пока ждём челнок, рой давит сильнее
QUEEN_SPAWN_SHARE = 0.6     # при королеве обычных жуков меньше: свиту она призывает сама
TARGET_SWITCH = 0.6         # MAR бросает цель ради жука, который вдвое ближе (с запасом)
BOSS_FOCUS = 220.0          # босс в приоритете, пока ближе этого никто не кусает
KNOCK_STEP = 0.35           # отбрасывание за попадание — доля силы knockback, в пикселях
SLOTS = 3                   # стволов одновременно


@dataclass
class Controls:
    move: tuple[float, float] = (0.0, 0.0)
    sprint: bool = False
    ability_target: tuple[float, float] | None = None  # точка в мире, если в этом кадре нажат пробел
    select: int | None = None                          # клавиша 1–3: слот оружия
    cycle: int = 0                                     # колесо мыши: ±1 слот


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
        self.weapons: list[Weapon] = [make_weapon("mar", b.weapons["mar"])]
        self.active = 0
        self.turrets = Turrets(b.weapons["turret"])
        self.fires = Fires()
        self.orbital = Orbital(b.orbital)
        self.splashes = Particles()
        self.flames = Particles(1500)
        self.decals = Decals()
        self.gems = Gems()
        self.progress = Progress(b.xp)
        self.upgrades = Upgrades(b.upgrades, b.rarities, b.weapons)
        self.session = Session(b.alert, b.evac, b.scaling, b.semi_boss_minutes)
        flocks = {k: (c.flock_min, c.flock_max) for k, c in b.enemies.items() if c.flock_max}
        self.spawner = Spawner(b.spawn, b.mix, self.rng, flocks)
        self.kills: Counter[str] = Counter()
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.damage_by: defaultdict[str, float] = defaultdict(float)  # полученный урон: вид жука или "acid"
        self.events: list[str] = []      # что случилось за последний шаг — для звука, баннеров, тряски
        self.blasts: list[Mark] = []     # взрывы орбитального удара за последний шаг
        self.booms: list[Explosion] = [] # взрывы ракет и гранат за последний шаг
        self.arcs: list[np.ndarray] = [] # молнии шокера за последний шаг: точки цепи
        self.evac_pos: np.ndarray | None = None
        self.target: int | None = None   # номер жука, в которого целится выбранный ствол
        self._offer: list[Offer] = []

    # ---------- оружие ----------
    @property
    def weapon(self) -> Weapon:
        return self.weapons[self.active]

    def owned(self) -> dict[str, tuple[int, bool]]:
        return {w.id: (w.level, w.mod) for w in self.weapons}

    def select(self, slot: int) -> None:
        if 0 <= slot < len(self.weapons) and slot != self.active:
            self.active = slot
            self.target = None
            self.events.append("switch")

    # ---------- выбор улучшений ----------
    @property
    def awaiting_choice(self) -> bool:
        return self.progress.pending > 0 and not self.session.over

    def offer(self) -> list[Offer]:
        if self.awaiting_choice and not self._offer:
            self._offer = self.upgrades.offer(self.rng, owned=self.owned(), free_slot=len(self.weapons) < SLOTS)
        return self._offer

    def choose(self, i: int) -> None:
        offer = self.offer()[i]
        up = offer.upgrade
        if offer.special == "weapon":
            self.weapons.append(make_weapon(up.weapon, self.balance.weapons[up.weapon]))
            self.active = len(self.weapons) - 1  # новый ствол сразу в руках
        elif up.weapon:
            w = next(w for w in self.weapons if w.id == up.weapon)
            if offer.special == "mod":
                w.mod = True
            else:
                w.level += 1
        old_max = self.stats["max_hp"]
        self.upgrades.apply(offer, self.stats)
        self.player.hp += max(0.0, self.stats["max_hp"] - old_max)  # новая броня приходит целой
        self.progress.pending -= 1
        self._offer = []

    # ---------- шаг ----------
    def step(self, dt: float, controls: Controls) -> None:
        self.events = []
        self.blasts = []
        self.booms = []
        self.arcs = []
        if self.session.over or self.awaiting_choice:
            return
        dt = min(dt, MAX_DT)
        b, s, p = self.balance, self.session, self.player

        p.move(dt, controls.move, controls.sprint, self.stats)
        if controls.select is not None:
            self.select(controls.select)
        if controls.cycle:
            self.select((self.active + controls.cycle) % len(self.weapons))
        if controls.ability_target is not None:
            target = np.array(controls.ability_target, np.float32)
            self.events.append("orbital" if self.orbital.trigger(target, p.pos, self.stats) else "orbital_not_ready")

        for event in s.update(dt):
            kind = SEMI_BOSS if event == "semi_boss" else QUEEN
            self.swarm.spawn(kind, edge_positions(self.rng, 1, p.pos, VIEW_HALF, 120), s.boss_hp_mult, s.damage_mult)
            self.events.append(event)

        boost = {Phase.EVAC: EVAC_SPAWN_BOOST, Phase.QUEEN: QUEEN_SPAWN_SHARE}.get(s.phase, 1.0)
        for name, pos in self.spawner.update(dt, s.elapsed, self.swarm.count(), p.pos, VIEW_HALF, boost):
            kind = KINDS.index(name)
            limit = b.enemies[name].max_alive
            if limit:  # сверх лимита (мозговые) — рабочими
                room = max(0, limit - self.swarm.count(kind))
                self.swarm.spawn(WORKER, pos[room:], s.hp_mult, s.damage_mult)
                pos = pos[:room]
            self.swarm.spawn(kind, pos, s.hp_mult, s.damage_mult)
        far = self.swarm.far_from(p.pos, b.spawn.despawn_distance)
        if len(far):
            # Отставших — за край экрана по ходу движения: убежать от толпы насовсем нельзя
            side = front_side(self.rng, p.facing)
            self.swarm.pos[far] = edge_positions(self.rng, len(far), p.pos, VIEW_HALF, b.spawn.margin, side)

        step = self.swarm.step(dt, p.pos, self.stats["radius"])
        acid = self.acid.step(dt, p.pos, self.stats["radius"])
        sources = [KINDS[k] for k in step.hit_kinds] + ["acid"] * len(acid)
        for amount, source in zip(step.hits + acid, sources, strict=True):
            if p.take_hit(amount, self.rng, self.stats["dodge"]):
                self.damage_taken += amount
                self.damage_by[source] += amount
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
        spitter = b.enemies["spitter"]
        for where, direction, damage in step.spits:
            self.acid.fire(where, direction[None], spitter.projectile_speed, damage)
            self.events.append("spit")
        if step.swings:
            self.events.append("swing")

        for w in self.weapons:
            w.tick(dt, self)
        self.weapon.update(dt, self)
        self.turrets.step(dt, self)
        self.fires.step(dt, self)
        self.damage_dealt += self.swarm.burn_tick(dt)
        self.resolve(self.bullets.step(dt, self.swarm))
        for mark in self.orbital.update(dt):
            caught = self.swarm.in_radius(mark.pos, mark.radius)
            self.deal(caught, np.full(len(caught), mark.damage, np.float32))
            self.splashes.burst(self.rng, mark.pos[None], 60, (100, 700), 0.8, (5, 12))
            self.blasts.append(mark)
            self.events.append("blast")
        self._collect_dead()

        gained = self.progress.add(self.gems.step(dt, p.pos, self.stats["pickup_radius"]))
        if gained:
            self.events.append("level_up")
        self.splashes.step(dt)
        self.flames.step(dt, drag=2.5)
        self.decals.step(dt)

        if self.evac_pos is not None:
            inside = float(np.linalg.norm(p.pos - self.evac_pos)) < b.evac.radius
            if s.evac_tick(dt, inside):
                self.events.append("victory")
        if p.hp <= 0 and not s.over:
            s.player_died()
            self.events.append("defeat")

    def pick_target(self, reach: float) -> int | None:
        """Цель выбранного ствола; запоминается в self.target."""
        self.target = self._choose_target(reach)
        return self.target

    def _choose_target(self, reach: float) -> int | None:
        """Ближайший жук, но с «залипанием»: пока цель жива и в радиусе, ствол добивает её, а не размазывает
        урон по всей подходящей толпе. Переключение — только на жука заметно ближе (вцепившегося в десантника)."""
        sw, pos = self.swarm, self.player.pos
        nearest = sw.nearest(pos, reach)
        if nearest is not None and float(np.linalg.norm(sw.pos[nearest] - pos)) > BOSS_FOCUS:
            boss = self.nearest_boss(reach)
            if boss is not None:
                return boss
        current = self.target
        if current is None or nearest is None or not sw.alive[current]:
            return nearest
        d_current = float(np.linalg.norm(sw.pos[current] - pos))
        if d_current > reach:
            return nearest
        d_nearest = float(np.linalg.norm(sw.pos[nearest] - pos))
        return nearest if d_nearest < d_current * TARGET_SWITCH else current

    def nearest_boss(self, reach: float) -> int | None:
        sw, pos = self.swarm, self.player.pos
        idx = sw.alive_idx
        idx = idx[sw.kind[idx] >= SEMI_BOSS]
        if not len(idx):
            return None
        d = np.linalg.norm(sw.pos[idx] - pos, axis=1) - sw.radius[idx]
        i = int(np.argmin(d))
        return int(idx[i]) if d[i] <= reach else None

    def deal(self, idx: np.ndarray, amounts: np.ndarray, armor: bool = True) -> None:
        """Урон по жукам с учётом брони; в статистику идёт фактически нанесённый."""
        if len(idx):
            self.damage_dealt += self.swarm.hit(idx, amounts, armor)

    def resolve(self, imp: Impacts) -> None:
        """Последствия полёта пуль: урон, отбрасывание, поджог, взрывы."""
        self.deal(imp.bug, imp.damage)
        pushed = np.flatnonzero(np.abs(imp.knock).sum(1) > 0)
        self.swarm.knock(imp.bug[pushed], imp.knock[pushed] * KNOCK_STEP)
        burning = np.flatnonzero(imp.burn_time > 0)
        if len(burning):
            self.swarm.ignite(imp.bug[burning], float(imp.burn_time[burning].max()), float(imp.burn_dps[burning].max()))
        for e in imp.explosions:
            self._explode(e)

    def _explode(self, e: Explosion, secondary: bool = False) -> None:
        sw = self.swarm
        caught = sw.in_radius(e.pos, e.radius)
        self.deal(caught, np.full(len(caught), e.damage, np.float32))
        if e.knock and len(caught):
            away = sw.pos[caught] - e.pos
            away /= np.linalg.norm(away, axis=1, keepdims=True) + 1e-6
            sw.knock(caught, away * e.knock * KNOCK_STEP)
        self.splashes.burst(self.rng, e.pos[None], 14, (80, 360), 0.4, (4, 9))
        self.booms.append(e)
        self.events.append("explosion")
        if e.cluster and not secondary:  # кассета: малые заряды кольцом вокруг
            angles = self.rng.uniform(0, 2 * np.pi, e.cluster)
            dist = self.rng.uniform(0.6, 1.1, e.cluster) * e.radius
            for a, r in zip(angles, dist, strict=True):
                pos = e.pos + np.array([np.cos(a), np.sin(a)], np.float32) * r
                self._explode(Explosion(pos, e.cluster_radius, e.cluster_damage), secondary=True)

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
