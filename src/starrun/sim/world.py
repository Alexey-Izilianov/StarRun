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
from starrun.sim.terrain import DNA, FENCE, MAGNET, MEDKIT, MUD, SAND, TREE, UNDERBARREL, WEAPON_CRATE, WINDOW, Terrain
from starrun.sim.weapons import Fires, Turrets, Weapon, make_weapon

VIEW_HALF = (960.0, 540.0)  # полэкрана 1080p в 2D; 3D-камера задаёт свой прямоугольник (World.view_half)
MAX_DT = 1 / 30             # при подвисании кадра симуляция не перескакивает сквозь стены жуков
EVAC_SPAWN_BOOST = 1.3      # пока ждём челнок, рой давит сильнее
QUEEN_SPAWN_SHARE = 0.6     # при королеве обычных жуков меньше: свиту она призывает сама
TARGET_SWITCH = 0.6         # MAR бросает цель ради жука, который вдвое ближе (с запасом)
BOSS_FOCUS = 220.0          # босс в приоритете, пока ближе этого никто не кусает
KNOCK_STEP = 0.35           # отбрасывание за попадание — доля силы knockback, в пикселях
SLOTS = 3                   # стволов одновременно
SPAWN_RING = 4              # на сколько клеток можно сдвинуть точку появления жука с препятствия
SPAWN_HIDDEN = 30.0         # жук появляется хотя бы настолько за краем экрана
SIGHT_STEP = 20.0           # шаг проверки линии огня, пикселей
SIGHT_BATCH = 8             # сколько ближайших жуков проверять на линию огня за раз
EXPLOSION_CLEAR = 0.6       # взрыв выжигает лес в этой доле своего радиуса
SAND_SLOW = 0.7             # замедление при полном погружении в зыбучий песок
ITEM_REACH = 18.0           # находка подбирается при касании (к радиусу десантника)


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
    facing: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], np.float32))   # куда смотрит (и целится)
    heading: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0], np.float32))  # куда шёл последним
    sprinting: bool = False
    hurt: float = 0.0       # подсветка после попадания
    sink: float = 0.0       # погружение в зыбучий песок, 0..1
    vaulting: bool = False  # перелезает окно или забор
    chop: float = -1.0      # фаза замаха топором 0..1; < 0 — не рубит
    _rest: float = 0.0

    def move(self, dt: float, direction: tuple[float, float], sprint: bool, stats: Stats,
             slow: float = 1.0) -> None:
        d = np.array(direction, np.float32)
        n = float(np.linalg.norm(d))
        moving = n > 1e-6
        self.sprinting = sprint and moving and self.stamina > 0
        if moving:
            d /= n
            self.facing = self.heading = d
            speed = stats["speed"] * (stats["sprint_mult"] if self.sprinting else 1.0) * slow
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
    def __init__(self, balance: Balance, seed: int | None = None, biome: str | None = None) -> None:
        """biome — id из biomes.toml; без него — бесконечная ровная земля (так проверяется бой без карты)."""
        self.balance = b = balance
        self.rng = np.random.default_rng(seed)
        self.stats = Stats(b.player.stats, b.player.caps)
        self.player = Player(b.player.stamina, hp=self.stats["max_hp"], stamina=b.player.stamina.max)
        self.biome = biome
        self.view_half = VIEW_HALF   # видимая область вокруг десантника: жуки появляются за её краем
        self.terrain = Terrain(b.map, b.biomes[biome], b.items, self.rng) if biome else None
        items = self.terrain.items if self.terrain else []
        self.item_pos = np.array([pos for pos, _ in items], np.float32).reshape(-1, 2)
        self.item_kind = np.array([kind for _, kind in items], np.int8)
        self.item_alive = np.ones(len(items), bool)
        self.dna = 0
        self._cactus_left = 0.0
        if self.terrain:
            self.player.pos = self.terrain.center
            self.terrain.update_flow(self.player.pos, 0.0, force=True)  # спавну с первого кадра нужны пути
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

        p.move(dt, controls.move, controls.sprint, self.stats, self._slow())
        if self.terrain:
            self._on_map(dt, controls.move)
        if controls.select is not None:
            self.select(controls.select)
        if controls.cycle:
            self.select((self.active + controls.cycle) % len(self.weapons))
        if controls.ability_target is not None:
            target = np.array(controls.ability_target, np.float32)
            self.events.append("orbital" if self.orbital.trigger(target, p.pos, self.stats) else "orbital_not_ready")

        for event in s.update(dt):
            kind = SEMI_BOSS if event == "semi_boss" else QUEEN
            where = self._boss_point(edge_positions(self.rng, 1, p.pos, self.view_half, 120))
            self.swarm.spawn(kind, where, s.boss_hp_mult, s.damage_mult)
            self.events.append(event)

        boost = {Phase.EVAC: EVAC_SPAWN_BOOST, Phase.QUEEN: QUEEN_SPAWN_SHARE}.get(s.phase, 1.0)
        for name, edge in self.spawner.update(dt, s.elapsed, self.swarm.count(), p.pos, self.view_half, boost):
            pos, found = self._spawn_points(edge)
            pos = pos[found]  # не нашлось места — жук появится в другой раз
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
            edge = edge_positions(self.rng, len(far), p.pos, self.view_half, b.spawn.margin, side)
            pos, found = self._spawn_points(edge)
            self.swarm.pos[far[found]] = pos[found]

        step = self.swarm.step(dt, p.pos, self.stats["radius"], self.terrain)
        if step.doors:
            self.events.append("door_broken")
        acid = self.acid.step(dt, p.pos, self.stats["radius"], self.terrain)
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
            ring = ring.astype(np.float32)
            if self.terrain:
                ring, found = self.terrain.place(ring, 3)
                ring = ring[found]
            self.swarm.spawn(WORKER, ring, s.hp_mult, s.damage_mult)
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
        self.resolve(self.bullets.step(dt, self.swarm, self.terrain))
        for mark in self.orbital.update(dt):
            caught = self.swarm.in_radius(mark.pos, mark.radius)
            self.deal(caught, np.full(len(caught), mark.damage, np.float32))
            if self.terrain:
                self.terrain.clear_circle_px(mark.pos, mark.radius, (TREE, FENCE))
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
        nearest = self.nearest_bug(pos, reach)
        if nearest is not None and float(np.linalg.norm(sw.pos[nearest] - pos)) > BOSS_FOCUS:
            boss = self.nearest_boss(reach)
            if boss is not None:
                return boss
        current = self.target
        if current is None or nearest is None or not sw.alive[current]:
            return nearest
        if not self.in_sight(pos, sw.pos[[current]])[0]:  # цель ушла за дерево
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
        idx = idx[self.in_sight(pos, sw.pos[idx])]
        if not len(idx):
            return None
        d = np.linalg.norm(sw.pos[idx] - pos, axis=1) - sw.radius[idx]
        i = int(np.argmin(d))
        return int(idx[i]) if d[i] <= reach else None

    def nearest_bug(self, origin: np.ndarray, reach: float) -> int | None:
        """Ближайший жук, в которого можно попасть: на карте — без леса и стен на линии огня."""
        sw = self.swarm
        if self.terrain is None:
            return sw.nearest(origin, reach)
        idx = sw.in_radius(origin, reach)
        d2 = ((sw.pos[idx] - origin) ** 2).sum(1) - sw.radius[idx] ** 2
        idx = idx[np.argsort(d2)]
        # Линия огня — от ближних к дальним пачками: обычно ближайший виден, и проверять всю толпу незачем
        for start in range(0, len(idx), SIGHT_BATCH):
            batch = idx[start : start + SIGHT_BATCH]
            seen = batch[self.in_sight(origin, sw.pos[batch])]
            if len(seen):
                return int(seen[0])
        return None

    def toughest_bug(self, origin: np.ndarray, reach: float) -> int | None:
        sw = self.swarm
        if self.terrain is None:
            return sw.toughest(origin, reach)
        idx = sw.in_radius(origin, reach)
        idx = idx[self.in_sight(origin, sw.pos[idx])]
        return int(idx[np.argmax(sw.hp[idx])]) if len(idx) else None

    def in_sight(self, origin: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Маска точек, до которых линия огня от origin не упирается в лес, стену или закрытую дверь."""
        n = len(points)
        if self.terrain is None or not n:
            return np.ones(n, bool)
        rel = points - origin
        k = max(2, int(np.ceil(float(np.linalg.norm(rel, axis=1).max()) / SIGHT_STEP)))
        t = np.linspace(0, 1, k + 1, dtype=np.float32)[1:-1]
        pts = origin + rel[:, None, :] * t[None, :, None]
        return ~self.terrain.shot_blocked(pts.reshape(-1, 2)).reshape(n, len(t)).any(1)

    def burn_tree(self, origin: np.ndarray, direction: np.ndarray, reach: float, amount: float) -> None:
        """Огнемёт жжёт первое дерево на оси струи."""
        t = self.terrain
        if t is None:
            return
        for dist in np.arange(self.stats["radius"], reach, t.cell / 2):
            pos = origin + direction * dist
            if t.shot_blocked(pos)[0]:
                ix, iy = t.cells(pos)
                if t.damage_tree(int(ix[0]), int(iy[0]), amount):
                    self.events.append("tree_down")
                return

    def deal(self, idx: np.ndarray, amounts: np.ndarray, armor: bool = True) -> None:
        """Урон по жукам с учётом брони; в статистику идёт фактически нанесённый."""
        if len(idx):
            self.damage_dealt += self.swarm.hit(idx, amounts, armor)

    def resolve(self, imp: Impacts) -> None:
        """Последствия полёта пуль: урон, отбрасывание, поджог, взрывы."""
        self.deal(imp.bug, imp.damage)
        pushed = np.flatnonzero(np.abs(imp.knock).sum(1) > 0)
        self.swarm.knock(imp.bug[pushed], imp.knock[pushed] * KNOCK_STEP, self.terrain)
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
            sw.knock(caught, away * e.knock * KNOCK_STEP, self.terrain)
        if self.terrain:
            self.terrain.clear_circle_px(e.pos, e.radius * EXPLOSION_CLEAR, (TREE, FENCE))
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
            self.evac_pos = self._evac_point(self.player.pos + offset)
            self.events.append("evac")

    # ---------- карта ----------
    def _slow(self) -> float:
        """Множитель скорости десантника от того, где он стоит: окно, забор, топь, песок."""
        t = self.terrain
        if t is None:
            return 1.0
        c, p = t.cfg, self.player
        slow = c.vault_speed if t.kinds_at(p.pos)[0] in (WINDOW, FENCE) else 1.0
        if t.soil_at(p.pos)[0] == MUD:
            slow *= c.mud_speed
        return slow * (1 - SAND_SLOW * p.sink)

    def _on_map(self, dt: float, move: tuple[float, float]) -> None:
        """Десантник и карта: стены держат, двери открываются, топор рубит лес, песок затягивает,
        кактусы колются, находки подбираются. В конце — поле путей жуков от нового места."""
        t, p, c = self.terrain, self.player, self.balance.map
        assert t is not None
        r = self.stats["radius"]
        p.pos = t.resolve_player(p.pos, r)
        p.vaulting = t.kinds_at(p.pos)[0] in (WINDOW, FENCE)
        if t.update_doors(p.pos, r):
            self.events.append("door_open")

        d = np.array(move, np.float32)
        n = float(np.linalg.norm(d))
        tree = t.tree_ahead(p.pos, d / n, r) if n > 1e-6 else None
        if tree is None:
            p.chop = -1.0
        else:  # упёрся в дерево — замах, удар в конце замаха
            p.chop = max(p.chop, 0.5) + dt / c.chop_every
            if p.chop >= 1:
                p.chop -= 1
                self.events.append("chop")
                if t.damage_tree(*tree):
                    self.events.append("tree_down")

        if t.soil_at(p.pos)[0] == SAND:
            p.sink = min(1.0, p.sink + c.sand_sink_rate * dt)
        else:
            p.sink = max(0.0, p.sink - c.sand_recover * dt)
        if p.sink >= 1:
            self._hurt(c.sand_damage * dt, "sand", flash=False)

        if t.biome.tree == "cactus" and t.touching(p.pos, r, TREE):
            self._cactus_left -= dt
            if self._cactus_left <= 0:
                self._cactus_left = c.cactus_every
                self._hurt(c.cactus_damage, "cactus")
        else:
            self._cactus_left = 0.0

        self._pickup(r)
        t.update_flow(p.pos, dt)

    def _hurt(self, amount: float, source: str, flash: bool = True) -> None:
        """Урон от карты: от него не уклониться."""
        p = self.player
        p.hp -= amount
        self.damage_taken += amount
        self.damage_by[source] += amount
        if flash:
            p.hurt = 0.2
            self.events.append("player_hit")

    def _pickup(self, radius: float) -> None:
        idx = np.flatnonzero(self.item_alive)
        if not len(idx):
            return
        d = np.linalg.norm(self.item_pos[idx] - self.player.pos, axis=1)
        for i in idx[d < radius + ITEM_REACH]:
            self.item_alive[i] = False
            self._use_item(int(self.item_kind[i]))

    def _use_item(self, kind: int) -> None:
        p, items = self.player, self.balance.items
        if kind == MEDKIT:
            p.hp = min(self.stats["max_hp"], p.hp + items.medkit_heal * self.stats["max_hp"])
            self.events.append("medkit")
        elif kind == MAGNET:
            self.gems.pulled[self.gems.alive] = True
            self.events.append("magnet")
        elif kind == DNA:
            self.dna += items.dna_value
            self.events.append("dna")
        elif kind == UNDERBARREL:
            next(w for w in self.weapons if w.id == "mar").mod = True
            self.events.append("underbarrel")
        elif kind == WEAPON_CRATE:
            new = [wid for wid in self.balance.weapons if wid not in self.owned()]
            if len(self.weapons) < SLOTS and new:
                wid = new[int(self.rng.integers(len(new)))]
                self.weapons.append(make_weapon(wid, self.balance.weapons[wid]))
                self.active = len(self.weapons) - 1
                self.events.append(f"crate:{wid}")
            else:  # слоты заняты — опыт
                if self.progress.add(items.weapon_xp):
                    self.events.append("level_up")
                self.events.append("crate_xp")

    def _spawn_points(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Точки появления жуков за краем экрана; на карте — сдвинутые на свободные клетки, откуда есть путь
        к десантнику, а за краем карты — отражённые на другую сторону экрана. Сдвиг не должен затянуть
        точку на экран: на глазах у игрока жуки не появляются. Возвращает (точки, нашлось ли место)."""
        t = self.terrain
        if t is None or not len(pos):
            return pos, np.ones(len(pos), bool)
        lo = t.cfg.border * t.cell
        hi = t.size_px - lo
        pos = np.where((pos < lo) | (pos > hi), 2 * self.player.pos - pos, pos)
        placed, found = t.place(np.clip(pos, lo, hi), SPAWN_RING, reachable=True)
        rel = np.abs(placed - self.player.pos)
        found &= (rel[:, 0] > self.view_half[0] + SPAWN_HIDDEN) | (rel[:, 1] > self.view_half[1] + SPAWN_HIDDEN)
        return placed, found

    def _boss_point(self, pos: np.ndarray) -> np.ndarray:
        """Босс появляется всегда: не нашлось места с путём — ближайшее свободное."""
        placed, found = self._spawn_points(pos)
        if self.terrain and not found[0]:
            placed[0] = self.terrain.nearest_open(placed[0])
        return placed

    def _evac_point(self, pos: np.ndarray) -> np.ndarray:
        """Зона эвакуации — на открытом месте внутри карты; лес и заборы вокруг вырубаются."""
        t = self.terrain
        if t is None:
            return pos
        margin = (t.cfg.border + t.cfg.evac_clear + 1) * t.cell
        spot = t.nearest_open(np.clip(pos, margin, t.size_px - margin))
        t.carve_evac(spot)
        return spot
