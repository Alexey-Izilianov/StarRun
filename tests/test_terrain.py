from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import BIOME_IDS, Balance
from starrun.sim import terrain as T
from starrun.sim.combat import ROCKET, Acid, Bullets, Explosion
from starrun.sim.swarm import FLYER, WORKER, Swarm
from starrun.sim.terrain import Terrain
from starrun.sim.weapons import make_weapon
from starrun.sim.world import VIEW_HALF, Controls, World

DT = 1 / 30
C = 40.0  # клетка карты, px


def px(ix: float, iy: float) -> np.ndarray:
    """Центр клетки в пикселях."""
    return np.array([(ix + 0.5) * C, (iy + 0.5) * C], np.float32)


def blank(balance: Balance, biome: str = "forest", seed: int = 1) -> World:
    """Мир на карте, расчищенной до ровной земли без находок и спавна: препятствия ставит сам тест.
    Десантник — в центре клетки (100, 100)."""
    w = World(balance, seed, biome)
    t = w.terrain
    assert t is not None
    t.kind[:] = T.GROUND
    t.soil[:] = T.PLAIN
    t.building[:] = -1
    t.buildings.clear()
    t.doors.clear()
    b = balance.map.border
    t.kind[:b, :] = t.kind[-b:, :] = T.ROCK
    t.kind[:, :b] = t.kind[:, -b:] = T.ROCK
    w.item_alive[:] = False
    w.spawner.update = lambda *args, **kwargs: []  # type: ignore[method-assign]
    w.player.pos = px(100, 100)
    t.update_flow(w.player.pos, 0, force=True)
    return w


def room(t: Terrain, x0: int, y0: int, x1: int, y1: int, door: tuple[int, int] | None = None) -> None:
    """Постройка: стены по периметру, пол внутри, дверь в стене."""
    t.kind[y0 : y1 + 1, x0 : x1 + 1] = T.WALL
    t.kind[y0 + 1 : y1, x0 + 1 : x1] = T.FLOOR
    t.building[y0 : y1 + 1, x0 : x1 + 1] = len(t.buildings)
    t.buildings.append(T.Building(x0, y0, x1, y1))
    if door:
        x, y = door
        t.kind[y, x] = T.DOOR
        t.hp[y, x] = t.cfg.door_hp
        t.doors.append(door)


def item_here(w: World, kind: int) -> None:
    w.item_pos = w.player.pos[None].copy()
    w.item_kind = np.array([kind], np.int8)
    w.item_alive = np.ones(1, bool)


# ---------- генерация ----------
@pytest.mark.parametrize("biome", BIOME_IDS)
def test_generation_is_deterministic_and_bounded(balance: Balance, biome: str) -> None:
    a = Terrain(balance.map, balance.biomes[biome], balance.items, np.random.default_rng(5))
    b = Terrain(balance.map, balance.biomes[biome], balance.items, np.random.default_rng(5))
    assert np.array_equal(a.kind, b.kind) and np.array_equal(a.soil, b.soil)
    border = balance.map.border
    assert (a.kind[:border] == T.ROCK).all() and (a.kind[:, -border:] == T.ROCK).all()
    # поляна высадки свободна: ни леса, ни воды
    n, r = a.n, balance.map.start_clear
    ys, xs = np.ogrid[:n, :n]
    clearing = (xs - n // 2) ** 2 + (ys - n // 2) ** 2 <= r * r
    assert not np.isin(a.kind[clearing], (T.TREE, T.WATER)).any()
    assert len(a.buildings) >= 1
    for bid in range(len(a.buildings)):  # в каждой постройке дверь и ящик
        assert any(a.building[y, x] == bid for x, y in a.doors)
        assert any(kind in (T.WEAPON_CRATE, T.UNDERBARREL) and a.building[a.cells(pos)[::-1]][0] == bid
                   for pos, kind in a.items)
    assert sum(kind == T.UNDERBARREL for _, kind in a.items) == 1


def test_biomes_have_their_features(balance: Balance) -> None:
    def make(biome: str) -> Terrain:
        return Terrain(balance.map, balance.biomes[biome], balance.items, np.random.default_rng(2))

    assert (make("desert").soil == T.SAND).any()
    assert (make("jungle").kind == T.BRIDGE).any() and (make("jungle").kind == T.WATER).any()
    swamp = make("swamp")
    assert abs((swamp.soil == T.MUD).mean() - balance.biomes["swamp"].mud) < 0.05
    forest, desert = make("forest"), make("desert")
    assert (forest.kind == T.TREE).mean() > 2 * (desert.kind == T.TREE).mean()


# ---------- десантник и карта ----------
def test_walls_and_water_hold_the_marine(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 101] = T.WALL
    t.kind[100, 99] = T.WATER
    for d in ((1, 0), (-1, 0)):
        w.player.pos = px(100, 100)
        for _ in range(30):
            w.step(DT, Controls(move=d))
        assert abs(float(w.player.pos[0]) - px(100, 100)[0]) < C / 2  # не прошёл сквозь


def test_axe_chops_a_tree_in_the_way(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 101] = T.TREE
    t.hp[100, 101] = balance.biomes["forest"].tree_hp
    events: list[str] = []
    for _ in range(int((balance.biomes["forest"].tree_hp * balance.map.chop_every + 0.5) / DT)):
        w.step(DT, Controls(move=(1, 0)))
        events += w.events
    assert t.kind[100, 101] == T.STUMP
    assert events.count("chop") == balance.biomes["forest"].tree_hp and "tree_down" in events


@pytest.mark.parametrize("start, move, cacti", [
    ((100.5, 100.5), (1, 0), [(102, 100)]),                      # прямо
    ((100.6, 98.6), (0.7, 0.7), [(102, 100)]),                   # в угол клетки
    ((100.4, 99.4), (0.7, 0.7), [(102, 100), (101, 101)]),       # в угол между двумя: скользить некуда
])
def test_cactus_is_chopped_when_pressed(balance: Balance, start: tuple[float, float], move: tuple[float, float],
                                        cacti: list[tuple[int, int]]) -> None:
    """Если десантник упирается в кактус и колется — он его и рубит, с какой стороны ни подойди."""
    w = blank(balance, "desert")
    t = w.terrain
    assert t is not None
    for x, y in cacti:
        t.kind[y, x] = T.TREE
        t.hp[y, x] = balance.biomes["desert"].tree_hp
    w.player.pos = np.array(start, np.float32) * C
    for _ in range(int(2.5 / DT)):
        w.step(DT, Controls(move=move))
    assert any(t.kind[y, x] == T.STUMP for x, y in cacti)


def test_quicksand_sinks_slows_and_hurts(balance: Balance) -> None:
    w = blank(balance, "desert")
    t = w.terrain
    assert t is not None
    t.soil[80:121, 80:121] = T.SAND
    for _ in range(int(2.5 / DT)):
        w.step(DT, Controls())
    assert w.player.sink == 1.0 and w.damage_by["sand"] > 0
    start = w.player.pos.copy()
    for _ in range(int(1 / DT)):
        w.step(DT, Controls(move=(1, 0)))
    assert float(w.player.pos[0] - start[0]) < balance.player.stats["speed"] * 0.5


def test_cactus_pricks_but_a_pine_does_not(balance: Balance) -> None:
    for biome, hurts in (("desert", True), ("forest", False)):
        w = blank(balance, biome)
        t = w.terrain
        assert t is not None
        t.kind[100, 101] = T.TREE
        t.hp[100, 101] = 99  # не срубить за время теста
        for _ in range(int(1 / DT)):
            w.step(DT, Controls(move=(1, 0)))
        assert (w.damage_by["cactus"] > 0) == hurts


def test_window_and_fence_slow_the_marine(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 100] = T.WINDOW
    assert w._slow() == pytest.approx(balance.map.vault_speed)
    t.kind[100, 100] = T.FENCE
    assert w._slow() == pytest.approx(balance.map.vault_speed)
    t.kind[100, 100] = T.GROUND
    assert w._slow() == 1.0


def test_doors_open_before_the_marine_and_close_behind(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    room(t, 101, 97, 107, 103, door=(101, 100))
    assert t.update_doors(px(100, 100), 22) == 1
    assert t.kind[100, 101] == T.OPEN_DOOR
    assert not t.shot_blocked(px(101, 100))[0] and not t.bug_blocked(px(101, 100))[0]
    t.update_doors(px(90, 100), 22)
    assert t.kind[100, 101] == T.DOOR


# ---------- находки ----------
def test_medkit_magnet_dna(balance: Balance) -> None:
    w = blank(balance)
    w.player.hp = 50
    item_here(w, T.MEDKIT)
    w.step(DT, Controls())
    assert w.player.hp == pytest.approx(50 + balance.items.medkit_heal * balance.player.stats["max_hp"], abs=1)

    w.gems.drop(w.rng, np.array([[3000, 3000]], np.float32), np.array([5]))
    item_here(w, T.MAGNET)
    w.step(DT, Controls())
    assert w.gems.pulled[w.gems.alive].all()

    item_here(w, T.DNA)
    w.step(DT, Controls())
    assert w.dna == balance.items.dna_value and not w.item_alive.any()


def test_weapon_crate_and_underbarrel(balance: Balance) -> None:
    w = blank(balance)
    item_here(w, T.WEAPON_CRATE)
    w.step(DT, Controls())
    assert len(w.weapons) == 2 and w.active == 1
    assert any(e.startswith("crate:") for e in w.events)

    free = next(k for k in balance.weapons if k not in w.owned())
    w.weapons.append(make_weapon(free, balance.weapons[free]))  # слоты заняты — ящик даёт опыт
    xp = w.progress.xp + w.progress.level * 1000
    item_here(w, T.WEAPON_CRATE)
    w.step(DT, Controls())
    assert "crate_xp" in w.events and w.progress.xp + w.progress.level * 1000 > xp

    while w.awaiting_choice:  # опыт из ящика мог дать уровень
        w.choose(0)
    item_here(w, T.UNDERBARREL)
    w.step(DT, Controls())
    assert w.weapons[0].id == "mar" and w.weapons[0].mod


# ---------- жуки и карта ----------
def test_bugs_walk_around_a_wall(balance: Balance, rng: np.random.Generator) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[90:111, 104] = T.WALL  # стена между жуком и десантником, обход — через концы
    sw = Swarm(balance.enemies, 4, rng)
    bug = sw.spawn(WORKER, px(108, 100)[None])[0]
    player = px(100, 100)
    for _ in range(int(20 / DT)):
        t.update_flow(player, DT)
        sw.step(DT, player, 22, t)
        assert not t.bug_blocked(sw.pos[bug])[0]
    assert float(np.linalg.norm(sw.pos[bug] - player)) < 60


def test_bugs_break_a_door_to_reach_the_marine(balance: Balance, rng: np.random.Generator) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    room(t, 95, 95, 105, 105, door=(100, 95))
    sw = Swarm(balance.enemies, 4, rng)
    bug = sw.spawn(WORKER, px(100, 90)[None])[0]
    player = px(100, 100)
    broken = 0
    for _ in range(int(15 / DT)):
        t.update_flow(player, DT)
        broken += sw.step(DT, player, 22, t).doors
    assert broken == 1 and t.kind[95, 100] == T.FLOOR
    assert float(np.linalg.norm(sw.pos[bug] - player)) < 60


def test_bugs_go_around_a_building_when_the_marine_is_outside(balance: Balance, rng: np.random.Generator) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    room(t, 95, 95, 105, 105, door=(100, 95))
    t.kind[105, 100] = T.DOOR  # вторая дверь: насквозь короче, но ломать двери незачем
    t.hp[105, 100] = t.cfg.door_hp
    t.doors.append((100, 105))
    sw = Swarm(balance.enemies, 4, rng)
    bug = sw.spawn(WORKER, px(100, 90)[None])[0]
    player = px(100, 110)
    for _ in range(int(20 / DT)):
        t.update_flow(player, DT)
        assert not sw.step(DT, player, 22, t).doors
    assert float(np.linalg.norm(sw.pos[bug] - player)) < 60


def test_windows_and_mud_slow_bugs(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 101] = T.WINDOW
    t.soil[100, 102] = T.MUD
    speed = t.bug_speed(np.stack([px(101, 100), px(102, 100), px(103, 100)]))
    assert speed == pytest.approx([balance.map.window_bug_speed, balance.map.mud_speed, 1.0])


def test_tanker_smashes_trees(balance: Balance, rng: np.random.Generator) -> None:
    from starrun.sim.swarm import TANKER
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[95:106, 103] = T.TREE
    sw = Swarm(balance.enemies, 4, rng)
    sw.spawn(TANKER, px(105, 100)[None])
    for _ in range(int(4 / DT)):
        t.update_flow(px(100, 100), DT)
        sw.step(DT, px(100, 100), 22, t)
    assert (t.kind[95:106, 103] == T.STUMP).any()


# ---------- снаряды ----------
def test_bullets_and_acid_stop_at_walls(balance: Balance, rng: np.random.Generator) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 102] = T.WALL
    sw = Swarm(balance.enemies, 4, rng)
    bug = sw.spawn(WORKER, px(104, 100)[None])[0]
    hp = float(sw.hp[bug])
    bullets = Bullets()
    bullets.fire(px(100, 100), np.array([[1, 0]], np.float32), 1100, 50.0, 1.0)
    for _ in range(20):
        imp = bullets.step(DT, sw, t)
        sw.hit(imp.bug, imp.damage)
    assert float(sw.hp[bug]) == hp and not bullets.alive.any()

    bullets.fire(px(100, 100), np.array([[1, 0]], np.float32), 650, 45.0, 1.0, kind=ROCKET, blast=100)
    booms = []
    for _ in range(20):
        booms += bullets.step(DT, sw, t).explosions
    assert len(booms) == 1 and float(booms[0].pos[0]) < 103 * C  # рванула о стену, а не у жука

    acid = Acid()
    acid.fire(px(104, 100), np.array([[-1, 0]], np.float32), 320, 10)
    hits = []
    for _ in range(40):
        hits += acid.step(DT, px(100, 100), 22, t)
    assert not hits and not acid.alive.any()


def test_guns_do_not_aim_through_trees(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 102] = T.TREE
    hidden = w.swarm.spawn(WORKER, px(104, 100)[None])[0]
    assert w.nearest_bug(w.player.pos, 600) is None
    seen = w.swarm.spawn(WORKER, px(100, 106)[None])[0]
    assert w.nearest_bug(w.player.pos, 600) == seen
    assert w.pick_target(600) == seen and hidden != seen


def test_explosions_clear_forest(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[90:111, 105:116] = T.TREE
    w._explode(Explosion(px(110, 100), 100, 10))
    assert t.kind[100, 110] == T.STUMP
    assert t.kind[90, 105] == T.TREE  # дальние уцелели


# ---------- мир на настоящей карте ----------
def test_bugs_spawn_off_screen_on_open_ground(balance: Balance) -> None:
    w = World(balance, seed=3, biome="forest")
    spawned: list[tuple[np.ndarray, np.ndarray]] = []
    original = w.swarm.spawn

    def spy(kind: int, pos: np.ndarray, *args: float) -> np.ndarray:
        spawned.append((pos.copy(), w.player.pos.copy()))
        return original(kind, pos, *args)

    w.swarm.spawn = spy  # type: ignore[method-assign]
    for _ in range(int(20 / DT)):
        if w.awaiting_choice:
            w.choose(0)
        w.step(DT, Controls())
    t = w.terrain
    assert t is not None
    assert spawned
    for pos, player in spawned:
        if not len(pos):
            continue
        rel = np.abs(pos - player)
        assert ((rel[:, 0] > VIEW_HALF[0]) | (rel[:, 1] > VIEW_HALF[1])).all()
        assert t.spawnable(pos).all()


def test_evac_zone_lands_on_open_ground(balance: Balance) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[60:140, 60:140] = T.TREE
    spot = w._evac_point(px(100, 100) + [900, 0])
    ix, iy = (int(v) for v in spot // C)
    assert t.kind[iy, ix] in (T.GROUND, T.STUMP)
    r = balance.map.evac_clear - 1
    ys, xs = np.ogrid[: t.n, : t.n]
    near = (xs - ix) ** 2 + (ys - iy) ** 2 <= r * r
    assert not (t.kind[near] == T.TREE).any()


def test_marine_does_not_push_bugs_into_trees(balance: Balance, rng: np.random.Generator) -> None:
    w = blank(balance)
    t = w.terrain
    assert t is not None
    t.kind[100, 101] = T.TREE
    sw = Swarm(balance.enemies, 4, rng)
    bug = sw.spawn(WORKER, np.array([[px(100, 100)[0] + 15, px(100, 100)[1]]], np.float32))[0]
    sw.speed[bug] = 0
    sw.step(DT, px(100, 100), 22, t)
    assert not t.bug_blocked(sw.pos[bug])[0]


def test_a_quarter_minute_on_every_biome(balance: Balance) -> None:
    for biome in BIOME_IDS:
        w = World(balance, seed=4, biome=biome)
        for _ in range(int(15 / DT)):
            if w.awaiting_choice:
                w.choose(0)
            w.step(DT, Controls(move=(1, 0.3)))
        t = w.terrain
        assert t is not None
        idx = w.swarm.alive_idx
        walkers = idx[w.swarm.kind[idx] != FLYER]  # летающие — над лесом
        assert not t.bug_blocked(w.swarm.pos[walkers]).any()
