from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim import terrain as T
from starrun.sim.progression import Offer, Upgrades
from starrun.sim.swarm import TANKER, WORKER
from starrun.sim.weapons import make_weapon
from starrun.sim.world import MAX_HIT_FEEDBACK, Controls, World


def points(*pairs: tuple[float, float]) -> np.ndarray:
    return np.array(pairs, np.float32)


def fire_until_done(world: World, frames: int = 180) -> list[int]:
    hit = []
    for _ in range(frames):
        impact = world.bullets.step(1 / 120, world.swarm, world.terrain)
        hit.extend(impact.bug.tolist())
        world.resolve(impact)
        if not world.bullets.alive.any():
            break
    return hit


@pytest.mark.parametrize("upgrade_id", ["mar_ricochet", "sniper_armor_pierce", "rocket_fire"])
def test_behavior_cards_have_fixed_effect_and_require_owned_weapon(balance: Balance, upgrade_id: str) -> None:
    upgrade = next(u for u in balance.upgrades if u.id == upgrade_id)
    assert upgrade.fixed
    assert {Offer(upgrade, rarity, i).amount(balance.rarities[0].bonus)
            for i, rarity in enumerate(balance.rarities)} == {1.0}
    upgrades = Upgrades(balance.upgrades, balance.rarities, balance.weapons)
    assert upgrade not in upgrades.available()
    assert upgrade in upgrades.available({upgrade.weapon: (1, False)})
    assert upgrade not in upgrades.available({upgrade.weapon: (5, False)})


def test_ricochet_bends_towards_new_target_and_loses_damage(balance: Balance) -> None:
    world = World(balance, 42)
    a, b = world.swarm.spawn(WORKER, points((100, 0), (100, 100)))
    world.swarm.hp[[a, b]] = 100
    world.bullets.fire(points((0, 0))[0], points((1, 0)), 600, 10, 1,
                       ricochet=(2, 220, 0.65))
    assert fire_until_done(world) == [a, b]  # не возвращается в первого ещё живого жука
    assert world.swarm.hp[a] == pytest.approx(90)
    assert world.swarm.hp[b] == pytest.approx(93.5)


def test_ricochet_follows_piercing_and_is_deterministic(balance: Balance) -> None:
    results = []
    for _ in range(2):
        world = World(balance, 8)
        ids = world.swarm.spawn(WORKER, points((100, 0), (160, 0), (160, 110)))
        world.swarm.hp[ids] = 100
        world.bullets.fire(points((0, 0))[0], points((1, 0)), 600, 10, 1,
                           pierce=1, ricochet=(1, 220, 0.65))
        hit = fire_until_done(world)
        assert hit == ids.tolist()
        assert np.allclose(world.swarm.hp[ids], [90, 90, 93.5])
        results.append(world.swarm.hp.copy())
    assert np.array_equal(*results)


def test_ricochet_crosses_previous_target_without_hitting_it_again(balance: Balance) -> None:
    world = World(balance, 42)
    a, b, c = world.swarm.spawn(WORKER, points((100, 0), (100, 100), (100, -100)))
    world.swarm.hp[[a, b, c]] = 100
    world.bullets.fire(points((0, 0))[0], points((1, 0)), 600, 10, 1, ricochet=(2, 220, 0.65))
    assert fire_until_done(world) == [a, b, c]
    assert world.swarm.hp[[a, b, c]] == pytest.approx([90, 93.5, 95.775])


def test_ricochet_does_not_choose_target_behind_wall(balance: Balance) -> None:
    world = World(balance, 42, "forest")
    terrain = world.terrain
    assert terrain is not None
    terrain.kind[:] = T.GROUND
    terrain.kind[102, 100] = T.WALL
    pos = np.array([terrain.cell_center(100, 100), terrain.cell_center(100, 104)])
    first, hidden = world.swarm.spawn(WORKER, pos)
    world.swarm.hp[[first, hidden]] = 100
    world.bullets.fire(pos[0] - [150, 0], points((1, 0)), 600, 10, 1, ricochet=(1, 220, 0.65))
    assert fire_until_done(world) == [first]
    assert world.swarm.hp[hidden] == 100


def test_ricochet_history_retains_long_piercing_chain(balance: Balance) -> None:
    world = World(balance, 8)
    line = np.column_stack([100 + np.arange(17) * 60, np.zeros(17)]).astype(np.float32)
    pos = np.concatenate([line, np.array([[1060, 120], [1060, -120]], np.float32)])
    ids = world.swarm.spawn(WORKER, pos)
    world.swarm.hp[ids] = 1000
    world.bullets.fire(np.zeros(2, np.float32), points((1, 0)), 600, 10, 3,
                       pierce=16, ricochet=(2, 220, 0.65))
    assert fire_until_done(world, 480) == ids.tolist()


@pytest.mark.parametrize("modified", [False, True])
def test_sniper_armor_card_combines_with_rail(balance: Balance, modified: bool) -> None:
    hp = []
    for piercing in (False, True):
        world = World(balance, 42)
        tanker = world.swarm.spawn(TANKER, points((150, 0)))[0]
        sniper = make_weapon("sniper", balance.weapons["sniper"])
        sniper.mod = modified
        world.stats.add("sniper_armor_pierce", float(piercing))
        sniper.fire(world, world.swarm.pos[tanker], 1)
        fire_until_done(world)
        hp.append(float(world.swarm.hp[tanker]))
        assert world.hit_feedback[-1].armored is not piercing
        armor = float(world.swarm.armor[tanker])
        assert armor == (0 if piercing else 3)
        before = float(world.swarm.hp[tanker])
        world.deal(np.array([tanker]), np.array([10], np.float32))
        assert before - world.swarm.hp[tanker] == (10 if piercing else 7)
    assert hp == pytest.approx([63, 60])


@pytest.mark.parametrize("modified", [False, True])
def test_thermobaric_field_is_created_once_and_preserves_cluster(balance: Balance, modified: bool) -> None:
    world = World(balance, 42)
    rocket = make_weapon("rocket", balance.weapons["rocket"])
    rocket.mod = modified
    world.stats.add("rocket_fire", 1)
    rocket.fire(world, points((200, 0))[0], 1)
    fire_until_done(world)
    assert world.fires.alive.sum() == 1
    field = np.flatnonzero(world.fires.alive)[0]
    assert world.fires.life[field] == pytest.approx(2.5)
    assert world.fires.dps[field] == pytest.approx(9)
    assert world.fires.radius[field] == pytest.approx(65)
    assert len(world.booms) == (5 if modified else 1)
    target = world.swarm.spawn(WORKER, world.fires.pos[field][None])[0]
    world.fires.step(0.1, world)
    assert world.swarm.hp[target] == pytest.approx(9.1)


def test_hit_feedback_is_bounded_copied_and_cleared(balance: Balance) -> None:
    world = World(balance, 42)
    ids = world.swarm.spawn(WORKER, np.column_stack([np.arange(50) * 10 + 1000, np.zeros(50)]).astype(np.float32))
    world.deal(ids, np.full(len(ids), 5, np.float32))
    assert len(world.hit_feedback) == MAX_HIT_FEEDBACK
    recorded = world.hit_feedback[0].pos.copy()
    world.swarm.pos[ids[0]] += 100
    assert np.array_equal(world.hit_feedback[0].pos, recorded)
    assert world.hit_feedback[0].kind == WORKER
    world.session.player_died()
    world.step(1 / 60, Controls())
    assert world.hit_feedback == []


def test_hit_feedback_caps_repeated_pellets_to_remaining_hp(balance: Balance) -> None:
    world = World(balance, 42)
    tanker = world.swarm.spawn(TANKER, points((100, 0)))[0]
    world.swarm.hp[tanker] = 5
    world.deal(np.full(6, tanker), np.full(6, 4, np.float32), critical=np.array([True, False] * 3))
    assert len(world.hit_feedback) == 5
    assert sum(event.damage for event in world.hit_feedback) == 5
    assert all(event.armored for event in world.hit_feedback)
    assert world.hit_feedback[0].critical and not world.hit_feedback[1].critical


def test_mar_weapon_transmits_critical_and_ricochet_state(balance: Balance) -> None:
    world = World(balance, 2)
    world.stats.add("mar_ricochet", 1)
    world.stats.add("mar_crit", 1)  # потолок 50 %, серия даёт и критические, и обычные
    world.weapon.fire(world, points((200, 0))[0], 32)
    idx = np.flatnonzero(world.bullets.alive)
    assert (world.bullets.ricochet[idx] == 1).all()
    assert world.bullets.critical[idx].any() and not world.bullets.critical[idx].all()
    assert set(world.bullets.damage[idx]) == {10, 20}
