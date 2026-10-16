from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.swarm import BRAIN, FLYER, QUEEN, SPITTER, TANKER, WORKER, Swarm
from starrun.sim.world import Controls, World

ORIGIN = np.zeros(2, np.float32)
DT = 1 / 60


@pytest.fixture
def swarm(balance: Balance, rng: np.random.Generator) -> Swarm:
    return Swarm(balance.enemies, 64, rng)


def at(*xy: float) -> np.ndarray:
    return np.array([xy], np.float32)


def test_armor_softens_each_hit_but_not_below_one(swarm: Swarm, balance: Balance) -> None:
    t = swarm.spawn(TANKER, at(500, 0))[0]
    hp = swarm.hp[t]
    assert swarm.hit(np.array([t]), 10.0) == pytest.approx(10 - balance.enemies["tanker"].armor)
    assert swarm.hit(np.array([t]), 2.0) == pytest.approx(1.0)
    assert swarm.hp[t] == pytest.approx(hp - 8)
    swarm.ignite(np.array([t]), 1.0, 4.0)
    swarm.burn_tick(0.5)  # горение броню не замечает
    assert swarm.hp[t] == pytest.approx(hp - 10)


def test_spitter_keeps_distance_and_spits(swarm: Swarm, balance: Balance) -> None:
    s = swarm.spawn(SPITTER, at(150, 0))[0]
    swarm.cooldown[s] = 99
    for _ in range(90):
        swarm.step(DT, ORIGIN, 22)
    assert np.linalg.norm(swarm.pos[s]) > 200  # попятился
    swarm.pos[s] = (330, 0)
    swarm.cooldown[s] = 0
    spits = [sp for _ in range(60) for sp in swarm.step(DT, ORIGIN, 22).spits]
    assert len(spits) == 1
    where, direction, damage = spits[0]
    assert direction[0] < -0.9 and damage == pytest.approx(balance.enemies["spitter"].damage)


def test_spit_hurts_the_marine_in_the_world(balance: Balance) -> None:
    w = World(balance, seed=4)
    s = w.swarm.spawn(SPITTER, at(300, 0))[0]
    w.swarm.cooldown[s] = 0
    w.spawner.budget = -1e9  # без спавна толпы
    w.weapons[0].reload_left = 1e9  # MAR молчит, иначе стрелок не доживёт до плевка
    for _ in range(90):
        w.step(1 / 30, Controls())
    assert w.damage_taken >= balance.enemies["spitter"].damage
    assert set(w.damage_by) == {"acid"}  # статистика знает, кто ранил


def test_brain_aura_speeds_up_its_neighbours(swarm: Swarm, balance: Balance) -> None:
    swarm.spawn(BRAIN, at(800, 0))
    near = swarm.spawn(WORKER, at(800, 100))[0]
    far = swarm.spawn(WORKER, at(-800, 100))[0]
    swarm.speed[[near, far]] = 100
    start = np.linalg.norm(swarm.pos[[near, far]], axis=1)
    swarm.step(0.1, ORIGIN, 22)
    moved = start - np.linalg.norm(swarm.pos[[near, far]], axis=1)
    assert swarm.buffed[near] and not swarm.buffed[far]
    assert moved[0] > moved[1] * (1 + balance.enemies["brain"].aura_bonus) * 0.95


def test_flyer_weaves_on_the_way(swarm: Swarm) -> None:
    f = swarm.spawn(FLYER, at(0, 600))[0]
    xs = []
    for _ in range(120):
        swarm.step(DT, ORIGIN, 22)
        xs.append(swarm.pos[f, 0])
    assert max(np.abs(xs)) > 10  # не по прямой
    assert swarm.pos[f, 1] < 600


def test_stun_freezes_and_knockback_respects_immunity(swarm: Swarm) -> None:
    worker = swarm.spawn(WORKER, at(200, 0))[0]
    tank = swarm.spawn(TANKER, at(-200, 0))[0]
    queen = swarm.spawn(QUEEN, at(0, 600))[0]
    swarm.stun_for(np.array([worker, queen]), 1.0)
    swarm.step(0.2, ORIGIN, 22)
    assert swarm.pos[worker, 0] == pytest.approx(200, abs=3)  # оглушённый стоит (расталкивание — мелочь)
    assert swarm.stun[queen] <= 0  # королеву не оглушить
    swarm.knock(np.array([worker, tank]), np.array([[50, 0], [-50, 0]], np.float32))
    assert swarm.pos[worker, 0] > 240
    assert swarm.pos[tank, 0] == pytest.approx(-200 + swarm.speed[tank] * 0.2, abs=5)


def test_brain_limit_in_the_world(balance: Balance) -> None:
    w = World(balance, seed=6)
    w.session.elapsed = 20 * 60
    w.stats.add("max_hp", 1e9)
    w.player.hp = w.stats["max_hp"]
    w.weapons[0].reload_left = 1e9  # не стреляем — считаем всех
    for _ in range(int(20 / (1 / 30))):
        w.step(1 / 30, Controls())
    assert 0 < w.swarm.count(BRAIN) <= balance.enemies["brain"].max_alive
