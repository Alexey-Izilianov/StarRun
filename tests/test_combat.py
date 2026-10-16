from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.combat import Bullets, Gun, Orbital, fan
from starrun.sim.effects import Gems
from starrun.sim.progression import Stats
from starrun.sim.swarm import QUEEN, SEMI_BOSS, WARRIOR, WORKER, Swarm

ORIGIN = np.zeros(2, np.float32)


@pytest.fixture
def swarm(balance: Balance, rng: np.random.Generator) -> Swarm:
    return Swarm(balance.enemies, 64, rng)


@pytest.fixture
def stats(balance: Balance) -> Stats:
    return Stats(balance.player.stats, balance.player.caps)


def at(*xy: float) -> np.ndarray:
    return np.array([xy], np.float32)


def test_nearest_respects_range(swarm: Swarm) -> None:
    swarm.spawn(WORKER, at(300, 0))
    near = swarm.spawn(WORKER, at(0, 200))[0]
    assert swarm.nearest(ORIGIN, 600) == near
    assert swarm.nearest(ORIGIN, 150) is None


def test_worker_bites_on_contact_with_cooldown(swarm: Swarm, balance: Balance) -> None:
    i = swarm.spawn(WORKER, at(30, 0))[0]
    swarm.cooldown[i] = 0
    hits = [h for _ in range(60) for h in swarm.step(1 / 60, ORIGIN, 22).hits]
    assert hits == [balance.enemies["worker"].damage]  # за секунду ровно один укус
    assert np.linalg.norm(swarm.pos[i]) >= 22 + 14 - 0.01  # на десантника не наползает


def test_warrior_swing_can_be_dodged(swarm: Swarm, balance: Balance) -> None:
    w = balance.enemies["warrior"]
    i = swarm.spawn(WARRIOR, at(60, 0))[0]
    swarm.cooldown[i] = 0
    swarm.step(1 / 60, ORIGIN, 22)
    assert swarm.windup[i] > 0
    far = np.array([400, 0], np.float32)  # десантник отбежал, пока воин замахивался
    hits = [h for _ in range(int(w.windup * 60) + 2) for h in swarm.step(1 / 60, far, 22).hits]
    assert hits == []

    j = swarm.spawn(WARRIOR, at(-60, 0))[0]
    swarm.alive[i] = False
    swarm.cooldown[j] = 0
    hits = [h for _ in range(int(w.windup * 60) + 2) for h in swarm.step(1 / 60, ORIGIN, 22).hits]
    assert hits == [w.damage]


def test_bullets_kill_and_pierce(swarm: Swarm) -> None:
    a = swarm.spawn(WORKER, at(100, 0))[0]
    b = swarm.spawn(WORKER, at(160, 0))[0]
    bullets = Bullets()
    bullets.fire(ORIGIN, at(1, 0), 1100, np.array([10], np.float32), pierce=1, life=1)
    hit = []
    for _ in range(20):
        idx, dmg = bullets.step(1 / 120, swarm)
        swarm.hit(idx, dmg)
        hit += idx.tolist()
    assert hit == [a, b]
    assert set(swarm.collect_dead().tolist()) == {a, b}
    assert not bullets.alive.any()


def test_gun_magazine_and_reload(balance: Balance, stats: Stats) -> None:
    gun = Gun(balance.mar)
    shots = sum(gun.update(1 / 60, stats, True) for _ in range(600))  # 10 с огня
    mag, rate, reload = balance.mar.magazine, balance.mar.fire_rate, balance.mar.reload
    cycle = mag / rate + reload
    expected = mag * int(10 // cycle) + min(mag, int((10 % cycle) * rate) + 1)
    assert abs(shots - expected) <= 2
    gun2 = Gun(balance.mar)
    assert sum(gun2.update(1 / 60, stats, False) for _ in range(60)) == 0  # без цели не стреляет


def test_orbital_strike_waits_then_hits_area(balance: Balance, stats: Stats, swarm: Swarm) -> None:
    orb = Orbital(balance.orbital)
    assert orb.trigger(np.array([5000, 0], np.float32), ORIGIN, stats)
    assert orb.marks[0].pos[0] == pytest.approx(balance.orbital.max_range)  # дальше предела не ставится
    assert not orb.trigger(ORIGIN, ORIGIN, stats)  # перезарядка
    assert orb.update(balance.orbital.delay - 0.01) == []
    fired = orb.update(0.02)
    assert len(fired) == 1
    swarm.spawn(WORKER, at(800 + 100, 0))
    swarm.spawn(WORKER, at(800 + 400, 0))
    assert len(swarm.in_radius(fired[0].pos, fired[0].radius)) == 1


def test_bosses_dash_summon_and_spit(swarm: Swarm, balance: Balance) -> None:
    boss = swarm.spawn(SEMI_BOSS, at(500, 0))[0]
    queen = swarm.spawn(QUEEN, at(-900, 0))[0]
    summons, volleys, dashed = 0, 0, False
    for _ in range(int(11 * 30)):
        step = swarm.step(1 / 30, ORIGIN, 22)
        summons += len(step.summons)
        volleys += len(step.volleys)
        dashed |= bool(swarm.dash[boss] > 0)
    assert dashed
    assert summons >= 2  # и полубосс, и королева позвали подмогу
    assert volleys >= 2
    assert swarm.alive[queen]


def test_fan_spreads_evenly() -> None:
    dirs = fan(np.array([1, 0], np.float32), 5, 40)
    angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
    assert np.allclose(angles, [-20, -10, 0, 10, 20], atol=1e-4)


def test_gems_pull_and_overflow_keeps_xp(rng: np.random.Generator) -> None:
    gems = Gems(capacity=3)
    gems.drop(rng, np.array([[100, 0], [300, 0], [2000, 0], [2000, 50]], np.float32), np.array([1, 1, 1, 5]))
    assert gems.value[gems.alive].sum() == 8  # четвёртому места нет, опыт прибавился к лежащим
    got = sum(gems.step(1 / 60, ORIGIN, pickup_radius=150) for _ in range(60))
    assert got >= 1  # ближний долетел
    assert gems.alive.sum() == 2  # дальние лежат, пока десантник не подойдёт
