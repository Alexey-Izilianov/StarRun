import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.swarm import TANKER, WORKER
from starrun.sim.weapons import make_weapon
from starrun.sim.world import World
from starrun.ui.fx3d import Effects3D
from starrun.ui.gun_feedback import SpentCases, recoil_step


def test_spent_cases_have_a_fixed_pool_and_land_then_expire() -> None:
    cases = SpentCases(capacity=12)
    origin, direction = np.zeros(2, np.float32), np.array([0, 1], np.float32)
    for i in range(50):
        cases.emit(origin, direction, "mar", magazine=i % 2 == 0)
    assert len(cases.instances()) + len(cases.instances(magazine=True)) == 12
    for _ in range(120):
        cases.step(1 / 60)
    assert (cases.pos[:, 2] >= 1.3).all()
    assert np.abs(cases.vel[:, 2]).max() < 3
    for _ in range(120):
        cases.step(1 / 60)
    assert len(cases.instances()) == len(cases.instances(magazine=True)) == 0


def test_feedback_does_not_touch_simulation_rng(balance: Balance) -> None:
    world = World(balance, 42)
    expected = World(balance, 42)
    cases = SpentCases()
    cases.emit(world.player.pos, world.player.facing, "mar")
    assert world.rng.random() == expected.rng.random()
    cases.emit(world.player.pos, world.player.facing, "flame")
    assert len(cases.instances()) == 1  # огнемёт не выбрасывает гильзы


def test_recoil_has_distinct_weapon_impulses_and_consistent_decay() -> None:
    mar = recoil_step(0, 1 / 60, ["shot_mar"])
    shotgun = recoil_step(0, 1 / 60, ["shot_shotgun"])
    assert shotgun > mar > 0
    a, b = shotgun, shotgun
    for _ in range(30):
        a = recoil_step(a, 1 / 30, [])
    for _ in range(120):
        b = recoil_step(b, 1 / 120, [])
    assert a == pytest.approx(b)
    assert a < 1e-8


def test_reload_magazine_emits_once_and_not_again_after_switch(balance: Balance) -> None:
    world = World(balance, 42)
    fx = Effects3D.__new__(Effects3D)
    fx.rng = np.random.default_rng(7)
    fx.reset()
    fx._weapon_feedback(world)
    world.weapon.reload_left = 1
    fx._weapon_feedback(world)
    fx._weapon_feedback(world)
    world.weapons.append(make_weapon("shotgun", balance.weapons["shotgun"]))
    world.active = 1
    fx._weapon_feedback(world)
    world.active = 0
    fx._weapon_feedback(world)
    assert len(fx.cases.instances(magazine=True)) == 1


def test_armor_critical_and_normal_hits_have_distinct_bounded_feedback(balance: Balance) -> None:
    world = World(balance, 42)
    tank = world.swarm.spawn(TANKER, np.array([[100, 0]], np.float32))[0]
    workers = world.swarm.spawn(WORKER, np.array([[120, 0], [140, 0]], np.float32))
    world.deal(np.array([tank, *workers]), np.array([5, 5, 5], np.float32),
               critical=np.array([False, True, False]))
    fx = Effects3D.__new__(Effects3D)
    fx.rng = np.random.default_rng(7)
    fx.reset()
    fx._weapon_feedback(world)
    assert fx.fire.alive.sum() == 7  # три искры брони и четыре золотые частицы критического попадания
    assert fx.smoke.alive.sum() == 3
