from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.spawner import Spawner, edge_positions, front_side

HALF = (960.0, 540.0)
CENTER = np.array([500.0, -300.0], np.float32)


def test_spawn_points_are_beyond_screen_edge(rng: np.random.Generator) -> None:
    pos = edge_positions(rng, 2000, CENTER, HALF, margin=80)
    rel = np.abs(pos - CENTER)
    outside = (rel[:, 0] >= HALF[0] + 79) | (rel[:, 1] >= HALF[1] + 79)
    assert outside.all()
    assert (rel[:, 0] <= HALF[0] + 81).all() and (rel[:, 1] <= HALF[1] + 81).all()


def test_rush_comes_from_one_side(rng: np.random.Generator) -> None:
    pos = edge_positions(rng, 300, CENTER, HALF, margin=80, side="left")
    assert np.allclose(pos[:, 0], CENTER[0] - HALF[0] - 80)


def test_population_ramps_to_full_and_respects_cap(balance: Balance, rng: np.random.Generator) -> None:
    sp = Spawner(balance.spawn, rng)
    assert sp.target_population(0) == balance.spawn.start_population
    assert sp.target_population(balance.spawn.full_minute * 60) == balance.spawn.full_population
    assert sp.target_population(10_000) <= balance.spawn.cap


def test_warriors_only_after_their_minute(balance: Balance, rng: np.random.Generator) -> None:
    sp = Spawner(balance.spawn, rng)
    assert sp.warrior_share(60) == 0
    assert sp.warrior_share(balance.spawn.warrior_share_minute * 60) == pytest.approx(balance.spawn.warrior_share_max)


def test_spawn_rate_is_limited_and_tops_up_population(balance: Balance, rng: np.random.Generator) -> None:
    sp = Spawner(balance.spawn, rng)
    alive, t, dt = 0, 0.0, 1 / 60
    first = sum(len(p) for _, p in sp.update(dt, t, alive, CENTER, HALF))
    assert first <= balance.spawn.max_per_second * dt + 1
    for _ in range(600):  # 10 секунд
        t += dt
        alive += sum(len(p) for _, p in sp.update(dt, t, alive, CENTER, HALF))
    assert alive == sp.target_population(t)


def test_rush_happens_once_per_minute(balance: Balance, rng: np.random.Generator) -> None:
    sp = Spawner(balance.spawn, rng)
    crowd = 300  # больше целевой численности первых минут — обычный спавн молчит, остаётся только налёт
    sizes = [sum(len(p) for _, p in sp.update(0.1, step / 10, crowd, CENTER, HALF)) for step in range(120 * 10)]
    rushes = [s for s in sizes if s >= balance.spawn.rush_base]
    assert len(rushes) == 2


def test_front_side_follows_movement(rng: np.random.Generator) -> None:
    assert front_side(rng, np.array([1.0, 0.0])) == "right"
    assert front_side(rng, np.array([0.0, -1.0])) == "bottom"
    sides = {front_side(rng, np.array([0.7, 0.7])) for _ in range(200)}
    assert sides == {"right", "top"}
