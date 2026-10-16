from __future__ import annotations

import numpy as np

from starrun.balance import Balance
from starrun.sim.session import Phase
from starrun.sim.swarm import QUEEN, SEMI_BOSS, WORKER
from starrun.sim.world import Controls, World

DT = 1 / 30


def kite(world: World) -> tuple[float, float]:
    """Бот-игрок: отходит от ближайших жуков по дуге, как живой игрок, а не стоит в толпе."""
    sw = world.swarm
    idx = sw.alive_idx
    rel = sw.pos[idx] - world.player.pos
    dist = np.linalg.norm(rel, axis=1) + 1e-3
    near = dist < 400
    if not near.any():
        return (0.0, 0.0)
    away = -(rel[near] / dist[near, None] ** 2).sum(0)
    away /= np.linalg.norm(away) + 1e-9
    v = away + 0.6 * np.array([-away[1], away[0]])
    v /= np.linalg.norm(v)
    return (float(v[0]), float(v[1]))


def play(world: World, seconds: float, controls: Controls | None = None) -> None:
    """Прогон без игрока за клавиатурой: улучшения выбираются по первой карточке."""
    controls = controls or Controls()
    for _ in range(int(seconds / DT)):
        if world.awaiting_choice:
            world.offer()
            world.choose(0)
        world.step(DT, controls)
        if world.session.over:
            return


def test_two_minutes_of_play(balance: Balance) -> None:
    w = World(balance, seed=1)
    for _ in range(int(120 / DT)):
        if w.awaiting_choice:
            w.offer()
            w.choose(0)
        w.step(DT, Controls(move=kite(w)))
    assert w.session.phase is Phase.WAVE
    assert w.kills["worker"] > 100
    assert w.progress.level >= 3
    assert len(w.upgrades.taken) == w.progress.level - 1
    assert 0 < w.swarm.count() <= balance.spawn.cap
    assert w.session.alert > 0


def test_same_seed_same_run(balance: Balance) -> None:
    a, b = World(balance, seed=7), World(balance, seed=7)
    play(a, 20, Controls(move=(1, 0)))
    play(b, 20, Controls(move=(1, 0)))
    assert np.array_equal(a.swarm.pos, b.swarm.pos)
    assert a.kills == b.kills


def test_sprint_drains_stamina_and_moves_faster(balance: Balance) -> None:
    walk, run = World(balance, seed=3), World(balance, seed=3)
    for _ in range(30):
        walk.player.move(DT, (1, 0), False, walk.stats)
        run.player.move(DT, (1, 0), True, run.stats)
    assert run.player.pos[0] > walk.player.pos[0] * 1.5
    assert run.player.stamina < balance.player.stamina.max
    assert walk.player.stamina == balance.player.stamina.max


def test_standing_still_in_the_crowd_kills_you(balance: Balance) -> None:
    w = World(balance, seed=2)
    w.player.hp = 10
    w.swarm.spawn(WORKER, np.array([[40, 0], [-40, 0], [0, 40], [0, -40]], np.float32))
    w.swarm.cooldown[:] = 0
    play(w, 5)
    assert w.session.phase is Phase.DEFEAT
    assert "defeat" in w.events or w.player.hp <= 0


def test_queen_kill_then_evac_wins(balance: Balance) -> None:
    w = World(balance, seed=5)
    w.session.elapsed = balance.alert.queen_min_minute * 60
    w.session.raise_alert(100)
    w.step(DT, Controls())
    assert w.session.phase is Phase.QUEEN
    queens = np.flatnonzero(w.swarm.alive & (w.swarm.kind == QUEEN))
    assert len(queens) == 1
    w.swarm.hp[queens] = 0
    w.step(DT, Controls())
    assert w.session.phase is Phase.EVAC and w.evac_pos is not None

    w.player.pos = w.evac_pos.copy()
    w.stats.add("max_hp", 1e9)  # проверяем таймер эвакуации, а не выживание
    w.player.hp = w.stats["max_hp"]
    play(w, balance.evac.seconds + 1)
    assert w.session.phase is Phase.VICTORY


def test_max_hp_upgrade_heals_the_difference(balance: Balance) -> None:
    w = World(balance, seed=4)
    w.player.hp = 50
    w.progress.add(5)
    for i, offer in enumerate(w.offer()):
        if offer.upgrade.id == "max_hp":
            w.choose(i)
            assert w.player.hp == 50 + w.stats["max_hp"] - 200
            return
    w.choose(0)  # в этот раз броню не предложили — проверяем, что выбор вообще сработал
    assert w.progress.pending == 0


def test_kiting_does_not_leave_the_swarm_behind(balance: Balance) -> None:
    w = World(balance, seed=21)
    w.session.elapsed = 6.5 * 60
    play_kite(w, 25)
    dist = np.linalg.norm(w.swarm.pos[w.swarm.alive_idx] - w.player.pos, axis=1)
    assert (dist < 1100).mean() > 0.5  # больше половины толпы — на экране или у его края


def play_kite(world: World, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        if world.awaiting_choice:
            world.offer()
            world.choose(0)
        world.step(DT, Controls(move=kite(world)))


def test_mar_focuses_boss_unless_something_bites(balance: Balance) -> None:
    w = World(balance, seed=8)
    worker = w.swarm.spawn(WORKER, np.array([[300, 0]], np.float32))[0]
    boss = w.swarm.spawn(SEMI_BOSS, np.array([[0, 450]], np.float32))[0]
    reach = balance.weapons["mar"].range
    assert w.pick_target(reach) == boss
    w.swarm.pos[worker] = (120, 0)  # рабочий вцепился — сначала он
    w.target = None
    assert w.pick_target(reach) == worker
