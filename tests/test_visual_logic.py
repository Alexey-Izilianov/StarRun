from collections import deque

import numpy as np
import pytest

from starrun.balance import BIOME_IDS, Balance
from starrun.sim import terrain as T
from starrun.sim.world import Controls, World
from starrun.ui.level_details import placements
from starrun.ui.terrain_view import TerrainView
from starrun.ui.visibility import prop_opacity, shell_alpha


@pytest.mark.parametrize("biome", BIOME_IDS)
def test_paths_connect_buildings_without_destroying_water_or_walls(balance: Balance, biome: str) -> None:
    world = World(balance, 42, biome)
    t = world.terrain
    assert t is not None
    assert not np.isin(t.kind[t.paths], (T.TREE, T.WALL, T.WATER, T.ROCK, T.FENCE)).any()
    reached = set()
    start = (t.n // 2, t.n // 2)
    seen = {start}
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        bid = int(t.building[y, x])
        if bid >= 0:
            reached.add(bid)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = (x + dx, y + dy)
            if 0 <= p[0] < t.n and 0 <= p[1] < t.n and t.paths[p[1], p[0]] and p not in seen:
                queue.append(p)
                seen.add(p)
    assert reached == set(range(len(t.buildings)))
    a = placements(t, 0, 0, t.n, t.n)
    b = placements(t, 0, 0, t.n, t.n)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key])
        if key in ("grass", "fern", "scrub", "reed", "stone", "debris"):
            x, y = t.cells(a[key][:, :2])
            assert not t.paths[y, x].any()


def test_building_reveals_before_entry_and_closes_after_leaving(balance: Balance) -> None:
    world = World(balance, 3, "forest")
    t = world.terrain
    assert t is not None
    bd = t.buildings[0]
    view = TerrainView(None)
    world.player.pos = np.array([bd.x0 * t.cell - 75, (bd.y0 + bd.y1 + 1) * t.cell / 2])
    for _ in range(20):
        view.update(1 / 60, world)
    assert view.roof_alpha[0] < 0.15
    pos = np.array([[(bd.x0 + 0.5) * t.cell, (bd.y0 + 1.5) * t.cell]])
    assert prop_opacity(pos, world.player.pos, t, view.roof_alpha)[0] < 0.32
    world.player.pos -= 2000
    for _ in range(20):
        view.update(1 / 60, world)
    assert view.roof_alpha[0] == 1.0


def test_revealed_house_has_no_upper_shell_and_external_objects_stay_visible(balance: Balance) -> None:
    world = World(balance, 3, "forest")
    t = world.terrain
    assert t is not None
    bd = t.buildings[0]
    points = np.stack([t.cell_center(bd.x0, bd.y0), t.center])
    assert shell_alpha(points, t, {0: 0.0}).tolist() == [0.0, 1.0]
    assert shell_alpha(points, t, {0: 1.0}).tolist() == [1.0, 1.0]
    assert shell_alpha(points, t, {0: 0.5})[0] == pytest.approx(0.25)


@pytest.mark.parametrize("biome", ["forest", "desert"])
def test_chop_plays_full_cycle_and_hits_once_per_period(balance: Balance, biome: str) -> None:
    world = World(balance, 2, biome)
    t = world.terrain
    assert t is not None
    t.kind[:] = T.GROUND
    t.soil[:] = T.PLAIN
    world.player.pos = t.cell_center(100, 100)
    t.kind[100, 101] = T.TREE
    t.hp[100, 101] = 100
    phases, hits = [], 0
    for _ in range(96):
        world.step(1 / 120, Controls((1, 0)))
        phases.append(world.player.chop)
        hits += world.events.count("chop")
    assert min(phases[:10]) < 0.15 and max(phases) > 0.9
    assert hits == 2
