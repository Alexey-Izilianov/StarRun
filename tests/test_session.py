from __future__ import annotations

import pytest

from starrun.balance import Balance
from starrun.sim.session import Phase, Session


def make(balance: Balance) -> Session:
    return Session(balance.alert, balance.evac, balance.scaling, balance.semi_boss_minutes)


def run_until_queen(s: Session, dt: float = 0.5, on_boss: bool = False) -> list[float]:
    """Гонит время до выхода королевы; возвращает минуты, когда выходили полубоссы."""
    bosses = []
    while s.phase is Phase.WAVE and s.elapsed < 30 * 60:
        for event in s.update(dt):
            if event == "semi_boss":
                bosses.append(s.elapsed / 60)
                if on_boss:
                    s.semi_boss_killed()
    return bosses


def test_without_tasks_queen_comes_at_about_18_minutes(balance: Balance) -> None:
    s = make(balance)
    bosses = run_until_queen(s)
    assert s.phase is Phase.QUEEN
    assert s.elapsed / 60 == pytest.approx(100 / balance.alert.per_minute, abs=0.05)
    assert [round(m) for m in bosses] == [5, 10, 15]


def test_killing_semi_bosses_brings_queen_earlier(balance: Balance) -> None:
    s = make(balance)
    run_until_queen(s, on_boss=True)
    assert 12 <= s.elapsed / 60 < 17


def test_queen_never_before_minimum_even_at_full_alert(balance: Balance) -> None:
    s = make(balance)
    s.raise_alert(100)
    s.update(60.0)
    assert s.phase is Phase.WAVE
    s.update(11 * 60.0)
    assert s.phase is Phase.QUEEN
    assert s.elapsed / 60 == pytest.approx(12)


def test_enemy_scaling_by_minute(balance: Balance) -> None:
    s = make(balance)
    s.update(59.0)
    assert s.hp_mult == 1.0
    s.update(5 * 60.0)
    assert s.hp_mult == pytest.approx(1 + 5 * 0.08)
    assert s.damage_mult == pytest.approx(1 + 5 * 0.05)
    assert s.boss_hp_mult == pytest.approx(1 + 5 * 0.08 * balance.scaling.boss_share)


def test_evac_counts_only_inside_zone_then_victory(balance: Balance) -> None:
    s = make(balance)
    s.phase = Phase.QUEEN
    s.queen_killed()
    assert s.phase is Phase.EVAC
    assert not s.evac_tick(30, inside=False)
    assert s.evac_progress == 0
    assert not s.evac_tick(59, inside=True)
    assert s.evac_tick(1, inside=True)
    assert s.phase is Phase.VICTORY
    s.player_died()  # после победы смерть уже ничего не меняет
    assert s.phase is Phase.VICTORY


def test_death_ends_run(balance: Balance) -> None:
    s = make(balance)
    s.player_died()
    assert s.over and s.phase is Phase.DEFEAT
    assert s.update(10) == []
