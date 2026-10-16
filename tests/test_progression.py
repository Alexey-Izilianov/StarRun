from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.progression import Offer, Progress, Stats, Upgrades, roll_rarity, xp_to_next


def test_xp_curve_grows_and_gives_about_thirty_levels(balance: Balance) -> None:
    needs = [xp_to_next(level, balance.xp) for level in range(1, 40)]
    assert needs[0] == 5
    assert all(b > a for a, b in zip(needs, needs[1:], strict=False))
    # ~30 уровней за забег: суммарно несколько тысяч опыта, не десятки тысяч
    assert 3_000 < sum(needs[:30]) < 8_000


def test_progress_can_gain_several_levels_at_once(balance: Balance) -> None:
    p = Progress(balance.xp)
    gained = p.add(xp_to_next(1, balance.xp) + xp_to_next(2, balance.xp) + 2)  # 2 уровня и 2 опыта в остатке
    assert gained == 2
    assert (p.level, p.xp, p.pending) == (3, 2, 2)


def test_stats_combine_add_mult_and_cap() -> None:
    s = Stats({"speed": 100.0, "dodge": 0.0}, {"dodge": 0.6})
    s.add("speed", 20)
    s.multiply("speed", 1.1)
    s.multiply("speed", 1.1)
    assert s["speed"] == pytest.approx(120 * 1.21)
    s.add("dodge", 0.9)
    assert s["dodge"] == 0.6


def test_offer_is_three_distinct_cards_with_weapon_and_passive(balance: Balance, rng: np.random.Generator) -> None:
    ups = Upgrades(balance.upgrades, balance.rarities)
    for _ in range(200):
        cards = ups.offer(rng)
        ids = [c.upgrade.id for c in cards]
        assert len(ids) == 3 == len(set(ids))
        groups = {c.upgrade.group for c in cards}
        assert groups == {"weapon", "passive"}


def test_rarity_rolls_follow_weights(balance: Balance, rng: np.random.Generator) -> None:
    rolls = Counter(roll_rarity(rng, balance.rarities) for _ in range(20_000))
    assert rolls[0] / 20_000 == pytest.approx(0.55, abs=0.02)
    assert rolls[5] > 0  # мифическое выпадает, хоть и редко
    assert rolls[0] > rolls[1] > rolls[2] > rolls[3] > rolls[4] > rolls[5]


def test_apply_mult_and_add_upgrades(balance: Balance) -> None:
    ups = Upgrades(balance.upgrades, balance.rarities)
    stats = Stats(balance.player.stats, balance.player.caps)
    by_id = {u.id: u for u in balance.upgrades}
    blue, purple = balance.rarities[2], balance.rarities[3]

    ups.apply(Offer(by_id["speed"], blue, 2), stats)
    assert stats["speed"] == pytest.approx(balance.player.stats["speed"] * 1.14)

    ups.apply(Offer(by_id["regen"], blue, 2), stats)  # add: 0.6 × (0.14 / 0.07)
    assert stats["regen"] == pytest.approx(1.2)

    ups.apply(Offer(by_id["mar_pierce"], purple, 3), stats)  # целое, округляется вверх
    assert stats["mar_pierce"] == 3


def test_upgrade_disappears_after_max_rank(balance: Balance, rng: np.random.Generator) -> None:
    ups = Upgrades(balance.upgrades, balance.rarities)
    stats = Stats(balance.player.stats, balance.player.caps)
    pierce = next(u for u in balance.upgrades if u.id == "mar_pierce")
    for _ in range(pierce.max_rank):
        ups.apply(Offer(pierce, balance.rarities[0], 0), stats)
    assert pierce not in ups.available()
    assert all(c.upgrade.id != "mar_pierce" for _ in range(100) for c in ups.offer(rng))
