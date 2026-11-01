from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import arcade
import pytest

from starrun.app import Game
from starrun.balance import Balance
from starrun.sim.progression import Offer, Stats, Upgrades
from starrun.sim.world import World
from starrun.ui.choices import ChoiceCursor


@pytest.mark.parametrize(("key", "index"), [(arcade.key.A, 2), (arcade.key.LEFT, 2), (arcade.key.W, 2),
    (arcade.key.UP, 2), (arcade.key.D, 1), (arcade.key.RIGHT, 1), (arcade.key.S, 1), (arcade.key.DOWN, 1)])
def test_game_navigates_cards_without_applying_them(balance: Balance, key: int, index: int) -> None:
    world = World(balance, 42)
    world.progress.add(world.progress.needed)
    game = SimpleNamespace(held=set(), world=world, choice_cursor=ChoiceCursor(), audio=Mock(), choose=Mock())
    Game.on_key_press(game, key, 0)
    assert game.choice_cursor.index == index
    game.choose.assert_not_called()
    Game.on_key_press(game, key, 0)  # повтор зажатой клавиши не перескакивает ещё раз
    assert game.choice_cursor.index == index


@pytest.mark.parametrize("key", [arcade.key.ENTER, arcade.key.SPACE])
def test_confirm_uses_focused_card(balance: Balance, key: int) -> None:
    world = World(balance, 42)
    world.progress.add(world.progress.needed)
    game = SimpleNamespace(held=set(), world=world, choice_cursor=ChoiceCursor(2), choose=Mock())
    Game.on_key_press(game, key, 0)
    game.choose.assert_called_once_with(2)


def test_rerolls_preserve_pending_levels_and_stats_and_have_budget(balance: Balance) -> None:
    world = World(balance, 42)
    assert not world.reroll()
    world.progress.add(world.progress.needed * 5)
    pending = world.progress.pending
    for _ in range(balance.xp.rerolls):
        previous = {o.upgrade.id for o in world.offer()}
        assert world.reroll()
        assert previous.isdisjoint({o.upgrade.id for o in world.offer()})
        assert {o.upgrade.group for o in world.offer()} == {"weapon", "passive"}
        assert world.progress.pending == pending
        assert world.upgrades.taken == []
    assert world.rerolls_left == 0 and not world.reroll()
    world.choose(0)
    assert world.progress.pending == pending - 1
    assert world.rerolls_left == 0


def test_preview_matches_apply_and_capped_stats_are_excluded(balance: Balance) -> None:
    stats = Stats(balance.player.stats, balance.player.caps)
    upgrades = Upgrades(balance.upgrades, balance.rarities)
    offer = Offer(next(u for u in balance.upgrades if u.id == "max_hp"), balance.rarities[2], 2)
    before, after = stats.preview(offer, balance.rarities[0].bonus)
    assert stats["max_hp"] == before
    upgrades.apply(offer, stats)
    assert stats["max_hp"] == after
    stats.add("dodge", 0.59)
    dodge = Offer(next(u for u in balance.upgrades if u.id == "dodge"), balance.rarities[-1], 5)
    assert stats.preview(dodge, balance.rarities[0].bonus)[1] == 0.6
    upgrades.apply(dodge, stats)
    assert not any(u.stat == "dodge" for u in upgrades.available(stats=stats))


def test_reroll_retains_only_remaining_passive(balance: Balance) -> None:
    passive = next(u for u in balance.upgrades if u.group == "passive")
    pool = tuple(u for u in balance.upgrades if u.group == "weapon" and not u.weapon) + (passive,)
    upgrades = Upgrades(pool, balance.rarities)
    import numpy as np

    cards = upgrades.offer(np.random.default_rng(1), exclude={passive.id})
    assert any(o.upgrade.id == passive.id for o in cards)


def test_holding_confirm_does_not_spend_multiple_levels(balance: Balance) -> None:
    world = World(balance, 42)
    world.progress.add(world.progress.needed * 5)
    game = SimpleNamespace(held={arcade.key.SPACE, arcade.key.W}, world=world,
                           choice_cursor=ChoiceCursor(), _clear_actions=Mock())
    game.choose = lambda i: Game.choose(game, i)
    pending = world.progress.pending
    Game.choose(game, 0)
    Game.on_key_press(game, arcade.key.SPACE, 0)
    assert world.progress.pending == pending - 1
    assert arcade.key.W not in game.held
    Game.on_key_release(game, arcade.key.SPACE, 0)
    Game.on_key_press(game, arcade.key.SPACE, 0)
    assert world.progress.pending == pending - 2


def test_negative_reroll_budget_is_rejected(balance: Balance) -> None:
    with pytest.raises(ValueError, match="rerolls"):
        replace(balance, xp=replace(balance.xp, rerolls=-1)).validate()
