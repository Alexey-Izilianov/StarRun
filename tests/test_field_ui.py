from unittest.mock import Mock

import pytest

from starrun.balance import Balance
from starrun.sim.progression import Offer
from starrun.sim.world import World
from starrun.ui.hud import H, Hud, SettingRow, comparison_value, effect_text


def layout_hud() -> Hud:
    hud = Hud.__new__(Hud)
    hud.s = Mock()
    hud.t = Mock()
    hud.o = Mock()
    hud.mouse = (-1, -1)
    hud.buttons = []
    return hud


@pytest.mark.parametrize("width", [1440, 1920, 2520])
def test_cards_and_reroll_fit_viewport_and_keep_mouse_callbacks(balance: Balance, width: int) -> None:
    world = World(balance, 42)
    world.progress.add(world.progress.needed)
    hud, choose, reroll = layout_hud(), Mock(), Mock()
    hud.level_up(world, width, choose, 1, reroll)
    cards = [b for b in hud.buttons if b.key.startswith("card")]
    assert len(cards) == 3
    for i, button in enumerate(cards):
        l, b, r, top = button.box  # noqa: E741
        assert 0 < l < r < width and 0 < b < top < H
        assert hud.click((l + r) / 2, (b + top) / 2)
        choose.assert_called_with(i)
    refresh = next(b for b in hud.buttons if b.key == "reroll")
    assert hud.click((refresh.box[0] + refresh.box[2]) / 2, refresh.box[1] + 10)
    reroll.assert_called_once()


@pytest.mark.parametrize("count", [7, 9])
def test_setting_arrow_targets_do_not_overlap_rows(count: int) -> None:
    hud = layout_hud()
    rows = [SettingRow(str(i), "вкл", Mock(), Mock()) for i in range(count)]
    hud.settings(1440, rows, Mock(), Mock(), "Примечание")
    arrows = [b for b in hud.buttons if b.key.startswith("st_p")]
    assert len(arrows) == count
    for previous, current in zip(arrows, arrows[1:], strict=False):
        assert current.box[3] < previous.box[1]


def test_behavioral_upgrade_labels_describe_mechanics(balance: Balance) -> None:
    upgrades = {u.id: u for u in balance.upgrades}
    expected = {"mar_ricochet": "+1 рикошет", "sniper_armor_pierce": "Раскалывает броню",
                "rocket_fire": "Горящее поле"}
    for key, text in expected.items():
        assert effect_text(Offer(upgrades[key], balance.rarities[-1], 5), balance.rarities[0].bonus) == text
    assert comparison_value("rocket_fire", 0) == "Нет"
    assert comparison_value("rocket_fire", 1) == "Да"
