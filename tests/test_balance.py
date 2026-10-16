from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from starrun.balance import Balance, data_dir


def test_data_loads_and_matches_design(balance: Balance) -> None:
    assert balance.enemies["worker"].hp == 10
    assert balance.enemies["warrior"].windup == pytest.approx(0.8)
    assert balance.mar.fire_rate == 8
    assert [r.bonus for r in balance.rarities[:3]] == [0.07, 0.10, 0.14]
    assert len(balance.rarities) == 6


def _copy_data(tmp_path: Path) -> Path:
    folder = tmp_path / "data"
    shutil.copytree(data_dir(), folder)
    return folder


def test_typo_in_data_is_rejected(tmp_path: Path) -> None:
    folder = _copy_data(tmp_path)
    path = folder / "weapons.toml"
    path.write_text(path.read_text(encoding="utf-8").replace("fire_rate", "fire_rat"), encoding="utf-8")
    with pytest.raises(ValueError, match="fire_rat"):
        Balance.load(folder)


def test_upgrade_with_unknown_stat_is_rejected(tmp_path: Path) -> None:
    folder = _copy_data(tmp_path)
    path = folder / "upgrades.toml"
    path.write_text(path.read_text(encoding="utf-8").replace('stat = "speed"', 'stat = "sped"'), encoding="utf-8")
    with pytest.raises(ValueError, match="sped"):
        Balance.load(folder)
