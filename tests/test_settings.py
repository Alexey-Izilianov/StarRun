from __future__ import annotations

from pathlib import Path

from starrun.settings import Settings, resolutions


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    Settings(1600, 900, False, False).save(path)
    assert Settings.load(path) == Settings(1600, 900, False, False)


def test_broken_or_foreign_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    assert Settings.load(path) == Settings()                       # файла нет
    path.write_text("{не json", encoding="utf-8")
    assert Settings.load(path) == Settings()
    path.write_text('{"width": "big", "fullscreen": false, "volume": 3}', encoding="utf-8")
    assert Settings.load(path) == Settings(fullscreen=False)        # неверный тип и чужой ключ — мимо


def test_resolution_list() -> None:
    modes = [(800, 600), (1280, 720), (1920, 1080), (1920, 1080), (2560, 1440), (1600, 1000)]
    assert resolutions(modes, (1920, 1200)) == [(1280, 720), (1600, 1000), (1920, 1080), (1920, 1200)]
