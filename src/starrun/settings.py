"""Настройки игрока: разрешение, полный экран или окно, сглаживание. Хранятся в %APPDATA%\\StarRun\\settings.json;
битый или устаревший файл не мешает запуску — берутся значения по умолчанию."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

MIN_SIZE = (1280, 720)     # меньше — интерфейс 1080p не помещается


def settings_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "StarRun" / "settings.json"


@dataclass
class Settings:
    width: int = 0              # 0 — разрешение экрана
    height: int = 0
    fullscreen: bool = True
    antialiasing: bool = True   # MSAA ×4; применяется после перезапуска (задаётся при создании окна)
    volume: int = 8             # громкость 0..10: общая и по видам
    sfx_volume: int = 10
    voice_volume: int = 10
    ambient_volume: int = 7

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or settings_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        types = {f.name: {"int": int, "bool": bool}[str(f.type)] for f in fields(cls)}
        # чужие ключи и неверные типы пропускаются — файл мог править человек или старая версия
        values: dict[str, Any] = {k: v for k, v in raw.items() if k in types and type(v) is types[k]}
        return cls(**values)

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")


def resolutions(modes: list[tuple[int, int]], screen: tuple[int, int]) -> list[tuple[int, int]]:
    """Разрешения для выбора: режимы монитора не меньше MIN_SIZE и не больше экрана, без повторов, по возрастанию;
    разрешение экрана — всегда в списке."""
    good = {(w, h) for w, h in modes if w >= MIN_SIZE[0] and h >= MIN_SIZE[1] and w <= screen[0] and h <= screen[1]}
    good.add(screen)
    return sorted(good, key=lambda r: (r[0] * r[1], r[0]))
