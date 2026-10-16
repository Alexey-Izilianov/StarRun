"""Баланс из data/*.toml: все числа игры живут в данных, код читает их отсюда."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

Color = tuple[int, int, int]


def data_dir() -> Path:
    """Папка data/: рядом с исходниками при разработке, внутри сборки — в каталоге распаковки."""
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "data"
    return Path(__file__).resolve().parents[2] / "data"


def _build[T](cls: type[T], raw: dict[str, Any]) -> T:
    """Собирает dataclass из таблицы TOML; неизвестный ключ — ошибка (опечатка в данных не пройдёт молча)."""
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные ключи {sorted(unknown)}")
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return cls(**values)


@dataclass(frozen=True)
class Stamina:
    max: float
    drain: float
    regen: float
    regen_delay: float


@dataclass(frozen=True)
class PlayerCfg:
    stats: dict[str, float]
    caps: dict[str, float]
    stamina: Stamina


@dataclass(frozen=True)
class EnemyCfg:
    name: str
    hp: float
    speed: float
    damage: float
    radius: float
    attack_cooldown: float
    xp: int
    color: Color
    windup: float = 0.0
    reach: float = 0.0
    dash_every: float = 0.0
    dash_speed: float = 0.0
    dash_time: float = 0.0
    summon_every: float = 0.0
    summon_count: int = 0
    volley_every: float = 0.0
    volley_count: int = 0
    volley_spread_deg: float = 0.0
    acid_speed: float = 0.0
    acid_damage: float = 0.0


@dataclass(frozen=True)
class Scaling:
    hp_per_minute: float
    damage_per_minute: float
    boss_share: float


@dataclass(frozen=True)
class WeaponCfg:
    name: str
    damage: float
    fire_rate: float
    range: float
    bullet_speed: float
    spread_deg: float
    magazine: int
    reload: float


@dataclass(frozen=True)
class OrbitalCfg:
    name: str
    delay: float
    radius: float
    damage: float
    cooldown: float
    max_range: float


@dataclass(frozen=True)
class AlertCfg:
    per_minute: float
    semi_boss_kill: float
    queen_min_minute: float


@dataclass(frozen=True)
class EvacCfg:
    seconds: float
    distance: float
    radius: float


@dataclass(frozen=True)
class SpawnCfg:
    start_population: int
    full_population: int
    full_minute: float
    cap: int
    max_per_second: float
    margin: float
    despawn_distance: float
    rush_start: float
    rush_length: float
    rush_base: int
    rush_per_minute: int
    warrior_from_minute: float
    warrior_share_max: float
    warrior_share_minute: float


@dataclass(frozen=True)
class XpCfg:
    base: float
    linear: float
    quad: float


@dataclass(frozen=True)
class Rarity:
    name: str
    color: Color
    bonus: float
    weight: float


@dataclass(frozen=True)
class Upgrade:
    id: str
    name: str
    text: str
    stat: str
    mode: str
    group: str
    value: float = 0.0
    integer: bool = False
    max_rank: int = 0  # 0 — без ограничения


@dataclass(frozen=True)
class Balance:
    player: PlayerCfg
    enemies: dict[str, EnemyCfg]
    scaling: Scaling
    mar: WeaponCfg
    orbital: OrbitalCfg
    alert: AlertCfg
    semi_boss_minutes: tuple[float, ...]
    evac: EvacCfg
    spawn: SpawnCfg
    xp: XpCfg
    rarities: tuple[Rarity, ...]
    upgrades: tuple[Upgrade, ...] = field(default=())

    @classmethod
    def load(cls, folder: Path | None = None) -> Self:
        folder = folder or data_dir()

        def read(name: str) -> dict[str, Any]:
            with open(folder / name, "rb") as f:
                return tomllib.load(f)

        player, enemies, weapons, session, upgrades = (
            read(n) for n in ("player.toml", "enemies.toml", "weapons.toml", "session.toml", "upgrades.toml"))
        scaling = enemies.pop("scaling")
        balance = cls(
            player=PlayerCfg(stats={k: float(v) for k, v in player["stats"].items()},
                             caps={k: float(v) for k, v in player["caps"].items()},
                             stamina=_build(Stamina, player["stamina"])),
            enemies={k: _build(EnemyCfg, v) for k, v in enemies.items()},
            scaling=_build(Scaling, scaling),
            mar=_build(WeaponCfg, weapons["mar"]),
            orbital=_build(OrbitalCfg, weapons["orbital"]),
            alert=_build(AlertCfg, session["alert"]),
            semi_boss_minutes=tuple(session["bosses"]["semi_boss_minutes"]),
            evac=_build(EvacCfg, session["evac"]),
            spawn=_build(SpawnCfg, session["spawn"]),
            xp=_build(XpCfg, session["xp"]),
            rarities=tuple(_build(Rarity, r) for r in session["rarity"]),
            upgrades=tuple(_build(Upgrade, u) for u in upgrades["upgrade"]),
        )
        balance.validate()
        return balance

    def validate(self) -> None:
        """Связи между файлами, которые TOML сам не проверит."""
        for kind in ("worker", "warrior", "semi_boss", "queen"):
            if kind not in self.enemies:
                raise ValueError(f"enemies.toml: нет жука {kind}")
        for up in self.upgrades:
            if up.stat not in self.player.stats:
                raise ValueError(f"upgrades.toml: {up.id} меняет неизвестную характеристику {up.stat}")
            if up.mode not in ("mult", "add") or up.group not in ("weapon", "passive"):
                raise ValueError(f"upgrades.toml: {up.id} — неверный mode или group")
            if up.mode == "add" and up.value <= 0:
                raise ValueError(f"upgrades.toml: {up.id} — для mode=add нужен value > 0")
        if len({u.id for u in self.upgrades}) != len(self.upgrades):
            raise ValueError("upgrades.toml: повторяющиеся id")
        if not self.rarities or self.rarities[0].bonus <= 0:
            raise ValueError("session.toml: первая редкость должна давать бонус > 0")
