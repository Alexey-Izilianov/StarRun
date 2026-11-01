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
    windup: float = 0.0             # замах воина / раздувание мешка стрелка
    reach: float = 0.0              # досягаемость удара воина
    attack_range: float = 0.0       # дальность плевка стрелка
    keep_distance: float = 0.0      # держится на этой дистанции от десантника
    projectile_speed: float = 0.0
    flock_min: int = 0              # прилетает стаями
    flock_max: int = 0
    flying: bool = False
    armor: float = 0.0              # минус к урону каждого попадания
    knockback_immune: bool = False
    aura_radius: float = 0.0        # аура мозгового
    aura_bonus: float = 0.0
    max_alive: int = 0              # 0 — без ограничения
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
    """Один ствол. Общие поля есть у всех, особые — только у тех, кому нужны (иначе 0)."""

    name: str
    text: str
    damage: float
    fire_rate: float
    range: float
    mod_name: str
    mod_text: str
    bullet_speed: float = 0.0
    spread_deg: float = 0.0
    magazine: int = 0               # 0 — без магазина
    reload: float = 0.0
    pellets: int = 1
    cone_deg: float = 0.0
    pierce: int = 0
    radius: float = 0.0             # радиус взрыва
    knockback: float = 0.0
    burn_time: float = 0.0
    burn_dps: float = 0.0
    chain: int = 0
    jump: float = 0.0
    life: float = 0.0
    deploy_every: float = 0.0
    max_active: int = 0
    mod_every: float = 0.0
    mod_damage: float = 0.0
    mod_radius: float = 0.0
    mod_time: float = 0.0
    mod_count: int = 0
    mod_found: bool = False         # модификацию не выбрать карточкой — только найти на карте
    ricochet_range: float = 0.0
    ricochet_damage: float = 1.0
    field_time: float = 0.0
    field_dps: float = 0.0
    field_radius_share: float = 0.0


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


@dataclass(frozen=True)
class MixCfg:
    """Доля вида в толпе: 0 до from_minute, линейно до share к full_minute."""

    kind: str
    from_minute: float
    full_minute: float
    share: float


@dataclass(frozen=True)
class XpCfg:
    base: float
    linear: float
    quad: float
    rerolls: int = 3


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
    mode: str                       # mult, add; weapon и mod — карточки нового ствола и модификации
    group: str
    value: float = 0.0
    integer: bool = False
    max_rank: int = 0               # 0 — без ограничения
    weapon: str = ""                # улучшение конкретного ствола
    fixed: bool = False              # поведенческий эффект не масштабируется редкостью


@dataclass(frozen=True)
class MapCfg:
    size: int
    cell: float
    border: int
    start_clear: int
    flow_radius: tuple[int, int]
    flow_every: float
    chop_every: float
    door_hp: float
    window_bug_speed: float
    vault_speed: float
    mud_speed: float
    sand_bug_speed: float
    sand_sink_rate: float
    sand_recover: float
    sand_damage: float
    cactus_damage: float
    cactus_every: float
    evac_clear: int


@dataclass(frozen=True)
class ItemsCfg:
    medkit_heal: float
    dna_value: int
    weapon_xp: int
    medkits: int
    magnets: int
    dna: int


@dataclass(frozen=True)
class BiomeCfg:
    name: str
    ground: Color
    ground_alt: Color
    tree: str
    tree_color: Color
    tree_hp: int
    forest: float
    forest_scale: float
    buildings: int
    rivers: int
    quicksand: int
    mud: float


@dataclass(frozen=True)
class Balance:
    player: PlayerCfg
    enemies: dict[str, EnemyCfg]
    scaling: Scaling
    weapons: dict[str, WeaponCfg]
    orbital: OrbitalCfg
    alert: AlertCfg
    semi_boss_minutes: tuple[float, ...]
    evac: EvacCfg
    spawn: SpawnCfg
    mix: tuple[MixCfg, ...]
    xp: XpCfg
    rarities: tuple[Rarity, ...]
    map: MapCfg
    items: ItemsCfg
    biomes: dict[str, BiomeCfg]
    upgrades: tuple[Upgrade, ...] = field(default=())

    @classmethod
    def load(cls, folder: Path | None = None) -> Self:
        folder = folder or data_dir()

        def read(name: str) -> dict[str, Any]:
            with open(folder / name, "rb") as f:
                return tomllib.load(f)

        player, enemies, weapons, session, upgrades, biomes = (
            read(n) for n in ("player.toml", "enemies.toml", "weapons.toml", "session.toml", "upgrades.toml",
                              "biomes.toml"))
        map_cfg, items = biomes.pop("map"), biomes.pop("items")
        scaling = enemies.pop("scaling")
        orbital = weapons.pop("orbital")
        balance = cls(
            player=PlayerCfg(stats={k: float(v) for k, v in player["stats"].items()},
                             caps={k: float(v) for k, v in player["caps"].items()},
                             stamina=_build(Stamina, player["stamina"])),
            enemies={k: _build(EnemyCfg, v) for k, v in enemies.items()},
            scaling=_build(Scaling, scaling),
            weapons={k: _build(WeaponCfg, v) for k, v in weapons.items()},
            orbital=_build(OrbitalCfg, orbital),
            alert=_build(AlertCfg, session["alert"]),
            semi_boss_minutes=tuple(session["bosses"]["semi_boss_minutes"]),
            evac=_build(EvacCfg, session["evac"]),
            spawn=_build(SpawnCfg, session["spawn"]),
            mix=tuple(_build(MixCfg, m) for m in session["mix"]),
            xp=_build(XpCfg, session["xp"]),
            rarities=tuple(_build(Rarity, r) for r in session["rarity"]),
            map=_build(MapCfg, map_cfg),
            items=_build(ItemsCfg, items),
            biomes={k: _build(BiomeCfg, v) for k, v in biomes.items()},
            upgrades=tuple(_build(Upgrade, u) for u in upgrades["upgrade"]),
        )
        balance.validate()
        return balance

    def validate(self) -> None:
        """Связи между файлами, которые TOML сам не проверит."""
        if type(self.xp.rerolls) is not int or self.xp.rerolls < 0:
            raise ValueError("xp.rerolls должен быть целым неотрицательным числом")
        for kind in ENEMY_KINDS:
            if kind not in self.enemies:
                raise ValueError(f"enemies.toml: нет жука {kind}")
        for weapon in WEAPON_IDS:
            if weapon not in self.weapons:
                raise ValueError(f"weapons.toml: нет оружия {weapon}")
            for stat in (f"{weapon}_damage", f"{weapon}_rate"):
                if stat not in self.player.stats:
                    raise ValueError(f"player.toml: нет характеристики {stat}")
        for up in self.upgrades:
            if up.stat not in self.player.stats:
                raise ValueError(f"upgrades.toml: {up.id} меняет неизвестную характеристику {up.stat}")
            if up.mode not in ("mult", "add") or up.group not in ("weapon", "passive"):
                raise ValueError(f"upgrades.toml: {up.id} — неверный mode или group")
            if up.mode == "add" and up.value <= 0:
                raise ValueError(f"upgrades.toml: {up.id} — для mode=add нужен value > 0")
            if up.weapon and up.weapon not in self.weapons:
                raise ValueError(f"upgrades.toml: {up.id} — неизвестное оружие {up.weapon}")
        if len({u.id for u in self.upgrades}) != len(self.upgrades):
            raise ValueError("upgrades.toml: повторяющиеся id")
        for m in self.mix:
            if m.kind not in self.enemies or m.kind in ("worker", "semi_boss", "queen"):
                raise ValueError(f"session.toml: [[mix]] — неверный вид {m.kind}")
        for biome in BIOME_IDS:
            if biome not in self.biomes:
                raise ValueError(f"biomes.toml: нет биома {biome}")
        if not self.rarities or self.rarities[0].bonus <= 0:
            raise ValueError("session.toml: первая редкость должна давать бонус > 0")


# Порядок важен: индекс вида — его номер в массивах толпы, боссы — последними
ENEMY_KINDS = ("worker", "warrior", "spitter", "flyer", "tanker", "brain", "semi_boss", "queen")
WEAPON_IDS = ("mar", "shotgun", "sniper", "flame", "rocket", "turret", "shock")
BIOME_IDS = ("desert", "forest", "jungle", "swamp")  # порядок прохождения
