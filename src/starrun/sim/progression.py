"""Опыт, уровни, характеристики десантника и выбор улучшений с редкостями."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from starrun.balance import Rarity, Upgrade, WeaponCfg, XpCfg

MAX_WEAPON_LEVEL = 5


class Stats:
    """Итоговое значение = (база + сумма прибавок) × произведение множителей, с потолком из данных."""

    def __init__(self, base: dict[str, float], caps: dict[str, float]) -> None:
        self._base = dict(base)
        self._caps = caps
        self._add = dict.fromkeys(base, 0.0)
        self._mult = dict.fromkeys(base, 1.0)

    def __getitem__(self, name: str) -> float:
        value = (self._base[name] + self._add[name]) * self._mult[name]
        cap = self._caps.get(name)
        return min(value, cap) if cap is not None else value

    def add(self, name: str, amount: float) -> None:
        self._add[name] += amount

    def multiply(self, name: str, factor: float) -> None:
        self._mult[name] *= factor

    def names(self) -> list[str]:
        return list(self._base)


def xp_to_next(level: int, cfg: XpCfg) -> int:
    """Опыт, нужный с уровня level до следующего; растёт плавно, первые уровни — за 20–30 с."""
    n = level - 1
    return math.ceil(cfg.base + cfg.linear * n + cfg.quad * n * n)


@dataclass
class Progress:
    cfg: XpCfg
    level: int = 1
    xp: int = 0
    pending: int = 0  # уровни, за которые ещё не выбрано улучшение

    @property
    def needed(self) -> int:
        return xp_to_next(self.level, self.cfg)

    def add(self, amount: int) -> int:
        """Начисляет опыт; возвращает, сколько уровней получено (может быть больше одного)."""
        self.xp += amount
        gained = 0
        while self.xp >= self.needed:
            self.xp -= self.needed
            self.level += 1
            gained += 1
        self.pending += gained
        return gained


@dataclass(frozen=True)
class Offer:
    upgrade: Upgrade
    rarity: Rarity
    rarity_index: int

    @property
    def special(self) -> str:
        """"weapon" — новый ствол, "mod" — модификация, "" — обычное улучшение характеристики."""
        return self.upgrade.mode if self.upgrade.mode in ("weapon", "mod") else ""

    def amount(self, common_bonus: float) -> float:
        """Насколько изменится характеристика: доля для mult, абсолютная прибавка для add."""
        if self.special:
            return 0.0
        if self.upgrade.mode == "mult":
            return self.rarity.bonus
        value = self.upgrade.value * self.rarity.bonus / common_bonus
        return float(math.ceil(value - 1e-9)) if self.upgrade.integer else value


def roll_rarity(rng: np.random.Generator, rarities: tuple[Rarity, ...]) -> int:
    weights = np.array([r.weight for r in rarities], float)
    return int(rng.choice(len(rarities), p=weights / weights.sum()))


@dataclass
class Upgrades:
    """Пул улучшений забега: что предложить на уровне и как применить выбор."""

    pool: tuple[Upgrade, ...]
    rarities: tuple[Rarity, ...]
    weapons: dict[str, WeaponCfg] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)
    taken: list[Offer] = field(default_factory=list)

    def available(self, owned: dict[str, tuple[int, bool]] | None = None, free_slot: bool = False) -> list[Upgrade]:
        """Что можно предложить. owned — стволы в слотах: id → (уровень, есть ли модификация).
        Улучшения ствола — только пока он в слоте и не достиг 5-го уровня; на 5-м — карточка модификации;
        пока есть свободный слот — карточки новых стволов."""
        owned = owned or {}
        out = []
        for u in self.pool:
            if u.max_rank and self.ranks.get(u.id, 0) >= u.max_rank:
                continue
            if u.weapon and (u.weapon not in owned or owned[u.weapon][0] >= MAX_WEAPON_LEVEL):
                continue
            out.append(u)
        for wid, cfg in self.weapons.items():
            if wid not in owned and free_slot:
                out.append(Upgrade(f"new:{wid}", cfg.name, cfg.text, "", "weapon", "weapon", weapon=wid))
            elif wid in owned and owned[wid][0] >= MAX_WEAPON_LEVEL and not owned[wid][1] and not cfg.mod_found:
                out.append(Upgrade(f"mod:{wid}", f"{cfg.name}: {cfg.mod_name.lower()}", cfg.mod_text, "", "mod",
                                   "weapon", weapon=wid))
        return out

    def offer(self, rng: np.random.Generator, count: int = 3, owned: dict[str, tuple[int, bool]] | None = None,
              free_slot: bool = False) -> list[Offer]:
        """Три разные карточки: хотя бы одна для оружия и хотя бы одна пассивная, если такие остались."""
        left = self.available(owned, free_slot)
        picks: list[Upgrade] = []
        for group in ("weapon", "passive"):
            options = [u for u in left if u.group == group and u not in picks]
            if options and len(picks) < count:
                picks.append(options[int(rng.integers(len(options)))])
        rest = [u for u in left if u not in picks]
        while len(picks) < count and rest:
            picks.append(rest.pop(int(rng.integers(len(rest)))))
        rng.shuffle(picks)
        offers = []
        for up in picks:
            i = roll_rarity(rng, self.rarities)
            offers.append(Offer(up, self.rarities[i], i))
        return offers

    def apply(self, offer: Offer, stats: Stats) -> None:
        """Меняет характеристику; новый ствол и модификацию ставит мир — здесь только учёт."""
        amount = offer.amount(self.rarities[0].bonus)
        if offer.upgrade.mode == "mult":
            stats.multiply(offer.upgrade.stat, 1 + amount)
        elif offer.upgrade.mode == "add":
            stats.add(offer.upgrade.stat, amount)
        self.ranks[offer.upgrade.id] = self.ranks.get(offer.upgrade.id, 0) + 1
        self.taken.append(offer)
