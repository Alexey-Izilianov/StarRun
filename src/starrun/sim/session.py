"""Ход забега: время, шкала «Тревога улья», полубоссы, королева, эвакуация, рост силы жуков."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from starrun.balance import AlertCfg, EvacCfg, Scaling


class Phase(Enum):
    WAVE = auto()      # обычная волна, тревога растёт
    QUEEN = auto()     # королева на карте
    EVAC = auto()      # королева убита, ждём челнок в зоне
    VICTORY = auto()
    DEFEAT = auto()


@dataclass
class Session:
    alert_cfg: AlertCfg
    evac_cfg: EvacCfg
    scaling: Scaling
    semi_boss_minutes: tuple[float, ...]
    elapsed: float = 0.0
    alert: float = 0.0
    phase: Phase = Phase.WAVE
    evac_progress: float = 0.0  # секунды, простоянные в зоне
    _bosses_due: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._bosses_due = sorted(self.semi_boss_minutes)

    @property
    def minute(self) -> int:
        return int(self.elapsed // 60)

    @property
    def hp_mult(self) -> float:
        return 1 + self.scaling.hp_per_minute * self.minute

    @property
    def damage_mult(self) -> float:
        return 1 + self.scaling.damage_per_minute * self.minute

    @property
    def boss_hp_mult(self) -> float:
        return 1 + (self.hp_mult - 1) * self.scaling.boss_share

    @property
    def over(self) -> bool:
        return self.phase in (Phase.VICTORY, Phase.DEFEAT)

    def update(self, dt: float) -> list[str]:
        """Двигает время; возвращает события: "semi_boss" — пора выпускать полубосса, "queen" — королеву."""
        if self.over:
            return []
        self.elapsed += dt
        events: list[str] = []
        while self._bosses_due and self.elapsed >= self._bosses_due[0] * 60:
            self._bosses_due.pop(0)
            if self.phase is Phase.WAVE:
                events.append("semi_boss")
        if self.phase is Phase.WAVE:
            self.raise_alert(self.alert_cfg.per_minute * dt / 60)
            if self.alert >= 100 and self.elapsed >= self.alert_cfg.queen_min_minute * 60:
                self.phase = Phase.QUEEN
                events.append("queen")
        return events

    def raise_alert(self, amount: float) -> None:
        self.alert = min(100.0, self.alert + amount)

    def semi_boss_killed(self) -> None:
        self.raise_alert(self.alert_cfg.semi_boss_kill)

    def queen_killed(self) -> None:
        if self.phase is Phase.QUEEN:
            self.phase = Phase.EVAC
            self.evac_progress = 0.0

    def evac_tick(self, dt: float, inside: bool) -> bool:
        """Таймер эвакуации идёт, пока десантник в зоне; True — челнок забрал."""
        if self.phase is not Phase.EVAC:
            return False
        if inside:
            self.evac_progress += dt
        if self.evac_progress >= self.evac_cfg.seconds:
            self.phase = Phase.VICTORY
            return True
        return False

    def player_died(self) -> None:
        if not self.over:
            self.phase = Phase.DEFEAT
