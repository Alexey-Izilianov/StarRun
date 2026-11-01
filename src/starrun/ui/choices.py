"""Навигация по карточкам без зависимости от окна и графического контекста."""

from dataclasses import dataclass


@dataclass
class ChoiceCursor:
    index: int = 0

    def move(self, step: int, count: int) -> None:
        self.index = (self.index + step) % count if count else 0

    def reset(self) -> None:
        self.index = 0
