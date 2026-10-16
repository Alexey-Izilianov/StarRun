"""Окно игры: камеры, ввод, переключение между игрой, паузой, статистикой и выбором улучшения.

Запуск: python -m starrun [--windowed] [--seed N]
"""

from __future__ import annotations

import argparse
from enum import Enum, auto

import arcade
import numpy as np
from arcade.types import LRBT

from starrun.balance import Balance
from starrun.sim.world import Controls, World
from starrun.ui.hud import H, Hud
from starrun.ui.render import Shapes
from starrun.ui.scene import GROUND, SceneRenderer

MOVE_KEYS = {
    arcade.key.W: (0, 1), arcade.key.UP: (0, 1),
    arcade.key.S: (0, -1), arcade.key.DOWN: (0, -1),
    arcade.key.A: (-1, 0), arcade.key.LEFT: (-1, 0),
    arcade.key.D: (1, 0), arcade.key.RIGHT: (1, 0),
}
CHOICE_KEYS = {arcade.key.KEY_1: 0, arcade.key.KEY_2: 1, arcade.key.KEY_3: 2,
               arcade.key.NUM_1: 0, arcade.key.NUM_2: 1, arcade.key.NUM_3: 2}


class Mode(Enum):
    PLAY = auto()
    PAUSE = auto()
    STATS = auto()


class Game(arcade.Window):
    def __init__(self, windowed: bool, seed: int | None) -> None:
        super().__init__(1600, 900, "StarRun", fullscreen=not windowed, resizable=True, vsync=True,
                         update_rate=1 / 120, draw_rate=1 / 120)
        self.background_color = GROUND
        self.balance = Balance.load()
        self.seed = seed
        self.shapes = Shapes(self.ctx)
        self.world_cam = arcade.Camera2D()
        self.ui_cam = arcade.Camera2D()
        self.vw = 1920.0
        self._fit_cameras()
        self.held: set[int] = set()
        self.ability_pressed = False
        self.slot_pressed: int | None = None
        self.wheel = 0
        self.mouse_screen = (0.0, 0.0)
        self.shake = 0.0
        self.new_run()

    # ---------- забег ----------
    def new_run(self) -> None:
        self.world = World(self.balance, self.seed)
        self.scene = SceneRenderer(self.shapes)
        self.hud = Hud(self.shapes)
        self.mode = Mode.PLAY
        self.hud.notify("Высадка. Продержись до эвакуации")

    def _fit_cameras(self) -> None:
        """Мир всегда виден в масштабе 1080 по высоте; ширина — по пропорциям окна."""
        w, h = self.get_size()
        self.vw = H * w / max(1, h)
        projection = LRBT(-self.vw / 2, self.vw / 2, -H / 2, H / 2)
        for cam in (self.world_cam, self.ui_cam):
            cam.match_window(projection=False)
            cam.projection = projection
        self.ui_cam.position = (self.vw / 2, H / 2)

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self._fit_cameras()

    # ---------- ввод ----------
    def controls(self) -> Controls:
        dx = sum(MOVE_KEYS[k][0] for k in self.held if k in MOVE_KEYS)
        dy = sum(MOVE_KEYS[k][1] for k in self.held if k in MOVE_KEYS)
        sprint = arcade.key.LSHIFT in self.held or arcade.key.RSHIFT in self.held
        target = None
        if self.ability_pressed:
            self.ability_pressed = False
            p = self.world_cam.unproject(self.mouse_screen)
            target = (p.x, p.y)
        select, self.slot_pressed = self.slot_pressed, None
        cycle, self.wheel = self.wheel, 0
        return Controls((dx, dy), sprint, target, select, cycle)

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        self.held.add(symbol)
        w = self.world
        if symbol == arcade.key.F11:
            self.set_fullscreen(not self.fullscreen)
        elif w.session.over:
            if symbol in (arcade.key.ENTER, arcade.key.SPACE):
                self.new_run()
        elif w.awaiting_choice:
            if symbol in CHOICE_KEYS and CHOICE_KEYS[symbol] < len(w.offer()):
                self.choose(CHOICE_KEYS[symbol])
        elif symbol == arcade.key.ESCAPE:
            self.mode = Mode.PLAY if self.mode is not Mode.PLAY else Mode.PAUSE
        elif symbol == arcade.key.TAB:
            self.mode = Mode.PLAY if self.mode is Mode.STATS else Mode.STATS
        elif symbol == arcade.key.SPACE and self.mode is Mode.PLAY:
            self.ability_pressed = True
        elif symbol in CHOICE_KEYS and self.mode is Mode.PLAY:
            self.slot_pressed = CHOICE_KEYS[symbol]

    def on_mouse_scroll(self, x: int, y: int, scroll_x: float, scroll_y: float) -> None:
        if self.mode is Mode.PLAY and scroll_y:
            self.wheel = -1 if scroll_y > 0 else 1  # колесо от себя — предыдущий слот

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        self.held.discard(symbol)

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        self.mouse_screen = (x, y)
        p = self.ui_cam.unproject((x, y))
        self.hud.mouse = (p.x, p.y)

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        p = self.ui_cam.unproject((x, y))
        self.hud.click(p.x, p.y)

    def on_deactivate(self) -> None:
        # Окно потеряло фокус (Alt+Tab): клавиши «залипают», игра встаёт на паузу
        self.held.clear()
        if self.mode is Mode.PLAY and not self.world.session.over:
            self.mode = Mode.PAUSE

    def choose(self, i: int) -> None:
        self.world.choose(i)
        self.held.clear()  # клавиша, зажатая до карточек, не должна тянуть десантника после

    # ---------- цикл ----------
    def on_update(self, delta_time: float) -> None:
        w = self.world
        running = self.mode is Mode.PLAY and not w.awaiting_choice and not w.session.over
        if running:
            w.step(delta_time, self.controls())
            self._react(w.events)
            self.scene.update(delta_time, w)
        self.hud.update(delta_time if running else 0.0)
        self.shake = max(0.0, self.shake - delta_time * 30)
        jitter = np.random.uniform(-1, 1, 2) * self.shake
        self.world_cam.position = (float(w.player.pos[0] + jitter[0]), float(w.player.pos[1] + jitter[1]))

    def _react(self, events: list[str]) -> None:
        messages = {
            "semi_boss": "Полубосс на подходе",
            "queen": "Тревога 100 %: выходит королева",
            "evac": "Королева убита. Челнок идёт — держись в зоне",
            "orbital_not_ready": "",
        }
        for e in events:
            if messages.get(e):
                self.hud.notify(messages[e])
            if e == "player_hit":
                self.shake = max(self.shake, 4.0)
            elif e == "blast":
                self.shake = max(self.shake, 10.0)

    def world_to_ui(self, pos: np.ndarray) -> tuple[float, float]:
        cx, cy = self.world_cam.position
        return float(pos[0] - cx + self.vw / 2), float(pos[1] - cy + H / 2)

    def on_draw(self) -> None:
        self.clear()
        w = self.world
        with self.world_cam.activate():
            cx, cy = self.world_cam.position
            view = (cx - self.vw / 2, cy - H / 2, cx + self.vw / 2, cy + H / 2)
            self.scene.draw(w, view)
        with self.ui_cam.activate():
            self.hud.draw(w, self.vw, self.world_to_ui)
            if w.session.over:
                self.hud.end(w, self.vw, self.new_run, self.close)
            elif w.awaiting_choice:
                self.hud.level_up(w, self.vw, self.choose)
            elif self.mode is Mode.PAUSE:
                self.hud.pause(self.vw, self._resume, self.new_run, self.close)
            elif self.mode is Mode.STATS:
                self.hud.stats(w, self.vw)
            else:
                self.hud.buttons = []

    def _resume(self) -> None:
        self.mode = Mode.PLAY


def main() -> None:
    parser = argparse.ArgumentParser(description="StarRun")
    parser.add_argument("--windowed", action="store_true", help="в окне, а не на весь экран")
    parser.add_argument("--seed", type=int, default=None, help="зерно случайности — повторить забег")
    args = parser.parse_args()
    Game(args.windowed, args.seed)
    arcade.run()


if __name__ == "__main__":
    main()
