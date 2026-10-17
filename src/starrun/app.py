"""Окно игры: камеры, ввод, переключение между игрой, паузой, статистикой и выбором улучшения.

Запуск: python -m starrun [--windowed] [--seed N] [--biome desert|forest|jungle|swamp|plain] [--2d]
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from enum import Enum, auto

import arcade
import numpy as np
from arcade.types import LRBT

from starrun.balance import BIOME_IDS, Balance
from starrun.settings import Settings, resolutions
from starrun.sim.world import Controls, World
from starrun.ui.gl3d import Camera3D
from starrun.ui.hud import H, Hud, SettingRow
from starrun.ui.postfx import PostFX
from starrun.ui.render import Shapes
from starrun.ui.scene import GROUND, SceneRenderer
from starrun.ui.scene3d import Scene3D

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
    SETTINGS = auto()


class Game(arcade.Window):
    def __init__(self, windowed: bool, seed: int | None, biome: str = BIOME_IDS[0], flat: bool = False,
                 settings: Settings | None = None) -> None:
        # Сглаживание задаётся только при создании окна — поэтому настройки читаются до него
        self.settings = settings or Settings.load()
        if windowed:                                 # ключ командной строки важнее файла, но в файл не пишется
            self.settings = replace(self.settings, fullscreen=False)
        super().__init__(1600, 900, "StarRun", fullscreen=False, resizable=True, vsync=True,
                         update_rate=1 / 120, draw_rate=1 / 120, antialiasing=self.settings.antialiasing)
        self.draft = self.settings
        self.balance = Balance.load()
        self.seed = seed
        self.biome = biome
        self.shapes = Shapes(self.ctx)
        self.flat = flat                             # прежняя 2D-картинка вместо 3D
        self.cam3 = Camera3D()
        self.scene3d = None if flat else Scene3D(self.ctx)
        self.postfx = None if flat else PostFX(self.ctx)
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
        self.apply_display()
        self.new_run()

    # ---------- забег ----------
    def new_run(self) -> None:
        """Забег заново в том же биоме (пауза → «Начать заново»)."""
        biome = None if self.biome == "plain" else self.biome
        self.world = World(self.balance, self.seed, biome)
        self.scene = SceneRenderer(self.shapes)
        if self.scene3d:
            self.scene3d.reset()
            self._follow(0.0)
        self.hud = Hud(self.shapes)
        self.mode = Mode.PLAY
        self.background_color = self.balance.biomes[biome].ground if biome else GROUND
        name = self.balance.biomes[biome].name if biome else "Полигон"
        self.hud.notify(f"{name}. Высадка: продержись до эвакуации")

    def next_run(self) -> None:
        """После конца забега — следующий биом по кругу."""
        if self.biome in BIOME_IDS:
            self.biome = BIOME_IDS[(BIOME_IDS.index(self.biome) + 1) % len(BIOME_IDS)]
        self.new_run()

    def _fit_cameras(self) -> None:
        """Мир всегда виден в масштабе 1080 по высоте; ширина — по пропорциям окна."""
        w, h = self.get_size()
        self.vw = H * w / max(1, h)
        projection = LRBT(-self.vw / 2, self.vw / 2, -H / 2, H / 2)
        for cam in (self.world_cam, self.ui_cam):
            cam.match_window(projection=False)
            cam.projection = projection
        self.ui_cam.position = (self.vw / 2, H / 2)

    # ---------- настройки экрана ----------
    def _resolutions(self) -> list[tuple[int, int]]:
        screen = self.screen or self.display.get_default_screen()
        return resolutions([(m.width, m.height) for m in screen.get_modes()], (screen.width, screen.height))

    def apply_display(self) -> None:
        """Разрешение и режим из настроек. Полный экран — режимом монитора; окно — размером в физических
        пикселях (Windows масштабирует окно на 125 %, поэтому делим на коэффициент)."""
        s = self.settings
        screen = self.screen or self.display.get_default_screen()
        w, h = (s.width, s.height) if s.width and s.height else (screen.width, screen.height)
        if s.fullscreen:
            modes = [m for m in screen.get_modes() if (m.width, m.height) == (w, h)]
            native = (w, h) == (screen.width, screen.height)
            mode = None if native or not modes else max(modes, key=lambda m: m.rate or 0)
            self.set_fullscreen(True, screen=screen, mode=mode)
        else:
            self.set_fullscreen(False)
            ratio = self.get_pixel_ratio() or 1.0
            self.set_size(int(w / ratio), int(h / ratio))
            self.center_window()
        self._fit_cameras()

    def open_settings(self) -> None:
        self.draft = self.settings
        self.mode = Mode.SETTINGS

    def _setting_rows(self) -> list[SettingRow]:
        d = self.draft
        options = self._resolutions()
        current = (d.width, d.height) if d.width else options[-1]
        i = options.index(current) if current in options else len(options) - 1

        def res(step: int) -> None:
            w, h = options[(i + step) % len(options)]
            self.draft = replace(self.draft, width=w, height=h)

        def flip(field: str) -> None:
            self.draft = replace(self.draft, **{field: not getattr(self.draft, field)})

        return [SettingRow("Разрешение", f"{current[0]} × {current[1]}", lambda: res(-1), lambda: res(1)),
                SettingRow("Режим", "полный экран" if d.fullscreen else "окно",
                           lambda: flip("fullscreen"), lambda: flip("fullscreen")),
                SettingRow("Сглаживание", "вкл" if d.antialiasing else "выкл",
                           lambda: flip("antialiasing"), lambda: flip("antialiasing"))]

    def apply_settings(self) -> None:
        aa_changed = self.draft.antialiasing != self.settings.antialiasing
        self.settings = self.draft
        self.settings.save()
        self.apply_display()
        if aa_changed:
            self.hud.notify("Сглаживание изменится после перезапуска игры")
        self.mode = Mode.PAUSE

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
            if self.scene3d:
                w, h = self.get_size()
                g = self.cam3.ground_point(2 * self.mouse_screen[0] / w - 1, 2 * self.mouse_screen[1] / h - 1)
                target = (float(g[0]), float(g[1]))
            else:
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
                self.next_run()
        elif w.awaiting_choice:
            if symbol in CHOICE_KEYS and CHOICE_KEYS[symbol] < len(w.offer()):
                self.choose(CHOICE_KEYS[symbol])
        elif symbol == arcade.key.ESCAPE:
            self.mode = {Mode.PLAY: Mode.PAUSE, Mode.SETTINGS: Mode.PAUSE}.get(self.mode, Mode.PLAY)
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
            if self.scene3d:
                self.scene3d.update(delta_time, w)
        self.hud.update(delta_time if running else 0.0)
        self.shake = max(0.0, self.shake - delta_time * 30)
        self._follow(self.shake)

    def _follow(self, shake: float) -> None:
        """Камера — за десантником (с тряской); в 3D мир узнаёт, какой кусок земли виден: жуки — за его краем."""
        w = self.world
        jitter = np.random.uniform(-1, 1, 2) * shake
        self.world_cam.position = (float(w.player.pos[0] + jitter[0]), float(w.player.pos[1] + jitter[1]))
        if self.scene3d:
            width, height = self.get_size()
            self.cam3.shake = jitter
            self.cam3.update(w.player.pos, width / max(1, height))
            w.view_half = self.cam3.view_half()

    def _react(self, events: list[str]) -> None:
        heal = round(self.balance.items.medkit_heal * 100)
        messages = {
            "semi_boss": "Полубосс на подходе",
            "queen": "Тревога 100 %: выходит королева",
            "evac": "Королева убита. Челнок идёт — держись в зоне",
            "medkit": f"Аптечка: +{heal} % здоровья",
            "magnet": "Магнит: весь опыт летит к тебе",
            "dna": f"Образец ДНК — собрано {self.world.dna}",
            "underbarrel": "Найден подствольный гранатомёт для MAR",
            "crate_xp": f"Ящик с оружием: слоты заняты, +{self.balance.items.weapon_xp} опыта",
            "door_broken": "Жуки выломали дверь",
        }
        for e in events:
            if e.startswith("crate:"):
                self.hud.notify(f"Ящик с оружием: {self.balance.weapons[e[6:]].name}")
            elif messages.get(e):
                self.hud.notify(messages[e])
            if e == "player_hit":
                self.shake = max(self.shake, 4.0)
            elif e == "blast":
                self.shake = max(self.shake, 10.0)

    def world_to_ui(self, pos: np.ndarray) -> tuple[float, float]:
        if self.scene3d:
            x, y = self.cam3.to_ndc(pos)
            return (x + 1) / 2 * self.vw, (y + 1) / 2 * H
        cx, cy = self.world_cam.position
        return float(pos[0] - cx + self.vw / 2), float(pos[1] - cy + H / 2)

    def on_draw(self) -> None:
        self.clear()
        w = self.world
        if self.scene3d and self.postfx:
            self.scene3d.draw(w, self.cam3, self.scene, self.shapes)
            self.postfx.apply()              # резкость и контраст — только миру, HUD рисуется поверх
        else:
            with self.world_cam.activate():
                cx, cy = self.world_cam.position
                view = (cx - self.vw / 2, cy - H / 2, cx + self.vw / 2, cy + H / 2)
                self.scene.draw(w, view)
        with self.ui_cam.activate():
            self.hud.draw(w, self.vw, self.world_to_ui)
            if w.session.over:
                self.hud.end(w, self.vw, self.next_run, self.close)
            elif w.awaiting_choice:
                self.hud.level_up(w, self.vw, self.choose)
            elif self.mode is Mode.PAUSE:
                self.hud.pause(self.vw, self._resume, self.new_run, self.open_settings, self.close)
            elif self.mode is Mode.SETTINGS:
                self.hud.settings(self.vw, self._setting_rows(), self.apply_settings, self._back_to_pause,
                                  "Разрешение и режим — сразу, сглаживание — после перезапуска")
            elif self.mode is Mode.STATS:
                self.hud.stats(w, self.vw)
            else:
                self.hud.buttons = []

    def _resume(self) -> None:
        self.mode = Mode.PLAY

    def _back_to_pause(self) -> None:
        self.mode = Mode.PAUSE


def main() -> None:
    parser = argparse.ArgumentParser(description="StarRun")
    parser.add_argument("--windowed", action="store_true", help="в окне, а не на весь экран")
    parser.add_argument("--seed", type=int, default=None, help="зерно случайности — повторить забег")
    parser.add_argument("--biome", default=BIOME_IDS[0], choices=(*BIOME_IDS, "plain"),
                        help="с какого биома начать (дальше — по кругу); plain — ровная земля без карты")
    parser.add_argument("--2d", dest="flat", action="store_true", help="прежняя 2D-картинка вместо 3D")
    args = parser.parse_args()
    Game(args.windowed, args.seed, args.biome, args.flat)
    arcade.run()


if __name__ == "__main__":
    main()
