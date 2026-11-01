"""Проверка 3D в настоящем окне: карточки, интерьер, рубка, биомы и нагрузочная сцена.

Запуск: python tools/render_check.py [--out screenshots/rework] [--2d]. Звук выключен,
настройки пользователя не меняются. Снимки и время отрисовки сохраняются в out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arcade
import numpy as np
import pyglet

from starrun.app import Game
from starrun.settings import Settings
from starrun.sim import terrain as T
from starrun.sim.swarm import WORKER

CASES = ("upgrades", "interior", "chop", "jungle", "swamp")
EXTENDED = ("interior_north", "interior_south", "interior_east", "interior_west", "combat")


class RenderCheck(Game):
    def __init__(self, out: Path, flat: bool, extended: bool = False) -> None:
        self.cases = CASES + EXTENDED if extended else CASES
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.case = -1
        self.frame = 0
        self.times: list[float] = []
        self.update_times: list[float] = []
        self.report: dict[str, object] = {}
        super().__init__(True, 3, "forest", flat, Settings(1600, 900, False, True, volume=0))
        self.set_vsync(False)
        self.next_case()

    def next_case(self) -> None:
        self.case += 1
        if self.case >= len(self.cases):
            self.report["framebuffer"] = self.ctx.screen.size
            self.report["pixel_ratio"] = self.get_pixel_ratio()
            self.report["font"] = pyglet.font.load("Exo 2").name
            self.report["samples"] = self.config.samples
            self.out.joinpath("report.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            print(json.dumps(self.report), flush=True)
            pyglet.clock.schedule_once(lambda _dt: self.close(), 0)
            return
        name = self.cases[self.case]
        self.biome = {"chop": "desert", "jungle": "jungle", "swamp": "swamp"}.get(name, "forest")
        self.new_run()
        self.hud.banners.clear()
        w = self.world
        t = w.terrain
        assert t is not None
        if name == "upgrades":
            w.progress.add(w.progress.needed)
            self.on_key_press(arcade.key.D, 0)
            self.on_key_release(arcade.key.D, 0)
            assert self.choice_cursor.index == 1
        elif name.startswith("interior"):
            bd = t.buildings[0]
            x, y = next((x, y) for x, y in t.doors if t.building[y, x] == 0)
            offset = np.array([0, -55 if y == bd.y0 else 55]) if y in (bd.y0, bd.y1) else np.array(
                [-55 if x == bd.x0 else 55, 0])
            w.player.pos = t.cell_center(x, y) + offset
            if name != "interior":
                center = np.array([(bd.x0 + bd.x1 + 1) * t.cell / 2, (bd.y0 + bd.y1 + 1) * t.cell / 2])
                cx, cy = center
                w.player.pos = {
                    "interior_north": np.array([cx, (bd.y1 + 1) * t.cell + 65]),
                    "interior_south": np.array([cx, bd.y0 * t.cell - 65]),
                    "interior_east": np.array([(bd.x1 + 1) * t.cell + 65, cy]),
                    "interior_west": np.array([bd.x0 * t.cell - 65, cy]),
                }[name]
            if self.scene3d:
                self.cam3.distance = 1400 if name != "interior" else 1050
        elif name == "chop":
            t.kind[95:105, 95:105] = T.GROUND
            t.kind[100, 101] = T.TREE
            t.hp[100, 101] = 100
            w.player.pos = t.cell_center(100, 100)
            w.player.heading = w.player.facing = np.array([1, 0], np.float32)
            if self.scene3d:
                self.cam3.distance = 650
        else:
            if self.scene3d:
                self.cam3.distance = 1400
            angle = w.rng.uniform(0, 2 * np.pi, 320)
            minimum = 250 if name == "combat" else 100
            pos = w.player.pos + np.column_stack([np.cos(angle), np.sin(angle)]) * w.rng.uniform(minimum, 650, (320, 1))
            w.swarm.spawn(WORKER, pos.astype(np.float32), 1, 1)
            if name == "combat":
                t.kind[80:120, 80:120] = T.GROUND
                t.version += 1
                w.stats.add("mar_ricochet", 2)
                w.stats.add("mar_crit", 0.5)
                if self.scene3d:
                    self.cam3.distance = 900
        self.frame = 0
        self.times = []
        self.update_times = []
        self._follow(0)

    def on_update(self, delta_time: float) -> None:
        if self.case >= len(self.cases):
            return
        started = time.perf_counter()
        if self.cases[self.case] == "combat":
            self.world.step(1 / 60, self.controls())
            self.world.progress.pending = 0
        if self.cases[self.case] == "chop":
            self.world.player.chop = (self.frame % 48) / 48
            self.world.events = ["chop"] if self.frame % 48 == 0 else []
        self.scene.update(1 / 60, self.world)
        if self.scene3d:
            self.scene3d.update(1 / 60, self.world)
        self._follow(0)
        if self.frame > 15:
            self.update_times.append((time.perf_counter() - started) * 1000)

    def on_draw(self) -> None:
        if self.case >= len(self.cases):
            return
        started = time.perf_counter()
        super().on_draw()
        self.ctx.finish()
        elapsed = (time.perf_counter() - started) * 1000
        self.frame += 1
        if self.frame > 15:
            self.times.append(elapsed)
        if self.frame == 60:
            name = self.cases[self.case]
            arcade.get_image().save(self.out / f"{name}.png")
            self.report[name] = {"draw_mean_ms": round(float(np.mean(self.times)), 2),
                                 "draw_p99_ms": round(float(np.percentile(self.times, 99)), 2)}
            if self.update_times:
                self.report[name]["update_mean_ms"] = round(float(np.mean(self.update_times)), 2)
                self.report[name]["work_mean_ms"] = round(float(np.mean(self.times) + np.mean(self.update_times)), 2)
            self.next_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("screenshots/rework"))
    parser.add_argument("--2d", dest="flat", action="store_true")
    parser.add_argument("--extended", action="store_true", help="четыре стороны дома и действующий бой")
    args = parser.parse_args()
    RenderCheck(args.out, args.flat, args.extended)
    arcade.run()
