"""Звук: разовые звуки по событиям мира (world.events) и по разнице состояний между кадрами (новые пули, погибшие
жуки, подобранный опыт), петли (фон биома, гомон роя, рёв огнемёта) и фразы десантника по рации.

Звуки — assets/sounds/*.wav из tools/sfx/sfx.py; варианты одного звука — «имя_0», «имя_1»…, берётся случайный,
тон чуть плавает. Дальние звуки тише и уходят в сторону по панораме. Каждый звук не чаще своего gap — сотня
жуков, погибших за кадр, не превращается в сотню одновременных шлепков. Фразы не перебивают друг друга:
важные ждут очереди, остальные пропускаются, у каждой — своя пауза до повтора.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import arcade
import numpy as np

from starrun.settings import Settings
from starrun.sim.combat import BULLET, GRENADE, PELLET, ROCKET, SNIPER, TURRET_SHOT
from starrun.sim.terrain import BRIDGE, DOOR, FLOOR, MUD, OPEN_DOOR, SAND
from starrun.sim.world import World

SOUNDS = Path(__file__).resolve().parents[3] / "assets" / "sounds"
HEARING = 1100.0              # px: дальше — звук на минимальной громкости
STEP_DISTANCE = 72.0         # px пути между шагами: бег чаще, топь и песок реже
STEP_SURFACES = ("ground", "sand", "mud", "wood", "metal")
SHOT_SOUND = {BULLET: "mar", PELLET: "shotgun", SNIPER: "sniper", ROCKET: "rocket_launch", GRENADE: "grenade_launch"}
# Событие мира → звук и его громкость (позиции у этих событий нет — звучат «рядом»)
EVENT_SOUNDS = {
    "orbital": ("orbital_incoming", 0.8), "orbital_not_ready": ("ui_click", 0.4), "door_open": ("door", 0.7),
    "door_broken": ("door", 0.9), "player_hit": ("hurt", 0.8), "volley": ("queen_volley", 0.9), "spit": ("spit", 0.5),
    "swing": ("warrior_swing", 0.6), "blast": ("orbital_blast", 1.0), "level_up": ("level_up", 0.7),
    "tree_down": ("tree_fall", 0.6), "medkit": ("pickup", 0.8), "magnet": ("pickup", 0.8), "dna": ("pickup", 0.8),
    "underbarrel": ("pickup", 0.8), "switch": ("reload", 0.4), "queen": ("queen_roar", 1.0), "evac": ("dropship", 0.9),
}
# Фраза → (пауза до повтора, с, важная — ждёт очереди)
VOICE = {"drop": (999, True), "contact": (999, False), "reload": (30, False), "medkit": (45, False),
         "boss": (60, True), "queen": (999, True), "orbital": (25, False), "streak_0": (90, False),
         "streak_1": (90, False), "weapon": (40, False), "evac": (999, True), "win": (999, True)}
GAP = {"mar": 0.05, "turret": 0.08, "bug_die": 0.045, "gem": 0.035, "hurt": 0.2, "spit": 0.12, "warrior_swing": 0.1,
       "explosion": 0.04, "shock": 0.12, "chop": 0.08, "bug_bite": 0.15, "door": 0.3, "acid_sizzle": 0.3}


def sound_name(stem: str) -> str:
    """Имя звука по файлу: «mar_2» — вариант 2 звука «mar»; у фраз номер — часть имени («voice_streak_0»)."""
    if stem.startswith("voice_"):
        return stem
    return re.sub(r"_\d+$", "", stem)


@dataclass
class Loop:
    player: object | None = None   # pyglet Player
    target: float = 0.0
    level: float = 0.0


@dataclass
class Voice:
    busy_until: float = 0.0
    last: dict[str, float] = field(default_factory=dict)
    queue: list[str] = field(default_factory=list)


class Audio:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sounds: dict[str, list[arcade.Sound]] = {}
        for path in sorted(SOUNDS.glob("*.wav")):
            self.sounds.setdefault(sound_name(path.stem), []).append(arcade.load_sound(path))
        self.loops: dict[str, Loop] = {}
        self.rng = random.Random(3)
        self.reset(None)

    def reset(self, world: World | None) -> None:
        """Новый забег: петли — заново, счётчики — с нуля."""
        for loop in self.loops.values():
            self._stop(loop)
        self.loops = {}
        self.time = 0.0
        self.last: dict[str, float] = {}
        self.voice = Voice()
        self.prev_bullets: np.ndarray | None = None
        self.prev_alive: np.ndarray | None = None
        self.prev_gems: np.ndarray | None = None
        self.prev_reload = 0.0
        self.gem_streak = 0.0
        self.kills_said = 0
        self.contact_said = False
        self.step_pos = world.player.pos.copy() if world is not None else None
        self.step_distance = 0.0
        self.biome = world.biome or "" if world is not None else ""       # id биома: фон amb_<id>
        self.tree = world.terrain.biome.tree if world is not None and world.terrain is not None else ""
        if world is not None:
            self.say("drop")

    # ---------- громкость ----------
    def _gain(self, kind: str) -> float:
        s = self.settings
        part = {"sfx": s.sfx_volume, "voice": s.voice_volume, "ambient": s.ambient_volume}[kind]
        return s.volume / 10 * part / 10

    def play(self, name: str, gain: float = 1.0, pos: np.ndarray | None = None, here: np.ndarray | None = None,
             kind: str = "sfx", pitch: float = 0.05) -> None:
        variants = self.sounds.get(name)
        if not variants or self.time - self.last.get(name, -1.0) < GAP.get(name, 0.0):
            return
        self.last[name] = self.time
        pan = 0.0
        if pos is not None and here is not None:
            d = pos - here
            gain *= float(np.clip(1.25 - np.hypot(*d) / HEARING, 0.2, 1.0))
            pan = float(np.clip(d[0] / 900, -1, 1)) * 0.6
        volume = gain * self._gain(kind)
        if volume > 0.01:
            arcade.play_sound(self.rng.choice(variants), volume=volume, pan=pan,
                              speed=1 + self.rng.uniform(-pitch, pitch))

    def say(self, line: str) -> None:
        """Фраза десантника: не поверх другой; важная ждёт очереди, обычная — пропускается."""
        pause, important = VOICE[line]
        v = self.voice
        if self.time - v.last.get(line, -999.0) < pause:
            return
        if self.time < v.busy_until:
            if important and line not in v.queue:
                v.queue.append(line)
            return
        sound = self.sounds.get(f"voice_{line}")
        if not sound:
            return
        v.last[line] = self.time
        v.busy_until = self.time + sound[0].get_length() + 1.5
        if self._gain("voice") > 0.01:
            arcade.play_sound(sound[0], volume=self._gain("voice"))

    # ---------- кадр ----------
    def react(self, events: list[str], world: World) -> None:
        here = world.player.pos
        for e in events:
            if e.startswith("crate:"):
                self.play("pickup", 0.8)
                self.say("weapon")
            elif e == "chop":
                self.play("chop_cactus" if self.tree == "cactus" else "chop", 0.8)
            elif e in EVENT_SOUNDS:
                name, gain = EVENT_SOUNDS[e]
                self.play(name, gain, kind="sfx", pitch=0.0 if name in ("level_up", "queen_roar") else 0.05)
            if e == "orbital":
                self.say("orbital")
            elif e == "semi_boss":
                self.say("boss")
            elif e == "queen":
                self.say("queen")
            elif e == "evac":
                self.say("evac")
            elif e == "victory":
                self.say("win")
        for boom in world.booms[:3]:
            self.play("explosion", 1.0, boom.pos, here)

    def update(self, dt: float, world: World, paused: bool) -> None:
        self.time += dt
        if self.voice.queue and self.time >= self.voice.busy_until:
            self.say(self.voice.queue.pop(0))
        if paused:
            self.step_pos = world.player.pos.copy()
            self.step_distance = 0.0
            for loop in self.loops.values():
                loop.target = 0.0
        else:
            self._shots(world)
            self._deaths(world)
            self._gems(world)
            self._marine(world)
            self._footsteps(world)
            self._loop_targets(world)
        self._loops(dt)

    def _shots(self, world: World) -> None:
        bl = world.bullets
        prev = self.prev_bullets if self.prev_bullets is not None else bl.alive
        new = np.flatnonzero(bl.alive & ~prev)
        self.prev_bullets = bl.alive.copy()
        here = world.player.pos
        if len(new):
            kinds = set(bl.kind[new].tolist())
            for k in kinds & set(SHOT_SOUND):
                self.play(SHOT_SOUND[k], 0.75)
            if TURRET_SHOT in kinds:
                j = new[bl.kind[new] == TURRET_SHOT][0]
                self.play("turret", 0.6, bl.pos[j], here)
        if world.arcs:
            self.play("shock", 0.8)

    def _deaths(self, world: World) -> None:
        sw = world.swarm
        if self.prev_alive is not None:
            died = np.flatnonzero(self.prev_alive & ~sw.alive)
            if len(died):
                here = world.player.pos
                near = died[np.argsort(np.hypot(*(sw.pos[died] - here).T))[:2]]
                for i in near:
                    self.play("bug_die", 0.7, sw.pos[i], here, pitch=0.12)
        self.prev_alive = sw.alive.copy()
        total = sum(world.kills.values())
        if total >= self.kills_said + 250:
            self.kills_said = total // 250 * 250
            self.say(f"streak_{(total // 250) % 2}")
        if not self.contact_said and sw.count() and self.time > 6:
            d = np.hypot(*(sw.pos[sw.alive_idx] - world.player.pos).T)
            if d.min() < 600:
                self.contact_said = True
                self.say("contact")

    def _gems(self, world: World) -> None:
        """Опыт: звон с повышением тона, пока подбирается подряд."""
        g = world.gems.alive
        if self.prev_gems is not None and (self.prev_gems & ~g).any():
            self.gem_streak = min(1.0, self.gem_streak + 0.08) if self.time - self.last.get("gem", -1) < 0.4 else 0.0
            if self.time - self.last.get("gem", -1.0) >= GAP["gem"]:
                variants = self.sounds.get("gem")
                volume = 0.5 * self._gain("sfx")
                if variants and volume > 0.01:
                    self.last["gem"] = self.time
                    arcade.play_sound(variants[0], volume=volume, speed=1.0 + 0.5 * self.gem_streak)
        self.prev_gems = g.copy()

    def _marine(self, world: World) -> None:
        p = world.player
        if world.weapons:
            left = world.weapons[world.active].reload_left
            if left > 0 and self.prev_reload <= 0:
                self.play("reload", 0.6)
                if self.rng.random() < 0.3:
                    self.say("reload")
            self.prev_reload = left
        if p.hp < 0.3 * world.stats["max_hp"]:
            self.say("medkit")

    def _footsteps(self, world: World) -> None:
        """Шаги по пройденному пути после столкновений: упор в стену и рубка молчат."""
        p = world.player
        distance = 0.0 if self.step_pos is None else float(np.linalg.norm(p.pos - self.step_pos))
        self.step_pos = p.pos.copy()
        # Перенос в новый мир / отладочный телепорт не должен давать очередь шагов.
        if distance < 0.01 or distance > STEP_DISTANCE * 2 or p.vaulting or world.session.over:
            self.step_distance = 0.0
            return
        self.step_distance += distance
        if self.step_distance < STEP_DISTANCE:
            return
        self.step_distance %= STEP_DISTANCE
        surface = "sand" if world.biome == "desert" else "ground"
        t = world.terrain
        if t is not None:
            kind, soil = int(t.kinds_at(p.pos)[0]), int(t.soil_at(p.pos)[0])
            if kind == BRIDGE:
                surface = "wood"
            elif kind in (FLOOR, DOOR, OPEN_DOOR):
                surface = "metal"
            elif soil == MUD:
                surface = "mud"
            elif soil == SAND:
                surface = "sand"
        self.play(f"step_{surface}", 0.45 if p.sprinting else 0.32, pitch=0.08)

    # ---------- петли ----------
    def _loop_targets(self, world: World) -> None:
        sw = world.swarm
        here = world.player.pos
        idx = sw.alive_idx
        near = int((np.hypot(*(sw.pos[idx] - here).T) < 500).sum()) if len(idx) else 0
        flames = world.flames.alive
        burning = bool(flames.any()) and bool((np.hypot(*(world.flames.pos[flames] - here).T) < 260).any())
        self._want(f"amb_{self.biome}", 0.8, "ambient")
        self._want("swarm_loop", min(1.0, near / 35) * 0.7, "sfx")
        self._want("flame_loop", 0.8 if burning else 0.0, "sfx", fast=True)

    def _want(self, name: str, level: float, kind: str, fast: bool = False) -> None:
        loop = self.loops.setdefault(name, Loop())
        loop.target = level * self._gain(kind)
        if fast:
            loop.level = loop.target if loop.target > loop.level else loop.level

    def _loops(self, dt: float) -> None:
        """Петли плавно выходят на нужную громкость; молчащие — останавливаются."""
        for name, loop in self.loops.items():
            loop.level += (loop.target - loop.level) * min(1.0, dt * 3)
            if loop.player is None and loop.target > 0.01:
                sound = self.sounds.get(name)
                if sound:
                    loop.player = arcade.play_sound(sound[0], volume=0.0, loop=True)
            if loop.player is not None:
                loop.player.volume = max(0.0, loop.level)              # type: ignore[attr-defined]
                if loop.target <= 0.01 and loop.level < 0.01:
                    self._stop(loop)

    @staticmethod
    def _stop(loop: Loop) -> None:
        if loop.player is not None:
            arcade.stop_sound(loop.player)                             # type: ignore[arg-type]
            loop.player = None
        loop.level = 0.0

    def ui(self) -> None:
        self.play("ui_click", 0.6, pitch=0.0)

