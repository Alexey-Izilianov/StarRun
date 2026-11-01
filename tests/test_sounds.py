from __future__ import annotations

import random
import re
import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.settings import Settings
from starrun.sim import terrain as T
from starrun.sim.world import Controls, World
from starrun.ui.audio import STEP_SURFACES, Audio

ROOT = Path(__file__).resolve().parents[1]
SOUNDS = ROOT / "assets" / "sounds"


def names() -> set[str]:
    """Имена звуков, как их видит игра: mar_0, mar_1 → mar; фразы — как есть."""
    from starrun.ui.audio import sound_name

    return {sound_name(p.stem) for p in SOUNDS.glob("*.wav")}


def test_every_sound_the_game_asks_for_exists() -> None:
    """Опечатка в имени звука — тишина без ошибки; здесь — падение теста."""
    from starrun.ui import audio

    code = (ROOT / "src" / "starrun" / "ui" / "audio.py").read_text(encoding="utf-8")
    wanted = set(re.findall(r'self\.play\("(\w+)"', code))
    wanted |= {name for name, _ in audio.EVENT_SOUNDS.values()} | set(audio.SHOT_SOUND.values())
    wanted |= {f"voice_{line}" for line in audio.VOICE} | {"chop_cactus", "swarm_loop", "flame_loop", "turret"}
    wanted |= {f"step_{surface}" for surface in STEP_SURFACES}
    have = names()
    assert wanted <= have, wanted - have


def test_every_biome_has_ambience() -> None:
    import tomllib

    biomes = tomllib.loads((ROOT / "data" / "biomes.toml").read_text(encoding="utf-8"))
    ids = {k for k, v in biomes.items() if isinstance(v, dict) and "tree" in v}
    assert {f"amb_{b}" for b in ids} <= names()


def silent_audio(world: World) -> Audio:
    """Настоящая логика аудио, но без загрузки файлов и обращения к звуковому устройству."""
    audio = Audio.__new__(Audio)
    audio.settings = Settings()
    audio.sounds = {}
    audio.loops = {}
    audio.rng = random.Random(3)
    audio.reset(world)
    audio.play = Mock()
    return audio


def step_calls(audio: Audio) -> list[str]:
    return [call.args[0] for call in audio.play.call_args_list if call.args[0].startswith("step_")]


@pytest.mark.parametrize("dt", [1 / 30, 1 / 120])
def test_steps_follow_distance_and_sprint(balance: Balance, dt: float) -> None:
    counts = []
    for sprint in (False, True):
        world = World(balance, 42)
        audio = silent_audio(world)
        for _ in range(round(2 / dt)):
            world.step(dt, Controls((1, 0), sprint))
            audio.update(dt, world, paused=False)
        counts.append(len(step_calls(audio)))
    assert counts == [6, 10]  # одинаковый путь при одинаковом времени, независимо от частоты кадров


def test_wall_chopping_and_idle_do_not_make_steps(balance: Balance) -> None:
    world = World(balance, 42, "forest")
    t = world.terrain
    assert t is not None
    t.kind[:] = T.GROUND
    world.player.pos = t.cell_center(100, 100)
    t.kind[100, 101] = T.TREE
    audio = silent_audio(world)
    for _ in range(20):
        world.step(1 / 120, Controls((1, 0)))
        audio.update(1 / 120, world, paused=False)
    assert "chop" in world.events or world.player.chop >= 0
    assert step_calls(audio) == []
    for _ in range(120):
        world.step(1 / 120, Controls())
        audio.update(1 / 120, world, paused=False)
    assert step_calls(audio) == []


@pytest.mark.parametrize(("kind", "soil", "biome", "surface"), [
    (T.GROUND, T.PLAIN, "forest", "ground"),
    (T.GROUND, T.PLAIN, "desert", "sand"),
    (T.GROUND, T.SAND, "forest", "sand"),
    (T.GROUND, T.MUD, "swamp", "mud"),
    (T.BRIDGE, T.MUD, "swamp", "wood"),
    (T.FLOOR, T.MUD, "swamp", "metal"),
    (T.OPEN_DOOR, T.PLAIN, "forest", "metal"),
])
def test_steps_use_surface_underfoot(balance: Balance, kind: int, soil: int, biome: str, surface: str) -> None:
    world = World(balance, 42, biome)
    t = world.terrain
    assert t is not None
    t.kind[:] = kind
    t.soil[:] = soil
    audio = silent_audio(world)
    for _ in range(10):
        world.player.pos[0] += 8
        audio.update(1 / 30, world, paused=False)
    assert step_calls(audio) == [f"step_{surface}"]


def test_pause_vault_teleport_end_and_restart_clear_steps(balance: Balance) -> None:
    world = World(balance, 42)
    audio = silent_audio(world)
    for state in ("pause", "vault", "teleport", "end"):
        world.player.pos[0] += 60
        audio.update(1 / 30, world, paused=False)
        world.player.vaulting = state == "vault"
        if state == "end":
            world.session.player_died()
        world.player.pos[0] += 1000 if state == "teleport" else 20
        audio.update(1 / 30, world, paused=state == "pause")
        world.player.vaulting = False
        world.player.pos[0] += 20
        audio.update(1 / 30, world, paused=False)
        assert step_calls(audio) == []
        audio.reset(world)
    world = World(balance, 1)
    audio.reset(world)
    audio.update(1 / 30, world, paused=False)
    assert step_calls(audio) == []


def test_step_assets_are_valid_audio() -> None:
    for surface in STEP_SURFACES:
        paths = sorted(SOUNDS.glob(f"step_{surface}_*.wav"))
        assert len(paths) == 3
        samples = []
        for path in paths:
            with wave.open(str(path)) as sound:
                assert sound.getnchannels() == 1
                assert sound.getsampwidth() == 2
                assert sound.getframerate() == 44100
                assert 0.1 < sound.getnframes() / sound.getframerate() < 0.35
                pcm = sound.readframes(sound.getnframes())
            values = np.frombuffer(pcm, "<i2")
            assert 1000 < np.abs(values).max() < 32767
            assert abs(int(values[0])) < 5 and abs(int(values[-1])) < 5
            samples.append(pcm)
        assert len(set(samples)) == 3
