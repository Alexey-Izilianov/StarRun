from __future__ import annotations

import re
from pathlib import Path

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
    have = names()
    assert wanted <= have, wanted - have


def test_every_biome_has_ambience() -> None:
    import tomllib

    biomes = tomllib.loads((ROOT / "data" / "biomes.toml").read_text(encoding="utf-8"))
    ids = {k for k, v in biomes.items() if isinstance(v, dict) and "tree" in v}
    assert {f"amb_{b}" for b in ids} <= names()
