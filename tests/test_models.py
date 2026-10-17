from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[1] / "assets" / "models"
NAMES = sorted(p.stem for p in MODELS.glob("*.npz") if not p.stem.startswith("flora_"))
FLORA = sorted(p.stem for p in MODELS.glob("flora_*.npz"))


@pytest.mark.parametrize("name", NAMES)
def test_model_file_is_consistent(name: str) -> None:
    """Файл из tools/art/export.py: треугольники ссылаются на вершины, вершины — на кости, клипы — в пределах
    матриц, петли замкнуты, текстура рядом."""
    m = np.load(MODELS / f"{name}.npz")
    n = len(m["positions"])
    bones = len(m["bone_names"])
    assert m["normals"].shape == (n, 3) and m["uvs"].shape == (n, 2) and m["bones"].shape == (n,)
    assert m["indices"].max() < n and len(m["indices"]) % 3 == 0
    assert m["bones"].max() < bones
    assert (m["bone_parents"] < np.arange(bones)).all()          # родитель раньше ребёнка — скиннинг одним проходом
    start, count = m["sub_ranges"].T
    assert start[0] == 0 and (start + count)[-1] == len(m["indices"]) and "body" in m["sub_names"]
    data = m["clip_data"]
    assert data.shape[1:] == (bones, 3, 4)
    for (first, frames), loop in zip(m["clip_ranges"], m["clip_loop"], strict=True):
        assert first + frames <= len(data)
        if loop:
            assert np.allclose(data[first], data[first + frames - 1], atol=1e-3)
    assert (MODELS / f"{name}_albedo.png").exists()


def test_first_models_exist() -> None:
    assert {"marine", "worker", "warrior"} <= set(NAMES)


@pytest.mark.parametrize("name", FLORA)
def test_flora_file_is_consistent(name: str) -> None:
    """Дерево из tools/art/flora.py: цвет вершин в 0..1, стоит на земле, не выше 2,5 м (не закрывает бой)."""
    m = np.load(MODELS / f"{name}.npz")
    n = len(m["positions"])
    assert m["normals"].shape == (n, 3) and m["colors"].shape == (n, 4)
    assert m["indices"].max() < n and len(m["indices"]) % 3 == 0
    assert m["colors"].min() >= 0 and m["colors"].max() <= 1
    assert m["positions"][:, 2].min() > -0.1 and m["positions"][:, 2].max() < 2.5


def test_every_biome_tree_has_models() -> None:
    assert {n.rsplit("_", 1)[0] for n in FLORA} == {"flora_pine", "flora_palm", "flora_cactus", "flora_swamp"}
