"""Изолированный GPU A/B замер материала земли, без симуляции и рендера юнитов.

Перед изменением шейдера сохраните FS в файл, затем:
python tools/profile_ground.py --baseline .pytest_cache/ground3d-before.frag
Проверяются пиксели до/после и GPU-время, чередуя программы, чтобы сгладить дрейф частоты.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arcade
import numpy as np

from starrun.balance import Balance
from starrun.sim import terrain as T
from starrun.sim.world import World
from starrun.ui.gl3d import Camera3D
from starrun.ui.ground3d import Ground3D
from starrun.ui.mesh3d import RECEIVER_VS
from starrun.ui.postfx import PostFX


def profile(baseline: Path) -> dict[str, object]:
    window = arcade.Window(1280, 720, "Ground GPU check", visible=False, vsync=False, antialiasing=False)
    ctx = window.ctx
    size = (1600, 900)
    target = ctx.framebuffer(color_attachments=[ctx.texture(size, components=4)])
    old = Ground3D(ctx)
    old.program = ctx.program(vertex_shader=RECEIVER_VS, fragment_shader=baseline.read_text(encoding="utf-8"))
    new = Ground3D(ctx)
    cam = Camera3D()
    balance = Balance.load()
    report: dict[str, object] = {"size": size, "batch_draws": 40, "batches": 12}
    query = ctx.query(time=True)
    target.use()
    ctx.disable(ctx.BLEND, ctx.DEPTH_TEST)
    for biome, mask in (("forest", None), ("desert", "sand"), ("jungle", "water"), ("swamp", "mud")):
        world = World(balance, 42, biome)
        terrain = world.terrain
        assert terrain is not None
        center = terrain.center
        if mask is not None:
            cells = np.argwhere({"sand": terrain.soil == T.SAND, "water": terrain.kind == T.WATER,
                                 "mud": terrain.soil == T.MUD}[mask])
            if len(cells):
                y, x = cells[len(cells) // 2]
                center = terrain.cell_center(int(x), int(y))
        cam.update(center, size[0] / size[1])
        images = []
        for renderer in (old, new):
            renderer.draw(terrain, cam, 7.5)
            ctx.finish()
            images.append(np.frombuffer(target.read(components=4), np.uint8).astype(np.int16))
        delta = np.abs(images[0] - images[1])
        assert delta.max() <= 1, f"{biome}: pixel difference {delta.max()}"
        for _ in range(20):
            old.draw(terrain, cam, 7.5)
            new.draw(terrain, cam, 7.5)
        ctx.finish()
        times: dict[str, list[float]] = {"before": [], "after": []}
        for batch in range(12):
            order = (("before", old), ("after", new)) if batch % 2 == 0 else (("after", new), ("before", old))
            for name, renderer in order:
                with query:
                    for _ in range(40):
                        renderer.draw(terrain, cam, 7.5)
                times[name].append(query.time_elapsed / 1_000_000 / 40)
        report[biome] = {"max_channel_delta": int(delta.max()),
                         "different_channels": int(np.count_nonzero(delta)),
                         **{f"{name}_median_gpu_ms": round(float(np.median(values)), 4)
                            for name, values in times.items()}}
    # Повторные изменения размера: объект старого FBO/текстуры больше не нужен после _fit.
    postfx = PostFX(ctx)
    for fit in ((1600, 900), (1280, 720), (1920, 1080), (1600, 900)):
        postfx._fit(fit)
        assert postfx.fbo is not None and postfx.tex is not None
        assert postfx.fbo.size == fit and postfx.tex.size == fit
    report["postfx_resize"] = "ok"
    ctx.screen.use()
    window.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("screenshots/ground-profile.json"))
    args = parser.parse_args()
    result = profile(args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
