"""Непрерывный материал земли и воды; маски биома остаются привязаны к клеткам симуляции."""

from __future__ import annotations

import arcade
import numpy as np
from arcade.gl import BufferDescription

from starrun.sim import terrain as T
from starrun.ui.gl3d import Camera3D
from starrun.ui.mesh3d import RECEIVER_VS

FS = """
#version 330
uniform sampler2D u_fields;
uniform float u_size;
uniform float u_time;
uniform float u_desert;
uniform vec3 u_ground;
uniform vec3 u_alt;
in vec3 v_wp;
out vec4 f_color;
float hash(vec2 p) {
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
void main() {
    vec2 p = v_wp.xy;
    vec4 field = texture(u_fields, clamp(p / u_size, 0.0, 1.0));
    float broad = noise(p * 0.003) * 0.6 + noise(p * 0.012) * 0.4;
    float grain = noise(p * 0.3) - 0.5;
    grain *= 1.0 - smoothstep(0.8, 2.5, length(fwidth(p)));
    vec3 color = mix(u_ground, u_alt, broad) * (0.88 + noise(p * 0.065) * 0.22 + grain * 0.10);
    // Маски пространственно согласованы: дорогие материалы считаем только там,
    // где они влияют на цвет. При нулевой маске прежний mix возвращал color.
    if (field.a > 0.03) {
        float track = smoothstep(0.03, 0.7, field.a) * (0.65 + noise(p * 0.04) * 0.2);
        color = mix(color, mix(vec3(0.32, 0.29, 0.20), vec3(0.50, 0.42, 0.29), u_desert), track);
    }
    if (field.b > 0.1) {
        float mud = smoothstep(0.1, 0.85, field.b);
        color = mix(color, vec3(0.14, 0.18, 0.13) * (0.8 + broad * 0.4), mud * 0.85);
    }
    if (field.g > 0.1) {
        float sand = smoothstep(0.10, 0.8, field.g);
        float dune = 0.94 + sin(p.x * 0.065 + noise(p * 0.013) * 3.0) * 0.045;
        color = mix(color, vec3(0.59, 0.48, 0.29) * dune, sand);
    }
    if (field.r > 0.2) {
        float water = smoothstep(0.2, 0.75, field.r);
        float ripple = sin(p.x * 0.05 + u_time * 0.9) * sin(p.y * 0.075 - u_time * 1.1);
        vec3 wet = mix(vec3(0.10, 0.22, 0.25), vec3(0.20, 0.35, 0.37), broad) + max(0.0, ripple - 0.65) * 0.10;
        color = mix(color, wet, water);
    }
    f_color = vec4(color, 1.0);
}
"""


class Ground3D:
    def __init__(self, ctx: arcade.ArcadeContext) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=RECEIVER_VS, fragment_shader=FS)
        self.vertices = ctx.buffer(reserve=32)
        self.geometry = ctx.geometry([BufferDescription(self.vertices, "2f", ["in_pos"])], mode=ctx.TRIANGLE_STRIP)
        self.terrain: T.Terrain | None = None
        self.texture: arcade.gl.Texture2D | None = None

    def draw(self, terrain: T.Terrain | None, cam: Camera3D, time: float,
             rect: tuple[float, float, float, float] | None = None) -> None:
        if terrain is None:
            return
        p = self.program
        if terrain is not self.terrain:
            if self.texture is not None:
                self.texture.delete()
            data = np.stack([terrain.kind == T.WATER, terrain.soil == T.SAND,
                             terrain.soil == T.MUD, terrain.paths], -1).astype(np.uint8) * 255
            self.texture = self.ctx.texture((terrain.n, terrain.n), components=4, data=data.tobytes())
            self.texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
            self.texture.wrap_x = self.texture.wrap_y = self.ctx.CLAMP_TO_EDGE
            self.terrain = terrain
            p["u_fields"], p["u_size"] = 0, terrain.size_px
            p["u_ground"] = tuple(v / 255 for v in terrain.biome.ground)
            p["u_alt"] = tuple(v / 255 for v in terrain.biome.ground_alt)
            p["u_desert"] = float(terrain.biome.tree == "cactus")
        assert self.texture is not None
        # Scene3D уже вычисляет этот прямоугольник для отсечения — повторно четыре
        # раза инвертировать матрицу камеры для земли не требуется.
        l, b, r, top = cam.ground_rect() if rect is None else rect  # noqa: E741
        self.vertices.write(np.array([l, b, r, b, l, top, r, top], np.float32).tobytes())
        p["u_vp"] = tuple(cam.view_proj.T.ravel())
        p["u_time"] = time
        self.texture.use(0)
        self.geometry.render(p)
