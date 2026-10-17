"""Постобработка 3D-кадра перед HUD: адаптивная резкость (по мотивам AMD CAS), лёгкий контраст и насыщенность.

Кадр уже нарисован в основной буфер окна со сглаживанием MSAA. Arcade не умеет копировать основной буфер
(copy_framebuffer читает COLOR_ATTACHMENT0, а у основного — GL_BACK), поэтому копирование — прямыми вызовами
OpenGL: glBlitFramebuffer сводит сэмплы MSAA в обычную текстуру, затем полноэкранный проход пишет результат обратно.
"""

from __future__ import annotations

import arcade
from pyglet import gl

VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);     // треугольник на весь экран без буферов
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FS = """
#version 330
uniform sampler2D u_img;
uniform vec2 u_texel;
uniform float u_sharp;         // 0..1
uniform float u_contrast;
uniform float u_saturation;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 c = texture(u_img, v_uv).rgb;
    vec3 n = texture(u_img, v_uv + vec2(0.0, u_texel.y)).rgb;
    vec3 s = texture(u_img, v_uv - vec2(0.0, u_texel.y)).rgb;
    vec3 e = texture(u_img, v_uv + vec2(u_texel.x, 0.0)).rgb;
    vec3 w = texture(u_img, v_uv - vec2(u_texel.x, 0.0)).rgb;
    // резкость сильнее там, где перепад мал, и слабее у контрастных краёв — без ореолов
    vec3 mn = min(c, min(min(n, s), min(e, w)));
    vec3 mx = max(c, max(max(n, s), max(e, w)));
    vec3 amp = sqrt(clamp(min(mn, 1.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    vec3 k = -amp / mix(8.0, 5.0, u_sharp);
    vec3 col = (c + (n + s + e + w) * k) / (1.0 + 4.0 * k);
    col = (col - 0.5) * u_contrast + 0.5;
    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(lum), col, u_saturation);
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class PostFX:
    def __init__(self, ctx: arcade.ArcadeContext, sharp: float = 0.6, contrast: float = 1.06,
                 saturation: float = 1.1) -> None:
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.program["u_img"] = 0
        self.program["u_sharp"] = sharp
        self.program["u_contrast"] = contrast
        self.program["u_saturation"] = saturation
        self.geometry = ctx.geometry()          # пустая: вершины строит сам шейдер
        self.size = (0, 0)
        self.tex: arcade.gl.Texture2D | None = None
        self.fbo: arcade.gl.Framebuffer | None = None

    def _fit(self, size: tuple[int, int]) -> None:
        if size == self.size and self.tex is not None:
            return
        self.size = size
        self.tex = self.ctx.texture(size, components=4)
        self.tex.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.tex])
        self.program["u_texel"] = (1.0 / size[0], 1.0 / size[1])

    def apply(self) -> None:
        """Сводит MSAA основного буфера в текстуру и рисует её обратно с обработкой."""
        w, h = self.ctx.screen.size
        self._fit((w, h))
        assert self.fbo is not None and self.tex is not None
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        gl.glReadBuffer(gl.GL_BACK)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.fbo.glo)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDrawBuffer(gl.GL_BACK)
        self.ctx.screen.use(force=True)          # Arcade должен узнать, что связан основной буфер
        self.ctx.disable(self.ctx.BLEND, self.ctx.DEPTH_TEST)
        self.tex.use(0)
        self.geometry.render(self.program, mode=self.ctx.TRIANGLES, vertices=3)
        self.ctx.enable(self.ctx.BLEND)
