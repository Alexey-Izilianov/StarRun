"""3D-меши для отрисовки пакетами (инстансингом): модели со скелетом из assets/models и простые статичные меши
(деревья, стены, заборы, скалы), собранные в коде. Все экземпляры одного меша — один вызов видеокарты.

Скиннинг на видеокарте: у вершины одна кость (жёсткая привязка), матрицы костей всех клипов лежат в текстуре
(строка — кадр, по три тексела на кость), экземпляр несёт два кадра и долю смешивания между ними.
Свет: солнце с полусферическим окружением, блик по маске металл/гладкость, тень из карты теней (билинейно),
точечные источники от взрывов и выстрелов.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import arcade
import numpy as np
from arcade.gl import BufferDescription
from PIL import Image

# Точечный свет (взрывы, выстрелы, огонь — ui/fx3d.py): до восьми источников, затухание к радиусу квадратом
POINT_LIGHTS = """
uniform vec4 u_pl_pos[8];      // x, y, z, радиус (0 — нет источника)
uniform vec4 u_pl_col[8];      // цвет × сила
vec3 point_lights(vec3 n, vec3 wp) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 8; i++) {
        vec4 p = u_pl_pos[i];
        if (p.w <= 0.0) continue;
        vec3 d = p.xyz - wp;
        float dist = length(d);
        float a = clamp(1.0 - dist / p.w, 0.0, 1.0);
        float ndl = max(dot(n, d / max(dist, 1e-3)), 0.0) * 0.75 + 0.25;
        sum += u_pl_col[i].rgb * a * a * ndl;
    }
    return sum;
}
"""

LIGHT = POINT_LIGHTS + """
uniform vec3 u_sun;            // направление к солнцу
uniform vec3 u_eye;
uniform sampler2D u_shadow;    // глубина без аппаратного сравнения: sampler2DShadow Arcade не поддерживает
uniform mat4 u_light;          // вид-проекция солнца для карты теней
uniform float u_texel;         // размер тексела карты теней

float shadow(vec3 wp) {
    vec4 l = u_light * vec4(wp, 1.0);
    vec3 c = l.xyz / l.w * 0.5 + 0.5;
    if (c.x < 0.0 || c.x > 1.0 || c.y < 0.0 || c.y > 1.0 || c.z > 1.0) return 1.0;
    float z = c.z - 0.0015;
    // билинейное сглаживание сравнений по четырём соседним текселям: край тени плавный, а не ступеньками
    vec2 st = c.xy / u_texel - 0.5;
    vec2 f = fract(st);
    vec2 b = (floor(st) + 0.5) * u_texel;
    float s00 = step(z, texture(u_shadow, b).r);
    float s10 = step(z, texture(u_shadow, b + vec2(u_texel, 0.0)).r);
    float s01 = step(z, texture(u_shadow, b + vec2(0.0, u_texel)).r);
    float s11 = step(z, texture(u_shadow, b + vec2(u_texel, u_texel)).r);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

vec3 shade(vec3 albedo, vec3 n, vec3 wp, float metal, float gloss) {
    float sh = shadow(wp);
    float ndl = max(dot(n, u_sun), 0.0);
    vec3 sky = mix(vec3(0.36, 0.33, 0.29), vec3(0.62, 0.66, 0.74), n.z * 0.5 + 0.5);   // снизу — тёплый отсвет земли
    vec3 col = albedo * (sky * 0.62 + vec3(1.0, 0.94, 0.84) * ndl * sh * 0.9);
    vec3 v = normalize(u_eye - wp);
    vec3 h = normalize(u_sun + v);
    float spec = pow(max(dot(n, h), 0.0), mix(6.0, 48.0, gloss)) * gloss * (0.25 + 0.75 * sh);
    col += mix(vec3(1.0), albedo * 2.0, metal) * spec * 0.45;
    float rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.12;                             // кромка силуэта
    return col + vec3(rim) + albedo * point_lights(n, wp);
}
"""

SKINNED_VS = """
#version 330
uniform mat4 u_vp;
uniform sampler2D u_anim;
uniform float u_px;
in vec3 in_pos; in vec3 in_nrm; in vec2 in_uv; in float in_bone;
in vec4 i_a;       // x, y, z, курс
in vec4 i_b;       // кадр A, кадр B, доля B, масштаб
in vec4 i_c;       // цвет подсветки и её сила (вспышка попадания, горение)
out vec3 v_wp; out vec3 v_n; out vec2 v_uv; out vec4 v_tint;

mat4 bone(int frame, int b) {
    vec4 r0 = texelFetch(u_anim, ivec2(b * 3, frame), 0);
    vec4 r1 = texelFetch(u_anim, ivec2(b * 3 + 1, frame), 0);
    vec4 r2 = texelFetch(u_anim, ivec2(b * 3 + 2, frame), 0);
    return mat4(vec4(r0.x, r1.x, r2.x, 0.0), vec4(r0.y, r1.y, r2.y, 0.0), vec4(r0.z, r1.z, r2.z, 0.0),
                vec4(r0.w, r1.w, r2.w, 1.0));
}

void main() {
    int b = int(in_bone + 0.5);
    mat4 m = bone(int(i_b.x), b) * (1.0 - i_b.z) + bone(int(i_b.y), b) * i_b.z;
    vec3 p = (m * vec4(in_pos, 1.0)).xyz;
    vec3 n = mat3(m) * in_nrm;
    float c = cos(i_a.w), s = sin(i_a.w);
    mat2 r = mat2(c, s, -s, c);
    p.xy = r * p.xy;
    n.xy = r * n.xy;
    v_wp = i_a.xyz + p * u_px * i_b.w;
    v_n = normalize(n);
    v_uv = in_uv;
    v_tint = i_c;
    gl_Position = u_vp * vec4(v_wp, 1.0);
}
"""

SKINNED_FS = """
#version 330
uniform sampler2D u_albedo;
uniform sampler2D u_mask;
uniform vec4 u_glow;           // свечение подмеша: цвет и сила; 0 — обычный запечённый материал
in vec3 v_wp; in vec3 v_n; in vec2 v_uv; in vec4 v_tint;
out vec4 f_color;
""" + LIGHT + """
void main() {
    vec3 col;
    if (u_glow.a > 0.0) {
        col = min(u_glow.rgb * (0.9 + 0.12 * u_glow.a), vec3(1.0));
    } else {
        // смещение мип-уровня к резкости: модели на экране мелкие, иначе берётся размытый уровень
        vec4 mk = texture(u_mask, v_uv, -0.75);
        col = shade(texture(u_albedo, v_uv, -0.75).rgb, normalize(v_n), v_wp, mk.r, mk.a);
    }
    f_color = vec4(mix(col, v_tint.rgb, v_tint.a), 1.0);
}
"""

STATIC_VS = """
#version 330
uniform mat4 u_vp;
uniform vec2 u_player;
in vec3 in_pos; in vec3 in_nrm; in vec4 in_col;
in vec4 i_a;       // x, y, z, курс
in vec4 i_s;       // масштаб по x, y, z; прозрачность
in vec4 i_c;       // подмес цвета и его сила
out vec3 v_wp; out vec3 v_n; out vec4 v_col; out float v_fade;
void main() {
    // стоит между камерой и десантником (южнее, в полосе к камере) — растворяется сеткой, иначе его не видно
    vec2 rel = i_a.xy - u_player;
    v_fade = (rel.y < 40.0 && rel.y > -230.0 && abs(rel.x) < 70.0 + max(0.0, -rel.y) * 0.15) ? 1.0 : 0.0;
    float c = cos(i_a.w), s = sin(i_a.w);
    mat2 r = mat2(c, s, -s, c);
    vec3 p = in_pos * i_s.xyz;
    p.xy = r * p.xy;
    vec3 n = in_nrm / max(i_s.xyz, vec3(1e-4));
    n.xy = r * n.xy;
    v_wp = i_a.xyz + p;
    v_n = normalize(n);
    v_col = vec4(mix(in_col.rgb, i_c.rgb, i_c.a), in_col.a * i_s.w);
    gl_Position = u_vp * vec4(v_wp, 1.0);
}
"""

STATIC_FS = """
#version 330
in vec3 v_wp; in vec3 v_n; in vec4 v_col; in float v_fade;
out vec4 f_color;
const float BAYER[16] = float[](0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
""" + LIGHT + """
void main() {
    if (v_fade > 0.5) {                     // растворение сеткой 4×4: глубина честная, сортировать не нужно
        ivec2 q = ivec2(gl_FragCoord.xy) % 4;
        if ((BAYER[q.y * 4 + q.x] + 0.5) / 16.0 < 0.65) discard;
    }
    f_color = vec4(shade(v_col.rgb, normalize(v_n), v_wp, 0.0, 0.15), v_col.a);
}
"""

DEPTH_FS = """
#version 330
void main() {}
"""

RECEIVER_VS = """
#version 330
uniform mat4 u_vp;
in vec2 in_pos;
out vec3 v_wp;
void main() {
    v_wp = vec3(in_pos, 0.0);
    gl_Position = u_vp * vec4(v_wp, 1.0);
}
"""

RECEIVER_FS = """
#version 330
in vec3 v_wp;
out vec4 f_color;
""" + LIGHT + """
void main() {
    float s = shadow(v_wp);
    if (s > 0.99) discard;
    f_color = vec4(0.06, 0.05, 0.09, (1.0 - s) * 0.5);    // тень ложится на землю 2D-слоёв поверх
}
"""


def _flip(img: Image.Image) -> bytes:
    return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM).tobytes()


def load_texture(ctx: arcade.ArcadeContext, path: Path) -> arcade.gl.Texture2D:
    img = Image.open(path).convert("RGBA")
    tex = ctx.texture(img.size, components=4, data=_flip(img))
    tex.build_mipmaps()
    tex.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
    return tex


@dataclass
class Clip:
    first: int       # первая строка в текстуре анимаций
    frames: int      # кадров (у петли последний равен первому)
    loop: bool


class SkinnedModel:
    """Модель из tools/art/export.py. draw() рисует все экземпляры одним вызовом на подмеш."""

    FLOATS = 12

    def __init__(self, ctx: arcade.ArcadeContext, path: Path, capacity: int = 512) -> None:
        d = np.load(path)
        verts = np.concatenate([d["positions"], d["normals"], d["uvs"], d["bones"][:, None].astype(np.float32)], 1)
        self.vbo = ctx.buffer(data=verts.astype(np.float32).tobytes())
        self.ibo = ctx.buffer(data=d["indices"].astype(np.uint32).tobytes())
        self.capacity = capacity
        self.inst = ctx.buffer(reserve=capacity * self.FLOATS * 4)
        layout = [BufferDescription(self.vbo, "3f 3f 2f 1f", ["in_pos", "in_nrm", "in_uv", "in_bone"]),
                  BufferDescription(self.inst, "4f 4f 4f", ["i_a", "i_b", "i_c"], instanced=True)]
        self.geometry = ctx.geometry(layout, index_buffer=self.ibo, index_element_size=4)
        self.subs = {str(n): (int(s), int(c), tuple(g)) for n, (s, c), g in
                     zip(d["sub_names"], d["sub_ranges"], d["sub_glow"], strict=True)}
        anim = d["clip_data"]                       # (F, B, 3, 4)
        frames, bones = anim.shape[:2]
        self.anim = ctx.texture((bones * 3, frames), components=4, dtype="f4",
                                data=anim.reshape(frames, bones * 3, 4).astype(np.float32).tobytes())
        self.anim.filter = (ctx.NEAREST, ctx.NEAREST)
        self.clips = {str(n): Clip(int(f), int(c), bool(lp)) for n, (f, c), lp in
                      zip(d["clip_names"], d["clip_ranges"], d["clip_loop"], strict=True)}
        self.fps = float(d["fps"])
        self.bone_names = [str(b) for b in d["bone_names"]]
        stem = path.with_suffix("")
        self.albedo = load_texture(ctx, Path(f"{stem}_albedo.png"))
        self.mask = load_texture(ctx, Path(f"{stem}_mask.png"))

    def duration(self, clip: str) -> float:
        return (self.clips[clip].frames - 1) / self.fps

    def frames(self, clip: str, time: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Кадры A и B и доля B для времени клипа (секунды); петля зацикливается, иначе — стоп на последнем."""
        c = self.clips[clip]
        f = np.asarray(time, np.float64) * self.fps
        span = c.frames - 1
        f = np.mod(f, span) if c.loop else np.clip(f, 0, span)
        a = np.floor(f)
        b = np.minimum(a + 1, span)
        return c.first + a, c.first + b, f - a

    def draw(self, program: arcade.gl.Program, inst: np.ndarray, subs: tuple[str, ...], depth: bool = False) -> None:
        n = min(len(inst), self.capacity)
        if not n:
            return
        self.inst.write(inst[:n].astype(np.float32).tobytes())
        self.anim.use(1)
        if not depth:
            self.albedo.use(2)
            self.mask.use(3)
        for name in subs:
            if name not in self.subs:
                continue
            start, count, glow = self.subs[name]
            if not depth:
                program["u_glow"] = glow if glow[3] > 0 else (0.0, 0.0, 0.0, 0.0)
            self.geometry.render(program, first=start, vertices=count, instances=n)

    def all_subs(self, skip: tuple[str, ...] = ()) -> tuple[str, ...]:
        return tuple(s for s in self.subs if s not in skip)


class StaticMesh:
    """Статичный меш с цветом вершин; экземпляры: положение и курс, масштаб по осям и прозрачность, подмес цвета."""

    FLOATS = 12

    def __init__(self, ctx: arcade.ArcadeContext, parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                 capacity: int = 4096) -> None:
        pos, nrm, col, idx = merge(parts)
        verts = np.concatenate([pos, nrm, col], 1).astype(np.float32)
        self.vbo = ctx.buffer(data=verts.tobytes())
        self.ibo = ctx.buffer(data=idx.astype(np.uint32).tobytes())
        self.count = len(idx)
        self.capacity = capacity
        self.inst = ctx.buffer(reserve=capacity * self.FLOATS * 4)
        self.geometry = ctx.geometry(
            [BufferDescription(self.vbo, "3f 3f 4f", ["in_pos", "in_nrm", "in_col"]),
             BufferDescription(self.inst, "4f 4f 4f", ["i_a", "i_s", "i_c"], instanced=True)],
            index_buffer=self.ibo, index_element_size=4)

    def draw(self, program: arcade.gl.Program, inst: np.ndarray) -> None:
        n = min(len(inst), self.capacity)
        if not n:
            return
        self.inst.write(inst[:n].astype(np.float32).tobytes())
        self.geometry.render(program, vertices=self.count, instances=n)


def instances(pos: np.ndarray, yaw: np.ndarray | float = 0.0, z: np.ndarray | float = 0.0,
              scale: np.ndarray | tuple[float, float, float] = (1, 1, 1), alpha: np.ndarray | float = 1.0,
              tint: np.ndarray | tuple[float, float, float, float] = (0, 0, 0, 0)) -> np.ndarray:
    """Массив экземпляров статичного меша (n, 12)."""
    n = len(pos)
    out = np.zeros((n, 12), np.float32)
    out[:, 0:2] = pos
    out[:, 2] = z
    out[:, 3] = yaw
    out[:, 4:7] = scale
    out[:, 7] = alpha
    out[:, 8:12] = tint
    return out


# ---------------------------------------------------------------------------------------------- построение мешей
# Часть = (позиции, нормали, цвета RGBA 0..1, индексы); единицы — пиксели симуляции, z вверх.

Part = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def merge(parts: list[Part]) -> Part:
    pos, nrm, col, idx = [], [], [], []
    base = 0
    for p, n, c, i in parts:
        pos.append(p)
        nrm.append(n)
        col.append(c)
        idx.append(i + base)
        base += len(p)
    return np.concatenate(pos), np.concatenate(nrm), np.concatenate(col), np.concatenate(idx)


def _color(n: int, color: tuple[float, float, float], top: np.ndarray | None = None) -> np.ndarray:
    """Цвет вершин; top (0..1 по высоте части) — верх светлее, низ темнее: рисованный объём, как в запекании."""
    c = np.tile(np.array([*color, 1.0], np.float32), (n, 1))
    if top is not None:
        c[:, :3] *= (0.78 + 0.34 * top)[:, None]
    return np.clip(c, 0, 1)


Vec3 = tuple[float, float, float]


def box(size: Vec3, center: Vec3, color: Vec3) -> Part:
    sx, sy, sz = (s / 2 for s in size)
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    pos, nrm, idx = [], [], []
    for k, (n, quad) in enumerate(faces):
        for x, y, z in quad:
            pos.append((center[0] + x * sx, center[1] + y * sy, center[2] + z * sz))
            nrm.append(n)
        idx += [4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3]
    p = np.array(pos, np.float32)
    top = (p[:, 2] - p[:, 2].min()) / max(1e-6, np.ptp(p[:, 2]))
    return p, np.array(nrm, np.float32), _color(len(p), color, top), np.array(idx, np.uint32)


def lathe(profile: list[tuple[float, float]], color: tuple[float, float, float], seg: int = 12,
          center: tuple[float, float, float] = (0, 0, 0), scale: tuple[float, float] = (1, 1),
          jag: float = 0.0, seed: int = 0) -> Part:
    """Тело вращения по профилю [(радиус, высота)] снизу вверх: ствол, крона-конус, кактус, валун.
    jag — случайный разброс радиуса по кругу (неровная крона)."""
    rng = np.random.default_rng(seed)
    wob = 1 + rng.uniform(-jag, jag, seg)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    rings = []
    for r, z in profile:
        rr = r * wob
        rings.append(np.stack([center[0] + rr * ca * scale[0], center[1] + rr * sa * scale[1],
                               np.full(seg, center[2] + z)], 1))
    p = np.concatenate(rings).astype(np.float32)
    idx = []
    for i in range(len(profile) - 1):
        for j in range(seg):
            a, b = i * seg + j, i * seg + (j + 1) % seg
            c, d = a + seg, b + seg
            idx += [a, b, d, a, d, c]
    # нормаль перпендикулярна образующей (dr, dz) в плоскости радиус — высота: (dz, −dr)
    nrm = np.zeros_like(p)
    for i, (r, _z) in enumerate(profile):
        lo = profile[max(0, i - 1)]
        hi = profile[min(len(profile) - 1, i + 1)]
        dr, dz = hi[0] - lo[0], hi[1] - lo[1]
        n = np.stack([ca * dz, sa * dz, np.full(seg, -dr)], 1)
        if r < 1e-6:                                     # полюс: вверх у макушки, вниз у дна
            n = np.tile([0.0, 0.0, 1.0 if i == len(profile) - 1 else -1.0], (seg, 1))
        nrm[i * seg:(i + 1) * seg] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    top = (p[:, 2] - p[:, 2].min()) / max(1e-6, np.ptp(p[:, 2]))
    return p, nrm.astype(np.float32), _color(len(p), color, top), np.array(idx, np.uint32)


def blob(radius: float, center: tuple[float, float, float], color: tuple[float, float, float],
         squash: float = 1.0, jag: float = 0.12, seed: int = 0, seg: int = 10, rings: int = 6) -> Part:
    """Неровный шар: крона лиственного дерева, валун."""
    prof = []
    for i in range(rings + 1):
        a = -math.pi / 2 + math.pi * i / rings
        prof.append((radius * math.cos(a), radius * squash * (math.sin(a) + 1)))
    prof[0] = (0.0, prof[0][1])
    prof[-1] = (0.0, prof[-1][1])
    c = (center[0], center[1], center[2] - radius * squash)
    return lathe(prof, color, seg, c, jag=jag, seed=seed)


def leaf(length: float, width: float, yaw: float, droop: float, base: tuple[float, float, float],
         color: tuple[float, float, float]) -> Part:
    """Пальмовый лист: изогнутая вниз полоса из сегментов, видна с обеих сторон."""
    k = 6
    ca, sa = math.cos(yaw), math.sin(yaw)
    pts, idx = [], []
    for i in range(k + 1):
        t = i / k
        d = length * t
        z = base[2] + length * 0.25 * math.sin(t * math.pi * 0.6) - droop * t * t
        w = width * math.sin(math.pi * min(1.0, t * 1.1 + 0.08)) + 0.5
        cx, cy = base[0] + ca * d, base[1] + sa * d
        pts += [(cx - sa * w, cy + ca * w, z), (cx + sa * w, cy - ca * w, z)]
    for i in range(k):
        a = 2 * i
        idx += [a, a + 1, a + 3, a, a + 3, a + 2]
    p = np.array(pts, np.float32)
    idx2 = idx + [i for tri in zip(idx[0::3], idx[2::3], idx[1::3], strict=True) for i in tri]  # обратная сторона
    n = np.tile([0.0, 0.0, 1.0], (len(p), 1)).astype(np.float32)
    top = np.linspace(1, 0.4, len(p))
    return p, n, _color(len(p), color, top), np.array(idx2, np.uint32)
