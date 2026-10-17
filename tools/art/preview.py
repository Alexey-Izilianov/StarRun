# Превью для самопроверки: виды с игровой камеры (сверху под 55°, как в игре) и крупный план, раскадровка клипов
# одним листом. Рендер Eevee, свет — солнце и небо; результат — tools/art/previews/<имя>_*.png.
import os, math
import bpy
import numpy as np
from mathutils import Vector
import style

OUT = os.path.join(style.ART, "previews")
# вид: (наклон камеры, азимут — 0: камера с юга, модель смотрит на север, т. е. от камеры), дистанция ×
VIEWS = {"game": (55, 0, 1.0), "game_face": (55, 180, 1.0), "hero": (18, 150, 0.8), "side": (8, 90, 0.9),
         "top": (89, 0, 1.0)}


def _scene(res):
    scn = bpy.context.scene
    try:
        scn.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scn.render.engine = 'BLENDER_EEVEE'
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.film_transparent = False
    scn.view_settings.view_transform = 'AgX'
    scn.view_settings.look = 'AgX - Medium High Contrast'
    if hasattr(scn.eevee, "taa_render_samples"):
        scn.eevee.taa_render_samples = 32
    world = bpy.data.worlds.get("pv") or bpy.data.worlds.new("pv")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.32, 0.38, 0.48, 1)
    bg.inputs["Strength"].default_value = 0.7
    if "pv_sun" not in bpy.data.objects:
        sun = bpy.data.objects.new("pv_sun", bpy.data.lights.new("pv_sun", 'SUN'))
        sun.data.energy = 4.5
        sun.data.color = (1.0, 0.94, 0.84)
        sun.data.angle = math.radians(4)
        sun.rotation_euler = (math.radians(42), math.radians(12), math.radians(-30))
        scn.collection.objects.link(sun)
        bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
        g = bpy.context.active_object
        g.name = "pv_ground"
        gm = bpy.data.materials.new("pv_ground")
        gm.use_nodes = True
        b = gm.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.2, 0.15, 0.09, 1)       # пустынная почва, как в игре
        b.inputs["Roughness"].default_value = 0.95
        g.data.materials.append(gm)
        cam = bpy.data.objects.new("pv_cam", bpy.data.cameras.new("pv_cam"))
        scn.collection.objects.link(cam)
        scn.camera = cam
    return scn


def _shoot(path, target, pitch, yaw, dist, lens=50):
    scn = bpy.context.scene
    cam = scn.camera
    cam.data.lens = lens
    p, y = math.radians(pitch), math.radians(yaw)
    t = Vector(target)
    cam.location = t + Vector((math.sin(y) * math.cos(p) * dist, -math.cos(y) * math.cos(p) * dist, math.sin(p) * dist))
    cam.rotation_euler = (t - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _hide(model, hide):
    for name, o in model.objects.items():
        o.hide_render = name in hide


def model_views(model, name, target=(0, 0, 1.0), dist=4.0, views=("game", "game_face", "hero", "side"),
                hide=(), res=(900, 900)):
    _scene(res)
    model.rest()
    _hide(model, hide)
    paths = []
    for v in views:
        pitch, yaw, k = VIEWS[v]
        path = os.path.join(OUT, f"{name}_{v}.png")
        _shoot(path, target, pitch, yaw, dist * k)
        paths.append(path)
    sheet(paths, os.path.join(OUT, f"{name}_views.png"), cols=len(paths))
    print("[preview]", name, "виды готовы", flush=True)


def anim_strip(model, name, clips, phases=6, dist=4.0, target=(0, 0, 1.0), view="hero", show=None, res=(360, 360)):
    """Лист: строка на клип, phases кадров по клипу. show — {клип: подмеши, которые видны} (топор — в рубке)."""
    _scene(res)
    arm = model.arm
    paths = []
    pitch, yaw, k = VIEWS[view]
    for clip in clips:
        act = bpy.data.actions[clip]
        arm.animation_data.action = act
        visible = (show or {}).get(clip, ("body", "mar"))
        for n, o in model.objects.items():
            o.hide_render = n not in visible
        end = int(act.frame_end)
        for i in range(phases):
            f = round(i * end / (phases - (1 if not act.use_cyclic else 0)))
            bpy.context.scene.frame_set(min(f, end))
            path = os.path.join(OUT, f"_{name}_{clip}_{i}.png")
            _shoot(path, target, pitch, yaw, dist * k)
            paths.append(path)
    sheet(paths, os.path.join(OUT, f"{name}_anim.png"), cols=phases)
    model.rest()
    print("[preview]", name, "раскадровка готова", flush=True)


def sheet(paths, out, cols):
    imgs = [bpy.data.images.load(p, check_existing=False) for p in paths]
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / cols)
    canvas = np.zeros((rows * h, cols * w, 4), np.float32)
    for i, im in enumerate(imgs):
        px = np.empty(w * h * 4, np.float32)
        im.pixels.foreach_get(px)
        r, c = divmod(i, cols)
        canvas[(rows - 1 - r) * h:(rows - r) * h, c * w:(c + 1) * w] = px.reshape(h, w, 4)   # пиксели Blender — снизу вверх
        bpy.data.images.remove(im)
    out_img = bpy.data.images.new(os.path.basename(out), cols * w, rows * h, alpha=True)
    out_img.pixels.foreach_set(canvas.ravel())
    out_img.filepath_raw = out
    out_img.file_format = 'PNG'
    out_img.save()
    bpy.data.images.remove(out_img)
