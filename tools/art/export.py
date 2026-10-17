# Экспорт модели со скелетом в формат игры: assets/models/<имя>.npz (+ <имя>_albedo.png и <имя>_mask.png из
# запекания). Привязка жёсткая — у вершины одна кость, поэтому в игре скиннинг — одно умножение на матрицу кости.
#
# Состав .npz:
#   positions (n,3) f32, normals (n,3) f32, uvs (n,2) f32, bones (n,) u8, indices (m,) u32 — треугольники;
#   sub_names (k,) — подмеши: body, стволы, топор, glow_<ключ> (тела), <подмеш>_glow_<ключ>;
#   sub_ranges (k,2) — начало и длина в indices;
#   sub_glow (k,4) — цвет и сила свечения (0 — обычный запечённый материал);
#   bone_names, bone_parents (−1 — корень), bone_heads (b,3) — суставы в покое (точки крепления: дуло, кисть);
#   clip_names, clip_ranges (c,2) — первый кадр и число кадров в clip_data, clip_loop (c,) bool,
#   clip_data (F,b,3,4) f32 — матрица кости «покой → поза» по кадрам; fps.
# Оси — как в Blender: X вправо, Y вперёд (перёд модели), Z вверх, метры. Петля: последний кадр равен первому.
import os
import bpy
import numpy as np
import style
from rig import FPS


def _mesh_arrays(o, bone_index):
    me = o.data
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, np.int64)
    me.loop_triangles.foreach_get("loops", tris)
    loop_vert = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", loop_vert)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(o.matrix_world, np.float32)
    co = co @ M[:3, :3].T + M[:3, 3]
    nrm = np.empty(len(me.loops) * 3, np.float32)
    me.corner_normals.foreach_get("vector", nrm)
    nrm = nrm.reshape(-1, 3) @ np.linalg.inv(M[:3, :3])        # нормали — обратной транспонированной
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    uv = np.zeros((len(me.loops), 2), np.float32)
    if me.uv_layers.active:
        flat = np.empty(len(me.loops) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", flat)
        uv = flat.reshape(-1, 2)
    vbone = np.zeros(len(me.vertices), np.uint8)
    names = {g.index: g.name for g in o.vertex_groups}
    for v in me.vertices:
        if v.groups:
            vbone[v.index] = bone_index[names[max(v.groups, key=lambda g: g.weight).group]]
    # вершина игры — угол треугольника; одинаковые углы сливаются
    loops = tris
    rows = np.concatenate([co[loop_vert[loops]], nrm[loops], uv[loops], vbone[loop_vert[loops]][:, None].astype(np.float32)], 1)
    uniq, inverse = np.unique(np.round(rows, 5), axis=0, return_inverse=True)
    return uniq[:, :3], uniq[:, 3:6], uniq[:, 6:8], uniq[:, 8].astype(np.uint8), inverse.ravel().astype(np.uint32)


def export(model, extra=None):
    """Пишет .npz рядом с текстурами и сохраняет .blend для правок. Вызывать после bake()."""
    arm = model.arm
    model.rest()
    bpy.context.view_layer.update()
    bones = list(arm.data.bones)
    bone_index = {b.name: i for i, b in enumerate(bones)}
    meshes = [(name, o, (0, 0, 0, 0)) for name, o in model.objects.items() if o.type == 'MESH']
    for go, _ in model.glow_objects:
        owner, key = go.name.split("_glow_")
        col, _, _, glow = style.PAL[key]
        # свечение тела — «glow_<ключ>», подмеша — «<подмеш>_glow_<ключ>»: игра включает его вместе с подмешем
        meshes.append(("glow_" + key if owner == "body" else go.name, go, (*col, glow)))

    pos, nrm, uv, bone, idx, sub_names, sub_ranges, sub_glow = [], [], [], [], [], [], [], []
    base = start = 0
    for name, o, glow in meshes:
        p, n, t, b, i = _mesh_arrays(o, bone_index)
        pos.append(p); nrm.append(n); uv.append(t); bone.append(b); idx.append(i + base)
        sub_names.append(name)
        sub_ranges.append((start, len(i)))
        sub_glow.append(glow)
        base += len(p)
        start += len(i)

    rest_inv = [b.matrix_local.inverted() for b in bones]
    clip_names, clip_ranges, clip_loop, frames = [], [], [], []
    scn = bpy.context.scene
    for act in model.actions:
        arm.animation_data.action = act
        n = int(act.frame_end) + 1
        clip_names.append(act.name)
        clip_ranges.append((len(frames), n))
        clip_loop.append(bool(act.use_cyclic))
        for f in range(n):
            scn.frame_set(f)
            frames.append([np.array(arm.pose.bones[b.name].matrix @ rest_inv[i], np.float32)[:3]
                           for i, b in enumerate(bones)])
    model.rest()

    data = dict(
        positions=np.concatenate(pos).astype(np.float32), normals=np.concatenate(nrm).astype(np.float32),
        uvs=np.concatenate(uv).astype(np.float32), bones=np.concatenate(bone), indices=np.concatenate(idx),
        sub_names=np.array(sub_names), sub_ranges=np.array(sub_ranges, np.int64), sub_glow=np.array(sub_glow, np.float32),
        bone_names=np.array([b.name for b in bones]),
        bone_parents=np.array([bone_index[b.parent.name] if b.parent else -1 for b in bones], np.int16),
        bone_heads=np.array([tuple(b.head_local) for b in bones], np.float32),
        clip_names=np.array(clip_names), clip_ranges=np.array(clip_ranges, np.int64).reshape(-1, 2),
        clip_loop=np.array(clip_loop, bool), clip_data=np.array(frames, np.float32).reshape(-1, len(bones), 3, 4),
        fps=np.array(FPS),
    )
    for k, v in (extra or {}).items():
        data[k] = np.asarray(v)
    path = os.path.join(style.TEX, model.name + ".npz")
    np.savez_compressed(path, **data)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(style.ART, "blend", model.name + ".blend"))
    tris = len(data["indices"]) // 3
    print(f"[export] {model.name}: {len(data['positions'])} вершин, {tris} треугольников, {len(bones)} костей, "
          f"клипы {clip_names} → {path}", flush=True)
