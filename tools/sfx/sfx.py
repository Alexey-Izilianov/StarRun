"""Звуки StarRun синтезом, без чужих записей: выстрелы, взрывы, жуки, десантник, находки, фон биомов и фразы
десантника (голос Windows «Microsoft Pavel», обработанный как рация). Результат — assets/sounds/*.wav.

Приёмы: фильтр — умножение спектра (FFT) на плавную полосу; хвост — свёртка с затухающим шумом (отзвук);
вариации одного звука — другое зерно шума и чуть другой тон. Фоновые петли сшиты перекрёстным затуханием.
Запуск: .venv/Scripts/python tools/sfx/sfx.py [имена…]  (без имён — все; voice — только фразы)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np

SR = 44100
OUT = Path(__file__).resolve().parents[2] / "assets" / "sounds"
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------------------------- основа

def T(d: float, sr: int = SR) -> np.ndarray:
    return np.arange(int(sr * d)) / sr


def noise(d: float, sr: int = SR) -> np.ndarray:
    return rng.standard_normal(int(sr * d))


def band(x: np.ndarray, lo: float = 0.0, hi: float = 20000.0, order: int = 2, sr: int = SR) -> np.ndarray:
    """Полоса lo..hi со скатами как у фильтра Баттерворта порядка order (только амплитуда, фаза нулевая)."""
    f = np.fft.rfftfreq(len(x), 1 / sr)
    f[0] = 1e-3
    g = np.ones_like(f)
    if lo > 0:
        g /= np.sqrt(1 + (lo / f) ** (2 * order))
    if hi < sr / 2:
        g /= np.sqrt(1 + (f / hi) ** (2 * order))
    return np.fft.irfft(np.fft.rfft(x) * g, len(x))


def env(d: float, attack: float, decay: float, sr: int = SR) -> np.ndarray:
    """Огибающая: нарастание за attack, затем экспоненциальный спад с постоянной decay."""
    t = T(d, sr)
    return np.minimum(1.0, t / max(attack, 1e-5)) * np.exp(-np.maximum(0.0, t - attack) / decay)


def tone(freq: np.ndarray | float, d: float, shape: str = "sine", sr: int = SR) -> np.ndarray:
    """Осциллятор с меняющейся частотой (массив по отсчётам): фаза — накопленная сумма."""
    n = int(sr * d)
    f = np.broadcast_to(np.asarray(freq, float), (n,))
    ph = 2 * np.pi * np.cumsum(f) / sr
    if shape == "saw":
        return 2 * ((ph / (2 * np.pi)) % 1.0) - 1
    if shape == "square":
        return np.sign(np.sin(ph))
    return np.sin(ph)


def sweep(f0: float, f1: float, d: float, sr: int = SR) -> np.ndarray:
    """Частота от f0 к f1 по экспоненте."""
    return f0 * (f1 / f0) ** (T(d, sr) / d)


def fit(x: np.ndarray, d: float, sr: int = SR) -> np.ndarray:
    n = int(sr * d)
    return np.pad(x, (0, max(0, n - len(x))))[:n]


def mix(d: float, *parts: tuple[np.ndarray, float], sr: int = SR) -> np.ndarray:
    out = np.zeros(int(sr * d))
    for x, gain in parts:
        out += fit(x, d, sr) * gain
    return out


def at(x: np.ndarray, start: float, d: float, sr: int = SR) -> np.ndarray:
    """Звук x, начинающийся через start секунд, в отрезке длиной d."""
    return fit(np.concatenate([np.zeros(int(sr * start)), x]), d, sr)


def reverb(x: np.ndarray, length: float, wet: float, bright: float = 4000, sr: int = SR) -> np.ndarray:
    """Отзвук: свёртка с затухающим шумом (ранние отражения гуще, хвост темнее)."""
    ir = band(noise(length, sr), 80, bright, sr=sr) * env(length, 0.003, length / 5, sr)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    n = len(x) + len(ir)
    wetx = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[: len(x) + len(ir) // 2]
    return np.concatenate([x, np.zeros(len(wetx) - len(x))]) + wetx * wet


def drive(x: np.ndarray, k: float) -> np.ndarray:
    """Мягкое насыщение: плотнее и громче без щелчков."""
    return np.tanh(k * x / (np.abs(x).max() + 1e-9)) / np.tanh(k)


def clicks(d: float, times: list[float], lo: float = 1500, hi: float = 7000, width: float = 0.004,
           sr: int = SR) -> np.ndarray:
    """Щелчки хитина: короткие всплески шума в моменты times."""
    out = np.zeros(int(sr * d))
    for t0 in times:
        g = band(noise(width * 4, sr), lo, hi, sr=sr) * env(width * 4, 0.0003, width, sr) * rng.uniform(0.5, 1.0)
        i = int(t0 * sr)
        out[i:i + len(g)] += g[: max(0, len(out) - i)]
    return out


def finish(x: np.ndarray, peak: float = 0.89, fade: float = 0.004, sr: int = SR) -> np.ndarray:
    x = x - x.mean()
    n = max(1, int(fade * sr))
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x / (np.abs(x).max() + 1e-9) * peak


def loop_seam(x: np.ndarray, cross: float, sr: int = SR) -> np.ndarray:
    """Петля без шва: хвост длиной cross плавно накладывается на начало."""
    n = int(cross * sr)
    head, body, tail = x[:n], x[n:-n], x[-n:]
    w = np.linspace(0, 1, n)
    return np.concatenate([tail * (1 - w) + head * w, body])


def write(name: str, x: np.ndarray, peak: float = 0.89, sr: int = SR, loop: bool = False) -> None:
    x = x / (np.abs(x).max() + 1e-9) * peak if loop else finish(x, peak, sr=sr)
    OUT.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT / f"{name}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    print(f"[sfx] {name}: {len(x) / sr:.2f} с", flush=True)


# ---------------------------------------------------------------------------------------------- оружие

def ring(d: float, freqs: tuple[float, ...], decay: float, start: float = 0.0) -> np.ndarray:
    """Металлический звон: негармонические обертоны (как у стального корпуса), верхние гаснут быстрее."""
    out = np.zeros(int(SR * d))
    for k, f in enumerate(freqs, 1):
        f *= rng.uniform(0.97, 1.03)
        out += at(np.sin(2 * np.pi * f * T(d)) * env(d, 0.0005, decay / k ** 0.5), start, d) / k ** 0.3
    return out


def gunshot(d: float, body_decay: float, lo: float, hi: float, thump: tuple[float, float, float],
            room: float, wet: float, k: float, pitch: float, metal: float = 1.0) -> np.ndarray:
    """Выстрел: треск, тело, низкий толчок, звон ствольной коробки, лязг затвора через 40–60 мс и звон гильзы."""
    crack = band(noise(d), 2500, 12000) * env(d, 0.0003, 0.004)
    body = band(noise(d), lo * pitch, hi * pitch) * env(d, 0.0008, body_decay)
    f0, f1, td = thump
    low = tone(sweep(f0 * pitch, f1 * pitch, d), d) * env(d, 0.001, td)
    receiver = ring(d, tuple(f * pitch for f in (1870, 3020, 4410, 6230)), 0.045)
    bolt_at = rng.uniform(0.04, 0.06)
    bolt = clicks(d, [bolt_at], 1500, 7000, 0.003) * 1.6 + ring(d, (1240 * pitch, 2650 * pitch), 0.03, bolt_at)
    casing = ring(d, (5200, 7600, 9900), 0.05, min(d - 0.06, rng.uniform(0.16, 0.24)))
    shot = drive(reverb(mix(d, (crack, 0.6), (body, 1.0), (low, 0.9), (receiver, 0.35 * metal)), room, wet), k)
    return shot + fit(bolt, len(shot) / SR) * 0.35 * metal + fit(casing, len(shot) / SR) * 0.06 * metal


def weapons() -> None:
    for v in range(3):
        p = rng.uniform(0.93, 1.07)
        write(f"mar_{v}", gunshot(0.3, 0.035, 180, 3200, (120, 55, 0.05), 0.35, 0.25, 2.2, p), 0.7)
        write(f"turret_{v}", gunshot(0.22, 0.025, 400, 5000, (180, 90, 0.03), 0.25, 0.2, 2.0, p * 1.1), 0.55)
    for v in range(2):
        p = rng.uniform(0.95, 1.05)
        write(f"shotgun_{v}", gunshot(0.6, 0.09, 90, 2600, (80, 35, 0.12), 0.7, 0.35, 3.0, p))
        crack = gunshot(1.2, 0.05, 300, 6000, (140, 60, 0.05), 1.0, 0.45, 2.5, p)
        echo = np.concatenate([np.zeros(int(0.22 * SR)), band(crack, 200, 3000)])[: len(crack)] * 0.3
        write(f"sniper_{v}", crack + echo)
    d = 0.35
    write("grenade_launch", mix(d, (tone(sweep(190, 80, d), d) * env(d, 0.001, 0.07), 1.0),
                                (band(noise(d), 200, 1400) * env(d, 0.001, 0.04), 0.6)), 0.8)
    d = 1.2
    whoosh = band(noise(d), 400, 3500) * env(d, 0.03, 0.35)
    thump = tone(sweep(140, 50, d), d) * env(d, 0.002, 0.08)
    hiss = band(noise(d), 1500, 8000) * env(d, 0.06, 0.6)
    write("rocket_launch", drive(reverb(mix(d, (whoosh, 1.0), (thump, 0.8), (hiss, 0.35)), 0.6, 0.3), 2.0), 0.8)
    for v in range(3):                                         # шокер: жужжащий меандр, рваный решёткой, и треск
        d = 0.32
        f = 105 * rng.uniform(0.9, 1.1) + 25 * band(noise(d), 0, 40)
        gate = (band(noise(d), 0, 90) > -0.2).astype(float)
        buzz = band(tone(f, d, "square") * gate, 250, 7000) * env(d, 0.004, 0.12)
        crack = clicks(d, list(rng.uniform(0, 0.2, 14)), 2500, 11000, 0.002) * 2.5
        write(f"shock_{v}", reverb(mix(d, (buzz, 1.0), (crack, 0.8)), 0.25, 0.25), 0.7)
    d = 2.5                                                    # огнемёт: рокот пламени, петля
    roar = band(noise(d + 0.4), 90, 1600) * (0.75 + 0.25 * band(noise(d + 0.4), 0, 6) * 8)
    write("flame_loop", loop_seam(roar + 0.25 * band(noise(d + 0.4), 2000, 6000), 0.4), 0.55, loop=True)
    d = 0.5
    slide = band(noise(d), 1000, 5000) * at(env(0.12, 0.03, 0.04), 0.13, d)
    write("reload", mix(d, (clicks(d, [0.0], 1500, 6000, 0.006), 1.0), (slide, 0.3),
                        (clicks(d, [0.34], 1200, 5000, 0.008), 1.2)), 0.6)


# ---------------------------------------------------------------------------------------------- взрывы

def boom(d: float, size: float) -> np.ndarray:
    body = band(noise(d), 25, 900) * env(d, 0.003, 0.3 * size)
    sub = tone(sweep(70, 26, d), d) * env(d, 0.002, 0.35 * size)
    debris = clicks(d, list(rng.uniform(0.15, d * 0.7, int(30 * size))), 800, 5000, 0.01) * 0.6
    return drive(reverb(mix(d, (body, 1.0), (sub, 1.1), (debris, 0.5)), 1.2 * size, 0.4), 2.6)


def explosions() -> None:
    for v in range(3):
        write(f"explosion_{v}", boom(2.0, 1.0))
    write("orbital_blast", boom(3.5, 1.8))
    d = 1.5                                                    # нарастающий свист снаряда сверху
    whine = tone(sweep(380, 1500, d) * (1 + 0.01 * np.sin(2 * np.pi * 7 * T(d))), d) * (T(d) / d) ** 2
    rush = band(noise(d), 600, 6000) * (T(d) / d) ** 3
    write("orbital_incoming", mix(d, (whine, 0.6), (rush, 0.5)), 0.6)


# ---------------------------------------------------------------------------------------------- жуки

def bugs() -> None:
    for v in range(4):                                         # гибель: хруст панциря и влажный шлепок
        d = 0.5
        p = rng.uniform(0.85, 1.2)
        squelch = band(noise(d), 300 * p, 2200 * p) * env(d, 0.002, 0.07)
        drop = tone(sweep(650 * p, 140 * p, d), d) * env(d, 0.002, 0.05)
        crunch = clicks(d, list(rng.uniform(0, 0.05, 5)), 1500, 7000, 0.003)
        write(f"bug_die_{v}", reverb(mix(d, (squelch, 1.0), (drop, 0.4), (crunch, 0.9)), 0.3, 0.2), 0.6)
    for v in range(2):                                         # укус: дробь щелчков жвал
        d = 0.3
        times = list(np.cumsum(rng.uniform(0.012, 0.022, 9)))
        snap = clicks(d, [times[-1] + 0.02], 900, 4000, 0.01)
        write(f"bug_bite_{v}", clicks(d, times, 1800, 7500, 0.003) + snap, 0.5)
    d = 0.45
    write("warrior_swing", mix(d, (band(noise(d), 500, 2600) * np.sin(np.pi * T(d) / d) ** 3, 1.0),
                               (clicks(d, [0.3], 900, 5000, 0.01), 1.2)), 0.6)
    for v in range(2):                                         # плевок: «пту» и бульканье
        d = 0.45
        p = rng.uniform(0.9, 1.1)
        write(f"spit_{v}", mix(d, (band(noise(d), 500, 3000) * env(d, 0.004, 0.07), 1.0),
                               (tone(sweep(950 * p, 260 * p, d), d) * env(d, 0.003, 0.06), 0.6),
                               (tone(sweep(400, 900, 0.08), 0.08) * env(0.08, 0.002, 0.03), 0.3)), 0.55)
    d = 0.7
    flutter = 0.6 + 0.4 * (band(noise(d), 0, 40) > 0)
    write("acid_sizzle", band(noise(d), 2000, 9000) * env(d, 0.01, 0.25) * flutter, 0.5)
    d = 2.4                                                    # рёв королевы: пила с тремоло через «глотку»
    f = 52 + 9 * np.sin(2 * np.pi * 0.7 * T(d)) + 4 * band(noise(d), 0, 8)
    growl = band(tone(f, d, "saw") * (0.7 + 0.3 * np.sin(2 * np.pi * 17 * T(d))), 120, 1300)
    rasp = band(noise(d), 300, 2500) * 0.35
    shape = np.minimum(1, T(d) / 0.3) * np.exp(-np.maximum(0, T(d) - 0.9) / 0.6)
    write("queen_roar", drive(reverb((growl + rasp) * shape, 1.2, 0.4), 3.0))
    d = 0.8
    write("queen_volley", reverb(mix(d, (band(noise(d), 300, 2500) * env(d, 0.006, 0.12), 1.0),
                                     (tone(sweep(500, 120, d), d) * env(d, 0.004, 0.1), 0.7)), 0.5, 0.3))
    d = 1.4                                                    # призыв: гул и рой щелчков
    write("summon", mix(d, (clicks(d, list(rng.uniform(0, 1.1, 45)), 1500, 7000, 0.004), 1.0),
                        (band(noise(d), 40, 300) * np.sin(np.pi * T(d) / d), 0.8)), 0.65)
    d = 4.4                                                    # гомон роя: петля, громкость — от числа жуков рядом
    swarm = clicks(d, list(rng.uniform(0, d, int(70 * d))), 1500, 6500, 0.003) + band(noise(d), 120, 700) * 0.12
    write("swarm_loop", loop_seam(swarm, 0.4), 0.5, loop=True)


# ---------------------------------------------------------------------------------------------- десантник и мир

def marine() -> None:
    for v in range(3):                                         # попадание по броне: глухой удар и звон пластин
        d = 0.45
        thud = tone(sweep(130, 60, d), d) * env(d, 0.002, 0.08)
        clank = sum(np.sin(2 * np.pi * f * rng.uniform(0.95, 1.05) * T(d)) * env(d, 0.001, 0.12 / k)
                    for k, f in enumerate((910, 1440, 2230), 1))
        hit = band(noise(d), 400, 3000) * env(d, 0.001, 0.03)
        write(f"hurt_{v}", mix(d, (thud, 1.0), (clank, 0.25), (hit, 0.5)), 0.7)
    for v in range(3):                                         # топор по дереву: стук с резонансом и треск
        d = 0.4
        p = rng.uniform(0.9, 1.1)
        knock = band(noise(d), 150, 900) * env(d, 0.001, 0.05)
        res = sum(np.sin(2 * np.pi * f * p * T(d)) * env(d, 0.001, 0.07) for f in (310, 465))
        crack = band(noise(d), 2000, 6000) * env(d, 0.0005, 0.01)
        write(f"chop_{v}", mix(d, (knock, 1.0), (res, 0.4), (crack, 0.6)), 0.7)
    d = 0.35
    write("chop_cactus", mix(d, (band(noise(d), 300, 1800) * env(d, 0.002, 0.06), 1.0),
                             (tone(sweep(500, 200, d), d) * env(d, 0.002, 0.05), 0.4)), 0.6)
    d = 1.0
    write("tree_fall", mix(d, (clicks(d, list(np.sort(rng.uniform(0, 0.25, 12))), 600, 4000, 0.008), 1.0),
                           (band(noise(d), 800, 5000) * np.sin(np.pi * np.clip(T(d) / 0.8, 0, 1)) * 0.3, 1.0)), 0.6)
    for v in range(3):                                         # кристалл опыта: короткий чистый звон
        d = 0.25
        f = 1568 * 2 ** (v * 2 / 12)
        ring = tone(f, d) * env(d, 0.002, 0.07)
        write(f"gem_{v}", mix(d, (ring, 1.0), (tone(f * 1.5, d) * env(d, 0.002, 0.05), 0.5)), 0.3)
    d = 0.5
    notes = [1047, 1319, 1568]
    arp = [at(tone(f, 0.3) * env(0.3, 0.003, 0.09), i * 0.07, d) for i, f in enumerate(notes)]
    write("pickup", np.sum(arp, axis=0), 0.55)
    d = 1.3
    notes = [523, 659, 784, 1047]
    up = np.sum([at(tone(f, 0.9, "saw") * env(0.9, 0.005, 0.25), i * 0.09, d) for i, f in enumerate(notes)], axis=0)
    shimmer = band(noise(d), 4000, 12000) * env(d, 0.3, 0.3)
    write("level_up", reverb(band(up, 200, 5000) + 0.3 * shimmer, 0.8, 0.35), 0.55)
    d = 0.6
    write("door", mix(d, (band(noise(d), 700, 3500) * np.sin(np.pi * np.clip(T(d) / 0.45, 0, 1)) ** 2, 0.7),
                      (clicks(d, [0.45], 300, 2500, 0.015), 1.0)), 0.55)
    d = 0.08
    write("ui_click", band(noise(d), 1500, 6000) * env(d, 0.0005, 0.006), 0.4)
    d = 5.0                                                    # десантный корабль: гул турбин и свист
    rumble = band(noise(d), 30, 300) * np.minimum(1, T(d) / 1.5)
    whine = tone(sweep(600, 1400, d), d) * 0.15 * np.minimum(1, T(d) / 2)
    write("dropship", mix(d, (rumble, 1.0), (whine, 1.0), (band(noise(d), 1500, 5000) * 0.2, 1.0)), 0.75)


# ---------------------------------------------------------------------------------------------- фон биомов

AMB_SR = 22050


def ambient(name: str, layers: Callable[[float], np.ndarray], d: float = 26.0) -> None:
    write(f"amb_{name}", loop_seam(layers(d + 2), 2.0, AMB_SR), 0.45, AMB_SR, loop=True)


def wind(d: float, lo: float, hi: float) -> np.ndarray:
    slow = band(noise(d, AMB_SR), 0, 0.25, sr=AMB_SR)
    slow = 0.55 + 0.45 * slow / (np.abs(slow).max() + 1e-9)
    return band(noise(d, AMB_SR), lo, hi, sr=AMB_SR) * slow


def events(d: float, every: tuple[float, float], make: Callable[[], np.ndarray]) -> np.ndarray:
    out = np.zeros(int(AMB_SR * d))
    t = rng.uniform(*every)
    while t < d - 1.5:
        x = make()
        i = int(t * AMB_SR)
        out[i:i + len(x)] += x[: len(out) - i]
        t += rng.uniform(*every)
    return out


def chirp() -> np.ndarray:
    d = rng.uniform(0.25, 0.6)
    f = rng.uniform(2300, 3600) * (1 + 0.15 * np.sin(2 * np.pi * rng.uniform(10, 25) * T(d, AMB_SR)))
    return tone(f, d, sr=AMB_SR) * env(d, 0.01, d / 3, AMB_SR) * rng.uniform(0.2, 0.5)


def croak() -> np.ndarray:
    d = rng.uniform(0.25, 0.45)
    am = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * rng.uniform(20, 32) * T(d, AMB_SR)))
    voice = band(tone(rng.uniform(140, 220), d, "saw", AMB_SR) * am, 100, 900, sr=AMB_SR)
    return voice * env(d, 0.02, d / 2, AMB_SR) * rng.uniform(0.4, 0.8)


def bubble() -> np.ndarray:
    d = 0.06
    return tone(sweep(300, rng.uniform(600, 1000), d, AMB_SR), d, sr=AMB_SR) * env(d, 0.002, 0.02, AMB_SR) * 0.3


def ambients() -> None:
    ambient("desert", lambda d: wind(d, 80, 900) + 0.3 * wind(d, 1200, 4000))
    ambient("forest", lambda d: 0.6 * wind(d, 150, 1800) + events(d, (1.5, 4.0), chirp))
    ambient("jungle", lambda d: 0.35 * wind(d, 200, 1500) + 0.25 * band(noise(d, AMB_SR), 4500, 9000, sr=AMB_SR)
            * (0.6 + 0.4 * np.sin(2 * np.pi * 38 * T(d, AMB_SR))) + events(d, (0.8, 2.5), chirp))
    ambient("swamp", lambda d: 0.4 * wind(d, 80, 700) + events(d, (0.6, 2.2), croak) + events(d, (0.3, 1.2), bubble))


# ---------------------------------------------------------------------------------------------- фразы

LINES = {
    "drop": "Высадка завершена. Держимся до эвакуации!",
    "contact": "Контакт! Жуки на подходе!",
    "reload": "Перезаряжаюсь!",
    "medkit": "Мне нужна аптечка!",
    "boss": "Крупный жук! Сосредоточить огонь!",
    "queen": "Королева! Всем огонь по королеве!",
    "orbital": "Удар с орбиты! Ложись!",
    "streak_0": "Только мёртвый жук — хороший жук!",
    "streak_1": "Давайте, твари! Ещё!",
    "weapon": "Новое оружие. Посмотрим, как вам это.",
    "evac": "Эвакуация на подходе! Все к точке сбора!",
    "win": "Все на борту! Уходим!",
}
VOICE_SR = 22050


# Piper (MIT) и русский голос dmitri (CC0, датасет OHF-Voice) — нейросетевой синтез офлайн; скачиваются один раз
# в tools/sfx/piper_dl (в git не идут). Без них — голос Windows «Microsoft Pavel» (заметно роботизированнее).
PIPER_DIR = Path(__file__).with_name("piper_dl")
PIPER_ZIP = "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip"
PIPER_VOICE = "ru_RU-dmitri-medium"
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/"


def fetch_piper() -> Path:
    import urllib.request
    import zipfile

    PIPER_DIR.mkdir(exist_ok=True)
    exe = PIPER_DIR / "piper" / "piper.exe"
    if not exe.exists():
        archive = PIPER_DIR / "piper.zip"
        urllib.request.urlretrieve(PIPER_ZIP, archive)
        zipfile.ZipFile(archive).extractall(PIPER_DIR)
    for suffix in (".onnx", ".onnx.json"):
        model = PIPER_DIR / (PIPER_VOICE + suffix)
        if not model.exists():
            urllib.request.urlretrieve(VOICE_URL + PIPER_VOICE + suffix, model)
    return exe


def tts(lines: dict[str, str], folder: Path) -> None:
    """По wav на фразу в folder: Piper (голос dmitri), если не вышло — голос Windows через WinRT (voice.ps1).
    length_scale < 1 — речь быстрее (в бою не тянут слова), noise_w — живее ритм."""
    try:
        exe = fetch_piper()
        for name, text in lines.items():
            subprocess.run([str(exe), "--model", str(PIPER_DIR / f"{PIPER_VOICE}.onnx"), "--output_file",
                            str(folder / f"{name}.wav"), "--length_scale", "0.9", "--noise_scale", "0.7",
                            "--noise_w", "0.9", "--quiet"], input=text.encode("utf-8"), check=True)
        return
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"[sfx] Piper недоступен ({e}) — голос Windows", flush=True)
    table = folder / "lines.tsv"
    table.write_text("\n".join(f"{k}\t{v}" for k, v in lines.items()), encoding="utf-8")
    script = Path(__file__).with_name("voice.ps1")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
                    "-Lines", str(table), "-OutDir", str(folder)], check=True)


def radio(x: np.ndarray, sr: int) -> np.ndarray:
    """Рация: полоса 300–3800 Гц, лёгкий перегруз, шипение, щелчки тангенты в начале и конце."""
    v = drive(band(x, 300, 3800, sr=sr), 1.5)
    hiss = band(rng.standard_normal(len(v)), 2000, 6000, sr=sr) * 0.018
    pad = int(0.08 * sr)
    out = np.concatenate([np.zeros(pad), v + hiss, np.zeros(pad)])
    for i in (0, len(out) - pad):
        out[i:i + pad] += band(rng.standard_normal(pad), 1000, 5000, sr=sr) * env(pad / sr, 0.0005, 0.01, sr) * 0.4
    return out


def voice() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tts(LINES, Path(tmp))
        for name in LINES:
            with wave.open(str(Path(tmp) / f"{name}.wav")) as w:
                sr = w.getframerate()
                raw = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32768
            if sr != VOICE_SR:
                raw = np.interp(np.arange(0, len(raw), sr / VOICE_SR), np.arange(len(raw)), raw)
            write(f"voice_{name}", radio(raw, VOICE_SR), 0.85, VOICE_SR)


GROUPS = {"weapons": weapons, "explosions": explosions, "bugs": bugs, "marine": marine, "ambients": ambients,
          "voice": voice}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(GROUPS):
        GROUPS[name]()
    print("SFX DONE", flush=True)
