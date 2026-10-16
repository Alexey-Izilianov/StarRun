"""Прогон забегов ботом-игроком для проверки баланса — без окна, за десятки секунд.

Бот отходит от толпы по дуге, в спокойные моменты собирает кристаллы, бьёт с орбиты по самому
плотному скоплению и берёт улучшения по простому приоритету. Это не замена живому игроку,
а грубый замер: на какой минуте умирает «средний» игрок, сколько уровней набирает, доходит ли до эвакуации.

Запуск: .venv/Scripts/python tools/balance_bot.py [--runs 5] [--minutes 22]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

from starrun.balance import Balance
from starrun.sim.session import Phase
from starrun.sim.world import Controls, World

DT = 1 / 30
DANGER = 260.0     # ближе этого жуки — только отходить
SEE = 420.0        # от кого вообще уклоняться
PRIORITY = ("mod:", "_damage", "damage", "_rate", "max_hp", "new:", "regen", "speed", "pickup", "reload")
CLOSE = ("flame", "shotgun", "shock")        # чем отбиваться, когда жуки рядом
FAR = ("rocket", "mar", "shock", "sniper", "turret")  # турель сама не стреляет — в последнюю очередь


def steer(w: World) -> tuple[float, float]:
    sw, pos = w.swarm, w.player.pos
    idx = sw.alive_idx
    rel = sw.pos[idx] - pos
    dist = np.linalg.norm(rel, axis=1) + 1e-3
    near = dist < SEE
    if w.evac_pos is not None:  # эвакуация важнее всего: к зоне, лишь отклоняясь от толпы
        v = w.evac_pos - pos
        n = float(np.linalg.norm(v))
        if n < 60:
            return (0.0, 0.0)
        v = v / n
        if near.any():
            away = -(rel[near] / dist[near, None] ** 2).sum(0)
            v = v + 0.7 * away / (np.linalg.norm(away) + 1e-9)
        return tuple(v / (np.linalg.norm(v) + 1e-9))  # type: ignore[return-value]
    if near.any():
        away = -(rel[near] / dist[near, None] ** 2).sum(0)
        away /= np.linalg.norm(away) + 1e-9
        if dist.min() < DANGER:
            v = away + 0.6 * np.array([-away[1], away[0]])
            return tuple(v / np.linalg.norm(v))  # type: ignore[return-value]
    else:
        away = np.zeros(2)
    g = w.gems
    if g.alive.any():  # спокойно — идём за ближайшим кристаллом, чуть отклоняясь от толпы
        gi = np.flatnonzero(g.alive)
        to = g.pos[gi] - pos
        gd = np.linalg.norm(to, axis=1)
        k = int(np.argmin(gd))
        if gd[k] < 900:
            v = to[k] / (gd[k] + 1e-9) + 0.5 * away
            return tuple(v / (np.linalg.norm(v) + 1e-9))  # type: ignore[return-value]
    boss = w.nearest_boss(5000)
    if boss is not None and not near.any():  # спокойно — к боссу, пока он не в радиусе MAR
        v = w.swarm.pos[boss] - pos
        n = np.linalg.norm(v)
        if n > w.weapon.reach(w) * 0.8:
            return tuple(v / n)  # type: ignore[return-value]
    if w.evac_pos is not None:
        v = w.evac_pos - pos
        n = np.linalg.norm(v)
        return (0.0, 0.0) if n < 60 else tuple(v / n)  # type: ignore[return-value]
    return (float(away[0]), float(away[1]))


def in_danger(w: World) -> bool:
    idx = w.swarm.alive_idx
    return bool(len(idx)) and float(np.min(np.linalg.norm(w.swarm.pos[idx] - w.player.pos, axis=1))) < DANGER


def strike_target(w: World) -> tuple[float, float] | None:
    if not w.orbital.ready:
        return None
    idx = w.swarm.alive_idx
    if len(idx) < 25:
        return None
    pts = w.swarm.pos[idx]
    near = pts[np.linalg.norm(pts - w.player.pos, axis=1) < 700]
    if len(near) < 15:
        return None
    # самое плотное место — центр клетки 200 px с наибольшим числом жуков
    cells = np.floor(near / 200).astype(int)
    keys, counts = np.unique(cells, axis=0, return_counts=True)
    best = keys[np.argmax(counts)]
    if counts.max() < 8:
        return None
    c = (best + 0.5) * 200
    return (float(c[0]), float(c[1]))


def pick(w: World) -> int:
    offers = w.offer()
    for want in PRIORITY:
        for i, o in enumerate(offers):
            uid = o.upgrade.id
            prefix = want.endswith(":") and uid.startswith(want)    # mod:*, new:*
            suffix = want.startswith("_") and uid.endswith(want)    # *_damage, *_rate
            if uid == want or prefix or suffix:
                return i
    return 0


def weapon_slot(w: World) -> int | None:
    """Ствол по обстановке: основной — самый прокачанный; вблизи — ствол по площади, если он не слабее
    основного больше чем на уровень; на босса — снайперка такого же уровня."""
    levels = [x.level + 2 * x.mod for x in w.weapons]
    best = max(levels)
    owned = [x.id for x in w.weapons]

    def usable(wid: str) -> bool:
        return wid in owned and levels[owned.index(wid)] >= best - 1

    if w.nearest_boss(1000) is not None and usable("sniper"):
        return owned.index("sniper")
    if in_danger(w):
        for wid in CLOSE:
            if usable(wid):
                return owned.index(wid)
    candidates = [i for i, x in enumerate(w.weapons) if x.id != "turret" and levels[i] == best]
    return candidates[0] if candidates else levels.index(best)


def run(balance: Balance, seed: int, minutes: float) -> dict[str, float | str]:
    w = World(balance, seed)
    while not w.session.over and w.session.elapsed < minutes * 60:
        if w.awaiting_choice:
            w.choose(pick(w))
            continue
        w.step(DT, Controls(move=steer(w), sprint=in_danger(w), ability_target=strike_target(w),
                            select=weapon_slot(w)))
    return {"result": w.session.phase.name, "minute": w.session.elapsed / 60, "level": w.progress.level,
            "kills": sum(w.kills.values()), "queen": w.session.phase in (Phase.EVAC, Phase.VICTORY),
            "weapons": " ".join(f"{x.id}{x.level}{'*' if x.mod else ''}" for x in w.weapons)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--minutes", type=float, default=22)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # консоль Windows — cp1251
    balance = Balance.load()
    results = []
    for seed in range(args.runs):
        t = time.perf_counter()
        r = run(balance, seed, args.minutes)
        results.append(r)
        print(f"seed {seed}: {r['result']:8s} мин {r['minute']:5.1f}  уровень {r['level']:2d}  убито {r['kills']:5d}"
              f"  оружие {r['weapons']}  ({time.perf_counter() - t:.0f} с)", flush=True)
    minutes = [float(r["minute"]) for r in results]
    wins = sum(r["result"] == "VICTORY" for r in results)
    print(f"итого: побед {wins}/{len(results)}, средняя минута конца {statistics.fmean(minutes):.1f}, "
          f"средний уровень {statistics.fmean(float(r['level']) for r in results):.1f}")


if __name__ == "__main__":
    main()
