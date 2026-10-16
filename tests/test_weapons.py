from __future__ import annotations

import numpy as np
import pytest

from starrun.balance import Balance
from starrun.sim.combat import PELLET, SNIPER
from starrun.sim.swarm import SEMI_BOSS, TANKER, WORKER
from starrun.sim.weapons import Weapon, make_weapon
from starrun.sim.world import Controls, World

DT = 1 / 60


def arm(balance: Balance, wid: str, mod: bool = False, seed: int = 1) -> tuple[World, Weapon]:
    """Мир с одним стволом в руках, без спавна толпы (только расставленные тестом жуки)."""
    w = World(balance, seed)
    weapon = make_weapon(wid, balance.weapons[wid])
    weapon.mod = mod
    w.weapons, w.active = [weapon], 0
    return w, weapon


def put(w: World, kind: int, *xy: tuple[float, float]) -> np.ndarray:
    idx = w.swarm.spawn(kind, np.array(xy, np.float32))
    w.swarm.speed[idx] = 0  # стоят на месте — проверяем оружие, а не погоню
    w.swarm.cooldown[idx] = 99
    return idx


def fire_for(w: World, seconds: float) -> None:
    """Только оружие и снаряды, без хода сессии и спавна."""
    for _ in range(int(seconds / DT)):
        for weapon in w.weapons:
            weapon.tick(DT, w)
        w.weapon.update(DT, w)
        w.turrets.step(DT, w)
        w.fires.step(DT, w)
        w.swarm.burn_tick(DT)
        w.resolve(w.bullets.step(DT, w.swarm))


def test_magazine_and_reload(balance: Balance) -> None:
    w, mar = arm(balance, "mar")
    put(w, WORKER, (300, 0))
    w.swarm.hp[:] = 1e9
    c = balance.weapons["mar"]
    fire_for(w, c.magazine / c.fire_rate + 0.05)
    assert mar.reloading and mar.ammo == 0
    for _ in range(int((c.reload + 0.05) / DT)):
        mar.tick(DT, w)  # только перезарядка, без стрельбы
    assert not mar.reloading and mar.ammo == c.magazine


def test_reload_continues_in_background(balance: Balance) -> None:
    w, mar = arm(balance, "mar")
    mar.ammo, mar.reload_left = 0, 1.0
    w.weapons.append(make_weapon("shock", balance.weapons["shock"]))
    w.select(1)
    fire_for(w, 1.1)
    assert mar.ammo == mar.magazine(w)


def test_shotgun_fires_a_cone_of_pellets_and_knocks_back(balance: Balance) -> None:
    w, gun = arm(balance, "shotgun")
    w.stats.add("shotgun_pellets", 2)
    put(w, WORKER, (200, 0))
    gun.update(DT, w)
    pellets = np.flatnonzero(w.bullets.alive & (w.bullets.kind == PELLET))
    assert len(pellets) == balance.weapons["shotgun"].pellets + 2
    angles = np.degrees(np.arctan2(w.bullets.vel[pellets, 1], w.bullets.vel[pellets, 0]))
    assert np.abs(angles).max() <= balance.weapons["shotgun"].cone_deg / 2 + 1e-3
    w.swarm.hp[:] = 1e9
    before = w.swarm.pos[w.swarm.alive_idx[0]].copy()
    fire_for(w, 0.4)
    assert w.swarm.pos[w.swarm.alive_idx[0], 0] > before[0]  # отброшен от десантника


def test_sniper_picks_the_toughest_and_pierces(balance: Balance) -> None:
    w, rifle = arm(balance, "sniper")
    line = put(w, WORKER, (300, 0), (400, 0), (500, 0), (600, 0), (700, 0))
    tank = put(w, TANKER, (800, 0))
    assert np.allclose(rifle.aim(w), w.swarm.pos[tank[0]])  # самый живучий, хоть и дальше всех
    rifle.update(DT, w)
    shot = np.flatnonzero(w.bullets.alive & (w.bullets.kind == SNIPER))[0]
    assert w.bullets.pierce[shot] == balance.weapons["sniper"].pierce
    fire_for(w, 0.4)
    assert (w.swarm.hp[line] <= 0).sum() == balance.weapons["sniper"].pierce + 1  # пробил три, застрял в четвёртом


def test_railgun_mod_pierces_everything(balance: Balance) -> None:
    w, rifle = arm(balance, "sniper", mod=True)
    line = put(w, WORKER, *[(200 + 60 * i, 0) for i in range(10)])
    w.swarm.hp[line[-1]] = 500  # самый живучий — в конце линии
    fire_for(w, 0.3)
    assert (w.swarm.hp[line[:-1]] <= 0).all()


def test_flamethrower_burns_only_inside_the_cone(balance: Balance) -> None:
    w, flame = arm(balance, "flame")
    inside = put(w, WORKER, (150, 0), (150, 30))
    outside = put(w, WORKER, (0, 150), (400, 0))
    w.swarm.hp[:] = 1e9
    fire_for(w, 0.2)
    assert (w.swarm.burn_left[inside] > 0).all()
    assert (w.swarm.burn_left[outside] == 0).all()
    assert w.flames.alive.any()


def test_napalm_mod_leaves_burning_puddles(balance: Balance) -> None:
    w, flame = arm(balance, "flame", mod=True)
    put(w, WORKER, (150, 0))
    w.swarm.hp[:] = 1e9
    fire_for(w, 1.0)
    assert w.fires.alive.sum() >= 1


def test_rocket_explodes_over_an_area(balance: Balance) -> None:
    w, rocket = arm(balance, "rocket")
    target = put(w, WORKER, (500, 0))  # ближайший — в него и летит ракета
    around = put(w, WORKER, (540, 40), (520, -40))
    far = put(w, WORKER, (500, 300))
    tank = put(w, TANKER, (560, 0))
    fire_for(w, 1.2)
    assert (w.swarm.hp[np.r_[target, around]] <= 0).all()
    assert w.swarm.hp[far] > 0
    assert w.swarm.pos[tank[0], 0] == pytest.approx(560)  # танкер не отбрасывается


def test_cluster_mod_scatters_bomblets(balance: Balance) -> None:
    w, rocket = arm(balance, "rocket", mod=True)
    put(w, WORKER, (500, 0))
    w.swarm.hp[:] = 1e9
    imp_booms = []
    for _ in range(int(1.2 / DT)):
        w.booms = []
        fire_for(w, DT)
        imp_booms += w.booms
    assert len(imp_booms) == 1 + balance.weapons["rocket"].mod_count


def test_turret_deploys_and_shoots_on_its_own(balance: Balance) -> None:
    w, turret = arm(balance, "turret")
    worker = put(w, WORKER, (300, 0))
    w.swarm.hp[worker] = 1e5  # не 1e9: во float32 1e9 - 7 == 1e9
    fire_for(w, 0.5)
    assert w.turrets.alive.sum() == 1
    w.weapons.append(make_weapon("mar", balance.weapons["mar"]))
    w.select(1)
    w.weapons[1].ammo = 0
    w.weapons[1].reload_left = 99  # MAR молчит — стреляет только турель
    hp = w.swarm.hp[worker].copy()
    fire_for(w, 1.0)
    assert w.swarm.hp[worker] < hp


def test_turret_limit_drops_the_oldest(balance: Balance) -> None:
    w, turret = arm(balance, "turret", mod=True)
    put(w, WORKER, (300, 0))
    w.swarm.hp[:] = 1e9
    c = balance.weapons["turret"]
    fire_for(w, c.deploy_every * 4)
    assert w.turrets.alive.sum() <= c.max_active * c.mod_count


def test_shock_chains_through_distinct_bugs(balance: Balance) -> None:
    w, shock = arm(balance, "shock")
    put(w, WORKER, *[(200 + 120 * i, 0) for i in range(6)])
    w.swarm.hp[:] = 1e9
    shock.update(DT, w)
    chain = w.arcs[0]
    assert len(chain) == 1 + balance.weapons["shock"].chain  # десантник + 4 цели
    assert len({tuple(p) for p in chain[1:]}) == len(chain) - 1


def test_storm_mod_stuns_but_not_bosses(balance: Balance) -> None:
    w, shock = arm(balance, "shock", mod=True)
    bugs = put(w, WORKER, (200, 0), (320, 0))
    boss = put(w, SEMI_BOSS, (440, 0))
    w.swarm.hp[:] = 1e9
    shock.update(DT, w)
    assert (w.swarm.stun[bugs] > 0).all()
    assert w.swarm.stun[boss] == 0


def test_mar_mod_lobs_grenades(balance: Balance) -> None:
    w, mar = arm(balance, "mar", mod=True)
    put(w, WORKER, (400, 0))
    w.swarm.hp[:] = 1e9
    booms = 0
    for _ in range(int(4.2 / DT)):  # первая граната — через 3 с, плюс полёт
        w.booms = []
        fire_for(w, DT)
        booms += len(w.booms)
    assert booms >= 1


def test_new_weapon_card_fills_a_slot_and_takes_hands(balance: Balance) -> None:
    w = World(balance, seed=2)
    w.progress.add(5)
    cards = w.upgrades.available(w.owned(), free_slot=True)
    assert {c.id for c in cards if c.mode == "weapon"} == {f"new:{k}" for k in balance.weapons if k != "mar"}
    # улучшения ствола — только для того, что в слотах
    assert {c.weapon for c in cards if c.weapon and c.mode not in ("weapon", "mod")} == {"mar"}
    for _ in range(50):
        offer = w.offer()
        i = next((i for i, o in enumerate(offer) if o.special == "weapon"), None)
        if i is not None:
            w.choose(i)
            break
        w._offer = []
    assert len(w.weapons) == 2 and w.active == 1


def test_no_new_weapons_with_full_slots(balance: Balance) -> None:
    w = World(balance, seed=2)
    for wid in ("shotgun", "flame"):
        w.weapons.append(make_weapon(wid, balance.weapons[wid]))
    cards = w.upgrades.available(w.owned(), free_slot=len(w.weapons) < 3)
    assert not [c for c in cards if c.mode == "weapon"]


def test_weapon_levels_up_to_five_then_offers_mod(balance: Balance) -> None:
    w = World(balance, seed=2)
    mar = w.weapons[0]
    damage = next(u for u in balance.upgrades if u.id == "mar_damage")
    from starrun.sim.progression import Offer
    for _ in range(4):
        w.progress.pending = 1
        w._offer = [Offer(damage, balance.rarities[0], 0)]
        w.choose(0)
    assert mar.level == 5
    cards = w.upgrades.available(w.owned(), free_slot=False)
    assert "mod:mar" in {c.id for c in cards}
    assert not [c for c in cards if c.weapon == "mar" and c.mode in ("mult", "add")]
    w.progress.pending = 1
    w._offer = [Offer(next(c for c in cards if c.id == "mod:mar"), balance.rarities[0], 0)]
    w.choose(0)
    assert mar.mod
    assert "mod:mar" not in {c.id for c in w.upgrades.available(w.owned())}


def test_slot_keys_and_wheel(balance: Balance) -> None:
    w = World(balance, seed=2)
    for wid in ("shotgun", "flame"):
        w.weapons.append(make_weapon(wid, balance.weapons[wid]))
    w.step(DT, Controls(select=2))
    assert w.weapon.id == "flame"
    w.step(DT, Controls(cycle=1))
    assert w.weapon.id == "mar"
    w.step(DT, Controls(cycle=-1))
    assert w.weapon.id == "flame"
    w.step(DT, Controls(select=5))  # пустого слота нет — ничего не меняется
    assert w.weapon.id == "flame"
