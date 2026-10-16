"""Интерфейс в координатах 1080p (ширина зависит от пропорций экрана): HUD, карточки улучшений,
пауза, статистика, итог забега. Сдержанно: тонкие полосы, полупрозрачные панели, без градиентов."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import arcade
import numpy as np
import pyglet

from starrun.sim.progression import Offer
from starrun.sim.session import Phase
from starrun.sim.swarm import KINDS, QUEEN, SEMI_BOSS
from starrun.sim.weapons import Weapon
from starrun.sim.world import World
from starrun.ui.render import Shapes

FONT = ("Bahnschrift", "Segoe UI", "Arial")
H = 1080
INK = (238, 236, 228, 255)
QUIET = (186, 182, 172, 255)
PANEL = (14, 14, 16, 210)
DIM = (8, 8, 10, 170)
TRACK = (255, 255, 255, 38)
XP = (80, 210, 235, 255)
HP = (210, 64, 52, 255)
STAMINA = (220, 218, 210, 255)
ALERT = (235, 165, 40, 255)
EVAC = (90, 200, 240, 255)
EDGE = (255, 255, 255, 70)

STAT_NAMES = {
    "max_hp": "Здоровье", "speed": "Скорость", "sprint_mult": "Бег", "regen": "Регенерация",
    "dodge": "Уклонение", "pickup_radius": "Радиус подбора", "damage_mult": "Урон оружия",
    "range_mult": "Дальность", "reload_mult": "Перезарядка", "ammo_mult": "Боезапас",
    "orbital_rate": "Орбита: перезарядка", "orbital_radius": "Орбита: радиус",
}
FRACTION = {"dodge", "mar_crit"}           # доли — показываются в процентах
MULTIPLIER = {"damage_mult", "range_mult", "reload_mult", "ammo_mult", "orbital_rate", "orbital_radius",
              "sprint_mult"}
KIND_NAMES = {"worker": "Рабочие", "warrior": "Воины", "spitter": "Стрелки", "flyer": "Летающие",
              "tanker": "Танкеры", "brain": "Мозговые", "semi_boss": "Полубоссы", "queen": "Королева"}
SHORT = {"rocket": "Ракетница", "sniper": "Снайперка", "turret": "Турель"}  # в слот длинное имя не влезает
MOD_COLOR = (240, 200, 60, 255)


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def stat_text(name: str, value: float, base: float) -> str:
    if name in FRACTION:
        return f"{value * 100:.0f} %"
    if name in MULTIPLIER:
        return f"{(value / base - 1) * 100:+.0f} %" if base else f"{value:.2f}"
    if name == "regen":
        return f"{value:.1f} ХП/с"
    return f"{value:.0f}"


def effect_text(offer: Offer, common_bonus: float) -> str:
    if offer.special == "weapon":
        return "Новое оружие"
    if offer.special == "mod":
        return "Модификация"
    amount = offer.amount(common_bonus)
    stat = offer.upgrade.stat
    if offer.upgrade.mode == "mult" or stat in FRACTION:
        return f"+{amount * 100:.0f} %"
    if stat == "regen":
        return f"+{amount:.1f} ХП/с"
    return f"+{amount:.0f}"


class Texts:
    """Текстовые метки слоя: живут между кадрами по ключу (создавать arcade.Text дорого) и рисуются
    одним пакетом во flush(); метки, не обновлённые в этом кадре, прячутся."""

    def __init__(self) -> None:
        self.batch = pyglet.graphics.Batch()
        self._cache: dict[str, arcade.Text] = {}
        self._used: set[str] = set()

    def draw(self, key: str, text: str, x: float, y: float, size: float = 16, color: tuple[int, ...] = INK,
             anchor_x: str = "left", anchor_y: str = "baseline", bold: bool = False, width: int | None = None) -> None:
        t = self._cache.get(key)
        if t is None:
            t = arcade.Text(text, x, y, color, size, font_name=FONT, anchor_x=anchor_x, anchor_y=anchor_y,
                            bold=bold, width=width, multiline=width is not None, batch=self.batch)
            self._cache[key] = t
        if t.text != text:
            t.text = text
        if t.position != (x, y):
            t.position = (x, y)
        if t.color != color:
            t.color = color
        t.visible = True
        self._used.add(key)

    def flush(self) -> None:
        for key, t in self._cache.items():
            if key not in self._used and t.visible:
                t.visible = False
        self._used.clear()
        self.batch.draw()



@dataclass
class Button:
    key: str
    label: str
    box: tuple[float, float, float, float]   # l, b, r, t
    action: Callable[[], None]

    def contains(self, x: float, y: float) -> bool:
        l, b, r, t = self.box  # noqa: E741
        return l <= x <= r and b <= y <= t


@dataclass
class Banner:
    text: str
    left: float = 3.0


class Hud:
    def __init__(self, shapes: Shapes) -> None:
        self.s = shapes
        self.t = Texts()   # HUD
        self.o = Texts()   # оверлеи поверх затемнения
        self.banners: list[Banner] = []
        self.buttons: list[Button] = []
        self.mouse = (0.0, 0.0)

    def bar(self, x: float, y: float, w: float, h: float, share: float, color: tuple[int, ...]) -> None:
        self.s.rect(x, y, x + w, y + h, TRACK)
        if share > 0:
            self.s.rect(x, y, x + w * min(1.0, share), y + h, color)

    def notify(self, text: str) -> None:
        self.banners = [b for b in self.banners if b.text != text] + [Banner(text)]

    def update(self, dt: float) -> None:
        for b in self.banners:
            b.left -= dt
        self.banners = [b for b in self.banners if b.left > 0]

    # ---------- HUD ----------
    def draw(self, world: World, vw: float, world_to_ui: Callable[[np.ndarray], tuple[float, float]]) -> None:
        t, p, s, st = self.t, world.player, world.session, world.stats

        self.bar(16, H - 20, vw - 32, 8, world.progress.xp / world.progress.needed, XP)
        t.draw("level", f"Уровень {world.progress.level}", 24, H - 46, 14, QUIET)

        self.s.rect(16, H - 172, 360, H - 60, PANEL)
        t.draw("hp_l", "Здоровье", 32, H - 92, 17)
        t.draw("hp_v", f"{max(0, p.hp):.0f} / {st['max_hp']:.0f}", 344, H - 92, 17, QUIET, anchor_x="right")
        self.bar(32, H - 118, 312, 12, p.hp / st["max_hp"], HP)
        t.draw("st_l", "Выносливость", 32, H - 144, 14, QUIET)
        self.bar(32, H - 160, 312, 6, p.stamina / world.balance.player.stamina.max, STAMINA)

        cx = vw / 2
        self.s.rect(cx - 180, H - 160, cx + 180, H - 40, PANEL)
        t.draw("timer", clock(s.elapsed), cx, H - 86, 30, anchor_x="center", bold=True)
        if s.phase is Phase.WAVE:
            t.draw("alert", f"Тревога улья  {s.alert:.0f} %", cx, H - 120, 14, QUIET, anchor_x="center")
            self.bar(cx - 160, H - 144, 320, 6, s.alert / 100, ALERT)
        elif s.phase is Phase.QUEEN:
            t.draw("alert", "Королева на поле", cx, H - 120, 14, ALERT, anchor_x="center")
            queen = world.swarm.alive_idx[world.swarm.kind[world.swarm.alive_idx] == QUEEN]
            share = float(world.swarm.hp[queen[0]] / world.swarm.max_hp[queen[0]]) if len(queen) else 0.0
            self.bar(cx - 160, H - 144, 320, 6, share, HP)
        elif s.phase is Phase.EVAC:
            left = world.balance.evac.seconds - s.evac_progress
            t.draw("alert", f"Эвакуация: в зоне ещё {left:.0f} с", cx, H - 120, 14, EVAC, anchor_x="center")
            self.bar(cx - 160, H - 144, 320, 6, s.evac_progress / world.balance.evac.seconds, EVAC)

        self.s.rect(vw - 360, H - 140, vw - 16, H - 60, PANEL)
        t.draw("kills", f"Убито  {sum(world.kills.values())}", vw - 32, H - 92, 17, anchor_x="right", bold=True)
        t.draw("hint", "Tab — статистика · Esc — пауза", vw - 32, H - 122, 13, QUIET, anchor_x="right")

        self._slots(world, vw)
        self._pointers(world, vw, world_to_ui)
        for i, b in enumerate(reversed(self.banners[-2:])):
            alpha = int(255 * min(1.0, b.left / 0.5))
            t.draw(f"banner{i}", b.text, cx, H - 230 - i * 44, 26, (*INK[:3], alpha), anchor_x="center", bold=i == 0)
        self.s.flush()
        t.flush()

    def _slots(self, world: World, vw: float) -> None:
        t, orb = self.t, world.orbital
        sw, sh, gap = 190, 84, 12
        x = vw / 2 - (3 * sw + 2 * gap + 24 + 220) / 2
        y = 24
        for i in range(3):
            w = world.weapons[i] if i < len(world.weapons) else None
            on = w is not None and i == world.active
            self.s.rect(x, y, x + sw, y + sh, PANEL)
            self.s.outline(x, y, x + sw, y + sh, INK if on else EDGE, 2 if on else 1)
            t.draw(f"slot{i}k", str(i + 1), x + 12, y + sh - 24, 13, QUIET)
            if w is None:
                t.draw(f"slot{i}n", "—", x + sw / 2, y + 40, 18, QUIET, anchor_x="center")
                t.draw(f"slot{i}a", "", x + sw / 2, y + 14, 12, QUIET, anchor_x="center")
            else:
                name = SHORT.get(w.id, w.cfg.name)
                t.draw(f"slot{i}n", name, x + sw / 2, y + 40, 17, INK if on else QUIET, anchor_x="center", bold=on)
                for k in range(5):  # уровень ствола; золотые — есть модификация
                    color = (MOD_COLOR if w.mod else INK) if k < w.level else TRACK
                    px = x + sw - 14 - (4 - k) * 11
                    self.s.rect(px - 7, y + sh - 16, px, y + sh - 10, color)
                t.draw(f"slot{i}a", self._ammo(world, w), x + sw / 2, y + 14, 12,
                       ALERT if w.reloading else QUIET, anchor_x="center")
            x += sw + gap
        x += 24 - gap
        self.s.rect(x, y, x + 220, y + sh, PANEL)
        self.s.outline(x, y, x + 220, y + sh, EDGE)
        t.draw("ab_k", "Пробел", x + 12, y + sh - 24, 13, QUIET)
        t.draw("ab_n", world.balance.orbital.name, x + 110, y + 34, 17, anchor_x="center")
        total = orb.cooldown_total(world.stats)
        self.bar(x + 14, y + 12, 192, 5, 1 - orb.cooldown_left / total if total else 1, XP)

    @staticmethod
    def _ammo(world: World, w: Weapon) -> str:
        if w.reloading:
            return "перезарядка"
        if w.id == "turret":
            return f"турелей {int(world.turrets.alive.sum())}"
        mag = w.magazine(world)
        return f"{w.ammo} / {mag}" if mag else ""

    def _pointers(self, world: World, vw: float, world_to_ui: Callable[[np.ndarray], tuple[float, float]]) -> None:
        """Стрелки у края экрана: где полубосс, королева и зона эвакуации, если их не видно."""
        sw = world.swarm
        idx = sw.alive_idx
        targets: list[tuple[np.ndarray, tuple[int, ...]]] = [(sw.pos[i], HP) for i in idx[sw.kind[idx] >= SEMI_BOSS]]
        if world.evac_pos is not None:
            targets.append((world.evac_pos, EVAC))
        for pos, color in targets:
            x, y = world_to_ui(pos)
            if 0 <= x <= vw and 0 <= y <= H:
                continue
            cx, cy = vw / 2, H / 2
            dx, dy = x - cx, y - cy
            k = min((vw / 2 - 40) / abs(dx) if dx else 1e9, (H / 2 - 40) / abs(dy) if dy else 1e9)
            ex, ey = cx + dx * k, cy + dy * k
            ang = np.arctan2(dy, dx)
            tip = (ex + np.cos(ang) * 18, ey + np.sin(ang) * 18)
            l_pt = (ex + np.cos(ang + 2.5) * 16, ey + np.sin(ang + 2.5) * 16)
            r_pt = (ex + np.cos(ang - 2.5) * 16, ey + np.sin(ang - 2.5) * 16)
            arcade.draw_triangle_filled(*tip, *l_pt, *r_pt, color)

    # ---------- оверлеи ----------
    def _flush_overlay(self) -> None:
        self.s.flush()
        self.o.flush()

    def _dim(self, vw: float) -> None:
        self.s.rect(0, 0, vw, H, DIM)

    def _button(self, key: str, label: str, cx: float, y: float, action: Callable[[], None], w: float = 320) -> None:
        box = (cx - w / 2, y, cx + w / 2, y + 56)
        b = Button(key, label, box, action)
        hover = b.contains(*self.mouse)
        self.s.rect(*box, (40, 40, 44, 230) if hover else PANEL)
        self.s.outline(*box, INK if hover else EDGE, 1)
        self.o.draw(f"btn_{key}", label, cx, y + 20, 18, anchor_x="center")
        self.buttons.append(b)

    def level_up(self, world: World, vw: float, choose: Callable[[int], None]) -> None:
        self.buttons = []
        self._dim(vw)
        offers = world.offer()
        more = world.progress.pending - 1
        title = f"Уровень {world.progress.level - world.progress.pending + 1}: выбери улучшение"
        self.o.draw("lu_title", title, vw / 2, 800, 30, anchor_x="center", bold=True)
        if more > 0:
            self.o.draw("lu_more", f"и ещё {more} после этого", vw / 2, 764, 16, QUIET, anchor_x="center")
        cw, ch, gap = 400, 300, 32
        x0 = vw / 2 - (len(offers) * cw + (len(offers) - 1) * gap) / 2
        common = world.balance.rarities[0].bonus
        for i, offer in enumerate(offers):
            x, y = x0 + i * (cw + gap), 400
            card = Button(f"card{i}", "", (x, y, x + cw, y + ch), lambda i=i: choose(i))
            hover = card.contains(*self.mouse)
            color = MOD_COLOR if offer.special else (*offer.rarity.color, 255)
            self.s.rect(x, y, x + cw, y + ch, (26, 26, 30, 240) if hover else (18, 18, 22, 235))
            self.s.outline(x, y, x + cw, y + ch, color if hover else EDGE, 2 if hover else 1)
            self.s.rect(x, y + ch - 6, x + cw, y + ch, color)
            self.o.draw(f"c{i}k", str(i + 1), x + 20, y + ch - 44, 14, QUIET)
            label = "" if offer.special else offer.rarity.name
            self.o.draw(f"c{i}r", label, x + cw - 20, y + ch - 44, 14, color, anchor_x="right")
            self.o.draw(f"c{i}n", offer.upgrade.name, x + 20, y + ch - 96, 21, bold=True, width=cw - 40)
            self.o.draw(f"c{i}t", offer.upgrade.text, x + 20, y + ch - 170, 16, QUIET, width=cw - 40)
            size = 24 if offer.special else 30
            self.o.draw(f"c{i}e", effect_text(offer, common), x + 20, y + 30, size, color, bold=True)
            self.buttons.append(card)
        self.o.draw("lu_hint", "Клавиши 1–3 или мышь", vw / 2, 350, 14, QUIET, anchor_x="center")
        self._flush_overlay()

    def pause(self, vw: float, resume: Callable[[], None], restart: Callable[[], None],
              leave: Callable[[], None]) -> None:
        self.buttons = []
        self._dim(vw)
        self.o.draw("p_title", "Пауза", vw / 2, 700, 34, anchor_x="center", bold=True)
        self._button("resume", "Продолжить", vw / 2, 580, resume)
        self._button("restart", "Начать заново", vw / 2, 508, restart)
        self._button("leave", "Выйти из игры", vw / 2, 436, leave)
        self._flush_overlay()

    def stats(self, world: World, vw: float) -> None:
        self.buttons = []
        self._dim(vw)
        l, r, b, top = vw / 2 - 700, vw / 2 + 700, 200, 900  # noqa: E741
        self.s.rect(l, b, r, top, (18, 18, 22, 240))
        t = self.o
        t.draw("s_title", "Статистика десантника", l + 32, top - 56, 26, bold=True)
        t.draw("s_hint", "Tab — закрыть", r - 32, top - 52, 14, QUIET, anchor_x="right")
        base = world.balance.player.stats
        y = top - 110
        for name in STAT_NAMES:
            if name == "radius":
                continue
            t.draw(f"sn_{name}", STAT_NAMES[name], l + 32, y, 16, QUIET)
            t.draw(f"sv_{name}", stat_text(name, world.stats[name], base[name]), l + 420, y, 16, anchor_x="right")
            y -= 34
        x, x_end = l + 480, l + 880
        y = top - 110
        s = world.session
        rows = [("Время", clock(s.elapsed)), ("Уровень", str(world.progress.level)),
                ("Нанесено урона", f"{world.damage_dealt:,.0f}".replace(",", " ")),
                ("Получено урона", f"{world.damage_taken:,.0f}".replace(",", " "))]
        rows += [(KIND_NAMES[k], str(world.kills[k])) for k in KINDS if world.kills[k]]
        if world.damage_taken:
            worst, amount = max(world.damage_by.items(), key=lambda kv: kv[1])
            who = "кислота" if worst == "acid" else KIND_NAMES[worst].lower()
            rows.append(("Больше всего ранили", f"{who}, {amount / world.damage_taken:.0%}".replace("%", " %")))
        for i, (label, value) in enumerate(rows):
            t.draw(f"sr{i}l", label, x, y, 16, QUIET)
            t.draw(f"sr{i}v", value, x_end, y, 16, anchor_x="right")
            y -= 34
        x, y = l + 940, top - 110
        t.draw("s_w", "Оружие", x, y, 18, bold=True)
        y -= 34
        for i, w in enumerate(world.weapons):
            mod = f" · {w.cfg.mod_name.lower()}" if w.mod else ""
            t.draw(f"sw{i}", f"{w.cfg.name} — ур. {w.level}{mod}", x, y, 15, MOD_COLOR if w.mod else INK)
            y -= 28
        y -= 14
        t.draw("s_up", "Взятые улучшения", x, y, 18, bold=True)
        y -= 36
        recent = world.upgrades.taken[-9:]
        for i, offer in enumerate(reversed(recent)):
            label = {"weapon": "новое оружие", "mod": "модификация"}.get(offer.special, offer.rarity.name.lower())
            color = MOD_COLOR if offer.special else (*offer.rarity.color, 255)
            t.draw(f"su{i}", f"{offer.upgrade.name} — {label}", x, y, 14, color, width=400)
            y -= 30
        self._flush_overlay()

    def end(self, world: World, vw: float, restart: Callable[[], None], leave: Callable[[], None]) -> None:
        self.buttons = []
        self._dim(vw)
        won = world.session.phase is Phase.VICTORY
        title = "Эвакуация завершена" if won else "Десантник погиб"
        self.o.draw("e_title", title, vw / 2, 760, 38, EVAC if won else HP, anchor_x="center", bold=True)
        line = (f"Время {clock(world.session.elapsed)}  ·  уровень {world.progress.level}  ·  "
                f"убито {sum(world.kills.values())}")
        self.o.draw("e_line", line, vw / 2, 700, 20, QUIET, anchor_x="center")
        self._button("again", "Ещё забег", vw / 2, 560, restart)
        self._button("e_leave", "Выйти из игры", vw / 2, 488, leave)
        self._flush_overlay()

    def click(self, x: float, y: float) -> bool:
        for b in self.buttons:
            if b.contains(x, y):
                b.action()
                return True
        return False
