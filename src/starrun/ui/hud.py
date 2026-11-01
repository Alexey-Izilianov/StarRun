"""Полевой терминал в координатах 1080p: приборы бойца, вооружение и командные экраны."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import arcade
import numpy as np
import pyglet
from pyglet.font.base import Font

from starrun.sim.progression import Offer
from starrun.sim.session import Phase
from starrun.sim.swarm import KINDS, QUEEN, SEMI_BOSS
from starrun.sim.weapons import Weapon
from starrun.sim.world import World
from starrun.ui.icons import corners, icon
from starrun.ui.render import Shapes, rgba

FONT = ("Exo 2", "Bahnschrift", "Segoe UI")
_FONTS_LOADED = False
H = 1080
INK = (234, 236, 229, 255)
QUIET = (155, 168, 169, 255)
PANEL = (11, 20, 25, 216)
DIM = (7, 14, 19, 226)
TRACK = (255, 255, 255, 38)
XP = (80, 210, 235, 255)
HP = (210, 64, 52, 255)
STAMINA = (220, 218, 210, 255)
ALERT = (225, 179, 84, 255)
EVAC = (90, 200, 240, 255)
EDGE = (133, 157, 161, 80)

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
HAZARD_NAMES = {"acid": "кислота", "cactus": "кактусы", "sand": "зыбучий песок"}  # урон не от укусов
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
    if stat == "mar_ricochet":
        return f"+{amount:.0f} рикошет"
    if stat == "sniper_armor_pierce":
        return "Раскалывает броню"
    if stat == "rocket_fire":
        return "Горящее поле"
    if offer.upgrade.mode == "mult" or stat in FRACTION:
        return f"+{amount * 100:.0f} %"
    if stat == "regen":
        return f"+{amount:.1f} ХП/с"
    return f"+{amount:.0f}"


def comparison_value(stat: str, value: float) -> str:
    if stat in ("sniper_armor_pierce", "rocket_fire"):
        return "Да" if value > 0 else "Нет"
    if stat in ("mar_ricochet", "mar_pierce", "sniper_pierce", "shock_chain", "shotgun_pellets"):
        return f"{value:.0f}"
    if stat in FRACTION:
        return f"{value * 100:.0f}%"
    return f"{value:.2f}" if value < 10 else f"{value:.0f}"


class Texts:
    """Текстовые метки слоя: живут между кадрами по ключу (создавать arcade.Text дорого) и рисуются
    одним пакетом во flush(); метки, не обновлённые в этом кадре, прячутся."""

    def __init__(self) -> None:
        global _FONTS_LOADED
        if not _FONTS_LOADED:
            # Pyglet по умолчанию использует NEAREST для атласа глифов: при масштабе HUD
            # 1080 → физическое окно буквы становятся ступенчатыми даже со сглаживанием мира.
            Font.texture_min_filter = pyglet.gl.GL_LINEAR
            Font.texture_mag_filter = pyglet.gl.GL_LINEAR
            for path in (Path(__file__).resolve().parents[3] / "assets" / "fonts").glob("*.ttf"):
                pyglet.font.add_file(str(path))
            _FONTS_LOADED = True
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
        if t.font_size != size:
            t.font_size = size
        if t.width != width:
            t.width = width
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
class SettingRow:
    title: str
    value: str
    prev: Callable[[], None]
    next: Callable[[], None]


@dataclass
class Banner:
    text: str
    left: float = 3.0


class Hud:
    """Полевой терминал: компактные приборы в бою и отдельные командные экраны."""

    def __init__(self, shapes: Shapes) -> None:
        self.s = shapes
        self.t = Texts()
        self.o = Texts()
        self.banners: list[Banner] = []
        self.buttons: list[Button] = []
        self.mouse = (0.0, 0.0)

    def bar(self, x: float, y: float, w: float, h: float, share: float, color: tuple[int, ...]) -> None:
        self.s.rect(x, y, x + w, y + h, TRACK)
        if share > 0:
            self.s.rect(x, y, x + w * min(1.0, share), y + h, color)

    def segmented(self, x: float, y: float, w: float, h: float, share: float,
                  color: tuple[int, ...], count: int = 12) -> None:
        part = w / count
        for i in range(count):
            fill = float(np.clip(share * count - i, 0, 1))
            self.bar(x + i * part, y, part - 3, h, fill, color)

    def notify(self, text: str) -> None:
        self.banners = [b for b in self.banners if b.text != text] + [Banner(text)]

    def update(self, dt: float) -> None:
        for b in self.banners:
            b.left -= dt
        self.banners = [b for b in self.banners if b.left > 0]

    def plate(self, box: tuple[float, float, float, float], accent: tuple[int, ...] = EDGE,
              filled: bool = True) -> None:
        if filled:
            self.s.rect(*box, PANEL)
        l, b, r, top = box  # noqa: E741
        self.s.rect(l, b, r, b + 1, EDGE)
        corners(self.s, box, accent, 10, 1)

    def draw(self, world: World, vw: float, world_to_ui: Callable[[np.ndarray], tuple[float, float]]) -> None:
        t, p, session, stats = self.t, world.player, world.session, world.stats
        cx = vw / 2
        biome = world.balance.biomes[world.biome].name if world.biome else "Полигон"
        self.s.rect(18, H - 110, 310, H - 12, (9, 18, 23, 185))
        t.draw("operation", "КОРПУС / " + biome.upper(), 30, H - 36, 13, QUIET)
        t.draw("level", f"{world.progress.level:02d}", 30, H - 76, 26, INK, bold=True)
        t.draw("xp_label", "УРОВЕНЬ", 77, H - 57, 10, QUIET)
        t.draw("xp_value", f"{world.progress.xp} / {world.progress.needed} ОП", 77, H - 79, 12, INK)
        self.bar(30, H - 94, 260, 3, world.progress.xp / world.progress.needed, XP)

        self.plate((cx - 166, H - 128, cx + 166, H - 17))
        t.draw("timer", clock(session.elapsed), cx, H - 58, 32, INK, anchor_x="center", bold=True)
        share, color = session.alert / 100, ALERT
        title = f"ТРЕВОГА УЛЬЯ  /  {session.alert:.0f}%"
        if session.phase is Phase.QUEEN:
            title, color = "КОНТАКТ / КОРОЛЕВА", HP
            idx = world.swarm.alive_idx
            queens = idx[world.swarm.kind[idx] == QUEEN]
            share = float(world.swarm.hp[queens[0]] / world.swarm.max_hp[queens[0]]) if len(queens) else 0.0
        elif session.phase is Phase.EVAC:
            left = max(0.0, world.balance.evac.seconds - session.evac_progress)
            title, color = f"ЭВАКУАЦИЯ / {left:.0f} С В ЗОНЕ", EVAC
            share = session.evac_progress / world.balance.evac.seconds
        t.draw("phase", title, cx, H - 88, 12, color, anchor_x="center")
        self.segmented(cx - 145, H - 110, 290, 3, share, color, 20)

        right = vw - 30
        self.s.rect(right - 280, H - 110, vw - 18, H - 12, (9, 18, 23, 185))
        t.draw("counter_label", "ОБРАЗЦЫ ДНК      УСТРАНЕНО", right, H - 33, 10, QUIET, anchor_x="right")
        t.draw("dna", f"{world.dna:03d}", right - 152, H - 68, 24, INK, anchor_x="right", bold=True)
        t.draw("kills", f"{sum(world.kills.values()):04d}", right, H - 68, 24, INK,
               anchor_x="right", bold=True)
        self.s.rect(right - 125, H - 70, right - 124, H - 38, EDGE)
        t.draw("hint", "TAB  Статистика     ESC  Пауза", right, H - 94, 11, QUIET, anchor_x="right")

        self.plate((28, 28, 350, 166), HP if p.hp < stats["max_hp"] * 0.3 else EDGE)
        icon(self.s, "health", 65, 116, 32, HP)
        t.draw("hp", f"{max(0, p.hp):.0f}", 99, 104, 38, INK, bold=True)
        t.draw("hp_max", f"/ {stats['max_hp']:.0f}", 186, 108, 15, QUIET)
        t.draw("vitals_label", "СОСТОЯНИЕ БОЙЦА", 47, 145, 10, QUIET)
        self.segmented(47, 84, 284, 7, p.hp / stats["max_hp"], HP)
        self.bar(47, 53, 284, 3, p.stamina / world.balance.player.stamina.max, STAMINA)
        t.draw("stamina", "ФОРСАЖ" if p.sprinting else "ВЫНОСЛИВОСТЬ", 47, 63, 10, QUIET)
        t.draw("stamina_value", f"{p.stamina:.0f}%", 330, 63, 10, STAMINA, anchor_x="right")
        self._slots(world, vw)
        self._pointers(world, vw, world_to_ui)
        banners = [] if world.awaiting_choice else self.banners[-2:]
        for i, banner in enumerate(reversed(banners)):
            alpha = int(255 * min(1.0, banner.left / 0.5))
            t.draw(f"banner{i}", banner.text, cx, H - 176 - i * 32, 18, (*INK[:3], alpha),
                   anchor_x="center", bold=i == 0)
        self.s.flush()
        t.flush()

    def _slots(self, world: World, vw: float) -> None:
        t, right = self.t, vw - 28
        active = world.weapons[world.active] if world.weapons else None
        left = right - 340
        self.plate((left, 28, right, 166), ALERT)
        if active is not None:
            name = SHORT.get(active.id, active.cfg.name)
            t.draw("weapon_k", f"{world.active + 1} / ВООРУЖЕНИЕ", left + 18, 145, 10, ALERT)
            icon(self.s, active.id, left + 44, 101, 52, INK)
            t.draw("weapon_n", name, left + 88, 113, 20, INK, bold=True, width=230)
            ammo = str(active.ammo) if active.magazine(world) else "∞"
            if active.id == "turret":
                ammo = str(int(world.turrets.alive.sum()))
            t.draw("weapon_ammo", ammo, left + 88, 65, 34, ALERT if active.reloading else INK, bold=True)
            max_ammo = f"/ {active.magazine(world)}" if active.magazine(world) else ""
            t.draw("weapon_max", max_ammo, left + 160, 69, 15, QUIET)
            status = "ПЕРЕЗАРЯДКА" if active.reloading else "МОДИФИКАЦИЯ" if active.mod else "ГОТОВНОСТЬ"
            t.draw("weapon_status", status, right - 18, 46, 10, ALERT if active.reloading else QUIET,
                   anchor_x="right")
            for i in range(5):
                color = MOD_COLOR if active.mod else ALERT
                self.s.rect(left + 18 + i * 12, 44, left + 26 + i * 12, 48, color if i < active.level else TRACK)
            if active.reloading:
                share = 1 - active.reload_left / max(0.001, active.cfg.reload)
                self.bar(left + 18, 30, 304, 3, share, ALERT)
        other_slots = [i for i in range(3) if i != world.active]
        for offset, i in enumerate(other_slots):
            x, y = left + offset * 175, 178
            self.s.rect(x, y, x + 165, y + 48, (12, 20, 24, 195))
            self.s.rect(x, y, x + 2, y + 48, EDGE)
            weapon = world.weapons[i] if i < len(world.weapons) else None
            t.draw(f"slot{i}key", str(i + 1), x + 11, y + 17, 14, ALERT if weapon else QUIET)
            name = SHORT.get(weapon.id, weapon.cfg.name) if weapon else "Пустой слот"
            t.draw(f"slot{i}name", name, x + 34, y + 25, 12, INK if weapon else QUIET, width=123)
            if weapon:
                t.draw(f"slot{i}ammo", self._ammo(world, weapon), x + 34, y + 9, 10, QUIET)
        orb_x = left - 95
        orb = world.orbital
        ready = orb.cooldown_left <= 0
        color = ALERT if ready else QUIET
        icon(self.s, "orbital", orb_x, 103, 46, color)
        self.s.add_one(orb_x, 103, 70, rgba((*color[:3], 110)), ring=0.025)
        t.draw("ab_key", "ПРОБЕЛ", orb_x, 48, 10, color, anchor_x="center")
        t.draw("ab_time", "ОРБИТА" if ready else f"{orb.cooldown_left:.0f} С", orb_x, 68, 11, color,
               anchor_x="center")
        total = orb.cooldown_total(world.stats)
        self.bar(orb_x - 34, 32, 68, 3, 1 - orb.cooldown_left / total if total else 1, color)

    @staticmethod
    def _ammo(world: World, weapon: Weapon) -> str:
        if weapon.reloading:
            return "перезарядка"
        if weapon.id == "turret":
            return f"турелей {int(world.turrets.alive.sum())}"
        mag = weapon.magazine(world)
        return f"{weapon.ammo} / {mag}" if mag else "без магазина"

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

    def _flush_overlay(self) -> None:
        self.s.flush()
        self.o.flush()

    def _dim(self, vw: float) -> None:
        self.s.rect(0, 0, vw, H, DIM)
        self.s.rect(0, 0, 8, H, (ALERT[0], ALERT[1], ALERT[2], 100))

    def _header(self, vw: float, code: str, title: str, subtitle: str = "") -> tuple[float, float]:
        left, right = max(48.0, vw / 2 - 680), min(vw - 48, vw / 2 + 680)
        self.o.draw("menu_code", code, left, 946, 12, ALERT)
        self.o.draw("menu_title", title, left, 880, 46, INK, bold=True)
        self.o.draw("menu_subtitle", subtitle, left, 842, 15, QUIET)
        self.s.rect(left, 818, right, 819, EDGE)
        self.s.rect(left, 817, left + 72, 820, ALERT)
        self.o.draw("menu_brand", "STAR / RUN", right, 946, 12, QUIET, anchor_x="right")
        return left, right

    def _button(self, key: str, label: str, cx: float, y: float, action: Callable[[], None],
                w: float = 340, primary: bool = False, h: float = 58) -> None:
        box = (cx - w / 2, y, cx + w / 2, y + h)
        button = Button(key, label, box, action)
        hover = button.contains(*self.mouse)
        color = ALERT if primary or hover else INK
        self.s.rect(*box, (55, 47, 29, 245) if primary or hover else (20, 29, 34, 245))
        self.s.rect(box[0], y, box[0] + (4 if primary or hover else 2), y + h, color if hover or primary else EDGE)
        self.s.rect(box[0], y, box[2], y + 1, EDGE)
        if hover:
            corners(self.s, box, ALERT, 10, 1)
        self.o.draw(f"btn_{key}", label, cx, y + h / 2 - 8, 17, color, anchor_x="center", bold=primary)
        self.buttons.append(button)

    def level_up(self, world: World, vw: float, choose: Callable[[int], None], selected: int = 0,
                 reroll: Callable[[], None] | None = None) -> None:
        self.buttons = []
        self._dim(vw)
        level = world.progress.level - world.progress.pending + 1
        left, right = self._header(vw, "СНАРЯЖЕНИЕ / ПОЛЕВАЯ МОДЕРНИЗАЦИЯ", "Следующее преимущество",
                                   "Выберите улучшение для текущего забега")
        self.o.draw("lu_level", f"УРОВЕНЬ {level:02d}", right, 853, 16, INK, anchor_x="right")
        offers, gap = world.offer(), 20
        cw = (right - left - gap * (len(offers) - 1)) / max(1, len(offers))
        common = world.balance.rarities[0].bonus
        for i, offer in enumerate(offers):
            x, y, ch = left + i * (cw + gap), 300, 456
            focus = i == selected
            if focus:
                y += 10
            box = (x, y, x + cw, y + ch)
            card = Button(f"card{i}", "", box, lambda i=i: choose(i))
            color = MOD_COLOR if offer.special or offer.upgrade.fixed else (*offer.rarity.color, 255)
            self.s.rect(*box, (29, 38, 41, 255) if focus else (15, 23, 28, 250))
            self.s.rect(x, y + ch - 3, x + cw, y + ch, color)
            self.s.rect(x, y, x + cw, y + 1, ALERT if focus else EDGE)
            if focus:
                corners(self.s, box, ALERT, 22, 2)
            self.o.draw(f"card{i}number", f"0{i + 1}", x + 22, y + ch - 38, 14, QUIET)
            rarity = {"weapon": "НОВОЕ ОРУЖИЕ", "mod": "МОДИФИКАЦИЯ"}.get(offer.special, offer.rarity.name.upper())
            if offer.upgrade.fixed:
                rarity = "ТАКТИЧЕСКИЙ ПРИЁМ"
            self.o.draw(f"card{i}rarity", rarity, x + cw - 22, y + ch - 38, 11, color, anchor_x="right")
            up = offer.upgrade
            graphic = up.weapon or up.stat or up.id
            icon(self.s, graphic, x + 46, y + ch - 98, 44, color)
            self.o.draw(f"card{i}group", "БОЕЦ" if up.group == "passive" else "ВООРУЖЕНИЕ",
                        x + 83, y + ch - 94, 11, QUIET)
            name_size = 23 if cw >= 380 else 20
            self.o.draw(f"card{i}name", up.name, x + 22, y + ch - 154, name_size, INK, bold=True, width=int(cw - 44))
            self.o.draw(f"card{i}description", up.text, x + 22, y + ch - 237, 16, QUIET, width=int(cw - 44))
            compare = "Займёт свободный слот" if offer.special == "weapon" else "Меняет поведение оружия"
            if not offer.special:
                before, after = world.stats.preview(offer, common)
                compare = f"{comparison_value(up.stat, before)}   →   {comparison_value(up.stat, after)}"
            self.s.rect(x + 22, y + 111, x + cw - 22, y + 112, EDGE)
            self.o.draw(f"card{i}compare", compare, x + 22, y + 82, 15, QUIET)
            self.o.draw(f"card{i}effect", effect_text(offer, common), x + 22, y + 37, 27, color, bold=True)
            if up.weapon and up.weapon in world.owned():
                rank = world.owned()[up.weapon][0]
                line = f"УР. {rank} → {min(5, rank + 1)}" if not offer.special else "УР. 5 / МОД"
                self.o.draw(f"card{i}rank", line, x + cw - 22, y + 85, 11, QUIET, anchor_x="right")
            self.buttons.append(card)
        self.o.draw("lu_hint", "WASD / стрелки  Выбор      ENTER / ПРОБЕЛ  Принять      1–3 / МЫШЬ",
                    left, 257, 13, QUIET)
        if reroll is not None and world.rerolls_left > 0:
            self._button("reroll", f"R  Обновить выбор / {world.rerolls_left}", left + 175, 146, reroll, 350)
        else:
            self.o.draw("lu_exhausted", "Обновления выбора использованы", left, 170, 14, QUIET)
        more = world.progress.pending - 1
        self.o.draw("lu_more", f"Ещё улучшений: {more}" if more else "Одно улучшение на уровень",
                    right, 170, 13, QUIET, anchor_x="right")
        self._flush_overlay()

    def pause(self, vw: float, resume: Callable[[], None], restart: Callable[[], None],
              settings: Callable[[], None], leave: Callable[[], None]) -> None:
        self.buttons = []
        self._dim(vw)
        left, right = self._header(vw, "КОМАНДОВАНИЕ / ОПЕРАЦИЯ ПРИОСТАНОВЛЕНА", "Пауза", "ESC  Вернуться в бой")
        self.o.draw("pause_mark", "II", right - 130, 526, 140, (220, 179, 84, 70), anchor_x="center", bold=True)
        self.o.draw("pause_tip", "Ваше снаряжение и прогресс\nсохраняются до конца забега.",
                    right - 335, 412, 16, QUIET, width=330)
        for i, (key, label, callback) in enumerate((("resume", "Продолжить операцию", resume),
                ("restart", "Начать заново", restart), ("settings", "Настройки", settings),
                ("leave", "Выйти из игры", leave))):
            self._button(key, label, left + 220, 676 - i * 90, callback, 440, primary=i == 0)
        self._flush_overlay()

    def settings(self, vw: float, rows: list[SettingRow], apply: Callable[[], None], back: Callable[[], None],
                 note: str) -> None:
        self.buttons = []
        self._dim(vw)
        left, right = self._header(vw, "КОНФИГУРАЦИЯ / ОБОРУДОВАНИЕ", "Настройки", "Экран и звуковое окружение")
        cx, row_h = (left + right) / 2, min(64, 450 / max(1, len(rows)))
        for i, row in enumerate(rows):
            y = 734 - i * row_h
            if i % 2 == 0:
                self.s.rect(left, y - 7, right, y + 48, (21, 30, 35, 130))
            self.o.draw(f"setting{i}title", row.title, left + 20, y + 15, 18, INK)
            self._button(f"st_p{i}", "‹", right - 375, y - 5, row.prev, 50, h=min(52, row_h - 6))
            self.o.draw(f"setting{i}value", row.value, right - 210, y + 15, 17, ALERT, anchor_x="center")
            self._button(f"st_n{i}", "›", right - 45, y - 5, row.next, 50, h=min(52, row_h - 6))
        self.o.draw("settings_note", note, left, 240, 13, QUIET, width=int(right - left))
        self._button("st_apply", "Применить изменения", cx - 180, 126, apply, 340, primary=True)
        self._button("st_back", "Назад", cx + 180, 126, back, 340)
        self._flush_overlay()

    def stats(self, world: World, vw: float) -> None:
        self.buttons = []
        self._dim(vw)
        left, right = self._header(vw, "ПОЛЕВОЙ ОТЧЁТ / ТЕКУЩАЯ ОПЕРАЦИЯ", "Статистика бойца", "TAB  Закрыть отчёт")
        col = (right - left - 64) / 3
        headings = ("ХАРАКТЕРИСТИКИ", "РЕЗУЛЬТАТЫ", "КОМПЛЕКТ")
        for i, title in enumerate(headings):
            self.o.draw(f"stats_heading{i}", title, left + i * (col + 32), 762, 12, ALERT)
        base = world.balance.player.stats
        y = 710
        for name, label in STAT_NAMES.items():
            self.o.draw(f"stat_{name}_label", label, left, y, 15, QUIET)
            self.o.draw(f"stat_{name}_value", stat_text(name, world.stats[name], base[name]), left + col, y,
                        15, INK, anchor_x="right")
            self.s.rect(left, y - 14, left + col, y - 13, (255, 255, 255, 15))
            y -= 37
        x, y = left + col + 32, 710
        rows = [("Время", clock(world.session.elapsed)), ("Уровень", str(world.progress.level)),
                ("Образцы ДНК", str(world.dna)), ("Урон нанесён", f"{world.damage_dealt:,.0f}".replace(",", " ")),
                ("Урон получен", f"{world.damage_taken:,.0f}".replace(",", " "))]
        rows += [(KIND_NAMES[k], str(world.kills[k])) for k in KINDS if world.kills[k]]
        for i, (label, value) in enumerate(rows):
            self.o.draw(f"result{i}_label", label, x, y, 15, QUIET)
            self.o.draw(f"result{i}_value", value, x + col, y, 15, INK, anchor_x="right")
            y -= 37
        if world.damage_taken:
            worst, amount = max(world.damage_by.items(), key=lambda kv: kv[1])
            who = HAZARD_NAMES.get(worst) or KIND_NAMES[worst].lower()
            self.o.draw("stats_hazard", f"Главная угроза: {who}\n{amount / world.damage_taken:.0%} полученного урона",
                        x, y - 10, 13, QUIET, width=int(col))
        x, y = left + 2 * (col + 32), 706
        for i, weapon in enumerate(world.weapons):
            icon(self.s, weapon.id, x + 20, y + 4, 34, ALERT if i == world.active else QUIET)
            name = SHORT.get(weapon.id, weapon.cfg.name)
            self.o.draw(f"loadout{i}", f"{i + 1} / {name}", x + 48, y + 6, 16, INK, bold=True)
            line = f"Уровень {weapon.level}" + (f" · {weapon.cfg.mod_name}" if weapon.mod else "")
            self.o.draw(f"loadout{i}_mod", line, x + 48, y - 19, 12, QUIET, width=int(col - 48))
            y -= 73
        y -= 22
        self.o.draw("stats_upgrades_title", "ПОСЛЕДНИЕ УЛУЧШЕНИЯ", x, y, 11, ALERT)
        y -= 39
        for i, offer in enumerate(reversed(world.upgrades.taken[-5:])):
            color = MOD_COLOR if offer.special or offer.upgrade.fixed else (*offer.rarity.color, 255)
            self.o.draw(f"stats_upgrade{i}", offer.upgrade.name, x, y, 14, color, width=int(col))
            y -= 48
        self._flush_overlay()

    def end(self, world: World, vw: float, restart: Callable[[], None], leave: Callable[[], None]) -> None:
        self.buttons = []
        self._dim(vw)
        won = world.session.phase is Phase.VICTORY
        left, right = self._header(vw, "ИТОГОВЫЙ ОТЧЁТ / ОПЕРАЦИЯ ЗАВЕРШЕНА",
                                   "Эвакуация завершена" if won else "Связь с бойцом потеряна",
                                   "Группа принята на борт" if won else "Данные операции переданы командованию")
        color = EVAC if won else HP
        self.o.draw("end_status", "УСПЕШНО" if won else "БОЕЦ ПОГИБ", left, 735, 14, color)
        metrics = (("ВРЕМЯ ОПЕРАЦИИ", clock(world.session.elapsed)), ("УРОВЕНЬ БОЙЦА", str(world.progress.level)),
                   ("УСТРАНЕНО ЦЕЛЕЙ", str(sum(world.kills.values()))))
        width = (right - left - 48) / 3
        for i, (label, value) in enumerate(metrics):
            x = left + i * (width + 24)
            self.s.rect(x, 492, x + width, 666, (18, 27, 32, 240))
            self.o.draw(f"end_metric{i}", value, x + 24, 558, 48, INK, bold=True)
            self.o.draw(f"end_label{i}", label, x + 24, 518, 11, QUIET)
            self.s.rect(x, 492, x + 64, 495, color)
        cx = vw / 2
        self._button("again", "Следующая высадка", cx - 180, 334, restart, 340, primary=True)
        self._button("e_leave", "Выйти из игры", cx + 180, 334, leave, 340)
        self.o.draw("end_hint", "ENTER  Следующая высадка", cx, 270, 12, QUIET, anchor_x="center")
        self._flush_overlay()

    def click(self, x: float, y: float) -> bool:
        for button in self.buttons:
            if button.contains(x, y):
                button.action()
                return True
        return False
