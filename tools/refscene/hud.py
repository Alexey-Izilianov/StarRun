"""Draws the in-game HUD over the Blender render. Usage: python hud.py in.png out.png"""
import sys

from PIL import Image, ImageDraw, ImageFont

src, dst = sys.argv[1], sys.argv[2]
base = Image.open(src).convert("RGBA")
W, H = base.size
layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
d = ImageDraw.Draw(layer)

FONT = "C:/Windows/Fonts/bahnschrift.ttf"


def font(size, weight="Regular"):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return f


f_small, f_text, f_bold, f_timer = font(18), font(22), font(22, "SemiBold"), font(40, "SemiBold")
INK = (238, 236, 228, 255)
QUIET = (190, 186, 176, 255)
PANEL = (12, 12, 14, 150)
TRACK = (255, 255, 255, 40)
XP, HP, STAM, ALERT = (70, 210, 235, 255), (210, 64, 52, 255), (220, 218, 210, 255), (235, 165, 40, 255)


def bar(x, y, w, h, share, color):
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=TRACK)
    d.rounded_rectangle((x, y, x + max(h, int(w * share)), y + h), h // 2, fill=color)


def panel(x0, y0, x1, y1):
    d.rounded_rectangle((x0, y0, x1, y1), 10, fill=PANEL)


# experience along the top edge
bar(16, 12, W - 32, 8, 0.62, XP)
d.text((24, 28), "Уровень 12", font=f_small, fill=QUIET)

# health and stamina
panel(16, 56, 356, 160)
d.text((32, 66), "Здоровье", font=f_text, fill=INK)
d.text((340, 66), "140 / 200", font=f_text, fill=QUIET, anchor="ra")
bar(32, 98, 308, 12, 0.70, HP)
d.text((32, 118), "Выносливость", font=f_small, fill=QUIET)
bar(32, 144, 308, 6, 0.45, STAM)

# timer and hive alert
panel(W // 2 - 170, 40, W // 2 + 170, 138)
d.text((W // 2, 48), "08:42", font=f_timer, fill=INK, anchor="ma")
d.text((W // 2, 96), "Тревога улья  64 %", font=f_small, fill=QUIET, anchor="ma")
bar(W // 2 - 150, 120, 300, 6, 0.64, ALERT)

# DNA samples and tasks
panel(W - 356, 56, W - 16, 176)
d.text((W - 32, 66), "Образцы ДНК  14", font=f_bold, fill=INK, anchor="ra")
d.text((W - 32, 102), "Задачи  1 из 3", font=f_text, fill=INK, anchor="ra")
d.text((W - 32, 134), "Гнездо — выполнено", font=f_small, fill=QUIET, anchor="ra")

# off-screen task pointer on the right edge
cy = 560
d.polygon([(W - 14, cy), (W - 40, cy - 16), (W - 40, cy + 16)], fill=ALERT)
panel(W - 250, cy - 22, W - 50, cy + 22)
d.text((W - 62, cy), "Маяк  · 38 м", font=f_text, fill=INK, anchor="rm")

# weapons 1-3 and the active ability
slots = [("1", "MAR", True), ("2", "Дробовик", False), ("3", "Огнемёт", False)]
sw, sh, gap = 150, 78, 12
total = 3 * sw + 2 * gap + 24 + 200
x = W // 2 - total // 2
y = H - sh - 24
for key, name, on in slots:
    d.rounded_rectangle((x, y, x + sw, y + sh), 10, fill=PANEL, outline=INK if on else (255, 255, 255, 70), width=2 if on else 1)
    d.text((x + 14, y + 10), key, font=f_small, fill=QUIET)
    d.text((x + sw // 2, y + 46), name, font=f_bold if on else f_text, fill=INK, anchor="mm")
    x += sw + gap
x += 24 - gap
d.rounded_rectangle((x, y, x + 200, y + sh), 10, fill=PANEL, outline=(255, 255, 255, 70), width=1)
d.text((x + 14, y + 10), "Пробел", font=f_small, fill=QUIET)
d.text((x + 100, y + 42), "Удар с орбиты", font=f_text, fill=INK, anchor="mm")
bar(x + 14, y + sh - 16, 172, 5, 0.7, XP)

Image.alpha_composite(base, layer).convert("RGB").save(dst, optimize=True)
print("saved", dst)
