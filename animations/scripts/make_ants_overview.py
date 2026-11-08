"""one-slide schematic overview of the ant-colony attention model, based on the whiteboard sketch

four time steps: two food mounds (left = cued side), nest in the middle, arrow width = how many ants go there.
the differences are exaggerated on purpose, it is a picture for people who never saw the model.
writes animations/ants_overview.png and .svg

run from the repo root: python animations/scripts/make_ants_overview.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
C = dict(bg='#ffffff', ink='#1b2420', muted='#6b7670', food='#5fae7c', food_edge='#2f7a4f',
         trail='#d08a1e', cue='#e0542e', nest='#1b2420', ant='#2a2520', faint='#b9c2bd')

# schematic state per step: eaten share of each mound and the ants going each way
STEPS = [
    dict(t='T1', eaten=(0.0, 0.0), ants=(0.0, 0.0), cue=1.0,
         text='Błysk po lewej.\nOba miejsca są pełne.'),
    dict(t='T2', eaten=(0.3, 0.07), ants=(0.65, 0.2), cue=0.6,
         text='Większość mrówek idzie\nza błyskiem i zaczyna jeść.'),
    dict(t='T3', eaten=(0.62, 0.12), ants=(0.85, 0.2), cue=0.25,
         text='Szlak w lewo się wzmacnia,\nlewe jedzenie szybko znika.'),
    dict(t='T4', eaten=(0.93, 0.18), ants=(0.2, 0.8), cue=0.0,
         text='Po lewej prawie nic nie ma.\nMrówki przechodzą w prawo.'),
]

LX, RX, W, H = -2.4, 2.4, 1.6, 0.95   # mound centres, half-width and height


def dome(cx):
    th = np.linspace(0, np.pi, 80)
    return np.column_stack([cx + W * np.cos(th), H * np.sin(th)])


def draw_mound(ax, cx, y0, eaten, side, rng):
    full = dome(cx) + [0, y0]
    ax.add_patch(Polygon(full, closed=True, fill=False, ec=C['faint'], ls=(0, (4, 3)), lw=1.4, zorder=1))
    toward = 1 if side == 0 else -1
    if eaten < 0.85:
        # what is left: a lower mound with one bite on the side that faces the nest
        h = H * (1 - eaten)
        th = np.linspace(0, np.pi, 80)
        pts = np.column_stack([cx + W * np.cos(th), y0 + h * np.sin(th)])
        ax.add_patch(Polygon(pts, closed=True, fc=C['food'], ec=C['food_edge'], lw=2, zorder=2))
        if eaten > 0:
            r = 0.2 + 0.35 * eaten
            ax.add_patch(Circle((cx + toward * W * 0.62, y0 + h * 0.85), r, color=C['bg'], zorder=3))
    else:  # only crumbs left
        xs = np.linspace(cx - W, cx + W, 60)
        ys = y0 + 0.07 + 0.05 * np.abs(np.sin(xs * 7)) + rng.uniform(0, 0.04, xs.size)
        ax.fill_between(xs, y0, ys, color=C['food'], ec=C['food_edge'], lw=1.5, zorder=4)
    # baseline drawn last so the bite never cuts it
    ax.plot([cx - W - 0.05, cx + W + 0.05], [y0, y0], color=C['food_edge'], lw=2.4, zorder=5)


def draw_trail(ax, y0, side, amount):
    if amount <= 0:
        return
    x_end = LX + W * 0.75 if side == 0 else RX - W * 0.75
    start = (-0.18 if side == 0 else 0.18, y0 + 0.12)
    end = (x_end, y0 + 0.45)
    if amount < 0.3:  # a few ants: dotted line like on the sketch
        ax.annotate('', xy=end, xytext=start, zorder=6,
                    arrowprops=dict(arrowstyle='-|>', color=C['trail'], lw=2, ls=(0, (1.5, 2.5)), mutation_scale=16))
    else:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle=f'simple,head_width={0.6 + amount * 1.2},'
                                                            f'head_length={0.5 + amount * 0.6},tail_width={amount * 0.55}',
                                     mutation_scale=22, color=C['trail'], alpha=0.9, lw=0, zorder=6))
    # a few ants along the trail
    n = int(round(amount * 9))
    for k in range(n):
        f = (k + 0.5) / max(n, 1) * 0.8 + 0.1
        x = start[0] + (end[0] - start[0]) * f
        y = start[1] + (end[1] - start[1]) * f + (0.08 if k % 2 else -0.08)
        ax.add_patch(Circle((x, y), 0.055, color=C['ant'], zorder=7))


def draw_cue(ax, y0, strength):
    if strength <= 0:
        return
    cx, cy = LX - W - 0.05, y0 + H * 0.95
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        ax.plot([cx + 0.18 * np.cos(a), cx + (0.18 + 0.32 * strength) * np.cos(a)],
                [cy + 0.18 * np.sin(a), cy + (0.18 + 0.32 * strength) * np.sin(a)],
                color=C['cue'], lw=2.6, alpha=0.35 + 0.65 * strength, solid_capstyle='round', zorder=8)
    ax.add_patch(Circle((cx, cy), 0.12, color=C['cue'], alpha=0.35 + 0.65 * strength, zorder=8))


def main():
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=C['bg'])
    fig.text(0.04, 0.93, 'Uwaga jako kolonia mrówek', fontsize=30, fontweight='bold', color=C['ink'])
    fig.text(0.04, 0.885, 'mrówki = uwaga,  jedzenie = to, czego jeszcze nie sprawdziliśmy w danym miejscu',
             fontsize=15, color=C['muted'])

    ax = fig.add_axes([0.02, 0.1, 0.96, 0.76])
    ax.set_xlim(-5.6, 9.6)
    ax.set_ylim(-0.5, 4 * 1.75 - 0.3)
    ax.set_aspect('equal')
    ax.axis('off')
    for i, st in enumerate(STEPS):
        y0 = (3 - i) * 1.75
        ax.text(-5.5, y0 + 0.35, st['t'], fontsize=24, fontweight='bold', color=C['ink'], va='center')
        draw_mound(ax, LX, y0, st['eaten'][0], 0, rng)
        draw_mound(ax, RX, y0, st['eaten'][1], 1, rng)
        draw_trail(ax, y0, 0, st['ants'][0])
        draw_trail(ax, y0, 1, st['ants'][1])
        draw_cue(ax, y0, st['cue'])
        ax.add_patch(Circle((0, y0 + 0.12), 0.13, fc=C['bg'], ec=C['nest'], lw=2.5, zorder=9))
        if i < 3:
            ax.plot([-5.5, 9.4], [y0 - 0.35, y0 - 0.35], color='#eef0ee', lw=1, zorder=0)
        ax.text(4.7, y0 + 0.45, st['text'], fontsize=15.5, color=C['ink'], va='center', linespacing=1.4)

    # small legend
    lg = fig.add_axes([0.04, 0.025, 0.62, 0.06])
    lg.set_xlim(0, 18)
    lg.set_aspect('equal')
    lg.set_ylim(-0.5, 0.5)
    lg.axis('off')
    lg.add_patch(Circle((0.2, 0), 0.17, fc=C['bg'], ec=C['nest'], lw=2.5))
    lg.text(0.55, 0, 'gniazdo = punkt fiksacji', va='center', fontsize=12.5, color=C['muted'])
    lg.add_patch(FancyArrowPatch((5.0, 0), (6.0, 0), arrowstyle='simple,head_width=1.2,head_length=0.8,tail_width=0.45',
                                 mutation_scale=14, color=C['trail'], lw=0))
    lg.text(6.2, 0, 'grubość = ile mrówek', va='center', fontsize=12.5, color=C['muted'])
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        lg.plot([10.6 + 0.1 * np.cos(a), 10.6 + 0.3 * np.cos(a)], [0.1 * np.sin(a), 0.3 * np.sin(a)],
                color=C['cue'], lw=2)
    lg.text(11.05, 0, 'błysk (wskazówka)', va='center', fontsize=12.5, color=C['muted'])

    fig.text(0.70, 0.045, 'Schemat, różnice przerysowane.\nModel: 64 mrówki, feromon na szlaku,\nzużywanie jedzenia, 8% losowych wypraw.',
             fontsize=10.5, color=C['muted'], linespacing=1.4)

    out = ROOT / 'animations'
    fig.savefig(out / 'ants_overview.png', facecolor=C['bg'])
    fig.savefig(out / 'ants_overview.svg', facecolor=C['bg'])
    print('saved', out / 'ants_overview.png')


if __name__ == '__main__':
    main()
