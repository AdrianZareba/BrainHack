"""slow, minimal video of the posner experiment with Adrian's ant colony (nb05 rule)

valid and invalid cue side by side, white background, one trail per side, resource above each box,
share of ants per box. no readout and no result curves. writes animations/ants_experiment*.mp4 and .gif

the colony rule comes from parameter_analysis/scripts/ant_model.py (same as posner_ants.ipynb)

run from the repo root:
  ANTS_PANELS=valid ANTS_TARGET_ROUND=3 python animations/scripts/make_ants_video.py
  ANTS_PANELS=valid ANTS_TARGET_ROUND=10 python animations/scripts/make_ants_video.py
without ANTS_PANELS both the valid and the invalid cue are shown side by side
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'parameter_analysis' / 'scripts'))
import ant_model as am

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

P = am.params()
ANTS = int(P['ANTS'])
LAMP_ROUND = int(os.environ.get('ANTS_TARGET_ROUND', 3))  # the target appears in this round, the video ends after it
FPS = 10
ROUND_S = 4.5           # seconds per colony round
OUT_END, EAT_END = 0.42, 0.58
XC = P['CENTER']
PANELS = tuple(os.environ.get('ANTS_PANELS', 'valid,invalid').split(','))

C = dict(bg='#ffffff', screen='#f4f5f2', edge='#9aa29d', ink='#1b2420', muted='#6b7670',
         valid='#245fa9', invalid='#af4935', ant='#2a2520', food='#2f9e5b', pher='#d08a1e',
         lamp='#f2b705', cue='#111111', hill='#3f8f6b')


def one_colony(seed, horizon=LAMP_ROUND + 1):
    """nb05 rule for one colony, cue on the left, with ant counts and successes kept"""
    rng = np.random.default_rng(seed)
    cue, ph, res = P['CUE_STRENGTH'], np.full(2, P['INITIAL_PHEROMONE']), np.ones(2)
    out = []
    for _ in range(horizon):
        lw = P['PHEROMONE_EXPONENT'] * np.log(P['TRAIL_FLOOR'] + ph) + np.array([cue, 0.0])
        pl = (1 - P['EXPLORATION']) / (1 + np.exp(lw[1] - lw[0])) + P['EXPLORATION'] / 2
        n0 = rng.binomial(ANTS, pl)
        n = np.array([n0, ANTS - n0])
        k = rng.binomial(n, res)
        out.append(dict(cue=cue, ph=ph.copy(), res=res.copy(), n=n, k=k))
        res = np.maximum(res - P['CONSUMPTION'] * k / ANTS, 0)
        ph = (1 - P['EVAPORATION']) * ph + P['DEPOSIT'] * k / ANTS
        cue *= P['CUE_RETENTION']
    return out


# typical colony: closest to the mean-field trajectory of the cued share
mf = am.mean_field(P, LAMP_ROUND)
SEED = min(range(1, 500), key=lambda s: sum((r['n'][0] / ANTS - m) ** 2 for r, m in zip(one_colony(s), mf)))
COLONY = one_colony(SEED)


def plan(r):
    rng = np.random.default_rng(1000 + r)
    st = COLONY[r]
    side = np.where(np.arange(ANTS) < st['n'][0], 0, 1)
    rank = np.where(side == 0, np.arange(ANTS), np.arange(ANTS) - st['n'][0])
    ok = rank < st['k'][side]
    o = rng.permutation(ANTS)
    return dict(side=side[o], ok=ok[o], delay=rng.uniform(0, 0.18, ANTS), lane=rng.uniform(-1, 1, ANTS),
                wob=rng.uniform(0, 6.28, ANTS))


PLANS = [plan(r) for r in range(LAMP_ROUND)]


def live(r, phase):
    """resource, pheromone and cue shown inside a round, built from ants that already ate or came back"""
    st, pl = COLONY[r], PLANS[r]
    u = (phase - pl['delay']) / 0.82
    ate = np.array([np.sum(pl['ok'] & (pl['side'] == s) & (u > OUT_END + 0.05)) for s in (0, 1)])
    back = np.array([np.sum(pl['ok'] & (pl['side'] == s) & (u >= 1)) for s in (0, 1)])
    res = np.maximum(st['res'] - P['CONSUMPTION'] * ate / ANTS, 0)
    ph = st['ph'] * (1 - P['EVAPORATION'] * phase) + P['DEPOSIT'] * back / ANTS
    return res, ph, st['cue'] * P['CUE_RETENTION'] ** phase


# timeline entries: (stage, round, phase)
FRAMES = [('fix', 0, 0.0)] * int(FPS * 2) + [('cue', 0, 0.0)] * int(FPS * 2)
for r in range(LAMP_ROUND):
    FRAMES += [('run', r, i / (FPS * ROUND_S)) for i in range(int(FPS * ROUND_S))]
    if r == LAMP_ROUND - 1:
        stop = [f for f in FRAMES if f[1] == r and f[2] >= OUT_END + 0.08][0]
        i0 = FRAMES.index(stop)
        FRAMES[i0:i0] = [('lamp', r, stop[2])] * int(FPS * 5)
FRAMES += [('end', LAMP_ROUND - 1, 1.0)] * int(FPS * 3)

ONE = len(PANELS) == 1
fig = plt.figure(figsize=(10, 7.4) if ONE else (16, 6.6), dpi=110, facecolor=C['bg'])
gs = fig.add_gridspec(2, len(PANELS), height_ratios=[1, 2.3], hspace=0.04, wspace=0.08,
                      left=0.03, right=0.97, top=0.82 if ONE else 0.86, bottom=0.04)
AX = {name: (fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])) for i, name in enumerate(PANELS)}
HEAD = fig.text(0.5, 0.94, '', ha='center', color=C['ink'], fontsize=18)
# time-step strip: one marker per colony round, the current one filled
SW = min(0.13 * LAMP_ROUND, 0.84)
STEPS = fig.add_axes([0.5 - SW / 2, 0.85, SW, 0.05]) if ONE else None


def draw_steps(stage, r):
    STEPS.clear()
    STEPS.set_axis_off()
    STEPS.set_xlim(-0.5, LAMP_ROUND - 0.5)
    STEPS.set_ylim(-1, 1)
    STEPS.plot([0, LAMP_ROUND - 1], [0, 0], color=C['edge'], lw=1.5, zorder=1)
    for t in range(LAMP_ROUND):
        done = stage in ('run', 'lamp', 'end') and t <= r
        STEPS.scatter([t], [0], s=260, color=C['valid'] if done else C['bg'], ec=C['valid'], lw=2, zorder=2)
        STEPS.text(t, 0, f'T{t + 1}', ha='center', va='center', fontsize=9.5 if t < 9 else 7.5, zorder=3,
                   color=C['bg'] if done else C['valid'], fontweight='bold')
BOX = 0.12


def draw(name, stage, r, phase):
    ax_res, ax = AX[name]
    for a in (ax_res, ax):
        a.clear()
        a.set_xlim(-1, 1)
        a.set_xticks([]), a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    running = stage in ('run', 'lamp', 'end')
    res, ph, cue = live(r, phase) if running else (np.ones(2), np.full(2, P['INITIAL_PHEROMONE']), P['CUE_STRENGTH'])

    # resource above each box as a bar: dashed = full, green = what is left
    ax_res.set_ylim(0, 1.35)
    for s, c in ((0, -XC), (1, XC)):
        ax_res.add_patch(plt.Rectangle((c - BOX, 0), 2 * BOX, 1, fill=False, ec=C['edge'], ls='--', lw=1.2))
        ax_res.add_patch(plt.Rectangle((c - BOX, 0), 2 * BOX, res[s], color=C['hill'], alpha=0.75, lw=0))
        ax_res.text(c + BOX + 0.03, 0.5, f'zasób {res[s]:.2f}', color=C['hill'], fontsize=12, va='center')
    ax_res.plot([-1, 1], [0, 0], color=C['edge'], lw=1)
    title = 'TRAFNA WSKAZÓWKA' if name == 'valid' else 'NIETRAFNA WSKAZÓWKA'
    ax_res.text(-0.98, 1.2, title, color=C[name], fontsize=15, fontweight='bold')

    # screen
    ax.set_ylim(-0.62, 0.62)
    ax.add_patch(plt.Rectangle((-1, -0.62), 2, 1.24, color=C['screen'], zorder=0))
    for s in (0, 1):
        bx = -XC if s == 0 else XC
        ax.plot([0, bx], [0, 0], color=C['pher'], lw=max(24 * ph[s], 1), alpha=0.55, solid_capstyle='round', zorder=1)
        ax.text(bx / 2, -0.14, f'feromon {ph[s]:.2f}', color=C['pher'], ha='center', fontsize=11)
        ax.add_patch(plt.Rectangle((bx - BOX, -BOX * 2), 2 * BOX, 4 * BOX, fill=False, ec=C['edge'], lw=1.5, zorder=2))
    if stage != 'fix':
        a = min(cue / P['CUE_STRENGTH'], 1)
        if a > 0.03:
            ax.add_patch(plt.Rectangle((-XC - BOX - 0.02, -BOX * 2 - 0.04), 2 * BOX + 0.04, 4 * BOX + 0.08,
                                       fill=False, ec=C['cue'], lw=1 + 5 * a, alpha=a, zorder=3))
    if stage in ('lamp', 'end') or (stage == 'run' and r == LAMP_ROUND - 1 and phase >= OUT_END + 0.08):
        side = 0 if name == 'valid' else 1
        lx = -XC if side == 0 else XC
        share = COLONY[r]['n'][side] / ANTS
        contrast = P['PERIPHERAL_CONTRAST'] + (P['FOCAL_CONTRAST'] - P['PERIPHERAL_CONTRAST']) * share
        for rad, al in ((0.17, 0.12), (0.11, 0.25), (0.06, 0.6), (0.03, 1.0)):
            ax.add_patch(plt.Circle((lx, 0), rad, color=C['lamp'], alpha=al * contrast, lw=0, zorder=4))
        ax.scatter([lx], [0], marker='*', s=900, color=C['lamp'], ec=C['ink'], lw=1.2, zorder=9)
        ax.text(lx, -0.36, 'target', color=C['ink'], ha='center', fontsize=12, fontweight='bold')
    ax.plot([-0.035, 0.035], [0, 0], color=C['ink'], lw=2.2, zorder=6)
    ax.plot([0, 0], [-0.06, 0.06], color=C['ink'], lw=2.2, zorder=6)

    # ants and shares
    if running:
        pl = PLANS[r]
        u = np.clip((phase - pl['delay']) / 0.82, 0, 1)
        bx = np.where(pl['side'] == 0, -XC, XC)
        frac = np.where(u < OUT_END, u / OUT_END, np.where(u < EAT_END, 1, 1 - (u - EAT_END) / (1 - EAT_END)))
        at_box = (u >= OUT_END) & (u < EAT_END)
        xs = frac * bx + np.where(at_box, pl['lane'] * 0.08, 0)
        ys = np.where(at_box, np.cos(pl['wob']) * 0.17, pl['lane'] * 0.035 + np.where(u < OUT_END, 0.025, -0.025))
        home = (u <= 0) | (u >= 1)
        xs = np.where(home, pl['lane'] * 0.05, xs)
        ys = np.where(home, np.sin(pl['wob']) * 0.05, ys)
        carry = pl['ok'] & (u > OUT_END + 0.05) & (u < 1)
        ax.scatter(xs, ys, s=18, color=C['ant'], zorder=7, lw=0)
        ax.scatter(xs[carry] + 0.01, ys[carry] + 0.012, s=14, color=C['food'], zorder=8, lw=0)
        for s in (0, 1):
            ax.text(-XC if s == 0 else XC, 0.4, f'{round(100 * COLONY[r]["n"][s] / ANTS)}% mrówek',
                    color=C['ink'], ha='center', fontsize=15, fontweight='bold')


def update(i):
    stage, r, phase = FRAMES[i]
    for name in PANELS:
        draw(name, stage, r, phase)
    if ONE:
        draw_steps(stage, r)
    HEAD.set_text({'fix': 'punkt fiksacji', 'cue': 'wskazówka po lewej',
                   'lamp': f'krok T{r + 1}: pojawia się target'}.get(stage, f'krok czasowy T{r + 1}'))
    return []


if __name__ == '__main__':
    out = ROOT / 'animations'
    stem = 'ants_experiment' + ('_' + PANELS[0] if ONE else '') + f'_T{LAMP_ROUND}'
    mp4, gif = out / f'{stem}.mp4', out / f'{stem}.gif'
    print(f'colony seed {SEED}, {len(FRAMES)} frames = {len(FRAMES) / FPS:.0f} s')
    anim = animation.FuncAnimation(fig, update, frames=len(FRAMES), blit=False)
    anim.save(mp4, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000, extra_args=['-pix_fmt', 'yuv420p']),
              savefig_kwargs=dict(facecolor=C['bg']))
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = 'scale=' + ('800' if ONE else '1100') + ':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none'
    os.system(f'"{ff}" -y -loglevel error -i "{mp4}" -vf "{vf}" "{gif}"')
    print('saved', mp4, gif)
