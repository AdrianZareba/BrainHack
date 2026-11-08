"""figures for the report, built from ../results (run the two analysis scripts first)

usage: python make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import ant_model as M
import ant_sensitivity as S

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)

C = dict(valid='#245fa9', invalid='#af4935', ok='#3f8f6b', late='#c98a1c', noior='#8b8fa3', nofacil='#b5577a',
         ink='#1b2420', muted='#5b6862', accent='#2d6a55')
REG_COLORS = [C['ok'], C['late'], C['noior'], C['nofacil']]
REG_LABELS = ['działa', 'IOR za późno (po T10)', 'brak IOR', 'brak facylitacji']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def colony_trajectories():
    presets = [('domyślne', {}), ('CONSUMPTION = 0', dict(CONSUMPTION=0)), ('CUE_RETENTION = 1', dict(CUE_RETENTION=1)),
               ('EXPLORATION = 0.6', dict(EXPLORATION=0.6)), ('PHEROMONE_EXPONENT = 4', dict(PHEROMONE_EXPONENT=4)),
               ('TRAIL_FLOOR = 0', dict(TRAIL_FLOOR=0))]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True, sharey=True, layout='constrained')
    T = np.arange(1, 25)
    base = M.mean_field(M.params())
    for ax, (name, over) in zip(axes.flat, presets):
        mean, per = M.cued_focus(M.params(**over), n=1800)
        lo, hi = np.percentile(per, [10, 90], axis=1)
        reg = M.colony_metrics(mean)['regime']
        ax.axvspan(1, 10, color='#e3e8e4', alpha=0.6, lw=0)
        ax.fill_between(T, lo, hi, color=C['valid'], alpha=0.15, lw=0)
        ax.plot(T, base, '--', color=C['muted'], lw=1)
        ax.plot(T, mean, color=C['valid'], lw=2)
        ax.axhline(0.5, color=C['ink'], lw=0.7, alpha=0.5)
        ax.set_title(f'{name}\n[{REG_LABELS[S.REGIMES.index(reg)]}]', fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    for ax in axes[1]:
        ax.set_xlabel('runda T')
    for ax in axes[:, 0]:
        ax.set_ylabel('mrówki po stronie cue')
    fig.suptitle('Udział mrówek po stronie cue (1800 kolonii, pasmo 10-90 percentyl, przerywana = domyślne)')
    fig.savefig(FIG / 'colony_trajectories.png', dpi=150)


def colony_sobol():
    rows = read_csv(RES / 'colony' / 'sobol_indices.csv')
    outs = [('ok', 'czy działa'), ('win_depth', 'głębokość IOR do T10'), ('facil', 'facylitacja w T1')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout='constrained')
    order = sorted(S.KEYS, key=lambda k: -float(next(r['ST'] for r in rows if r['output'] == 'ok' and r['param'] == k)))
    y = np.arange(len(order))
    for ax, (key, title) in zip(axes, outs):
        st = [float(next(r['ST'] for r in rows if r['output'] == key and r['param'] == k)) for k in order]
        s1 = [float(next(r['S1'] for r in rows if r['output'] == key and r['param'] == k)) for k in order]
        ax.barh(y, st, color=C['accent'], label='ST (z interakcjami)')
        ax.barh(y, np.clip(s1, 0, None), height=0.4, color='white', alpha=0.6, label='S1 (sam parametr)')
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
    axes[0].set_yticks(y, order)
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Indeksy Sobola dla kolonii (81 920 przebiegów mean-field)')
    fig.savefig(FIG / 'colony_sobol.png', dpi=150)


def colony_oat_strips():
    fig, axes = plt.subplots(len(S.KEYS), 1, figsize=(8, 5.6), layout='constrained')
    cmap = ListedColormap(REG_COLORS)
    for ax, k in zip(axes, S.KEYS):
        x = np.linspace(*S.RANGES[k], 241)
        reg = S.metrics_batch(S.mf_batch({k: x}))['regime']
        ax.imshow(reg[None], aspect='auto', cmap=cmap, vmin=-0.5, vmax=3.5, extent=[x[0], x[-1], 0, 1])
        ax.axvline(M.DEFAULTS[k], color=C['ink'], lw=2)
        ax.set_yticks([])
        ax.set_ylabel(k, rotation=0, ha='right', va='center', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REG_COLORS]
    fig.legend(handles, REG_LABELS, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Reżim przy zmianie jednego parametru (pozostałe domyślne, czarna kreska = domyślna wartość)')
    fig.savefig(FIG / 'colony_oat_strips.png', dpi=150)


def colony_phase_maps():
    pairs = [('CONSUMPTION', 'EXPLORATION'), ('CONSUMPTION', 'PHEROMONE_EXPONENT'), ('CUE_RETENTION', 'CONSUMPTION')]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout='constrained')
    cmap = ListedColormap(REG_COLORS)
    for ax, (a, b) in zip(axes, pairs):
        xa, xb, m = S.phase_map(a, b, n=161)
        ax.imshow(m['regime'].T, origin='lower', aspect='auto', cmap=cmap, vmin=-0.5, vmax=3.5,
                  extent=[xa[0], xa[-1], xb[0], xb[-1]])
        ax.plot(M.DEFAULTS[a], M.DEFAULTS[b], 'o', mfc='white', mec=C['ink'])
        ax.set(xlabel=a, ylabel=b)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REG_COLORS]
    fig.legend(handles, REG_LABELS, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Mapy reżimów dla par parametrów (pozostałe domyślne, kółko = domyślne)')
    fig.savefig(FIG / 'colony_phase_maps.png', dpi=150)


def readout_oat():
    rows = read_csv(RES / 'readout' / 'oat_readout.csv')
    params = list(dict.fromkeys(r['param'] for r in rows))
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), layout='constrained')
    for ax, k in zip(axes.flat, params):
        rr = [r for r in rows if r['param'] == k]
        x = np.array([float(r['value']) for r in rr])
        ax.plot(x, [float(r['facil_z']) for r in rr], '-o', ms=3, color=C['valid'], label='facylitacja T1')
        ax.plot(x, [float(r['ior_z']) for r in rr], '-o', ms=3, color=C['invalid'], label='IOR (max do T10)')
        ax.axhline(3, color=C['muted'], ls='--', lw=0.8)
        ax.axhline(0, color=C['ink'], lw=0.5, alpha=0.4)
        ax.axvline(64 if k == 'ANTS' else M.DEFAULTS[k], color=C['muted'], ls=':', lw=1)
        if k in ('TEMPERATURE', 'MIN_WIDTH', 'ANTS'):
            ax.set_xscale('log')
        ax.set_title(k, fontsize=8)
    axes[0, 0].set_ylabel('z (1800 par)')
    axes[1, 0].set_ylabel('z (1800 par)')
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle('Odczyt: siła efektu po ponownym treningu dekodera (przerywana = próg z = 3, kropkowana = domyślne)')
    fig.savefig(FIG / 'readout_oat_z.png', dpi=150)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), layout='constrained')
    for ax, k in zip(axes.flat, params):
        rr = [r for r in rows if r['param'] == k]
        x = np.array([float(r['value']) for r in rr])
        ax.plot(x, [float(r['gain']) for r in rr], '-o', ms=3, color=C['accent'], label='wzmocnienie')
        ax.plot(x, [float(r['valid_error_T1']) for r in rr], '-o', ms=3, color=C['late'], label='błąd trafnego cue w T1')
        ax.axvline(64 if k == 'ANTS' else M.DEFAULTS[k], color=C['muted'], ls=':', lw=1)
        if k in ('TEMPERATURE', 'MIN_WIDTH', 'ANTS'):
            ax.set_xscale('log')
        ax.set_title(k, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle('Odczyt: wzmocnienie (błąd oszczędzony na jednostkę uwagi) i trudność zadania')
    fig.savefig(FIG / 'readout_oat_gain.png', dpi=150)


def readout_gp_maps():
    data = json.load(open(RES / 'readout' / 'gp_maps.json'))
    bounds = json.load(open(RES / 'readout' / 'active_summary.json'))['bounds']
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
    for col, mp in enumerate(data['maps']):
        for row, key in enumerate(('slice', 'marginal')):
            ax = axes[row, col]
            (xlo, xhi, xlg), (ylo, yhi, ylg) = bounds[mp['x']], bounds[mp['y']]
            im = ax.imshow(np.array(mp[key]).T, origin='lower', aspect='auto', cmap='Greens', vmin=0, vmax=1,
                           extent=[xlo, xhi, ylo, yhi])
            ax.plot(M.DEFAULTS[mp['x']], M.DEFAULTS[mp['y']], 'x', color=C['ink'], ms=8, mew=2)
            ax.set(xlabel=mp['x'], ylabel=mp['y'])
            ax.set_title('pozostałe domyślne' if key == 'slice' else 'uśrednione po pozostałych', fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label='P(efekt wykrywalny)')
    fig.suptitle('Odczyt: granica wykrywalności z modelu zastępczego (proces gaussowski, 336 dekoderów)')
    fig.savefig(FIG / 'readout_gp_maps.png', dpi=150)


if __name__ == '__main__':
    colony_trajectories()
    colony_sobol()
    colony_oat_strips()
    colony_phase_maps()
    readout_oat()
    readout_gp_maps()
    print('figures ->', FIG)
