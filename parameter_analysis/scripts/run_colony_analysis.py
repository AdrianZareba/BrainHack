"""colony part of the parameter analysis (cpu, about 1 minute)

writes to ../results/colony/:
  critical_values.csv    one-at-a-time ok-interval for each parameter around the defaults
  sobol_indices.csv      S1 / ST (with 95% conf) for four outputs
  sobol_pairs.csv        strongest second-order interactions for "is the model ok"
  morris.csv             morris mu* / sigma (cheap screening)
  regime_fractions.json  how much of the parameter box gives each regime
  nearest_failure.json   shortest path from the defaults to a broken model (20k directions)
  mean_field_check.json  mean-field vs 1800 stochastic colonies on 40 random settings
  oat_sweeps.npz         full one-at-a-time sweeps (focus trajectories + metrics)

usage: python run_colony_analysis.py
"""
import csv
import json
from pathlib import Path

import numpy as np

import ant_model as M
import ant_sensitivity as S

OUT = Path(__file__).resolve().parent.parent / 'results' / 'colony'
OUT.mkdir(parents=True, exist_ok=True)


def write_csv(name, header, rows):
    with open(OUT / name, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    # one-at-a-time sweeps and critical values
    sweeps = S.oat_sweeps()
    np.savez_compressed(OUT / 'oat_sweeps.npz', **{f'{k}__{m}': v[m] for k, v in sweeps.items()
                                                   for m in ('x', 'f', 'facil', 'cross', 'win_depth', 'regime')})
    crit = S.critical_values(sweeps)
    rows = []
    for k, (lo, hi, why) in crit.items():
        d = M.DEFAULTS[k]
        rel = [abs(v - d) / d for v in (lo, hi) if v is not None]
        below, above = why.split(' | ')
        rows.append([k, d, S.RANGES[k][0], S.RANGES[k][1],
                     '' if lo is None else round(lo, 4), '' if hi is None else round(hi, 4),
                     '' if not rel else round(min(rel), 3), below, above])
    write_csv('critical_values.csv',
              ['param', 'default', 'range_lo', 'range_hi', 'ok_from', 'ok_to', 'min_rel_change_to_break',
               'regime_below', 'regime_above'], rows)

    # sobol (saltelli design, 81920 mean-field runs)
    problem, res, frac, X, m = S.sobol_analysis(2 ** 13)
    rows = []
    for out_name, r in res.items():
        for i, k in enumerate(S.KEYS):
            rows.append([out_name, k, round(r['S1'][i], 4), round(r['S1_conf'][i], 4),
                         round(r['ST'][i], 4), round(r['ST_conf'][i], 4)])
    write_csv('sobol_indices.csv', ['output', 'param', 'S1', 'S1_conf95', 'ST', 'ST_conf95'], rows)
    s2 = res['ok']['S2']
    pairs = sorted(((S.KEYS[i], S.KEYS[j], s2[i, j]) for i in range(len(S.KEYS))
                    for j in range(i + 1, len(S.KEYS))), key=lambda t: -t[2])
    write_csv('sobol_pairs.csv', ['param_a', 'param_b', 'S2_ok'], [[a, b, round(v, 4)] for a, b, v in pairs])
    json.dump(dict(n_runs=int(len(X)), fractions=frac,
                   S1_sum_ok=float(res['ok']['S1'].sum())), open(OUT / 'regime_fractions.json', 'w'), indent=2)

    # morris screening
    mo, n_morris = S.morris_analysis()
    rows = [[out_name, k, round(v['mu_star'][i], 4), round(v['sigma'][i], 4)]
            for out_name, v in mo.items() for i, k in enumerate(S.KEYS)]
    write_csv('morris.csv', ['output', 'param', 'mu_star', 'sigma'], rows)

    # inverse objective: nearest failure from the defaults
    nf = S.nearest_failure(20000)
    order = np.argsort(np.nan_to_num(nf['t'], nan=9))[:20]
    nearest = [dict(radius=float(nf['t'][i]), regime=S.REGIMES[nf['regime'][i]],
                    point={k: float(nf['lo'][j] + np.clip(nf['x0'][j] + nf['t'][i] * nf['D'][i, j], 0, 1) * nf['span'][j])
                           for j, k in enumerate(S.KEYS)}) for i in order]
    stats = dict(n_directions=20000, morris_runs=int(n_morris),
                 radius_q05=float(np.nanquantile(nf['t'], 0.05)), radius_median=float(np.nanquantile(nf['t'], 0.5)),
                 never_fails=float(np.isnan(nf['t']).mean()),
                 first_failure_regime={r: float((nf['regime'] == i).mean()) for i, r in enumerate(S.REGIMES)})
    json.dump(dict(stats=stats, nearest=nearest), open(OUT / 'nearest_failure.json', 'w'), indent=2)

    # how good is the mean-field shortcut
    rng = np.random.default_rng(3)
    U = rng.random((40, len(S.KEYS)))
    Xr = np.array([[S.RANGES[k][0] + u * (S.RANGES[k][1] - S.RANGES[k][0]) for u, k in zip(row, S.KEYS)] for row in U])
    mf = S.mf_batch(S.to_dict(Xr))
    errs = [float(np.abs(M.cued_focus(M.params(**dict(zip(S.KEYS, Xr[i]))), n=1800)[0] - mf[i]).max())
            for i in range(len(Xr))]
    json.dump(dict(max_abs=max(errs), median_abs=float(np.median(errs)), per_setting=errs),
              open(OUT / 'mean_field_check.json', 'w'), indent=2)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
