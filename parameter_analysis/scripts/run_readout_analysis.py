"""readout part of the parameter analysis: every setting needs a freshly trained decoder

meant for a gpu (colab t4: ~80 s for the sweeps, ~5 min for the active search).
on a cpu it works too, just slower (~1 s per decoder per core is a rough guide)

writes to ../results/readout/:
  oat_readout.csv        one-at-a-time sweeps with retraining: z of facilitation / IOR, gain, valid error at T1
  active_points.csv      every evaluated point of the 6-d boundary search
  active_summary.json    gp length scales, total-order importance, P(ok) over the box
  gp_maps.json           P(ok) on 25x25 grids for three parameter pairs (slice at defaults and average over others)

usage: python run_readout_analysis.py [--skip-oat] [--skip-active]
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import ant_model as M

OUT = Path(__file__).resolve().parent.parent / 'results' / 'readout'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESH = 3.0  # z needed for both facilitation (T1) and IOR (any T<=10), 1800 paired trials

OAT_VALUES = {
    'PERIPHERAL_CONTRAST': np.linspace(0, 1.2, 13),
    'FOCAL_CONTRAST': np.linspace(0, 2.0, 13),
    'FRAME_NOISE': np.linspace(0, 1.0, 13),
    'BACKGROUND_NOISE': np.linspace(0, 2.0, 13),
    'BACKGROUND_BASELINE_READS': np.array([0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 100]),
    'TEMPERATURE': np.geomspace(0.02, 3.0, 13),
    'CENTER': np.linspace(0.15, 0.95, 13),
    'MIN_WIDTH': np.geomspace(0.01, 0.3, 13),
    'PRESENT_RATE': np.linspace(0.1, 1.0, 13),
}
ANTS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]

RB = {  # (lo, hi, log-scale?) box for the 6-d search
    'PERIPHERAL_CONTRAST': (0.0, 0.8, False),
    'FOCAL_CONTRAST': (0.2, 2.0, False),
    'FRAME_NOISE': (0.0, 1.0, False),
    'BACKGROUND_NOISE': (0.0, 2.0, False),
    'BACKGROUND_BASELINE_READS': (0, 32, False),
    'TEMPERATURE': (0.03, 2.0, True),
}
RK = list(RB)


def run_oat():
    jobs = [(k, float(v)) for k, vals in OAT_VALUES.items() for v in vals]
    plist = [M.params(**{k: (int(v) if k == 'BACKGROUND_BASELINE_READS' else v)}) for k, v in jobs]
    t = time.time()
    rows = M.evaluate(plist, device=DEVICE, chunk=39)
    print(f'{len(plist)} decoders in {time.time() - t:.0f} s')
    table = [[k, v, r['facil_z'], r['ior_z'], r['gain'], float(r['valid'][0])] for (k, v), r in zip(jobs, rows)]
    # the ant count only changes the test colonies, so the default decoder is reused
    dec0 = M.train_decoders_batched([M.params()], 83, DEVICE)[0]
    for a in ANTS_VALUES:
        p = M.params(ANTS=a)
        res, hist, cs = M.test_posner(p, dec0, 50151, device=DEVICE)
        r = M.readout_metrics(res, hist, cs)
        table.append(['ANTS', a, r['facil_z'], r['ior_z'], r['gain'], float(r['valid'][0])])
    with open(OUT / 'oat_readout.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['param', 'value', 'facil_z', 'ior_z', 'gain', 'valid_error_T1', 'ok'])
        for row in table:
            w.writerow([row[0], round(row[1], 4)] + [round(x, 4) for x in row[2:]] + [int(row[2] > THRESH and row[3] > THRESH)])


def u_to_params(u):
    out = {}
    for x, k in zip(u, RK):
        lo, hi, lg = RB[k]
        v = np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo))) if lg else lo + x * (hi - lo)
        out[k] = int(round(v)) if k == 'BACKGROUND_BASELINE_READS' else float(v)
    return M.params(**out)


def run_active():
    from scipy.stats import qmc
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    def run(U):
        rows = M.evaluate([u_to_params(u) for u in U], device=DEVICE, chunk=48)
        return np.array([[r['facil_z'], r['ior_z'], r['gain']] for r in rows])

    def make_gp():
        kernel = ConstantKernel() * Matern(length_scale=np.full(len(RK), 0.3), nu=2.5) + WhiteKernel(0.5)
        return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2)

    score = lambda Y: np.clip(np.minimum(Y[:, 0], Y[:, 1]), -5, 15)
    U = qmc.Sobol(len(RK), seed=1).random(192)
    Y = run(U)
    rounds = []
    for rnd in range(3):
        gp = make_gp().fit(U, score(Y))
        cand = qmc.Sobol(len(RK), seed=100 + rnd).random(8192)
        mu, sd = gp.predict(cand, return_std=True)
        straddle = 1.96 * sd - np.abs(mu - THRESH)  # high = uncertain and close to the threshold
        pick = []
        for i in np.argsort(-straddle):  # greedy, keep picks spread out
            if all(np.linalg.norm(cand[i] - cand[j]) > 0.12 for j in pick):
                pick.append(i)
            if len(pick) == 48:
                break
        Unew = cand[pick]
        Ynew = run(Unew)
        U, Y = np.vstack([U, Unew]), np.vstack([Y, Ynew])
        rounds.append(dict(n=len(U), near_boundary=float((np.abs(score(Ynew) - THRESH) < 2).mean())))
        print('round', rnd, rounds[-1])
    gp = make_gp().fit(U, score(Y))

    uni = qmc.Sobol(len(RK), seed=11).random(2 ** 14)
    A, B = qmc.Sobol(len(RK), seed=7).random(20000), qmc.Sobol(len(RK), seed=8).random(20000)
    fA = gp.predict(A) > THRESH
    importance = {}
    for i, k in enumerate(RK):  # jansen total-order estimator on the binary outcome of the surrogate
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        importance[k] = float(0.5 * np.mean((fA.astype(float) - (gp.predict(ABi) > THRESH)) ** 2) / fA.var())
    json.dump(dict(initial_points=192, rounds=rounds, n_evaluated=len(U), ok_evaluated=int((score(Y) > THRESH).sum()),
                   p_ok_box=float((gp.predict(uni) > THRESH).mean()),
                   length_scales=dict(zip(RK, gp.kernel_.k1.k2.length_scale.tolist())),
                   total_order_importance=importance, bounds={k: list(RB[k]) for k in RK}),
              open(OUT / 'active_summary.json', 'w'), indent=2)
    with open(OUT / 'active_points.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'u_{k}' for k in RK] + ['facil_z', 'ior_z', 'gain', 'ok'])
        for u, y in zip(U, Y):
            w.writerow([round(x, 5) for x in u] + [round(x, 4) for x in y] + [int(min(y[0], y[1]) > THRESH)])

    # p(ok) maps for the report: slice at the defaults and average over the other parameters
    def unit(k):
        lo, hi, lg = RB[k]
        d = M.DEFAULTS[k]
        return (np.log(d) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (d - lo) / (hi - lo)
    u_def = np.array([unit(k) for k in RK])
    g = np.linspace(0, 1, 25)
    G1, G2 = np.meshgrid(g, g, indexing='ij')
    R = qmc.Sobol(len(RK), seed=5).random(256)
    maps = []
    for a, b in [('FOCAL_CONTRAST', 'FRAME_NOISE'), ('PERIPHERAL_CONTRAST', 'FOCAL_CONTRAST'),
                 ('FRAME_NOISE', 'BACKGROUND_NOISE')]:
        ia, ib = RK.index(a), RK.index(b)
        P = np.tile(u_def, (625, 1))
        P[:, ia], P[:, ib] = G1.ravel(), G2.ravel()
        sl = (gp.predict(P) > THRESH).astype(float)
        mg = np.zeros(625)
        for r in R:
            Q = np.tile(r, (625, 1))
            Q[:, ia], Q[:, ib] = G1.ravel(), G2.ravel()
            mg += gp.predict(Q) > THRESH
        maps.append(dict(x=a, y=b, slice=sl.reshape(25, 25).tolist(), marginal=(mg / len(R)).reshape(25, 25).tolist()))
    json.dump(dict(grid_unit=g.tolist(), maps=maps), open(OUT / 'gp_maps.json', 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-oat', action='store_true')
    ap.add_argument('--skip-active', action='store_true')
    args = ap.parse_args()
    print('device:', DEVICE)
    if not args.skip_oat:
        run_oat()
    if not args.skip_active:
        run_active()
    print('done ->', OUT)
