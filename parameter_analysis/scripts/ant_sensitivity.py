"""sensitivity and breaking-point analysis of the ant-colony model (colony part)

the colony mean is almost deterministic (mean-field matches 1800 stochastic colonies
to ~0.002), so the colony part can be evaluated for 1e5-1e6 parameter sets in seconds
"""
from __future__ import annotations

import numpy as np

import ant_model as M

# physically meaningful ranges used for sweeps, sobol and boundary search
RANGES = {
    'CUE_STRENGTH': (0.0, 1.5),
    'CUE_RETENTION': (0.0, 1.0),
    'INITIAL_PHEROMONE': (0.0, 4.0),
    'EVAPORATION': (0.0, 1.0),
    'DEPOSIT': (0.0, 2.5),
    'CONSUMPTION': (0.0, 1.2),
    'PHEROMONE_EXPONENT': (0.0, 5.0),
    'TRAIL_FLOOR': (0.0, 1.5),
    'EXPLORATION': (0.0, 1.0),
}
KEYS = list(RANGES)
H = 24
WINDOW = M.WINDOW
DELTA = M.DELTA
REGIMES = ['ok', 'late_ior', 'no_ior', 'no_facil']


def mf_batch(P, horizon=H):
    """vectorised mean-field. P: dict key -> array [N] (missing keys use defaults).
    returns cued-side focus [N, horizon]"""
    n = max(np.size(v) for v in P.values()) if P else 1
    g = {k: np.broadcast_to(np.asarray(P.get(k, M.DEFAULTS[k]), float), (n,)) for k in KEYS}
    cue = g['CUE_STRENGTH'].copy()
    ph0 = g['INITIAL_PHEROMONE'].copy()
    ph1 = ph0.copy()
    r0 = np.ones(n)
    r1 = np.ones(n)
    out = np.empty((n, horizon))
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for t in range(horizon):
            lw0 = g['PHEROMONE_EXPONENT'] * np.log(g['TRAIL_FLOOR'] + ph0) + cue
            lw1 = g['PHEROMONE_EXPONENT'] * np.log(g['TRAIL_FLOOR'] + ph1)
            d = np.nan_to_num(lw1 - lw0, nan=0.0, posinf=700, neginf=-700)
            pl = (1 - g['EXPLORATION']) / (1 + np.exp(np.clip(d, -700, 700))) + g['EXPLORATION'] / 2
            out[:, t] = pl
            s0 = pl * r0
            s1 = (1 - pl) * r1
            r0 = np.maximum(r0 - g['CONSUMPTION'] * s0, 0)
            r1 = np.maximum(r1 - g['CONSUMPTION'] * s1, 0)
            ph0 = (1 - g['EVAPORATION']) * ph0 + g['DEPOSIT'] * s0
            ph1 = (1 - g['EVAPORATION']) * ph1 + g['DEPOSIT'] * s1
            cue = g['CUE_RETENTION'] * cue
    return out


def metrics_batch(f, delta=DELTA, window=WINDOW):
    """vectorised colony_metrics. f: [N, H] cued focus.
    returns dict of arrays; regime coded as index into REGIMES"""
    n, h = f.shape
    facil = f[:, 0] - 0.5
    # peak of the early facilitation: searched only inside the plotted window, otherwise
    # a late second peak in oscillating runs would hide a clear IOR dip
    peak = f[:, :window].argmax(1)
    idx = np.arange(h)[None, :]
    after = idx >= peak[:, None]
    below = after & (f < 0.5 - delta)
    has = below.any(1)
    cross = np.where(has, below.argmax(1) + 1, np.nan).astype(float)
    in_win = after & (idx < window)
    win_min = np.where(in_win, f, np.inf).min(1)
    win_depth = np.where(np.isfinite(win_min), 0.5 - win_min, -1.0)
    depth = 0.5 - np.where(after, f, np.inf).min(1)
    regime = np.where(facil < delta, 3,
                      np.where(~has, 2, np.where(cross > window, 1, 0)))
    margin = np.minimum(facil - delta, win_depth - delta)
    # number of crossings of 0.5 inside the horizon (oscillation detector)
    sgn = np.sign(f - 0.5)
    flips = (np.diff(sgn, axis=1) != 0).sum(1)
    return dict(facil=facil, cross=cross, depth=depth, win_depth=win_depth,
                peak=peak + 1, regime=regime, margin=margin, flips=flips)


def default_vector():
    return np.array([M.DEFAULTS[k] for k in KEYS], float)


def to_dict(X):
    return {k: X[:, i] for i, k in enumerate(KEYS)}


# ------------------------------------------------------------- analyses

def oat_sweeps(n=241):
    """one-at-a-time sweeps over RANGES; returns {key: {x, metrics...}}"""
    out = {}
    for k, (lo, hi) in RANGES.items():
        x = np.linspace(lo, hi, n)
        f = mf_batch({k: x})
        m = metrics_batch(f)
        out[k] = dict(x=x, f=f, **m)
    return out


def critical_values(sweeps):
    """for each param: the interval around the default where regime == ok"""
    res = {}
    for k, s in sweeps.items():
        ok = s['regime'] == 0
        x = s['x']
        d = M.DEFAULTS[k]
        i0 = int(np.abs(x - d).argmin())
        if not ok[i0]:
            res[k] = (np.nan, np.nan, 'default not ok')
            continue
        lo_i = i0
        while lo_i > 0 and ok[lo_i - 1]:
            lo_i -= 1
        hi_i = i0
        while hi_i < len(x) - 1 and ok[hi_i + 1]:
            hi_i += 1
        lo_reg = REGIMES[s['regime'][lo_i - 1]] if lo_i > 0 else 'range edge'
        hi_reg = REGIMES[s['regime'][hi_i + 1]] if hi_i < len(x) - 1 else 'range edge'
        res[k] = (x[lo_i] if lo_i > 0 else None, x[hi_i] if hi_i < len(x) - 1 else None,
                  f'{lo_reg} | {hi_reg}')
    return res


def phase_map(k1, k2, n=121):
    a = np.linspace(*RANGES[k1], n)
    b = np.linspace(*RANGES[k2], n)
    A, B = np.meshgrid(a, b, indexing='ij')
    m = metrics_batch(mf_batch({k1: A.ravel(), k2: B.ravel()}))
    return a, b, {key: v.reshape(n, n) for key, v in m.items()}


def sobol_analysis(n_base=2 ** 13, seed=0):
    """saltelli/sobol indices for facil, window depth, crossing round and P(ok)"""
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol
    problem = dict(num_vars=len(KEYS), names=KEYS, bounds=[list(RANGES[k]) for k in KEYS])
    X = sobol_sample.sample(problem, n_base, calc_second_order=True, seed=seed)
    m = metrics_batch(mf_batch(to_dict(X)))
    outputs = {
        'facil': m['facil'],
        'win_depth': m['win_depth'],
        'cross': np.nan_to_num(m['cross'], nan=H + 1),
        'ok': (m['regime'] == 0).astype(float),
    }
    res = {}
    for name, y in outputs.items():
        Si = sobol.analyze(problem, y, calc_second_order=True, seed=seed)
        res[name] = dict(S1=Si['S1'], ST=Si['ST'], S1_conf=Si['S1_conf'], ST_conf=Si['ST_conf'], S2=Si['S2'])
    frac = {r: float((m['regime'] == i).mean()) for i, r in enumerate(REGIMES)}
    return problem, res, frac, X, m


def morris_analysis(r=200, seed=0):
    """morris elementary effects: the cheap screening step (r*(k+1) runs)"""
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris
    problem = dict(num_vars=len(KEYS), names=KEYS, bounds=[list(RANGES[k]) for k in KEYS])
    X = morris_sample.sample(problem, r, num_levels=8, seed=seed)
    m = metrics_batch(mf_batch(to_dict(X)))
    out = {}
    for name in ('facil', 'win_depth'):
        Si = morris.analyze(problem, X, m[name], num_levels=8, seed=seed)
        out[name] = dict(mu_star=np.array(Si['mu_star']), sigma=np.array(Si['sigma']))
    return out, len(X)


def nearest_failure(n_dirs=20000, steps=400, seed=0, sparse=None):
    """'inverse optimisation': from the default, walk along random directions in the
    range-normalised space and record the first point where the model stops being ok.
    the shortest such distance is the robustness radius; the direction tells which
    combination of parameters breaks the model with the smallest change.
    sparse=k restricts directions to k non-zero parameters"""
    rng = np.random.default_rng(seed)
    lo = np.array([RANGES[k][0] for k in KEYS])
    hi = np.array([RANGES[k][1] for k in KEYS])
    span = hi - lo
    x0 = (default_vector() - lo) / span
    D = rng.normal(size=(n_dirs, len(KEYS)))
    if sparse:
        mask = np.zeros_like(D)
        for i in range(n_dirs):
            mask[i, rng.choice(len(KEYS), sparse, replace=False)] = 1
        D *= mask
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    ts = np.linspace(0, 1.0, steps + 1)[1:]
    first_t = np.full(n_dirs, np.nan)
    first_reg = np.full(n_dirs, -1)
    alive = np.ones(n_dirs, bool)
    for t in ts:
        idx = np.where(alive)[0]
        if not len(idx):
            break
        z = x0 + t * D[idx]
        inside = ((z >= 0) & (z <= 1)).all(1)
        Xr = lo + np.clip(z, 0, 1) * span
        m = metrics_batch(mf_batch(to_dict(Xr)))
        fail = (m['regime'] != 0) & inside
        first_t[idx[fail]] = t
        first_reg[idx[fail]] = m['regime'][fail]
        alive[idx[fail | ~inside]] = False
    return dict(D=D, t=first_t, regime=first_reg, x0=x0, lo=lo, span=span)
