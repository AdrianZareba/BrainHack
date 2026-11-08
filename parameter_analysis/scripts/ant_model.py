"""parametrised copy of the ant-colony Posner model from posner_ants.ipynb

the rules are copied 1:1 from posner_ants.ipynb, only the
module-level constants became a dict so we can sweep them. nothing in the
dynamics was changed
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn

DEFAULTS = dict(
    # colony
    CUE_STRENGTH=0.30,
    CUE_RETENTION=0.72,
    INITIAL_PHEROMONE=1.00,
    EVAPORATION=0.42,
    DEPOSIT=0.55,
    CONSUMPTION=0.32,
    PHEROMONE_EXPONENT=1.55,
    TRAIL_FLOOR=0.12,
    EXPLORATION=0.08,
    ANTS=64,
    # readout
    PERIPHERAL_CONTRAST=0.10,
    FOCAL_CONTRAST=0.94,
    FRAME_NOISE=0.13,
    BACKGROUND_NOISE=0.48,
    BACKGROUND_BASELINE_READS=4,
    TEMPERATURE=0.20,
    # stimulus
    CENTER=0.57,
    CENTER_JITTER=0.07,
    MIN_WIDTH=0.044,
    WIDTH_RANGE=0.019,
    # decoder training
    TRAIN_SIZE=60_000,
    VALIDATION_SIZE=6_000,
    PRESENT_RATE=0.75,
    HIDDEN_WIDTHS=(96, 64),
    LEARNING_RATE=0.003,
    BATCH_SIZE=512,
    EPOCHS=20,
    # protocol
    HORIZON=24,
    TRIALS=1800,
)

COLONY_KEYS = ['CUE_STRENGTH', 'CUE_RETENTION', 'INITIAL_PHEROMONE', 'EVAPORATION', 'DEPOSIT',
               'CONSUMPTION', 'PHEROMONE_EXPONENT', 'TRAIL_FLOOR', 'EXPLORATION', 'ANTS']
READOUT_KEYS = ['PERIPHERAL_CONTRAST', 'FOCAL_CONTRAST', 'FRAME_NOISE', 'BACKGROUND_NOISE',
                'BACKGROUND_BASELINE_READS', 'TEMPERATURE']

GRID = torch.linspace(-1, 1, 61)
SIDES = (GRID > 0).long()
UNIFORM = torch.full_like(GRID, 1 / len(GRID))


def params(**over):
    p = dict(DEFAULTS)
    for k, v in over.items():
        if k not in p:
            raise KeyError(k)
        p[k] = v
    return p


# ---------------------------------------------------------------- colony

def ant_colonies(p, cue_side, seed, *, consumption=None, cue_retention=None, initial_cue=None):
    """same rule as nb05; returns dict of [round, colony, side] arrays (numpy)"""
    consumption = p['CONSUMPTION'] if consumption is None else consumption
    cue_retention = p['CUE_RETENTION'] if cue_retention is None else cue_retention
    initial_cue = p['CUE_STRENGTH'] if initial_cue is None else initial_cue
    ants = int(round(p['ANTS']))
    rng = np.random.default_rng(seed)
    cue = np.eye(2)[cue_side] * initial_cue
    pheromone = np.full_like(cue, p['INITIAL_PHEROMONE'])
    resource = np.ones_like(cue)
    hist = {'focus': [], 'cue': [], 'food': [], 'pheromone': []}
    for _ in range(int(p['HORIZON'])):
        log_weight = p['PHEROMONE_EXPONENT'] * np.log(p['TRAIL_FLOOR'] + pheromone) + cue
        left_weight = 1 / (1 + np.exp(log_weight[:, 1] - log_weight[:, 0]))
        left_probability = (1 - p['EXPLORATION']) * left_weight + p['EXPLORATION'] / 2
        n_left = rng.binomial(ants, left_probability)
        count = np.stack((n_left, ants - n_left), axis=1)
        successful = rng.binomial(count, resource)
        success_share = successful / ants
        hist['focus'].append(count / ants)
        hist['cue'].append(cue)
        hist['food'].append(resource)
        hist['pheromone'].append(pheromone)
        resource = np.maximum(resource - consumption * success_share, 0)
        pheromone = (1 - p['EVAPORATION']) * pheromone + p['DEPOSIT'] * success_share
        cue = cue_retention * cue
    return {k: np.stack(v) for k, v in hist.items()}


def mean_field(p, horizon=None):
    """deterministic version (binomials replaced by expectations), cue on side 0.
    returns focus on the cued side per round. used as a fast sanity proxy"""
    H = int(p['HORIZON'] if horizon is None else horizon)
    cue = np.array([p['CUE_STRENGTH'], 0.0])
    ph = np.full(2, p['INITIAL_PHEROMONE'])
    res = np.ones(2)
    out = []
    for _ in range(H):
        lw = p['PHEROMONE_EXPONENT'] * np.log(p['TRAIL_FLOOR'] + ph) + cue
        pl = (1 - p['EXPLORATION']) / (1 + np.exp(lw[1] - lw[0])) + p['EXPLORATION'] / 2
        share = np.array([pl, 1 - pl])
        out.append(pl)
        succ = share * res
        res = np.maximum(res - p['CONSUMPTION'] * succ, 0)
        ph = (1 - p['EVAPORATION']) * ph + p['DEPOSIT'] * succ
        cue = p['CUE_RETENTION'] * cue
    return np.array(out)


def cued_focus(p, n=1800, seed=51151, **kw):
    """mean share of ants on the cued side per round, plus per-colony array"""
    cue_side = np.arange(n) % 2
    h = ant_colonies(p, cue_side, seed, **kw)
    f = h['focus'][:, np.arange(n), cue_side]  # [round, colony]
    return f.mean(1), f


DELTA = 0.01   # minimum focus deviation from 50% that counts as an effect
WINDOW = 10    # rounds shown on the main plot (STAGES = 1..10)


def colony_metrics(f_mean, delta=DELTA, window=WINDOW):
    """summarise the cued-side focus trajectory.
    facil   : focus(T1) - 0.5
    cross   : first round (1-based) with focus < 0.5 - delta after the peak, nan if none
    depth   : 0.5 - min focus after the peak (positive = IOR-like overshoot)
    regime  : ok / late_ior / no_ior / no_facil
    margin  : signed distance to failure (>0 healthy) for boundary search"""
    f = np.asarray(f_mean)
    facil = f[0] - 0.5
    peak = int(np.argmax(f[:window]))  # early peak only, a late second peak must not hide the dip
    after = f[peak:]
    below = np.where(after < 0.5 - delta)[0]
    cross = float(peak + below[0] + 1) if len(below) else float('nan')
    depth = 0.5 - after.min()
    win_depth = 0.5 - f[peak:window].min() if peak < window else -1.0
    if facil < delta:
        regime = 'no_facil'
    elif math.isnan(cross):
        regime = 'no_ior'
    elif cross > window:
        regime = 'late_ior'
    else:
        regime = 'ok'
    margin = min(facil - delta, win_depth - delta)
    return dict(facil=facil, cross=cross, depth=depth, win_depth=win_depth,
                peak_round=peak + 1, regime=regime, margin=margin)


# ---------------------------------------------------------------- readout

def make_generator(seed):
    return torch.Generator().manual_seed(seed)


def make_basis(p, side, rng):
    centers = torch.tensor([-p['CENTER'], p['CENTER']])
    center = centers[side] + p['CENTER_JITTER'] * (torch.rand(len(side), generator=rng) - 0.5)
    width = p['MIN_WIDTH'] + p['WIDTH_RANGE'] * torch.rand(len(side), generator=rng)
    log_profile = -0.5 * ((GRID - center[:, None]) / width[:, None]).square()
    return log_profile.softmax(dim=-1)


def make_frame(p, basis, focus, background_error, noise, present):
    contrast = p['PERIPHERAL_CONTRAST'] + (p['FOCAL_CONTRAST'] - p['PERIPHERAL_CONTRAST']) * focus[:, SIDES]
    lamp = basis / basis.amax(dim=1, keepdim=True)
    signal = (contrast * lamp * present[:, None]
              + p['FRAME_NOISE'] * noise
              - p['BACKGROUND_NOISE'] * background_error)
    return (signal / p['TEMPERATURE']).softmax(dim=-1)


def decoder_examples(p, size, seed):
    rng = make_generator(seed)
    side = torch.randint(2, (size,), generator=rng)
    left_share = torch.rand(size, generator=rng)
    focus = torch.stack((left_share, 1 - left_share), dim=-1)
    present = torch.rand(size, generator=rng) < p['PRESENT_RATE']
    rounds = torch.randint(1, int(p['HORIZON']) + 1, (size, 1), generator=rng).float()
    basis = make_basis(p, side, rng)
    background_error = (torch.randn(size, len(GRID), generator=rng)
                        / (rounds + p['BACKGROUND_BASELINE_READS']).sqrt())
    frame_noise = torch.randn(size, len(GRID), generator=rng)
    inputs = make_frame(p, basis, focus, background_error, frame_noise, present)
    targets = torch.where(present[:, None], basis, UNIFORM)
    return inputs, targets


def cross_entropy(model, inputs, targets):
    return -(targets * model(inputs).log_softmax(dim=-1)).sum(dim=-1).mean()


def train_decoder(p, seed, device='cpu'):
    torch.manual_seed(seed)
    shuffle_rng = make_generator(seed + 300)
    tx, ty = decoder_examples(p, int(p['TRAIN_SIZE']), seed + 500)
    vx, vy = decoder_examples(p, int(p['VALIDATION_SIZE']), seed + 600)
    tx, ty, vx, vy = (t.to(device) for t in (tx, ty, vx, vy))
    w1, w2 = p['HIDDEN_WIDTHS']
    model = nn.Sequential(nn.Linear(61, w1), nn.GELU(), nn.Linear(w1, w2), nn.GELU(),
                          nn.Linear(w2, 61)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p['LEARNING_RATE'])
    best, best_w = float('inf'), None
    for _ in range(int(p['EPOCHS'])):
        model.train()
        order = torch.randperm(len(tx), generator=shuffle_rng).to(device)
        for b in order.split(int(p['BATCH_SIZE'])):
            loss = cross_entropy(model, tx[b], ty[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = cross_entropy(model, vx, vy).item()
            if vl < best:
                best = vl
                best_w = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_w)
    return model.cpu().eval()


def distribution_errors(prediction, basis):
    true_center = (basis * GRID).sum(dim=-1, keepdim=True)
    loc = (prediction * (GRID - true_center).square()).sum(dim=-1)
    w1 = (prediction - basis).cumsum(dim=-1)[..., :-1].abs().sum(dim=-1) * (GRID[1] - GRID[0])
    return loc, w1


@torch.inference_mode()
def test_posner(p, decoder, seed, *, equal_reads=False, trials=None, device='cpu', **colony_settings):
    """returns errors {'valid','invalid'}: [round, trial] tensors, colony history, cue_side"""
    trials = int(p['TRIALS'] if trials is None else trials)
    H = int(p['HORIZON'])
    base = int(p['BACKGROUND_BASELINE_READS'])
    rng = make_generator(seed)
    cue_side = torch.arange(trials) % 2
    shape = (trials, 61)
    hist = ant_colonies(p, cue_side.numpy(), seed + 1000, **colony_settings)
    focus = torch.tensor(hist['focus'], dtype=torch.float32)
    bsum = torch.randn(H + base, *shape, generator=rng).cumsum(dim=0)
    noise = torch.randn(shape, generator=rng)
    present = torch.ones(trials, dtype=torch.bool)
    decoder = decoder.to(device)
    out = {}
    for cond, tside in (('valid', cue_side), ('invalid', 1 - cue_side)):
        basis = make_basis(p, tside, make_generator(seed + 2000))
        errs = []
        for t in range(1, H + 1):
            reads = base + (1 if equal_reads else t)
            frame = make_frame(p, basis, focus[t - 1], bsum[reads - 1] / reads, noise, present)
            pred = decoder(frame.to(device)).softmax(dim=-1).cpu()
            errs.append(distribution_errors(pred, basis)[0])
        out[cond] = torch.stack(errs)
    return out, hist, cue_side


def readout_metrics(res, hist=None, cue_side=None, window=WINDOW):
    """paired valid-invalid gap per round with z scores.
    gain = error saved per unit of extra focus on the cued side (slope over rounds 1..window)"""
    d = (res['valid'] - res['invalid'])[:window]  # >0 means valid worse (IOR)
    m = d.mean(1)
    se = d.std(1) / d.shape[1] ** 0.5
    z = m / se
    out = dict(gap=m.numpy(), z=z.numpy(), valid=res['valid'][:window].mean(1).numpy(),
               invalid=res['invalid'][:window].mean(1).numpy(),
               facil_z=float(-z[0]), ior_z=float(z.max()), facil=float(-m[0]), ior=float(m.max()),
               dz_facil=float(-m[0] / d[0].std()), dz_ior=float((m / d.std(1)).max()))
    if hist is not None:
        n = len(cue_side)
        fc = hist['focus'][:window, np.arange(n), cue_side.numpy()].mean(1) - 0.5
        out['gain'] = float(-np.polyfit(fc, out['gap'], 1)[0])
    return out


# ---------------------------------------------------------------- batched training (gpu)

class StackedMLP(nn.Module):
    """m independent 61-w1-w2-61 decoders evaluated with bmm"""

    def __init__(self, m, widths=(96, 64), seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dims = [61, *widths, 61]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            lin = [nn.Linear(a, b) for _ in range(m)]  # same init scheme as nn.Linear
            self.W.append(nn.Parameter(torch.stack([l.weight.T.detach() for l in lin])))
            self.b.append(nn.Parameter(torch.stack([l.bias.detach() for l in lin])[:, None, :]))

    def forward(self, x):  # x [m, batch, 61]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.W) - 1:
                x = nn.functional.gelu(x)
        return x

    def single(self, i):
        dims = [w.shape[1] for w in self.W] + [61]
        seq = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.GELU(), nn.Linear(dims[1], dims[2]),
                            nn.GELU(), nn.Linear(dims[2], dims[3]))
        with torch.no_grad():
            for j, k in enumerate((0, 2, 4)):
                seq[k].weight.copy_(self.W[j][i].T.cpu())
                seq[k].bias.copy_(self.b[j][i, 0].cpu())
        return seq.eval()


def train_decoders_batched(plist, seed, device='cpu'):
    """train one decoder per parameter dict in plist, all at once.
    every model gets its own synthetic dataset generated with its own readout params"""
    p0 = plist[0]
    m = len(plist)
    data = [decoder_examples(p, int(p0['TRAIN_SIZE']), seed + 500) for p in plist]
    val = [decoder_examples(p, int(p0['VALIDATION_SIZE']), seed + 600) for p in plist]
    tx = torch.stack([d[0] for d in data]).to(device)
    ty = torch.stack([d[1] for d in data]).to(device)
    vx = torch.stack([d[0] for d in val]).to(device)
    vy = torch.stack([d[1] for d in val]).to(device)
    del data, val
    model = StackedMLP(m, p0['HIDDEN_WIDTHS'], seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p0['LEARNING_RATE'])
    shuffle_rng = make_generator(seed + 300)
    best = torch.full((m,), float('inf'), device=device)
    best_state = [t.detach().clone() for t in model.parameters()]

    def ce(x, y):  # per-model mean cross-entropy [m]
        return -(y * model(x).log_softmax(-1)).sum(-1).mean(-1)

    n = tx.shape[1]
    for _ in range(int(p0['EPOCHS'])):
        order = torch.randperm(n, generator=shuffle_rng).to(device)
        for bidx in order.split(int(p0['BATCH_SIZE'])):
            loss = ce(tx[:, bidx], ty[:, bidx]).sum()  # params are disjoint, so grads stay separate
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = torch.cat([ce(vx[:, i:i + 2000], vy[:, i:i + 2000])[None] for i in range(0, vx.shape[1], 2000)]).mean(0)
            better = vl < best
            best = torch.where(better, vl, best)
            for s, t in zip(best_state, model.parameters()):
                s[better] = t.detach()[better]
    with torch.no_grad():
        for s, t in zip(best_state, model.parameters()):
            t.copy_(s)
    return [model.single(i) for i in range(m)]


def evaluate(plist, seed=83, test_seed=50151, device='cpu', chunk=32):
    """train a decoder for every setting (in gpu chunks) and run the posner test on it"""
    rows = []
    for c in range(0, len(plist), chunk):
        part = plist[c:c + chunk]
        decs = train_decoders_batched(part, seed, device)
        for p, d in zip(part, decs):
            res, hist, cs = test_posner(p, d, test_seed, device=device)
            rows.append(readout_metrics(res, hist, cs))
    return rows
