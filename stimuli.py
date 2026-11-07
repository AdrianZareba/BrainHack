"""Synthetic X/O discrimination sequences, not a pixel-level human replication.

All images are 16x32. The peripheral cue is a square outline; the target
is an X or O inside it. Cue location and target class/location are independent.
No condition labels, target onset, or SOA are passed to the neural network.
"""
import torch


def make_batch(cfg, n, generator, soa_ms=None, valid=None, catch=None):
    """Return frames [B,T,1,H,W], per-frame labels, and evaluation metadata.

    Labels are 0=absent, 1=X, 2=O. Target persists to the end of the sequence.
    A common padded horizon prevents batch shape from revealing each onset.
    Pre-cue fixation is jittered by 0..3 steps, independently of all conditions.
    """
    dt = cfg['dt_ms']
    soas = cfg['train_soas_ms'] if soa_ms is None else [soa_ms]
    soa = torch.tensor(soas)[torch.randint(len(soas), (n,), generator=generator)]
    pre = 4 + torch.randint(4, (n,), generator=generator)
    onset = pre + soa // dt
    horizon = 7 + max(cfg['train_soas_ms'] + cfg['eval_soas_ms']) // dt + cfg['post_target_ms'] // dt
    cue_side = torch.randint(2, (n,), generator=generator)
    is_valid = torch.randint(2, (n,), generator=generator).bool() if valid is None else torch.full((n,), bool(valid))
    side = torch.where(is_valid, cue_side, 1 - cue_side)
    cls = 1 + torch.randint(2, (n,), generator=generator)
    absent = torch.rand(n, generator=generator) < cfg['catch_probability'] if catch is None else torch.full((n,), bool(catch))
    x = torch.randn(n, horizon, 1, 16, 32, generator=generator) * cfg['noise_sd']
    y = torch.zeros(n, horizon, dtype=torch.long)
    for i in range(n):
        cx = 8 + 16 * int(cue_side[i])
        start, stop = int(pre[i]), int(pre[i]) + cfg['cue_ms'] // dt
        x[i, start:stop, 0, 4, cx-4:cx+5] += cfg['contrast']
        x[i, start:stop, 0, 12, cx-4:cx+5] += cfg['contrast']
        x[i, start:stop, 0, 4:13, cx-4] += cfg['contrast']
        x[i, start:stop, 0, 4:13, cx+4] += cfg['contrast']
        if not absent[i]:
            cx = 8 + 16 * int(side[i])
            # Equal squared pixel energy: X has 9 pixels; O has 12.
            glyph = torch.zeros(5, 5)
            if cls[i] == 1:
                glyph.fill_diagonal_(1)
                glyph += torch.fliplr(torch.eye(5))
                glyph.clamp_(max=1)
            else:
                glyph[0, 1:4] = 1; glyph[4, 1:4] = 1
                glyph[1:4, 0] = 1; glyph[1:4, 4] = 1
            glyph *= (9 / glyph.square().sum()).sqrt() * cfg['contrast']
            x[i, int(onset[i]):, 0, 6:11, cx-2:cx+3] += glyph
            y[i, int(onset[i]):] = cls[i]
    return x, y, {'onset': onset, 'class': cls, 'catch': absent, 'valid': is_valid, 'soa_ms': soa}
