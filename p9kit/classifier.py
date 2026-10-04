"""A small CNN that separates real point sources from the junk a stack throws up.

The training pictures are produced by the search pipeline itself, so they carry the same
noise correlations as the data the classifier will see: a classifier trained on idealised
white-noise stamps scores real detections near 0.5 and is useless.
"""
from __future__ import annotations

import numpy as np

from .images import ImageSet, add_point_source, matched_filter, remove_static_sky, stack

STAMP_SIZE = 21


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("the classifier needs: pip install 'p9kit[cnn]'") from exc
    return torch, nn


def noise_level(image):
    med = np.median(image)
    return 1.4826 * np.median(np.abs(image - med)) + 1e-9


def cut_stamp(image, x, y, size: int = STAMP_SIZE, norm=None):
    """Small picture around (x, y) in units of the image noise, clipped and scaled."""
    half = size // 2
    x, y = int(round(x)), int(round(y))
    if x - half < 0 or y - half < 0 or x + half + 1 > image.shape[1] or y + half + 1 > image.shape[0]:
        return None
    patch = np.asarray(image, dtype=np.float32)[y - half:y + half + 1, x - half:x + half + 1]
    return (np.clip(patch / (norm or noise_level(image)), -5, 30) / 10.0).astype(np.float32)


def make_training_stamps(rng, n_sets: int = 20, npix: int = 256, rate_range=(1.0, 6.0),
                         good_mag=(19.5, 22.3), junk_mag=(19.0, 21.5), n_good: int = 8,
                         n_wrong: int = 6, max_cosmics: int = 8, frame_factory=None):
    """Build labelled stamps: 1 = object moving at the stacked speed, 0 = cosmic ray or streak.

    ``frame_factory(movers, seed)`` returns ``(ImageSet, cosmic_rays)``; the default makes
    simulated frames, but real frames with injected sources can be supplied instead.
    """
    from .images import simulate_frames

    def default_factory(movers, seed):
        return simulate_frames(movers=movers, seed=seed, npix=npix, n_cosmics=2.0)

    factory = frame_factory or default_factory
    margin = 45
    X, labels = [], []
    for k in range(n_sets):
        rate = float(rng.uniform(*rate_range))
        angle = float(rng.uniform(0, 360))
        good = [dict(mag=float(rng.uniform(*good_mag)), rate=rate, angle=angle,
                     x0=float(rng.uniform(margin, npix - margin)),
                     y0=float(rng.uniform(margin, npix - margin))) for _ in range(n_good)]
        wrong = [dict(mag=float(rng.uniform(*junk_mag)), rate=rate * float(rng.uniform(1.8, 3.0)),
                      angle=angle + float(rng.uniform(50, 310)),
                      x0=float(rng.uniform(margin, npix - margin)),
                      y0=float(rng.uniform(margin, npix - margin))) for _ in range(n_wrong)]
        images, cosmics = factory(good + wrong, 300 + k)
        stacked = stack(images, rate, angle, frames=remove_static_sky(images.frames))
        dt = np.asarray(images.times, dtype=float) - float(images.times[0])
        vx = rate / images.pixel_scale * np.cos(np.radians(angle))
        vy = rate / images.pixel_scale * np.sin(np.radians(angle))
        for mv in good:
            stamp = cut_stamp(stacked, mv["x0"], mv["y0"])
            if stamp is not None:
                X.append(stamp); labels.append(1.0)
        for mv in wrong:
            wx = mv["rate"] / images.pixel_scale * np.cos(np.radians(mv["angle"])) - vx
            wy = mv["rate"] / images.pixel_scale * np.sin(np.radians(mv["angle"])) - vy
            stamp = cut_stamp(stacked, mv["x0"] + wx * dt.mean(), mv["y0"] + wy * dt.mean())
            if stamp is not None:
                X.append(stamp); labels.append(0.0)
        for (index, cx, cy) in list(cosmics)[:max_cosmics]:
            stamp = cut_stamp(stacked, cx - vx * dt[int(index)], cy - vy * dt[int(index)])
            if stamp is not None:
                X.append(stamp); labels.append(0.0)
    return np.asarray(X)[:, None], np.asarray(labels, dtype=np.float32)


def build_cnn():
    torch, nn = _require_torch()
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 5 * 5, 32), nn.ReLU(), nn.Linear(32, 1), nn.Flatten(0))


class RealBogusClassifier:
    """Trained CNN wrapper: ``score(stacked_image, x, y)`` returns 1 for real, 0 for junk."""

    def __init__(self, net=None):
        self.net = net

    def train(self, X, y, epochs: int = 40, batch_size: int = 64, lr: float = 2e-3, seed: int = 0,
              test_fraction: float = 0.2):
        torch, nn = _require_torch()
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        split = int((1 - test_fraction) * len(y))
        train_idx, test_idx = perm[:split], perm[split:]
        self.net = build_cnn()
        pos_weight = torch.as_tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        Xt, yt = torch.as_tensor(X[train_idx]), torch.as_tensor(y[train_idx])
        for _ in range(epochs):
            order = torch.randperm(len(yt))
            for b in range(0, len(yt), batch_size):
                j = order[b:b + batch_size]
                opt.zero_grad()
                loss_fn(self.net(Xt[j]), yt[j]).backward()
                opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(self.net(torch.as_tensor(X[test_idx]))) > 0.5).numpy()
        return float((pred == y[test_idx]).mean())

    def score(self, stacked_image, x, y):
        torch, _ = _require_torch()
        if self.net is None:
            raise RuntimeError("classifier is not trained")
        stamp = cut_stamp(stacked_image, x, y)
        if stamp is None:
            return 0.0
        with torch.no_grad():
            return float(torch.sigmoid(self.net(torch.as_tensor(stamp[None, None]))).item())
