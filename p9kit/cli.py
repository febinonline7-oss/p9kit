"""Command line interface: ``p9kit clustering``, ``p9kit demo``, ``p9kit search``."""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import __version__
from .images import (blind_search, build_mask, depth_50, recovery_curve, simulate_frames,
                     split_half_test)
from .stats import clustering, luck_test


def _load_orbits(path):
    import pandas as pd
    frame = pd.read_csv(path)
    rename = {"om": "node", "w": "argperi", "i": "inc"}
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    needed = ["a", "inc", "node", "argperi"]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    return frame


def cmd_clustering(args):
    frame = _load_orbits(args.orbits)
    if args.q_min is not None and "q" in frame:
        frame = frame[frame["q"] > args.q_min]
    sample = {k: frame[k].to_numpy() for k in ("a", "inc", "node", "argperi")}
    direction, strength = clustering(sample)
    rng = np.random.default_rng(args.seed)
    test = luck_test(sample, rng, n_null=args.n_null, isotropic=args.isotropic)
    result = dict(n_objects=int(len(frame)), mean_direction_deg=round(float(direction), 1),
                  clustering_R=round(float(strength), 3), p_value=test.p_value,
                  chance=test.one_in(), verdict=test.verdict())
    print(json.dumps(result, indent=2))
    return result


def cmd_demo(args):
    """End-to-end check on simulated images: hide an object, then find it blind."""
    truth = dict(mag=args.mag, rate=3.2, angle=250.0, x0=150.0, y0=120.0)
    images, _ = simulate_frames(movers=[truth], seed=args.seed)
    mask = build_mask(images)
    candidates = blind_search(images, mask=mask)
    found = [c for c in candidates
             if np.hypot(c["x"] - truth["x0"], c["y"] - truth["y0"]) < 5]
    rng = np.random.default_rng(args.seed)
    mags, eff = recovery_curve(images, np.arange(20.0, 23.01, 0.5), rng, per_magnitude=6, mask=mask)
    result = dict(recovered=bool(found), candidates=len(candidates),
                  best_snr=round(found[0]["snr"], 1) if found else None,
                  depth_50=round(depth_50(mags, eff), 2))
    if found:
        result["split_half"] = {k: round(v, 1) for k, v in split_half_test(images, found[0]).items()}
    print(json.dumps(result, indent=2))
    return result


def cmd_search(args):
    from .ztf import fetch_patch
    images = fetch_patch(args.ra, args.dec, size_arcmin=args.size)
    mask = build_mask(images)
    rng = np.random.default_rng(args.seed)
    mags, eff = recovery_curve(images, np.arange(20.0, 23.51, 0.5), rng, per_magnitude=args.injections,
                               mask=mask)
    candidates = blind_search(images, mask=mask, progress=200)
    junk = blind_search(images, mask=mask, frames=-np.asarray(images.frames), progress=None)
    level = float(np.percentile([c["snr"] for c in junk], 90)) if junk else 5.0
    survivors = [c for c in candidates if c["snr"] >= level]
    result = dict(n_images=len(images.frames), baseline_days=round(images.baseline_days, 2),
                  masked_fraction=round(float(mask.mean()), 3), depth_50=round(depth_50(mags, eff), 2),
                  n_candidates=len(candidates), junk_level=round(level, 1),
                  n_survivors=len(survivors),
                  survivors=[{**{k: round(float(v), 2) for k, v in c.items()},
                              "split_half": {k: round(v, 1) for k, v in split_half_test(images, c).items()}}
                             for c in survivors[:5]])
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="p9kit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"p9kit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("clustering", help="measure perihelion clustering in a table of orbits")
    p.add_argument("orbits", help="CSV with columns a, e, i/inc, om/node, w/argperi")
    p.add_argument("--q-min", type=float, default=None, help="keep only objects above this perihelion")
    p.add_argument("--n-null", type=int, default=500)
    p.add_argument("--isotropic", action="store_true", help="null without survey selection effects")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_clustering)

    p = sub.add_parser("demo", help="run the image pipeline on simulated data")
    p.add_argument("--mag", type=float, default=21.8)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("search", help="download a ZTF patch and search it (needs network)")
    p.add_argument("ra", type=float)
    p.add_argument("dec", type=float)
    p.add_argument("--size", type=float, default=6.0, help="cutout size in arcminutes")
    p.add_argument("--injections", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
