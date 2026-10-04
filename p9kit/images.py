"""Shift-and-stack search for slow-moving objects, with the checks that make a null result mean something."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

ARCSEC_PER_PIX_ZTF = 1.01


@dataclass(frozen=True)
class ImageSet:
    """Aligned frames in units of their own noise, with observation times in days.

    ``limit_mags`` is the single-frame 5-sigma limiting magnitude of each frame, which is
    what lets fake objects be injected with a physically meaningful brightness.
    """

    frames: np.ndarray
    times: np.ndarray
    limit_mags: np.ndarray | None = None
    pixel_scale: float = ARCSEC_PER_PIX_ZTF
    seeing_arcsec: float = 2.0

    def __post_init__(self):
        if len(self.frames) != len(self.times):
            raise ValueError("frames and times must have the same length")

    @property
    def psf_sigma_pix(self):
        return self.seeing_arcsec / 2.355 / self.pixel_scale

    @property
    def baseline_days(self):
        return float(np.ptp(self.times))


def flux_for_magnitude(mag, limit_mag, snr_at_limit: float = 5.0):
    """Peak height, in units of the frame noise, of a source of magnitude ``mag``."""
    return snr_at_limit * 10 ** (-0.4 * (np.asarray(mag, dtype=float) - np.asarray(limit_mag, dtype=float)))


def add_point_source(image, x, y, peak, sigma_pix):
    """Add a Gaussian point source of the given peak height, in place."""
    half = int(np.ceil(4 * sigma_pix))
    xs = np.arange(max(0, int(x) - half), min(image.shape[1], int(x) + half + 1))
    ys = np.arange(max(0, int(y) - half), min(image.shape[0], int(y) + half + 1))
    if xs.size == 0 or ys.size == 0:
        return
    gx, gy = np.meshgrid(xs, ys)
    image[np.ix_(ys, xs)] += peak * np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma_pix ** 2))


def remove_static_sky(frames):
    """Subtract the per-pixel median: stars stay still, so the median *is* the static sky."""
    frames = np.asarray(frames, dtype=np.float32)
    return (frames - np.median(frames, axis=0)).astype(np.float32)


def build_mask(images: ImageSet, levels=((5, 7), (20, 15), (100, 45)), scatter_factor: float = 3.0):
    """Pixels to ignore: bright stars (with a radius that grows with brightness) and wobbly pixels.

    Undersampled bright stars never subtract perfectly, and their scars dominate any blind
    search, so the area around them must be excluded before, not after, looking at candidates.
    """
    frames = np.asarray(images.frames, dtype=float)
    static = np.median(frames, axis=0)
    noise = 1.4826 * np.median(np.abs(static - np.median(static))) + 1e-9
    smooth = ndimage.gaussian_filter(static, images.psf_sigma_pix)
    mask = np.zeros(static.shape, dtype=bool)
    for level, radius in levels:
        core = smooth > level * noise
        if core.any():
            mask |= ndimage.binary_dilation(core, ndimage.generate_binary_structure(2, 2),
                                            iterations=int(radius))
    # Pixels that vary a lot *and* sit on something static are bad subtractions, not movers:
    # restricting the scatter cut to star positions keeps real moving objects out of the mask.
    scatter = frames.std(axis=0)
    median_scatter = np.median(scatter)
    on_a_star = smooth > 3 * noise
    wobbly = ((scatter > scatter_factor * median_scatter) & on_a_star) | (scatter > 2 * scatter_factor * median_scatter)
    mask |= ndimage.binary_dilation(wobbly, np.ones((9, 9), bool))
    return mask


def patch_quality(mask, max_masked_fraction: float = 0.4):
    """How much of a patch is unusable, and whether it is worth searching at all.

    A crowded field can be almost entirely masked by bright-star scars; searching what is
    left mostly produces artefacts, so it is cheaper to pick a different patch.
    """
    fraction = float(np.asarray(mask, dtype=bool).mean())
    return dict(masked_fraction=fraction, usable=bool(fraction <= max_masked_fraction),
                max_masked_fraction=float(max_masked_fraction))


def matched_filter(image, sigma_pix):
    """Signal-to-noise map: smooth with the PSF shape, then divide by the robust noise."""
    smoothed = ndimage.gaussian_filter(np.asarray(image, dtype=float), sigma_pix)
    med = np.median(smoothed)
    mad = np.median(np.abs(smoothed - med))
    return (smoothed - med) / (1.4826 * mad + 1e-9)


def stack(images: ImageSet, rate_arcsec_day, angle_deg, subpixel: bool = False, frames=None):
    """Shift every frame back along a trial path and average them."""
    frames = images.frames if frames is None else frames
    dt = np.asarray(images.times, dtype=float) - float(images.times[0])
    dx = rate_arcsec_day * dt / images.pixel_scale * np.cos(np.radians(angle_deg))
    dy = rate_arcsec_day * dt / images.pixel_scale * np.sin(np.radians(angle_deg))
    out = np.zeros(frames[0].shape, dtype=np.float32)
    if subpixel:
        for frame, ddx, ddy in zip(frames, dx, dy):
            out += ndimage.shift(frame, (-ddy, -ddx), order=1, mode="constant", cval=0.0)
        return out / len(frames)
    ny, nx = out.shape
    for frame, shift_y, shift_x in zip(frames, np.round(dy).astype(int), np.round(dx).astype(int)):
        sy, sx = -int(shift_y), -int(shift_x)
        ys0, ys1 = max(0, sy), min(ny, ny + sy)
        xs0, xs1 = max(0, sx), min(nx, nx + sx)
        if ys1 > ys0 and xs1 > xs0:
            out[ys0:ys1, xs0:xs1] += frame[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
    return out / len(frames)


def find_peaks(snr_map, threshold: float = 5.0, min_sep: int = 5, edge: int = 25,
               mask=None, max_peaks: int = 8):
    """Local maxima above ``threshold``, ignoring masked pixels and a border."""
    m = np.array(snr_map, dtype=float, copy=True)
    if edge:
        m[:edge] = m[-edge:] = 0
        m[:, :edge] = m[:, -edge:] = 0
    if mask is not None and mask.shape == m.shape:
        m[mask] = 0
    ys, xs = np.where(m > threshold)
    if xs.size == 0:
        return []
    peaks = []
    for k in np.argsort(-m[ys, xs])[:400]:
        x, y = int(xs[k]), int(ys[k])
        window = m[max(0, y - min_sep):y + min_sep + 1, max(0, x - min_sep):x + min_sep + 1]
        if m[y, x] < window.max():
            continue
        if all((x - px) ** 2 + (y - py) ** 2 > min_sep ** 2 for px, py, _ in peaks):
            peaks.append((float(x), float(y), float(m[y, x])))
        if len(peaks) >= max_peaks:
            break
    return peaks


def rate_angle_grid(images: ImageSet, rate_min: float = 1.0, rate_max: float = 6.0,
                    tolerance_pix: float = 2.0):
    """Trial speeds and directions, spaced so nothing smears by more than ``tolerance_pix``."""
    baseline = max(images.baseline_days, 0.2)
    d_rate = max(tolerance_pix * images.pixel_scale / baseline, 0.1)
    grid = []
    for rate in np.arange(rate_min, rate_max + 1e-9, d_rate):
        d_angle = np.degrees(min(1.0, tolerance_pix * images.pixel_scale / max(rate * baseline, 1e-3)))
        grid += [(float(rate), float(a)) for a in np.arange(0, 360, max(d_angle, 2.0))]
    return grid


def blind_search(images: ImageSet, grid=None, threshold: float = 5.0, mask=None, frames=None,
                 progress=None):
    """Search every trial speed and keep the best peak per position."""
    grid = grid or rate_angle_grid(images)
    cleaned = remove_static_sky(images.frames if frames is None else frames)
    best = {}
    for k, (rate, angle) in enumerate(grid):
        snr = matched_filter(stack(images, rate, angle, frames=cleaned), images.psf_sigma_pix)
        for x, y, s in find_peaks(snr, threshold, mask=mask):
            key = (int(x) // 6, int(y) // 6)
            if s > best.get(key, (0.0,))[0]:
                best[key] = (s, x, y, rate, angle)
        if progress and (k + 1) % progress == 0:
            print(f"  searched {k + 1}/{len(grid)} speeds", flush=True)
    rows = [dict(x=x, y=y, snr=s, rate=rate, angle=angle) for s, x, y, rate, angle in best.values()]
    return sorted(rows, key=lambda r: -r["snr"])


def inject(images: ImageSet, movers, frames=None):
    """Add fake moving objects (dicts with mag, rate, angle, x0, y0) to a copy of the frames."""
    frames = np.array(images.frames if frames is None else frames, dtype=float, copy=True)
    limits = images.limit_mags
    dt = np.asarray(images.times, dtype=float) - float(images.times[0])
    for i in range(len(frames)):
        limit = 20.5 if limits is None else float(np.asarray(limits)[i])
        for mv in movers:
            dx = mv["rate"] * dt[i] / images.pixel_scale * np.cos(np.radians(mv["angle"]))
            dy = mv["rate"] * dt[i] / images.pixel_scale * np.sin(np.radians(mv["angle"]))
            add_point_source(frames[i], mv["x0"] + dx, mv["y0"] + dy,
                             flux_for_magnitude(mv["mag"], limit), images.psf_sigma_pix)
    return frames


def recovery_curve(images: ImageSet, magnitudes, rng, per_magnitude: int = 10, mask=None,
                   threshold: float = 5.0, rate_range=(1.0, 6.0), scorer=None, margin: int = 45):
    """Hide objects of known brightness in the real frames and count how many come back.

    Without this curve a non-detection carries no information, so it is part of the search,
    not an optional extra.
    """
    ny, nx = images.frames[0].shape
    found = []
    for mag in np.asarray(magnitudes, dtype=float):
        hits = 0
        for _ in range(per_magnitude):
            for attempt in range(500):
                x0, y0 = rng.uniform(margin, nx - margin), rng.uniform(margin, ny - margin)
                if mask is None or not mask[int(y0), int(x0)]:
                    break
            else:
                raise ValueError("the mask leaves nowhere to inject: this patch is unusable "
                                 f"({float(mask.mean()):.0%} masked)")
            mv = dict(mag=float(mag), rate=float(rng.uniform(*rate_range)),
                      angle=float(rng.uniform(0, 360)), x0=x0, y0=y0)
            cleaned = remove_static_sky(inject(images, [mv]))
            stacked = stack(images, mv["rate"], mv["angle"], frames=cleaned)
            peaks = find_peaks(matched_filter(stacked, images.psf_sigma_pix), threshold, mask=mask)
            near = any(np.hypot(px - x0, py - y0) < 5 for px, py, _ in peaks)
            if near and (scorer is None or scorer(stacked, x0, y0) > 0.5):
                hits += 1
        found.append(hits / per_magnitude)
    return np.asarray(magnitudes, dtype=float), np.asarray(found, dtype=float)


def depth_50(magnitudes, efficiency):
    """Faintest magnitude at which half the hidden objects are still recovered."""
    magnitudes, efficiency = np.asarray(magnitudes, float), np.asarray(efficiency, float)
    for k in range(len(magnitudes) - 1):
        if efficiency[k] >= 0.5 > efficiency[k + 1]:
            frac = (efficiency[k] - 0.5) / max(efficiency[k] - efficiency[k + 1], 1e-9)
            return float(magnitudes[k] + frac * (magnitudes[k + 1] - magnitudes[k]))
    return float(magnitudes[0] if efficiency[0] < 0.5 else magnitudes[-1])


def split_half_test(images: ImageSet, candidate, mask=None):
    """Re-measure a candidate in each half of the data.

    A real moving object keeps roughly 1/sqrt(2) of its signal in each half; an artefact
    confined to a few frames does not. This is the cheapest way to kill a false candidate.
    """
    half = len(images.frames) // 2
    if half < 2:
        raise ValueError("need at least 4 frames to split")
    # The static sky comes from every frame, not from each half: with few frames a slow
    # object sits on nearly the same pixels and would subtract part of itself away.
    cleaned = remove_static_sky(images.frames)

    def measure(sl):
        subset = ImageSet(np.asarray(images.frames)[sl], np.asarray(images.times)[sl],
                          None if images.limit_mags is None else np.asarray(images.limit_mags)[sl],
                          images.pixel_scale, images.seeing_arcsec)
        stacked = stack(subset, candidate["rate"], candidate["angle"], frames=cleaned[sl])
        snr = matched_filter(stacked, subset.psf_sigma_pix)
        y, x = int(candidate["y"]), int(candidate["x"])
        return float(snr[max(0, y - 2):y + 3, max(0, x - 2):x + 3].max())

    full = measure(slice(None))
    return dict(full=full, first_half=measure(slice(0, half)), second_half=measure(slice(half, None)),
                expected_per_half=full / np.sqrt(2))


def simulate_frames(n_frames: int = 18, n_nights: int = 3, movers=(), n_stars: int = 120,
                    npix: int = 256, seed: int = 1, n_cosmics: float = 2.0,
                    limit_mag: float = 20.5, pixel_scale: float = ARCSEC_PER_PIX_ZTF,
                    seeing_arcsec: float = 2.0, night_length_days: float = 0.08):
    """Make a realistic test field: fixed stars, noise, cosmic rays and optional moving objects.

    Used for the package's own tests, for training the classifier on pipeline-like stamps,
    and as a stand-in when a real download is unavailable. Returns ``(ImageSet, cosmic_rays)``.
    """
    rng = np.random.default_rng(seed)
    sigma = seeing_arcsec / 2.355 / pixel_scale
    per_night = max(1, n_frames // n_nights)
    times = np.concatenate([night + np.linspace(0, night_length_days, per_night)
                            for night in range(n_nights)])[:n_frames]
    star_x, star_y = rng.uniform(0, npix, n_stars), rng.uniform(0, npix, n_stars)
    faint_end, bright_end, slope = 21.0, 17.5, 0.3      # realistic: a few bright stars, many faint
    lo, hi = 10 ** (slope * bright_end), 10 ** (slope * faint_end)
    star_mag = np.log10(lo + rng.random(n_stars) * (hi - lo)) / slope
    frames, cosmics = [], []
    for index, t in enumerate(times):
        image = rng.normal(0.0, 1.0, (npix, npix))
        for x, y, mag in zip(star_x, star_y, star_mag):
            add_point_source(image, x, y, flux_for_magnitude(mag, limit_mag), sigma)
        for _ in range(rng.poisson(n_cosmics)):
            cx, cy = int(rng.integers(20, npix - 20)), int(rng.integers(20, npix - 20))
            image[cy, cx] += rng.uniform(8, 25)
            cosmics.append((index, float(cx), float(cy)))
        for mv in movers:
            dt = t - times[0]
            dx = mv["rate"] * dt / pixel_scale * np.cos(np.radians(mv["angle"]))
            dy = mv["rate"] * dt / pixel_scale * np.sin(np.radians(mv["angle"]))
            add_point_source(image, mv["x0"] + dx, mv["y0"] + dy,
                             flux_for_magnitude(mv["mag"], limit_mag), sigma)
        frames.append(image.astype(np.float32))
    images = ImageSet(np.asarray(frames), times, np.full(len(times), limit_mag),
                      pixel_scale, seeing_arcsec)
    return images, cosmics
