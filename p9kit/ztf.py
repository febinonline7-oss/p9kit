"""Fetch public ZTF image cutouts from IRSA.

Three details decide whether a download is usable, and all three were learned the hard way:
only ``ipac_gid == 1`` images are public (the rest return 404), ``infobits == 0`` marks clean
frames, and the frames must be **spread over several days** — within one night a distant
object moves less than a pixel and is removed along with the static sky.
"""
from __future__ import annotations

import io

import numpy as np

from .images import ARCSEC_PER_PIX_ZTF, ImageSet

IBE_SEARCH = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci"
IBE_DATA = "https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci"
REQUIRED_COLUMNS = {"filtercode", "obsjd", "filefracday", "field", "ccdid", "qid", "imgtypecode"}


def query_metadata(ra_deg, dec_deg, timeout: int = 120):
    """Table of every ZTF science image covering a position (astroquery, then a plain request)."""
    import pandas as pd
    import requests

    errors = []
    try:
        from astroquery.ipac.irsa.ibe import Ibe
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
        table = Ibe.query_region(coordinate=coord, mission="ztf", dataset="products", table="sci")
        frame = table.to_pandas()
    except Exception as exc:  # pragma: no cover - network path
        errors.append(repr(exc))
        response = requests.get(IBE_SEARCH, timeout=timeout,
                                params={"POS": f"{ra_deg},{dec_deg}", "INTERSECT": "OVERLAPS", "ct": "csv"})
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"IRSA returned unexpected columns, missing {sorted(missing)}; {errors}")
    frame["obsjd"] = frame["obsjd"].astype(float)
    return frame


def choose_window(metadata, filters=("zr", "zg"), window_days: float = 10.0, max_images: int = 24,
                  min_baseline_days: float = 2.0, min_nights: int = 3, min_limit_mag: float = 19.3,
                  max_seeing: float = 3.3):
    """Pick public, clean images that share one observing window of a few days."""
    frame = metadata[metadata["filtercode"].isin(list(filters))]
    for column, keep in (("ipac_gid", lambda f: f["ipac_gid"] == 1),
                         ("infobits", lambda f: f["infobits"] == 0),
                         ("maglimit", lambda f: f["maglimit"] > min_limit_mag),
                         ("seeing", lambda f: f["seeing"] < max_seeing)):
        if column in frame:
            frame = frame[keep(frame)]
    if len(frame) == 0:
        raise RuntimeError("no public, clean images in these filters")
    frame = frame.sort_values("obsjd").reset_index(drop=True)
    jd = frame["obsjd"].values
    best, best_count = None, 0
    for window in (window_days, 2 * window_days, 4 * window_days):
        for start in jd:
            selected = (jd >= start) & (jd <= start + window)
            nights = len(set(np.floor(jd[selected] - 0.5)))
            if (np.ptp(jd[selected]) >= min_baseline_days and nights >= min_nights
                    and selected.sum() > best_count):
                best, best_count = selected, int(selected.sum())
        if best_count >= 8:
            break
    if best is None:
        raise RuntimeError("could not find images spread over at least "
                           f"{min_baseline_days} days at this position")
    return frame[best].head(max_images)


def cutout_url(row, ra_deg, dec_deg, size_arcmin: float):
    """Build the IRSA cutout URL for one metadata row."""
    ff = str(int(row["filefracday"]))
    return (f"{IBE_DATA}/{ff[:4]}/{ff[4:8]}/{ff[8:]}/"
            f"ztf_{ff}_{int(row['field']):06d}_{row['filtercode']}_c{int(row['ccdid']):02d}_"
            f"{row['imgtypecode']}_q{int(row['qid'])}_sciimg.fits"
            f"?center={ra_deg},{dec_deg}&size={size_arcmin}arcmin&gzip=false")


def download_cutouts(selection, ra_deg, dec_deg, size_arcmin: float = 6.0, timeout: int = 120,
                     verbose: bool = True):
    """Download the selected cutouts. Returns ``(arrays, headers, julian_dates, limit_mags)``."""
    import requests
    from astropy.io import fits

    arrays, headers, jds, limits = [], [], [], []
    for _, row in selection.iterrows():
        response = requests.get(cutout_url(row, ra_deg, dec_deg, size_arcmin), timeout=timeout)
        if response.status_code != 200:
            if verbose:
                print(f"  skipped one image (HTTP {response.status_code})")
            continue
        with fits.open(io.BytesIO(response.content)) as hdul:
            data, header = hdul[0].data, hdul[0].header
        if data is None or data.size == 0:
            continue
        arrays.append(np.asarray(data, dtype=float))
        headers.append(header)
        jds.append(float(row["obsjd"]))
        limits.append(float(header.get("MAGLIM", row.get("maglimit", 20.5))))
    if len(arrays) < 6:
        raise RuntimeError(f"only {len(arrays)} usable images were downloaded")
    return arrays, headers, np.asarray(jds), np.asarray(limits)


def align_and_normalise(arrays, headers, ra_deg, dec_deg, background_box: int = 32):
    """Centre every cutout on the target, trim to a common size, and scale to its own noise.

    Returns ``(frames, kept_indices)``; frames cut short at a chip edge are dropped.
    """
    from astropy.wcs import WCS
    from scipy import ndimage

    sizes = np.array([min(a.shape) for a in arrays])
    size = int(np.median(sizes))
    size -= size % 2
    kept = [k for k, a in enumerate(arrays) if min(a.shape) >= size]
    frames = []
    for k in kept:
        image = np.nan_to_num(arrays[k], nan=float(np.nanmedian(arrays[k])))
        try:
            x, y = WCS(headers[k]).all_world2pix(ra_deg, dec_deg, 0)
        except Exception:  # pragma: no cover - only with a broken header
            x, y = image.shape[1] / 2, image.shape[0] / 2
        centred = ndimage.shift(image, (image.shape[0] / 2 - y, image.shape[1] / 2 - x),
                                order=1, mode="nearest")
        cy, cx = centred.shape[0] // 2, centred.shape[1] // 2
        cut = centred[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
        residual = cut - ndimage.median_filter(cut, background_box)
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        frames.append((residual / mad).astype(np.float32))
    return np.asarray(frames), kept


def fetch_patch(ra_deg, dec_deg, size_arcmin: float = 6.0, filters=("zr", "zg"),
                window_days: float = 10.0, max_images: int = 24, verbose: bool = True):
    """One call from sky position to a ready-to-search :class:`~p9kit.images.ImageSet`."""
    metadata = query_metadata(ra_deg, dec_deg)
    selection = choose_window(metadata, filters, window_days, max_images)
    if verbose:
        print(f"  {len(metadata)} images cover this spot; using {len(selection)} "
              f"over {np.ptp(selection['obsjd']):.1f} days")
    arrays, headers, jds, limits = download_cutouts(selection, ra_deg, dec_deg, size_arcmin,
                                                    verbose=verbose)
    frames, kept = align_and_normalise(arrays, headers, ra_deg, dec_deg)
    return ImageSet(frames, jds[kept] - jds[kept].min(), limits[kept], ARCSEC_PER_PIX_ZTF,
                    float(np.median([h.get("SEEING", 2.0) for h in headers])))
