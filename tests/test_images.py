import numpy as np
import pytest

from p9kit import images as im


@pytest.fixture(scope="module")
def blank_field():
    return im.simulate_frames(n_frames=12, n_nights=3, seed=7)[0]


@pytest.fixture(scope="module")
def field_with_object(blank_field):
    free = np.argwhere(~im.build_mask(blank_field))            # put it where no star is in the way
    free = free[(free[:, 0] > 60) & (free[:, 0] < 190) & (free[:, 1] > 60) & (free[:, 1] < 190)]
    y0, x0 = free[len(free) // 2]
    mover = dict(mag=20.5, rate=3.2, angle=250.0, x0=float(x0), y0=float(y0))
    images, _ = im.simulate_frames(n_frames=12, n_nights=3, movers=[mover], seed=7)
    return images, mover


def test_image_set_rejects_mismatched_times():
    with pytest.raises(ValueError):
        im.ImageSet(np.zeros((3, 8, 8)), np.zeros(2))


def test_removing_the_static_sky_kills_the_stars(blank_field):
    cleaned = im.remove_static_sky(blank_field.frames)
    assert np.abs(cleaned).max() < 0.5 * np.abs(blank_field.frames).max()


def test_stacking_at_the_right_speed_finds_the_object(field_with_object):
    images, mover = field_with_object
    cleaned = im.remove_static_sky(images.frames)
    right = im.matched_filter(im.stack(images, mover["rate"], mover["angle"], frames=cleaned),
                              images.psf_sigma_pix)
    wrong = im.matched_filter(im.stack(images, 1.0, 30.0, frames=cleaned), images.psf_sigma_pix)
    y, x = int(mover["y0"]), int(mover["x0"])
    assert right[y, x] > 5.0
    assert wrong[y, x] < right[y, x] / 2


def test_integer_and_subpixel_stacking_agree(field_with_object):
    images, mover = field_with_object
    cleaned = im.remove_static_sky(images.frames)
    a = im.stack(images, mover["rate"], mover["angle"], frames=cleaned)
    b = im.stack(images, mover["rate"], mover["angle"], frames=cleaned, subpixel=True)
    y, x = int(mover["y0"]), int(mover["x0"])
    assert b[y - 3:y + 4, x - 3:x + 4].max() == pytest.approx(a[y - 3:y + 4, x - 3:x + 4].max(), rel=0.4)


def test_blind_search_recovers_an_injected_object(field_with_object):
    images, mover = field_with_object
    candidates = im.blind_search(images, mask=im.build_mask(images))
    assert any(np.hypot(c["x"] - mover["x0"], c["y"] - mover["y0"]) < 5 for c in candidates)


def test_find_peaks_respects_the_mask_and_the_border():
    snr = np.zeros((80, 80))
    snr[40, 40] = 20.0
    snr[5, 5] = 50.0                       # inside the excluded border
    assert [(x, y) for x, y, _ in im.find_peaks(snr)] == [(40.0, 40.0)]
    mask = np.zeros_like(snr, dtype=bool)
    mask[38:43, 38:43] = True
    assert im.find_peaks(snr, mask=mask) == []


def test_mask_covers_bright_stars_but_not_the_whole_field(blank_field):
    mask = im.build_mask(blank_field)
    assert 0.0 < mask.mean() < 0.6


def test_grid_step_shrinks_when_the_baseline_grows():
    short = im.ImageSet(np.zeros((4, 32, 32)), np.array([0.0, 0.5, 1.0, 1.5]))
    long = im.ImageSet(np.zeros((4, 32, 32)), np.array([0.0, 4.0, 8.0, 12.0]))
    assert len(im.rate_angle_grid(long)) > len(im.rate_angle_grid(short))


def test_recovery_curve_falls_with_magnitude_and_gives_a_sensible_depth(blank_field):
    rng = np.random.default_rng(11)
    mags, eff = im.recovery_curve(blank_field, [20.0, 21.0, 23.0], rng, per_magnitude=4,
                                  mask=im.build_mask(blank_field))
    assert eff[0] >= eff[-1]
    assert eff[0] > 0.5 and eff[-1] == 0.0
    assert 20.0 <= im.depth_50(mags, eff) <= 23.0


def test_flux_scale_matches_the_limiting_magnitude():
    assert im.flux_for_magnitude(20.5, 20.5) == pytest.approx(5.0)
    assert im.flux_for_magnitude(25.5, 20.5) == pytest.approx(0.05)


def test_split_half_keeps_signal_for_a_real_object(field_with_object):
    images, mover = field_with_object
    result = im.split_half_test(images, dict(x=mover["x0"], y=mover["y0"], rate=mover["rate"],
                                             angle=mover["angle"]))
    # Both halves keep real signal. They lose more than the ideal 1/sqrt(2) because each
    # half has a shorter baseline, so the object moves less and blends into the static sky.
    halves = (result["first_half"], result["second_half"])
    assert min(halves) > 3.0
    assert min(halves) / max(halves) > 0.2


def test_split_half_rejects_a_signal_present_in_only_one_half():
    mover = dict(mag=20.0, rate=3.0, angle=45.0, x0=120.0, y0=120.0)
    images, _ = im.simulate_frames(n_frames=12, n_nights=3, movers=[mover], seed=5)
    frames = np.array(images.frames)
    frames[6:] = im.simulate_frames(n_frames=6, n_nights=3, seed=9)[0].frames[:6]  # object only early
    spliced = im.ImageSet(frames, images.times, images.limit_mags)
    result = im.split_half_test(spliced, dict(x=mover["x0"], y=mover["y0"], rate=mover["rate"],
                                              angle=mover["angle"]))
    halves = (result["first_half"], result["second_half"])
    assert max(halves) > 4.0
    assert min(halves) / max(halves) < 0.1


def test_patch_quality_flags_a_crowded_field():
    clean = np.zeros((100, 100), dtype=bool)
    clean[:10] = True
    crowded = np.ones((100, 100), dtype=bool)
    assert im.patch_quality(clean)["usable"] is True
    assert im.patch_quality(crowded)["usable"] is False
    assert im.patch_quality(crowded)["masked_fraction"] == pytest.approx(1.0)


def test_recovery_curve_refuses_an_unusable_patch(blank_field):
    with pytest.raises(ValueError, match="unusable"):
        im.recovery_curve(blank_field, [21.0], np.random.default_rng(0), per_magnitude=1,
                          mask=np.ones(blank_field.frames[0].shape, dtype=bool))
