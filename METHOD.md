# Method

## Orbits: from a clustering pattern to a probability map

1. **Forward model** (`population.simulate_detected`). Draw a distant-object population, apply a
   candidate planet's confining effect, place each object at a random point in its orbit weighted
   by time, and keep the ones a survey would have detected.
2. **Summary statistics** (`stats.summary_statistics`). Reduce a sample to eleven order-independent
   numbers: mean perihelion and node direction as vector components (so the 0/360 degree wrap
   cannot bite), the mean orbit pole and its spread, the same perihelion statistics for the most
   distant half, and the median inclination.
3. **Inference** (`inference.train_posterior`). Train a neural posterior on simulations drawn from
   the prior. A spline flow (`nsf`) is the default: in coverage tests a masked autoregressive flow
   was systematically overconfident on samples of this size.
4. **Check before use** (`inference.coverage_check`). Simulate many data sets, and count how often
   the truth lands inside its own 68% interval. If that fraction is far below 0.68, the posterior
   is overconfident and should not be quoted.
5. **Predict** (`sky.predict_sky`). Turn posterior samples into positions, magnitudes and sky
   motion rates. The planet's position along its orbit is unknown, so the mean anomaly is drawn
   uniformly in time, which naturally favours the distant, slow part of the orbit.

## Images: from frames to a measured limit

1. **Prepare** (`ztf.fetch_patch`). Download public, clean cutouts that share one observing window
   of several days, centre them on the target and scale each to its own noise.
2. **Remove the static sky** (`remove_static_sky`). The per-pixel median over frames is the static
   sky; what is left moves, or is junk.
3. **Mask** (`build_mask`). Ignore bright stars with a radius that grows with brightness, and
   pixels whose scatter marks a bad subtraction. `patch_quality` reports whether enough of the
   field survives to be worth searching.
4. **Search** (`blind_search`). For every trial speed and direction, slide the frames back along
   that path, stack, matched-filter, and keep the best peak per position. Step sizes come from the
   requirement that nothing smears by more than a couple of pixels over the whole run.
5. **Reject artefacts** (`classifier.RealBogusClassifier`). A small CNN trained on stamps made by
   this same pipeline — stamps made from idealised white noise do not work, because the stacking
   correlates the noise.
6. **Measure the depth** (`recovery_curve`, `depth_50`). Inject objects of known brightness into the
   real frames and count how many come back. Without this, "we found nothing" says nothing.
7. **Compare with junk**. Re-run the identical search on the brightness-flipped frames; every real
   object becomes a dip, so whatever is found is junk with the same noise.
8. **Verify survivors** (`split_half_test`). A real moving object keeps signal in both halves of
   the data. Artefacts confined to a few frames do not, which kills most false candidates in
   seconds.
9. **Feed back** (`sky.apply_nondetection`). Down-weight every possible planet the search should
   have seen and did not, and report what fraction of the posterior that removes.
