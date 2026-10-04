# Limitations

Read this before quoting any number from p9kit.

## The orbit model is phenomenological, not N-body

`p9kit.population` does not integrate orbits. It reproduces the *qualitative* effects reported
for a distant perturber — perihelia anti-aligned with the planet, orbit poles clustered near the
planet's pole, both stronger for a more massive or closer planet — with a forcing strength that
scales as mass / (semi-minor axis)^3. That makes a realisation cost milliseconds instead of CPU
days, which is what allows tens of thousands of simulations on a laptop.

What follows from this:

* **Masses and distances are exploratory.** They inherit the shape of the hand-written rule.
* **The method, not the number, is the contribution.** The same inference code will accept an
  N-body simulator; `examples/` shows the REBOUND setup and how long a 4 Gyr run takes.

## The survey model is deliberately simple

`SurveyModel` applies a limiting magnitude that is shallower near the galactic plane and away
from the ecliptic. Real surveys have pointing histories, nightly depths and tracking efficiencies.
Selection effects can imitate clustering, so any significance from this model is indicative only.
A published survey simulator should be used before claiming a detection significance.

## Small samples

The distant-object samples involved are a few dozen objects. Posteriors are therefore wide, and
conclusions can change when the sample cut changes (for example perihelion > 30 AU versus > 42 AU).
`p9kit clustering --q-min` exists so that this dependence is easy to check and report.

## What the imaging side does *not* assume

The image search measures its own performance from your data: the recovery curve is measured by
injecting sources into the real frames, the junk level comes from re-searching the same frames
with their brightness flipped, and every survivor is re-measured in each half of the data.
Those numbers stand on their own.

Known imaging caveats:

* Frames must span several days. Within one night a distant object moves less than a pixel and is
  removed along with the static sky.
* Crowded fields are not searchable this way. Undersampled bright stars leave scars of hundreds of
  sigma; `patch_quality` reports how much had to be masked, and above roughly 40% a different field
  is the better choice.
* Candidate significance is not calibrated to a formal false-alarm probability. Use the junk
  comparison and the split-half test instead.
