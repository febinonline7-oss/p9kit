# Notebooks

Two notebooks that run start to finish in Google Colab, no installation needed beyond the first cell.

| Notebook | What it does | Runtime |
|---|---|---|
| `stage_a_orbits_colab.ipynb` | Downloads distant objects from the JPL Small-Body Database, measures clustering, trains a neural posterior, checks its coverage, and produces a sky probability map | 20-30 min |
| `stage_b_images_colab.ipynb` | Downloads public ZTF cutouts of a chosen patch, searches them by shift-and-stack, trains the artefact CNN, measures the depth and verifies candidates | 15-25 min |

Both notebooks pre-date the package and call the same algorithms inline; porting them to
import `p9kit` directly is tracked in the issue list and is a good first contribution.
