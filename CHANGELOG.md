# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.1] - 2026-10-05

### Changed
* Flattened the repository layout so the package and tests sit in two folders.
* Archived on Zenodo; releases now have a DOI.

First public release, extracted from a working search of two ZTF fields.

### Added
* Forward model of distant-object orbits under a hypothetical planet (`p9kit.population`).
* Clustering statistics and a selection-aware significance test (`p9kit.stats`).
* Neural posterior estimation with a coverage check (`p9kit.inference`).
* Sky, brightness and motion predictions from a posterior (`p9kit.sky`).
* Shift-and-stack search, brightness-scaled star masking, recovery curves, split-half
  verification and patch-quality reporting (`p9kit.images`).
* CNN artefact rejection trained on pipeline-generated stamps (`p9kit.classifier`).
* Public ZTF cutout retrieval from IRSA (`p9kit.ztf`).
* Command line interface: `p9kit demo`, `p9kit clustering`, `p9kit search`.
