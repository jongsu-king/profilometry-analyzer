# Examples

* `validate_synthetic.py` – the synthetic and analytical validation table of the README.
* `make_screenshots.py` – regenerates `docs/images/*.png` (needs the GUI extra; run with
  `QT_QPA_PLATFORM=offscreen`).

Synthetic scans with known values: `profilometry-analyzer synth --out demo` writes nine
film-edge scans of three samples, a scratch with burrs, a sine for roughness, a wafer pair
for stress and `truth.json`.
