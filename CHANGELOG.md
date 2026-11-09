# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org).

## [0.1.0] - unreleased

First version.

- Readers for CSV / TXT / TSV / xlsx with header, unit-row and metadata detection; units
  from the header or assumed with `UNIT_ASSUMED`; stylus force, scan length, speed,
  stylus radius and date kept; values as read preserved; processing history.
- Spike removal; manual levelling with two reference regions (one or separate levels);
  automatic robust levelling (Huber, RANSAC) with the steps masked; optional polynomial
  bow as a separate step.
- Edge detection and classification (edge, groove, ridge, staircase, periodic, multi);
  plane-fit step heights at the half-height point; ISO 5436-1 type A1 evaluation of
  grooves and ridges; burr exclusion; uncertainty from the residuals (integrated
  autocorrelation time) and the plane slope difference; manual edges and regions.
- ISO 21920-2 roughness parameters with the ISO 16610-21 Gaussian filter; setting class
  from the measured Ra; sampling and evaluation-length warnings.
- Stoney film stress with a substrate table and assumption checks.
- Batch per sample, CSV / xlsx / PNG / SVG / PDF / JSON output, `verify`.
- Desktop application (PySide6 + PyQtGraph) with draggable cursors and edge lines.
- Command-line output keeps working on narrow code pages: a piped or redirected stream on
  Windows (cp1252) is switched to UTF-8, so reports with λ no longer fail.
- Origin-style research outputs: figures as multi-layer graphs with panel letters,
  dimension lines (h, d, W) and a residual layer of the plane fits; roughness figure with
  parameter table, amplitude distribution and material ratio (Abbott-Firestone) curve;
  stress figure with fit residuals; batch column chart with SD / SE / 95 % CI error bars;
  journal size presets (single, 1.5, double column, slide), dpi, font size, display units,
  TIFF output and editable text in SVG / PDF (`--figure-size`, `--dpi`, `--font-size`,
  `--x-display`, `--z-display`, `--no-annotate`, `--no-residuals`, `--error-bar`).
- Origin-ready worksheets (`--formats origin`): CSV and xlsx with Long Name, Units and
  Comments rows.
- Batch statistics: standard error, coefficient of variation, median and the 95 %
  confidence interval of the mean (Student t) per sample.
- Desktop: Plot Details dialog (units, axis limits, fonts, line width, layers), residual
  layer and dimension lines on the live graph, data reader snapping to data points,
  Shift-click two-point Δx/Δz, overlay of scans with optional offset, worksheet tab with
  copy as tab-separated text, batch column chart, Export Figure dialog, copy graph.
- Synthetic test data generator with stylus tracing; 115 tests.
- Not validated against real instrument scans yet.
