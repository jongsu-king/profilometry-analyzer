# Methods

All lengths are stored as x in µm and z in nm. The profile as read (`raw_x`, `raw_z`,
file units) is never modified; every processing step returns a new profile with one more
`history` entry, and the project file stores the complete history.

## 1. Reading

Rows of cells (text lines or worksheet rows) are scanned for numeric blocks of at least
10 rows and two columns. The text row above a block is its header; if it holds only
units it is the unit row and the header is the row above. All other text rows are
metadata (`key: value`, `key, value`, `key<TAB>value`). Scan parameters are recognised by
generic words only (`force`, `radius`, `speed`, `length`, `points`, `date`) and converted
to mg, µm and µm/s. No instrument-specific column name is assumed.

Column choice: the first lateral-looking column (`x`, `lateral`, `distance`, `position`,
`scan length`…) is x; among height-looking columns (`z`, `height`, `profile`, `raw`…) a raw
one wins over processed ones (`level`, `normalized`, `filtered`, `roughness`, …).

Units: brackets or a trailing `/ unit` / `in unit` in the header; a unit row; otherwise
the lateral unit is the one that makes the scan length match the length in the metadata,
or fall inside the instrument range (50 µm–55 mm); the height unit is assumed to be Å.
Assumed units raise `UNIT_ASSUMED` (warning) in every output.

## 2. Spikes

A spike is a run of at most three points deviating from a 7-point running median by more
than 8 robust sigmas. Step edges follow the running median and are never flagged. Spike
points are excluded from all fits and interpolated in the stored heights.

## 3. Edges and plateaus

The height is smoothed with a Gaussian of σ = one stylus radius (at least three points)
and differentiated. Runs where `|dz/dx − median| > k σ_d` (k = 6, σ_d from the MAD) are
grown while the derivative stays above 2 σ_d with the same sign; runs of the same sign
closer than two smoothing widths are merged; runs of opposite sign are never merged. A
run is an edge when the smoothed height changes across it by more than five times the
height noise (and `--min-step`).

Plateaus lie between edges, shortened by a margin of two stylus radii next to each edge.
A plateau shorter than `max(3 r, 10 dx)` is dissolved: its two edges become one
transition, or, if the net height change is below the threshold (a burr top, dust, a
narrow line), a *narrow feature* that is excluded from all fits and reported. Edges whose
neighbouring plateau medians differ by less than the threshold are removed.

Finally each edge centre is moved to the **half-height point**, where the profile crosses
the middle between the two fitted plateau lines (the derivative centroid is pulled
sideways by burrs and asymmetric walls).

Classification: one edge = `edge`; down–up = `groove`; up–down = `ridge`; all in one
direction = `staircase`; at least four alternating edges with same-direction spacings of
coefficient of variation < 0.25 = `periodic`; otherwise `multi`.

## 4. Levelling and bow

*Automatic*: edges are detected on the profile as read; a polynomial trend shared by all
plateaus, with one offset per plateau, is fitted robustly — Huber weights (k = 1.345,
scale from the MAD, iteratively reweighted) or RANSAC (two-point slopes within one
plateau, consensus within 3·√2 noise) — so a step never tilts the line. The lowest
plateau is set to zero.

*Manual*: a line through the points of the reference regions. With one level (default)
the regions share one line; the program checks this with a shared-slope / separate-offset
fit and warns (`LEVELING_REGIONS_DIFFER`) if the offsets differ by more than three
residual sigmas. With separate levels only the slope is shared.

*Bow* (`--bow 2…6`): the shared trend is a polynomial; the peak-to-valley of the part
beyond the straight line is reported (`BOW_REMOVED`). It is one levelling step in the
history, so a height computed on a bow-corrected profile is always labelled.

## 5. Step height

### Plane fits

Each plane is a least-squares line through the usable points. Points beyond `k` robust
sigmas (k = 4) are excluded iteratively; raised runs are grown until the residual falls
below one sigma so a whole burr is excluded, and runs of at least three points are
reported as burrs (`BURR_EXCLUDED`). Raised rims inside the excluded zones next to an edge
or a groove (A1 `W/3` zones) are also reported.

### Film edge, staircase, periodic

The height of an edge is `right(x_e) − left(x_e)` at the half-height point `x_e`.
Periodic lines: mean |height| over all edges, sample standard deviation, n and the mean
period.

### Groove and ridge: ISO 5436-1 type A1

With W the distance between the half-height points and c the centre:

* upper level: **one** line fitted over `c − 1.5 W … c − W/2 − W/3` and
  `c + W/2 + W/3 … c + 1.5 W` (the groove and W/3 next to each edge are ignored);
* lower level: a line over the central third, `c ± W/6`;
* depth: upper line minus lower line at c.

The regions are additionally kept clear of the detected transitions plus the margin
(`A1_REGION_TRUNCATED`, info, when this shortens them; warning when the scan does not
reach c ± 1.5 W). A ridge (stripe of film) is evaluated with the levels exchanged.
Source for the wording: the ISO 5436 text as quoted by Song & Vorburger, *J. Res. NIST*
**96**, 271 (1991); the ISO 5436-1:2000 §6.1.1 note ("wide ridges with flat tops are
equivalent") was read in the public preview. Clause 7 of ISO 5436-1:2000 itself was not
available.

Stylus checks: a spherical tip of radius r reaches the floor of a rectangular groove of
width W and depth d only if `(W/2)² ≥ 2 r d − d²` (d < r) or `W ≥ 2 r` (d ≥ r); otherwise
`NARROW_FEATURE` says depth and width are underestimated. Grooves narrower than 3 r are
flagged as shaped by the stylus. Real styli are cones with a rounded tip, so steep walls
are traced worse than this estimate.

### Uncertainty (single scan, k = 1)

* `u_fit = √(var_left(x_e) + var_right(x_e))` from the line covariances. Each covariance
  is the ordinary one multiplied by `n / n_eff`, where `n_eff = n / τ` and `τ` is the
  integrated autocorrelation time of the residuals (Sokal's window, c = 5).
* `u_slope = |m_right − m_left| · gap / 2`: how much the height changes across the
  excluded gap between the two fitted regions when the planes are not parallel.
* `u = √(u_fit² + u_slope²)`.

Instrument calibration and the scatter between positions on the sample are not in `u`;
the batch table gives the mean, standard deviation and n per sample. On synthetic data u
is well calibrated for white noise and too small by a factor of about 1.2–2.3 for
roughness correlated over a few percent of the plane length (`CORRELATED_RESIDUALS`).

### Range checks

`SMALL_STEP` below twice the 20 nm lower end of the vertical range (warning below 20 nm
too), `STEP_BELOW_NOISE` below 3 u or 5× the noise, `SOFT_FILM_HIGH_FORCE` for a film
marked soft scanned with more than 3 mg (polymers: start at 1–3 mg),
`SOFT_FILM_FORCE_UNKNOWN` when the force is not known.

## 6. Roughness

1. Uniform grid (linear interpolation if the spacing varies by more than 1 %), spikes
   interpolated.
2. Form removal: least-squares polynomial (default order 1).
3. λs: Gaussian low-pass (skipped when λs < 2 dx).
4. λc: the Gaussian mean line is the waviness; roughness = primary − mean line.
5. λc/2 discarded at each end; evaluation length = the largest whole number of sections
   (lsc = λc), centred.

Gaussian weighting function (ISO 16610-21): `s(x) = exp(−π (x/(α λc))²) / (α λc)`,
`α = √(ln 2/π)`, truncated at ±λc and normalised; the mean line transmits 50 % at λc.

Parameters (ISO 21920-2): Ra, Rq, Rsk, Rku, Rt over the evaluation length; Rp, Rv, Rz as
means of the section values. Rsm: mean width of profile elements; runs of one sign whose
extreme is below 10 % of Rz or whose width is below 1 % of the section length are merged
(ISO 4287 convention; the ISO 21920-2 discrimination was not checked).

Default cut-offs: setting class from the measured Ra, iterated until stable:

| class | λs (µm) | λc (mm) | largest spacing (µm) | Ra up to (µm) |
|---|---|---|---|---|
| Sc1 | 2.5 | 0.08 | 0.5 | 0.02 |
| Sc2 | 2.5 | 0.25 | 0.5 | 0.1 |
| Sc3 | 2.5 | 0.8 | 0.5 | 2 |
| Sc4 | 8 | 2.5 | 1.5 | 10 |
| Sc5 | 25 | 8 | 5 | – |

(ISO 21920-3 classes as summarised by Polytec, 2023; original text not checked.)
Warnings: `SAMPLING_TOO_COARSE`, `EVALUATION_TOO_SHORT` (fewer than the nominal five
sections; error when less than one), `FILTER_LONGER_THAN_DATA`, `STEP_IN_ROUGHNESS`.

## 7. Film stress (Stoney)

Curvature of each scan from `z = a (x − x0)² + b (x − x0) + c`:
`κ = 2a / (1 + b²)^{3/2}`, with the covariance of `a` inflated like the plane fits.

`σ_f = M_s · h_s² / (6 h_f) · (κ_after − κ_before)`, `M_s = E_s / (1 − ν_s)`.

Sign: scans on the film side; κ > 0 for a concave surface; σ > 0 = tensile. Back-side
scans: `--backside`. (The sign convention of the instrument software was not checked.)

Substrates (editable, sources in `core/stress.py` and `profilometry-analyzer substrates`):

| key | M (GPa) | source |
|---|---|---|
| si100 | 180.3 | c11 + c12 − 2c12²/c11 with c11 165.7, c12 63.9 GPa (Hopcroft, Nix & Kenny, J. MEMS 19, 229 (2010)) |
| si111 | 229.1 | 6c44(c11 + 2c12)/(c11 + 2c12 + 4c44), c44 79.6 GPa (same) |
| eagle-xg | 95.6 | Corning EAGLE XG: E 73.6 GPa, ν 0.23 |
| borofloat33 | 80.0 | SCHOTT BOROFLOAT 33: E 64 GPa, ν 0.2 |
| fused-silica | 86.5 | Corning 7980: E 72.7 GPa, ν 0.16 |
| soda-lime | 87.5 | typical 70 GPa, 0.2 — **unverified** |

Checks: `STONEY_THICK_FILM` (h_f/h_s > 1 %, error above 5 %), `STONEY_LARGE_DEFLECTION`
(sag over the scan > h_s/10), `BOW_NOT_SPHERICAL` (a quartic fits more than twice better),
`CURVATURE_CHANGE_NOT_SIGNIFICANT` (|Δκ| < 2 u), `SCAN_MISMATCH` (length or spacing
differ), `STRESS_NO_REFERENCE` (no scan before deposition: flat assumed),
`SUBSTRATE_THICKNESS_TYPICAL`, `SUBSTRATE_VALUE_UNVERIFIED`.

## 8. Figures, statistics and worksheets

**Dimension lines.** For a film edge the height line is drawn beside the edge, between
the two plane fits evaluated at the half-height position (with extension lines), and
labelled `h = value ± u`. For an ISO 5436-1 A1 groove or ridge the depth line runs at
the groove centre from the upper line to the bottom line (`d`, or `h` for a ridge), and
the width line spans W between the half-height crossings, drawn inside the feature. The
numbers are the reported results; drawing changes nothing.

**Residual layer.** For every plane fit, levelled height minus the fitted line, for the
points that stayed in the fit; points offered to the fit but rejected (burr flanks,
outliers) are shown separately. The dashed lines are ±2σ of the plane residuals
(median over the planes). Correlated residuals - waviness, roughness - are visible here
before they appear as `CORRELATED_RESIDUALS`.

**Roughness curves.** Material ratio (Abbott-Firestone) curve: at each height c, the
fraction of evaluated points at or above c (uniform sampling, so equal to the fraction
of the evaluation length). Amplitude distribution: histogram density of the evaluated
roughness profile, bin width by the Freedman-Diaconis rule (10-200 bins). No
material-ratio parameter (Rmr, Rdc, Rk family) is derived from them in this version.

**Batch statistics.** Per sample: n, mean, sample standard deviation (n − 1), standard
error SD/√n, coefficient of variation SD/|mean| in %, median, minimum, maximum and the
half-width of the 95 % confidence interval of the mean, t(0.975, n − 1)·SE. With one scan
the spread entries are empty. These describe scan-to-scan scatter; the single-scan
uncertainty `u` (section 5) is reported separately.

**Worksheets.** Three header rows - Long Name, Units, Comments - then the data; empty
cells for missing values; xlsx with the header rows shaded and frozen. Column roles
(X, Y, yEr, L) are shown in the desktop worksheet as `A(X)`, `B(Y)`…
