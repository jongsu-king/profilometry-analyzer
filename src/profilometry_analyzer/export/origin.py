"""Worksheets laid out for OriginLab Origin (and any other plotting program).

Each table starts with three header lines - **Long Name**, **Units** and **Comments** -
followed by the data, which is the order of Origin's worksheet header rows.  When
importing the CSV into Origin, set three header lines and assign them to Long Name,
Units and Comments; in Excel-based workflows the ``.xlsx`` version carries the same
three rows (bold) above the data.  Column designations (X, Y, yEr, L for labels) are
kept in :attr:`Column.designation` and shown in the desktop worksheet view.

Missing values are written as empty cells.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..annotations import plane_residuals

if TYPE_CHECKING:  # pragma: no cover
    from ..analysis import RoughnessAnalysis, StepResult, StressAnalysis
    from ..batch import BatchResult

HEADER_ROWS = ("Long Name", "Units", "Comments")


@dataclass
class Column:
    long_name: str
    units: str
    comments: str
    values: Sequence[Any] | np.ndarray
    designation: str = "Y"
    """``X``, ``Y``, ``yEr`` (error bar of the Y column on its left) or ``L`` (label)."""

    def cell(self, i: int) -> Any:
        if i >= len(self.values):
            return ""
        v = self.values[i]
        if v is None:
            return ""
        if isinstance(v, float | np.floating):
            return float(v) if np.isfinite(v) else ""
        if isinstance(v, np.integer | np.bool_):
            return int(v)
        return v


def column_letters(n: int) -> list[str]:
    """``A, B, ..., Z, AA, AB, ...`` (Origin's short names)."""
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def n_rows(cols: Sequence[Column]) -> int:
    return max((len(c.values) for c in cols), default=0)


def write_origin_csv(path: str | Path, cols: Sequence[Column]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow([c.long_name for c in cols])
        w.writerow([c.units for c in cols])
        w.writerow([c.comments for c in cols])
        for i in range(n_rows(cols)):
            w.writerow([c.cell(i) for c in cols])
    return p


def write_origin_xlsx(path: str | Path, sheets: Mapping[str, Sequence[Column]]) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    fill = PatternFill("solid", fgColor="EDEDED")
    for k, (name, cols) in enumerate(sheets.items()):
        ws = wb.active if k == 0 else wb.create_sheet()
        assert ws is not None
        ws.title = name[:31]
        ws.append([c.long_name for c in cols])
        ws.append([c.units for c in cols])
        ws.append([c.comments for c in cols])
        for row in ws.iter_rows(min_row=1, max_row=3):
            for cell in row:
                cell.font = Font(bold=cell.row == 1, italic=cell.row == 3)
                cell.fill = fill
        for i in range(n_rows(cols)):
            ws.append([c.cell(i) for c in cols])
        ws.freeze_panes = "A4"
        for j, c in enumerate(cols, start=1):
            width = max(len(c.long_name), len(c.units), 10)
            ws.column_dimensions[ws.cell(1, j).column_letter].width = min(width + 2, 40)
    wb.save(p)
    return p


# ---------------------------------------------------------------- worksheets
def _nan_outside(x: np.ndarray, mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    out[mask] = values[mask]
    return out


def step_worksheet(res: StepResult) -> list[Column]:
    """Point table of a step result: position, heights as read and levelled, the plane
    fits (empty outside their ranges), the fit residuals and the validity flag."""
    p, src = res.profile, res.source
    x = p.x_um
    m = res.measurement
    fit = np.full(x.shape, np.nan)
    resid = np.full(x.shape, np.nan)
    used = np.zeros(x.shape, dtype=bool)
    for pl in m.planes:
        rng = (x >= pl.x_start_um) & (x <= pl.x_stop_um)
        fit[rng] = pl.line.at(x[rng])
        resid[rng] = p.z_nm[rng] - fit[rng]
        if pl.used is not None:
            used |= np.asarray(pl.used, dtype=bool) & rng
    cols = [
        Column("Position", "µm", src.name, x, "X"),
        Column(
            "Height as read",
            "nm",
            f"file unit {src.z_unit} ({src.z_unit_source})",
            src.z_nm if src.n == p.n else np.full(x.shape, np.nan),
        ),
        Column("Height levelled", "nm", res.level.method, np.where(p.valid, p.z_nm, np.nan)),
        Column("Plane fit", "nm", m.method, fit),
        Column("Fit residual", "nm", "levelled - plane fit", resid),
        Column("Used in fit", "", "1 = point entered a plane fit", used.astype(int)),
        Column("Valid", "", "0 = spike removed", p.valid.astype(int)),
    ]
    return cols


def step_edges_worksheet(res: StepResult) -> list[Column]:
    rows = res.measurement.per_edge
    if not rows:
        return []

    def col(key: str) -> list[Any]:
        return [r.get(key) for r in rows]

    return [
        Column("Edge", "", res.source.name, col("edge"), "X"),
        Column("Position", "µm", "half-height crossing", col("x_um")),
        Column("Height", "nm", "right plane - left plane", col("height_nm")),
        Column("u(height)", "nm", "k = 1", col("u_nm"), "yEr"),
        Column("u fit", "nm", "regression part", col("u_fit_nm")),
        Column("u slope", "nm", "plane slope difference part", col("u_slope_nm")),
        Column("Transition width", "µm", "", col("width_um")),
    ]


def roughness_worksheet(res: RoughnessAnalysis) -> list[Column]:
    r = res.result
    lc = f"λc = {r.lambda_c_um:g} µm"
    return [
        Column("Position", "µm", res.source.name, r.x_um, "X"),
        Column(
            "Primary profile", "nm", f"λs = {r.lambda_s_um or 0:g} µm, form removed", r.primary_nm
        ),
        Column("Mean line", "nm", f"Gaussian, {lc}", r.waviness_nm),
        Column("Roughness profile", "nm", "primary - mean line", r.roughness_nm),
        Column("Evaluated", "", "1 = inside the evaluation length", r.evaluated.astype(int)),
    ]


def roughness_curves_worksheet(res: RoughnessAnalysis) -> list[Column]:
    from ..core.roughness import amplitude_distribution, material_ratio_curve

    r = res.result
    rr = r.roughness_nm[r.evaluated]
    mr, c = material_ratio_curve(rr)
    h, dens = amplitude_distribution(rr)
    return [
        Column("Material ratio", "%", "Abbott-Firestone curve", mr, "X"),
        Column("Height", "nm", "level c", c),
        Column("Height", "nm", "bin centre", h, "X"),
        Column("Amplitude density", "1/nm", "integrates to 1", dens),
    ]


def roughness_parameters_worksheet(res: RoughnessAnalysis) -> list[Column]:
    pr = res.result.parameters
    names = list(pr)
    units = ["µm" if k == "Rsm" else "" if k in ("Rsk", "Rku") else "nm" for k in names]
    return [
        Column("Parameter", "", "ISO 21920-2", names, "L"),
        Column("Value", "", res.source.name, [pr[k] for k in names]),
        Column("Unit", "", "", units, "L"),
    ]


def stress_worksheet(res: StressAnalysis) -> list[Column]:
    from ..annotations import parabola_residuals

    cols: list[Column] = []
    for label, prof, fit in (
        ("before", res.pre, res.result.pre),
        ("after", res.post, res.result.post),
    ):
        if prof is None or fit is None:
            continue
        cols += [
            Column("Position", "mm", f"{label}: {prof.name}", prof.x_um / 1e3, "X"),
            Column("Height", "µm", label, prof.z_nm / 1e3),
            Column(
                "Fit residual",
                "nm",
                f"R = {fit.radius_m:.5g} m",
                parabola_residuals(prof.x_um, prof.z_nm, fit),
            ),
        ]
    return cols


def batch_worksheet(batch: BatchResult) -> list[Column]:
    s = batch.samples
    return [
        Column("Sample", "", "", [v.sample for v in s], "X"),
        Column("Thickness", "nm", "mean of the scans", [v.mean_nm for v in s]),
        Column("SD", "nm", "sample standard deviation (n - 1)", [v.std_nm for v in s], "yEr"),
        Column("SE", "nm", "standard error of the mean", [v.sem_nm for v in s]),
        Column("95% CI", "nm", "half-width, Student t", [v.ci95_nm for v in s]),
        Column("CV", "%", "SD / mean", [v.cv_percent for v in s]),
        Column("N", "", "scans", [v.n for v in s]),
        Column("Min", "nm", "", [v.min_nm for v in s]),
        Column("Max", "nm", "", [v.max_nm for v in s]),
    ]


def batch_scans_worksheet(batch: BatchResult) -> list[Column]:
    items = [it for it in batch.items if it.result is not None]
    return [
        Column("Sample", "", "", [it.sample for it in items], "L"),
        Column("Scan", "", "", [it.path.name for it in items], "X"),
        Column("Thickness", "nm", "", [it.result.thickness_nm for it in items if it.result]),
        Column(
            "u(thickness)",
            "nm",
            "single scan, k = 1",
            [it.result.measurement.u_nm for it in items if it.result],
            "yEr",
        ),
        Column("Kind", "", "", [it.result.measurement.kind for it in items if it.result], "L"),
    ]


def plane_residual_worksheet(res: StepResult) -> list[Column]:
    cols: list[Column] = []
    for r in plane_residuals(res.measurement, res.profile):
        cols += [
            Column("Position", "µm", r.name, r.x_um, "X"),
            Column("Residual", "nm", f"σ = {r.sigma_nm:.3g} nm", r.r_nm),
        ]
    return cols


__all__ = [
    "HEADER_ROWS",
    "Column",
    "batch_scans_worksheet",
    "batch_worksheet",
    "column_letters",
    "plane_residual_worksheet",
    "roughness_curves_worksheet",
    "roughness_parameters_worksheet",
    "roughness_worksheet",
    "step_edges_worksheet",
    "step_worksheet",
    "stress_worksheet",
    "write_origin_csv",
    "write_origin_xlsx",
]
