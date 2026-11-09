"""Origin-style outputs: descriptive statistics, dimension lines, residuals, worksheets,
roughness distributions and publication figures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from profilometry_analyzer.analysis import (
    StepSettings,
    analyze_roughness,
    analyze_step,
    analyze_stress,
)
from profilometry_analyzer.annotations import fmt_uncertainty, plane_residuals, step_dimensions
from profilometry_analyzer.batch import run_batch
from profilometry_analyzer.cli import main
from profilometry_analyzer.core.roughness import (
    RoughnessSettings,
    amplitude_distribution,
    material_ratio_curve,
)
from profilometry_analyzer.core.stats import describe
from profilometry_analyzer.core.stress import StressSettings
from profilometry_analyzer.export import origin as O
from profilometry_analyzer.export import plots
from profilometry_analyzer.readers import read_profile
from profilometry_analyzer.synthetic import demo_set


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("origin_demo")
    demo_set(out)
    return out


@pytest.fixture(scope="module")
def groove(demo: Path):  # type: ignore[no-untyped-def]
    return analyze_step(read_profile(demo / "scratch_1.csv"))


@pytest.fixture(scope="module")
def edge(demo: Path):  # type: ignore[no-untyped-def]
    return analyze_step(read_profile(demo / "filmA_1.csv"))


# ---------------------------------------------------------------- statistics
def test_describe_matches_hand_calculation() -> None:
    d = describe([1.0, 2.0, 3.0])
    se = 1.0 / np.sqrt(3.0)
    assert d["n"] == 3 and d["mean"] == pytest.approx(2.0) and d["sd"] == pytest.approx(1.0)
    assert d["se"] == pytest.approx(se)
    assert d["ci_half"] == pytest.approx(4.302652729749464 * se)  # t(0.975, 2)
    assert d["cv_percent"] == pytest.approx(50.0)
    assert d["median"] == 2.0 and d["min"] == 1.0 and d["max"] == 3.0
    one = describe([5.0, None, float("nan")])
    assert one["n"] == 1 and one["mean"] == 5.0 and one["sd"] is None and one["ci_half"] is None
    assert describe([])["mean"] is None


def test_batch_reports_cv_and_confidence_interval(demo: Path) -> None:
    res = run_batch(demo)
    s = {x.sample: x for x in res.samples}["filmA"]
    assert s.ci95_nm == pytest.approx(4.302652729749464 * s.std_nm / np.sqrt(3))  # type: ignore[operator]
    assert s.cv_percent == pytest.approx(100 * s.std_nm / s.mean_nm)  # type: ignore[operator]
    row = s.row()
    assert row["thickness_ci95_nm"] == s.ci95_nm and row["thickness_cv_percent"] == s.cv_percent


# ---------------------------------------------------------------- annotations
def test_dimension_lines(edge, groove) -> None:  # type: ignore[no-untyped-def]
    (d,) = step_dimensions(edge.measurement)
    assert d.kind == "height" and d.offset
    assert d.value == pytest.approx(edge.thickness_nm, abs=0.05)
    assert d.label.startswith("h = ") and "±" in d.label
    dims = step_dimensions(groove.measurement)
    kinds = {x.kind: x for x in dims}
    assert kinds["height"].value == pytest.approx(groove.thickness_nm, abs=0.05)
    assert kinds["height"].label.startswith("d = ")
    assert kinds["width"].value == pytest.approx(groove.measurement.width_um)
    # the width line lies inside the groove, above the bottom
    assert kinds["height"].z1_nm < kinds["width"].z0_nm < kinds["height"].z0_nm
    angstrom = step_dimensions(groove.measurement, "Å")
    assert "Å" in angstrom[0].label


def test_fmt_uncertainty_plain_notation() -> None:
    assert fmt_uncertainty(2.5e-5) == "0.000025"
    assert fmt_uncertainty(0.0836) == "0.084"
    assert fmt_uncertainty(25.3) == "25"


def test_plane_residuals_exclude_only_rejected_points(groove) -> None:  # type: ignore[no-untyped-def]
    res = plane_residuals(groove.measurement, groove.profile)
    assert len(res) == 2
    for r in res:
        assert r.r_nm.size > 100
        assert np.std(r.r_nm) == pytest.approx(r.sigma_nm, rel=0.3)
        # points rejected from a plane (burr flanks, outliers) - never the other level
        if r.r_excluded_nm.size:
            assert np.max(np.abs(r.r_excluded_nm)) < 0.5 * groove.thickness_nm


# ---------------------------------------------------------------- roughness curves
def test_material_ratio_and_amplitude_distribution() -> None:
    x = np.linspace(0, 400, 40001)
    r = 10.0 * np.sin(2 * np.pi * x / 40.0)
    mr, c = material_ratio_curve(r)
    assert mr[0] == 0.0 and mr[-1] == 100.0
    assert np.all(np.diff(c) <= 1e-12)  # heights fall as the material ratio grows
    assert np.interp(50.0, mr, c) == pytest.approx(0.0, abs=0.05)
    assert c[0] == pytest.approx(10.0, abs=1e-3) and c[-1] == pytest.approx(-10.0, abs=1e-3)
    h, dens = amplitude_distribution(r)
    width = h[1] - h[0]
    assert float(np.sum(dens) * width) == pytest.approx(1.0, rel=1e-6)
    assert dens[0] == pytest.approx(dens[-1], rel=0.05)  # symmetric
    assert amplitude_distribution(np.zeros(10))[0].size == 0


# ---------------------------------------------------------------- worksheets
def test_origin_csv_has_three_header_rows(groove, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cols = O.step_worksheet(groove)
    p = O.write_origin_csv(tmp_path / "w.csv", cols)
    rows = list(csv.reader(p.open(encoding="utf-8-sig")))
    assert rows[0][:3] == ["Position", "Height as read", "Height levelled"]
    assert rows[1][:3] == ["µm", "nm", "nm"]
    assert rows[2][0] == "scratch_1"
    assert len(rows) == 3 + groove.profile.n
    fit_col = rows[0].index("Plane fit")
    assert rows[3][fit_col] == ""  # outside the plane ranges the fit column stays empty
    assert [c.designation for c in cols][0] == "X"
    assert O.column_letters(28)[-2:] == ["AA", "AB"]


def test_origin_xlsx_sheets(groove, demo: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from openpyxl import load_workbook

    batch = run_batch(demo)
    p = O.write_origin_xlsx(
        tmp_path / "w.xlsx",
        {
            "Profile": O.step_worksheet(groove),
            "Edges": O.step_edges_worksheet(groove),
            "Samples": O.batch_worksheet(batch),
        },
    )
    wb = load_workbook(p)
    ws = wb["Samples"]
    assert [c.value for c in ws[1]][:3] == ["Sample", "Thickness", "SD"]
    assert [c.value for c in ws[2]][:3] == [None, "nm", "nm"]
    assert ws.freeze_panes == "A4"
    assert ws.max_row == 3 + len(batch.samples)
    assert wb["Edges"]["C1"].value == "Height"


def test_roughness_and_stress_worksheets(demo: Path) -> None:
    rr = analyze_roughness(
        read_profile(demo / "roughness" / "sine_A12nm_40um.csv"),
        RoughnessSettings(lambda_c_um=250.0, lambda_s_um=0.0),
    )
    cols = O.roughness_worksheet(rr)
    assert cols[3].long_name == "Roughness profile" and len(cols[3].values) == rr.result.x_um.size
    curves = O.roughness_curves_worksheet(rr)
    assert curves[0].units == "%" and len(curves[0].values) == len(curves[1].values)
    params = O.roughness_parameters_worksheet(rr)
    assert "Ra" in params[0].values
    st = analyze_stress(
        read_profile(demo / "stress" / "wafer_pre.csv"),
        read_profile(demo / "stress" / "wafer_post.csv"),
        StressSettings(film_thickness_nm=500.0),
    )
    sc = O.stress_worksheet(st)
    assert len(sc) == 6 and sc[0].units == "mm"
    assert float(np.std(np.asarray(sc[2].values))) < 5.0  # residual of a parabola: noise only


# ---------------------------------------------------------------- figures
def test_figure_options_and_presets() -> None:
    o = plots.figure_options("single")
    assert (o.width_mm, o.height_mm, o.font_pt) == (85.0, 75.0, 8.0)
    c = plots.figure_options("100x60", dpi=600)
    assert (c.width_mm, c.height_mm, c.dpi) == (100.0, 60.0, 600)
    with pytest.raises(ValueError):
        plots.figure_options("huge")
    with pytest.raises(ValueError):
        plots.FigureOptions(x_unit="cm")
    with pytest.raises(ValueError):
        plots.FigureOptions(dpi=10)
    assert plots.z_scale("auto", np.array([0.0, 6000.0])) == (1e-3, "µm")
    assert plots.z_scale("Å") == (10.0, "Å")
    assert plots.x_scale("mm") == (1e-3, "mm")


def test_step_figure_layers_size_and_formats(groove, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from PIL import Image

    o = plots.figure_options("single", dpi=150)
    fig = plots.plot_step(groove, title="scratch", options=o)
    try:
        w, h = fig.get_size_inches()
        assert w * 25.4 == pytest.approx(85.0) and h * 25.4 == pytest.approx(75.0)
        assert len(fig.axes) == 3  # as read, levelled, residuals
        texts = [t.get_text() for t in fig.axes[1].texts]
        assert any(t.startswith("d = ") for t in texts) and any(t.startswith("W = ") for t in texts)
        files = plots.save(fig, tmp_path / "s", ["png", "tif", "svg", "pdf"])
    finally:
        plots.close(fig)
    assert [p.suffix for p in files] == [".png", ".tiff", ".svg", ".pdf"]
    with Image.open(files[1]) as im:
        assert im.size[0] == pytest.approx(85 / 25.4 * 150, abs=2)
    svg = files[2].read_text(encoding="utf-8")
    assert "<text" in svg and "W = " in svg  # text stays editable text
    bare = plots.plot_step(
        groove, options=plots.FigureOptions(show_raw=False, residuals=False, annotate=False)
    )
    try:
        assert len(bare.axes) == 1 and not any(
            t.get_text().startswith("d =") for t in bare.axes[0].texts
        )
    finally:
        plots.close(bare)
    with pytest.raises(ValueError):
        plots.save(bare, tmp_path / "x", ["bmp"])


def test_roughness_stress_batch_figures(demo: Path) -> None:
    rr = analyze_roughness(
        read_profile(demo / "roughness" / "sine_A12nm_40um.csv"),
        RoughnessSettings(lambda_c_um=250.0, lambda_s_um=0.0),
    )
    fig = plots.plot_roughness(rr, options=plots.figure_options("double"))
    try:
        assert len(fig.axes) == 5
    finally:
        plots.close(fig)
    st = analyze_stress(
        read_profile(demo / "stress" / "wafer_pre.csv"),
        read_profile(demo / "stress" / "wafer_post.csv"),
        StressSettings(film_thickness_nm=500.0),
    )
    fig = plots.plot_stress(st)
    try:
        assert len(fig.axes) == 2 and fig.axes[-1].get_xlabel() == "x (mm)"
    finally:
        plots.close(fig)
    batch = run_batch(demo)
    for err in ("sd", "se", "ci95"):
        fig = plots.plot_batch(batch, options=plots.FigureOptions(error_bar=err))
        try:
            ax = fig.axes[0]
            assert len(ax.patches) == len(batch.samples)
            assert sum(t.get_text().startswith("n = ") for t in ax.texts) == len(batch.samples)
        finally:
            plots.close(fig)


def test_cli_origin_tiff_and_figure_options(demo: Path, tmp_path: Path) -> None:
    out = tmp_path / "o"
    f = str(demo / "scratch_1.csv")
    assert (
        main(
            [
                "step",
                f,
                "-o",
                str(out),
                "--formats",
                "origin,tif",
                "--figure-size",
                "single",
                "--dpi",
                "120",
                "--z-display",
                "A",
            ]
        )
        == 0
    )
    names = {p.name for p in out.iterdir()}
    assert {
        "scratch_1_step_origin.csv",
        "scratch_1_step_origin.xlsx",
        "scratch_1_step.tiff",
    } <= names
    assert (
        main(
            [
                "batch",
                str(demo),
                "-o",
                str(out / "b"),
                "--formats",
                "origin,png",
                "--error-bar",
                "ci95",
            ]
        )
        == 0
    )
    assert (out / "b" / "batch_origin.csv").exists() and (out / "b" / "batch_samples.png").exists()
    with pytest.raises(SystemExit):
        main(["step", f, "--formats", "bmp"])


def test_step_settings_unchanged_by_figures(groove) -> None:  # type: ignore[no-untyped-def]
    """Drawing never changes the numbers that verify compares."""
    before = groove.key_numbers()
    fig = plots.plot_step(groove)
    plots.close(fig)
    assert groove.key_numbers() == before
    assert isinstance(groove.settings, StepSettings)
