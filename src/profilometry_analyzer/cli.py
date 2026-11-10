"""Command-line interface: ``profilometry-analyzer <command> ...``.

Commands: ``info``, ``level``, ``step``, ``roughness``, ``stress``, ``batch``, ``verify``,
``synth``, ``substrates``, ``gui``, ``update-check``.  Every analysis prints its result
and its notices; with ``--output`` it also writes tables, figures and a project file
(JSON) that ``verify`` can check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from . import notices as N
from .analysis import StepSettings, analyze_roughness, analyze_step, analyze_stress
from .core.leveling import level_auto, level_none, level_regions
from .core.roughness import RoughnessSettings
from .core.stress import SUBSTRATES, StressSettings
from .instrument import DEFAULT_INSTRUMENT
from .notices import Notice
from .readers import ReadError, read_profile

FIG_FORMATS = ("png", "tiff", "svg", "pdf")


# ------------------------------------------------------------------ parsing helpers
def _pair(text: str) -> tuple[float, float]:
    sep = ":" if ":" in text else ".." if ".." in text else None
    if sep is None:
        raise argparse.ArgumentTypeError(f"expected a range a:b, got {text!r}")
    a, b = text.split(sep, 1)
    return float(a), float(b)


def _pairs(text: str) -> list[tuple[float, float]]:
    return [_pair(t) for t in text.split(",") if t.strip()]


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.split(",") if t.strip()]


def _formats(text: str) -> list[str]:
    out = [t.strip().lower() for t in text.split(",") if t.strip()]
    out = ["tiff" if t == "tif" else t for t in out]
    allowed = {"csv", "xlsx", "json", "origin", *FIG_FORMATS}
    bad = [t for t in out if t not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown format(s) {bad}; use {sorted(allowed)}")
    return out


def _add_reader(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("reading")
    g.add_argument("--x-col", help="lateral column (0-based index or header text)")
    g.add_argument("--z-col", help="height column (0-based index or header text)")
    g.add_argument("--x-unit", help="lateral unit when the file does not say (Å, nm, µm/um, mm)")
    g.add_argument("--z-unit", help="height unit when the file does not say (Å, nm, µm/um, mm)")
    g.add_argument("--delimiter", help="column delimiter (default: detected)")
    g.add_argument("--encoding", help="text encoding (default: UTF-8, CP949, Latin-1)")
    g.add_argument("--sheet", help="Excel worksheet (name or 0-based index)")
    g.add_argument("--block", type=int, help="numeric table of the file (0-based)")


def _reader_options(a: argparse.Namespace) -> dict[str, Any]:
    keys = ("x_col", "z_col", "x_unit", "z_unit", "delimiter", "encoding", "sheet", "block")
    return {k: getattr(a, k) for k in keys if getattr(a, k, None) is not None}


def _add_scan(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("scan")
    g.add_argument("--tip-radius", type=float, help="stylus radius in µm (default: file, else 2)")
    g.add_argument(
        "--cone-angle",
        type=float,
        help="full angle of the stylus cone in degrees (default: file, else 60; 0 = sphere only)",
    )
    g.add_argument("--force", type=float, help="stylus force in mg (if not in the file)")
    g.add_argument("--soft", action="store_true", help="the film is soft (polymer): force check")


def _add_step(p: argparse.ArgumentParser) -> None:
    _add_scan(p)
    g = p.add_argument_group("levelling and steps")
    g.add_argument("--crop", type=_pair, help="analyse only x range a:b (µm)")
    g.add_argument("--level", choices=("auto", "regions", "none"), default="auto")
    g.add_argument("--level-method", choices=("huber", "ransac", "lsq"), default="huber")
    g.add_argument("--level-regions", type=_pairs, help="reference regions a:b,c:d (µm)")
    g.add_argument(
        "--separate-levels",
        action="store_true",
        help="reference regions on different levels (shared slope only)",
    )
    g.add_argument("--bow", type=int, default=1, help="polynomial order (1 = tilt only, 2-6 bow)")
    g.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "edge", "groove", "ridge", "periodic", "staircase", "multi"),
    )
    g.add_argument(
        "--edges", type=_floats, help="edge positions x1,x2,... (µm) instead of detection"
    )
    g.add_argument("--regions", type=_pairs, help="manual step: reference a:b, measurement c:d")
    g.add_argument("--smooth", type=float, help="derivative smoothing sigma (µm)")
    g.add_argument("--edge-k", type=float, default=6.0, help="edge threshold in sigma")
    g.add_argument("--min-step", type=float, default=0.0, help="smallest step to report (nm)")
    g.add_argument("--margin", type=float, help="edge margin (µm; default 2 stylus radii)")
    g.add_argument("--min-plane", type=float, help="shortest usable plane (µm)")
    g.add_argument("--burr-k", type=float, default=4.0, help="raised-point exclusion (sigma)")
    g.add_argument(
        "--single-scale",
        action="store_true",
        help="look for edges at the base smoothing only (no search for wide, gentle edges)",
    )
    g.add_argument(
        "--no-widen",
        action="store_true",
        help="keep the detected transitions even when the plateau next to them is not flat",
    )
    g.add_argument(
        "--no-tip-check",
        action="store_true",
        help="do not look for the points the stylus could not touch",
    )
    g.add_argument("--no-despike", action="store_true")
    g.add_argument("--spike-k", type=float, default=8.0)


def _step_settings(a: argparse.Namespace) -> StepSettings:
    level = a.level
    if a.level_regions and level == "auto":
        level = "regions"
    return StepSettings(
        despike=not a.no_despike,
        spike_k=a.spike_k,
        crop_um=a.crop,
        level=level,
        level_method=a.level_method,
        level_regions_um=a.level_regions,
        level_separate=a.separate_levels,
        bow_order=a.bow,
        mode=a.mode,
        edges_um=a.edges,
        regions_um=a.regions,
        smooth_um=a.smooth,
        edge_k=a.edge_k,
        min_step_nm=a.min_step,
        margin_um=a.margin,
        min_plane_um=a.min_plane,
        burr_k=a.burr_k,
        multiscale=not a.single_scale,
        widen=not a.no_widen,
        tip_radius_um=a.tip_radius,
        cone_angle_deg=a.cone_angle,
        stylus_force_mg=a.force,
        soft_film=a.soft,
        tip_check=not a.no_tip_check,
    )


def _add_output(p: argparse.ArgumentParser, default: str = "csv,json,png") -> None:
    g = p.add_argument_group("output")
    g.add_argument("-o", "--output", help="folder for tables, figures and the project file")
    g.add_argument(
        "--formats",
        type=_formats,
        default=_formats(default),
        help=f"comma list of csv,xlsx,json,origin,png,tiff,svg,pdf (default {default});"
        " origin = worksheets with Long Name / Units / Comments rows (CSV + xlsx)",
    )
    g.add_argument("--style", choices=("origin", "classic"), default="origin")
    f = p.add_argument_group("figures")
    f.add_argument(
        "--figure-size",
        metavar="SIZE",
        help="single, onehalf, double, slide or WIDTHxHEIGHT in mm (default: own size)",
    )
    f.add_argument("--dpi", type=int, default=300, help="raster resolution (default 300)")
    f.add_argument("--font-size", type=float, metavar="PT", help="label font size in points")
    f.add_argument("--x-display", choices=("µm", "um", "mm"), help="x axis unit of the figures")
    f.add_argument(
        "--z-display", choices=("auto", "Å", "A", "nm", "µm", "um"), help="height axis unit"
    )
    f.add_argument("--no-annotate", action="store_true", help="no dimension lines")
    f.add_argument("--no-residuals", action="store_true", help="no residual layer")
    f.add_argument(
        "--error-bar", choices=("sd", "se", "ci95"), default="sd", help="batch chart error bars"
    )
    g.add_argument("--json", action="store_true", help="print the result as JSON")


def _print_notices(notices: Sequence[Notice], indent: str = "  ") -> None:
    for n in notices:
        print(f"{indent}{n}")


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def _fig_options(a: argparse.Namespace, stress: bool = False) -> Any:
    from .export import plots

    xu = {"um": "µm"}.get(a.x_display, a.x_display) if a.x_display else None
    zu = {"um": "µm", "A": "Å"}.get(a.z_display, a.z_display) if a.z_display else None
    if stress:
        xu, zu = xu or "mm", zu or "µm"
    return plots.figure_options(
        a.figure_size,
        dpi=a.dpi,
        font_pt=a.font_size,
        x_unit=xu,
        z_unit=zu,
        annotate=not a.no_annotate,
        residuals=not a.no_residuals,
        error_bar=a.error_bar,
    )


def _origin(stem: Path, sheets: dict[str, Any], formats: list[str]) -> None:
    if "origin" not in formats:
        return
    from .export.origin import write_origin_csv, write_origin_xlsx

    sheets = {k: v for k, v in sheets.items() if v}
    first = next(iter(sheets.values()))
    print(f"  wrote {write_origin_csv(stem.with_name(stem.name + '_origin.csv'), first)}")
    print(f"  wrote {write_origin_xlsx(stem.with_name(stem.name + '_origin.xlsx'), sheets)}")


def _figures(
    fig_func: Any,
    obj: Any,
    stem: Path,
    formats: list[str],
    style: str,
    title: str,
    options: Any = None,
) -> None:
    figs = [f for f in formats if f in FIG_FORMATS]
    if not figs:
        return
    from .export import plots

    fig = fig_func(obj, title=title, style=style, options=options)
    try:
        for p in plots.save(fig, stem, figs):
            print(f"  wrote {p}")
    finally:
        plots.close(fig)


# ------------------------------------------------------------------ commands
def cmd_info(a: argparse.Namespace) -> int:
    prof = read_profile(a.file, **_reader_options(a))
    s = prof.summary()
    if a.json:
        print(json.dumps(N._plain(s), indent=2, ensure_ascii=False))
        return 0
    print(f"{s['name']}  ({s['path']})")
    print(
        f"  points {s['points']} (valid {s['valid_points']}), length {s['length_um']:.6g} µm,"
        f" spacing {s['dx_um']:.4g} µm"
    )
    print(f"  columns: x = {prof.reader.get('x_column')!r}, z = {prof.reader.get('z_column')!r}")
    print(
        f"  units: x {s['x_unit']} ({s['x_unit_source']}), z {s['z_unit']} ({s['z_unit_source']})"
    )
    print(f"  height range {s['z_min_nm']:.6g} .. {s['z_max_nm']:.6g} nm")
    for k, v in s["scan"].items():
        if k != "sources" and v not in (None, False):
            print(f"  {k}: {v}  [{s['scan']['sources'].get(k, '')}]")
    if s["metadata"]:
        print(f"  metadata keys: {', '.join(list(s['metadata'])[:25])}")
    for note in prof.reader.get("column_notes", []) + prof.reader.get("metadata_notes", []):
        print(f"  note: {note}")
    _print_notices(prof.notices)
    return 0


def cmd_level(a: argparse.Namespace) -> int:
    prof = read_profile(a.file, **_reader_options(a))
    if a.tip_radius:
        prof = prof.with_scan(tip_radius_um=a.tip_radius)
    from .core.despike import despike

    p = despike(prof) if not a.no_despike else prof
    if a.level_regions:
        lp, lev = level_regions(p, a.level_regions, same_level=not a.separate_levels, order=a.bow)
    elif a.method == "none":
        lp, lev = level_none(p)
    else:
        lp, lev, _ = level_auto(p, method=a.method, order=a.bow)
    print(
        f"{prof.name}: {lev.method}, slope {lev.slope_nm_per_um:.6g} nm/µm"
        f" ({lev.slope_nm_per_um:.4g} mrad), order {lev.order}, residual {lev.sigma_nm:.4g} nm"
    )
    _print_notices(lp.notices)
    if a.output:
        from .export.tables import profile_rows, write_csv

        out = Path(a.output)
        stem = out / f"{Path(a.file).stem}_levelled"
        rows = profile_rows(lp, {"z_as_read_nm": prof.z_nm} if prof.n == lp.n else None)
        print(f"  wrote {write_csv(stem.with_suffix('.csv'), rows)}")
        info = {
            "level": lev.to_dict(),
            "history": lp.history,
            "notices": [n.to_dict() for n in lp.notices],
        }
        jp = stem.with_suffix(".json")
        jp.write_text(json.dumps(N._plain(info), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  wrote {jp}")
    return 0


def _report_step(res: Any) -> None:
    m = res.measurement
    head = f"{res.source.name}: {m.kind} ({m.method})"
    if m.thickness_nm is not None:
        what = {"groove": "depth", "ridge": "height", "periodic": "mean step"}.get(m.kind, "step")
        head += f"  {what} = {m.thickness_nm:.5g} ± {_fmt(m.u_nm, 3)} nm"
        if m.kind == "edge" and m.value_nm is not None:
            head += f" ({'up' if m.value_nm > 0 else 'down'} at {m.position_um:.5g} µm)"
        if m.width_um and m.kind in ("groove", "ridge"):
            head += f", W = {m.width_um:.4g} µm"
    print(head)
    if m.kind == "periodic":
        s = m.stats
        print(f"  n = {s['n']}, s.d. = {_fmt(s['std_nm'])} nm, period = {_fmt(s['period_um'])} µm")
    if m.kind in ("staircase", "multi", "periodic") or len(m.per_edge) > 1:
        for r in m.per_edge:
            print(
                f"   edge {r['edge']}: x = {_fmt(r.get('x_um'), 6)} µm, height ="
                f" {_fmt(r.get('height_nm'), 5)} ± {_fmt(r.get('u_nm'), 3)} nm"
            )
    print(
        f"  levelling {res.level.method}, slope {res.level.slope_nm_per_um:.4g} nm/µm,"
        f" stylus radius {res.tip_radius_um:g} µm ({res.tip_radius_source}),"
        f" cone {res.cone_angle_deg:g}° ({res.cone_angle_source})"
    )
    _print_notices(res.notices)


def _write_step_outputs(res: Any, a: argparse.Namespace) -> None:
    from .export import plots
    from .export.project import save_project
    from .export.tables import notice_rows, profile_rows, write_csv, write_xlsx

    out = Path(a.output)
    stem = out / Path(res.source.path or res.source.name).stem
    if "csv" in a.formats:
        print(f"  wrote {write_csv(stem.with_name(stem.name + '_step.csv'), [res.row()])}")
        if res.measurement.per_edge:
            write_csv(stem.with_name(stem.name + "_edges.csv"), res.measurement.per_edge)
    if "xlsx" in a.formats:
        sheets = {
            "Result": [res.row()],
            "Edges": res.measurement.per_edge,
            "Planes": [pl.to_dict() for pl in res.measurement.planes],
            "Notices": notice_rows(res.notices),
            "Profile": profile_rows(res.profile),
        }
        print(f"  wrote {write_xlsx(stem.with_name(stem.name + '_step.xlsx'), sheets)}")
    if "json" in a.formats:
        pj = save_project(res, stem.with_name(stem.name + "_step.json"), DEFAULT_INSTRUMENT)
        print(f"  wrote {pj}")
    from .export import origin as O

    _origin(
        stem.with_name(stem.name + "_step"),
        {
            "Profile": O.step_worksheet(res),
            "Edges": O.step_edges_worksheet(res),
            "Residuals": O.plane_residual_worksheet(res),
        },
        a.formats,
    )
    _figures(
        plots.plot_step,
        res,
        stem.with_name(stem.name + "_step"),
        a.formats,
        a.style,
        res.source.name,
        _fig_options(a),
    )


def cmd_step(a: argparse.Namespace) -> int:
    st = _step_settings(a)
    code = 0
    results = []
    for f in a.files:
        try:
            prof = read_profile(f, **_reader_options(a))
            res = analyze_step(prof, st, DEFAULT_INSTRUMENT)
        except (ReadError, ValueError) as exc:
            print(f"{f}: error: {exc}", file=sys.stderr)
            code = 1
            continue
        results.append(res)
        if a.json:
            print(json.dumps(N._plain(res.to_dict()), indent=2, ensure_ascii=False))
        else:
            _report_step(res)
        if a.output:
            _write_step_outputs(res, a)
    return code


def cmd_roughness(a: argparse.Namespace) -> int:
    prof = read_profile(a.file, **_reader_options(a))
    if a.tip_radius or a.cone_angle is not None:
        prof = prof.with_scan(tip_radius_um=a.tip_radius or None, cone_angle_deg=a.cone_angle)
    st = RoughnessSettings(
        lambda_c_um=a.lc,
        lambda_s_um=a.ls,
        n_sections=a.sections,
        form_order=a.form,
        region_um=a.range,
    )
    res = analyze_roughness(prof, st, despike_first=not a.no_despike, crop_um=a.crop)
    r = res.result
    if a.json:
        print(json.dumps(N._plain(res.to_dict()), indent=2, ensure_ascii=False))
    else:
        cls = f" (class {r.setting_class})" if r.setting_class else ""
        print(
            f"{prof.name}: λc = {r.lambda_c_um:g} µm, λs = {_fmt(r.lambda_s_um)} µm{cls},"
            f" {r.n_sections} section(s),"
            f" evaluation {r.evaluation_um[1] - r.evaluation_um[0]:.5g} µm"
        )
        pr = r.parameters
        for k in ("Ra", "Rq", "Rz", "Rp", "Rv", "Rt"):
            print(f"  {k:<4}= {pr[k]:.5g} nm")
        print(f"  Rsk = {pr['Rsk']:.4g}\n  Rku = {pr['Rku']:.4g}\n  Rsm = {pr['Rsm']:.5g} µm")
        if r.stylus:
            print(
                f"  stylus {r.stylus['radius_um']:g} µm, {r.stylus['cone_angle_deg']:g}° cone:"
                f" {r.stylus['unreached_percent']:.2g} % of the evaluation length not touched"
            )
        _print_notices(res.notices)
    if a.output:
        from .export import plots
        from .export.project import save_project
        from .export.tables import notice_rows, write_csv, write_xlsx

        stem = Path(a.output) / f"{Path(a.file).stem}_roughness"
        if "csv" in a.formats:
            print(f"  wrote {write_csv(stem.with_suffix('.csv'), [res.row()])}")
        if "xlsx" in a.formats:
            sheets = {
                "Parameters": [res.row()],
                "Sections": r.sections,
                "Notices": notice_rows(res.notices),
            }
            print(f"  wrote {write_xlsx(stem.with_suffix('.xlsx'), sheets)}")
        if "json" in a.formats:
            print(f"  wrote {save_project(res, stem.with_suffix('.json'))}")
        from .export import origin as O

        _origin(
            stem,
            {
                "Profile": O.roughness_worksheet(res),
                "Curves": O.roughness_curves_worksheet(res),
                "Parameters": O.roughness_parameters_worksheet(res),
            },
            a.formats,
        )
        _figures(plots.plot_roughness, res, stem, a.formats, a.style, prof.name, _fig_options(a))
    return 0


def cmd_stress(a: argparse.Namespace) -> int:
    ro = _reader_options(a)
    post = read_profile(a.post, **ro)
    pre = read_profile(a.pre, **ro) if a.pre else None
    st = StressSettings(
        substrate=a.substrate,
        biaxial_modulus_gpa=a.modulus,
        youngs_modulus_gpa=a.youngs,
        poisson_ratio=a.poisson,
        substrate_thickness_um=a.hs,
        film_thickness_nm=a.hf,
        region_um=a.range,
        backside=a.backside,
        despike=not a.no_despike,
    )
    res = analyze_stress(pre, post, st)
    r = res.result
    if a.json:
        print(json.dumps(N._plain(res.to_dict()), indent=2, ensure_ascii=False))
    else:
        print(f"stress = {r.stress_mpa:.5g} ± {r.u_stress_mpa:.3g} MPa ({r.kind})")
        if r.pre:
            print(f"  before: R = {r.pre.radius_m:.5g} m ({r.pre.kappa_per_um * 1e6:.4g} 1/m)")
        print(f"  after:  R = {r.post.radius_m:.5g} m ({r.post.kappa_per_um * 1e6:.4g} 1/m)")
        print(f"  biaxial modulus {r.biaxial_modulus_gpa:.5g} GPa ({r.modulus_source})")
        print(
            f"  h_s = {r.substrate_thickness_um:g} µm, h_f = {r.film_thickness_nm:g} nm;"
            " positive = tensile"
        )
        _print_notices(res.notices)
    if a.output:
        from .export import plots
        from .export.project import save_project
        from .export.tables import notice_rows, write_csv, write_xlsx

        stem = Path(a.output) / f"{Path(a.post).stem}_stress"
        if "csv" in a.formats:
            print(f"  wrote {write_csv(stem.with_suffix('.csv'), [res.row()])}")
        if "xlsx" in a.formats:
            sheets = {"Stress": [res.row()], "Notices": notice_rows(res.notices)}
            print(f"  wrote {write_xlsx(stem.with_suffix('.xlsx'), sheets)}")
        if "json" in a.formats:
            print(f"  wrote {save_project(res, stem.with_suffix('.json'))}")
        from .export import origin as O

        _origin(stem, {"Scans": O.stress_worksheet(res)}, a.formats)
        _figures(
            plots.plot_stress,
            res,
            stem,
            a.formats,
            a.style,
            Path(a.post).stem,
            _fig_options(a, stress=True),
        )
    return 0


def cmd_batch(a: argparse.Namespace) -> int:
    from .batch import run_batch

    st = _step_settings(a)
    src: Any = a.inputs[0] if len(a.inputs) == 1 and Path(a.inputs[0]).is_dir() else a.inputs
    res = run_batch(
        src, st, grouping=a.group, reader_options=_reader_options(a), recursive=a.recursive
    )
    if not res.items:
        print("no supported files found", file=sys.stderr)
        return 1
    if a.json:
        print(
            json.dumps(
                N._plain({"scans": res.scan_rows(), "samples": res.sample_rows()}),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(
            f"{'sample':<20} {'n':>3} {'mean (nm)':>11} {'SD (nm)':>9} {'CV (%)':>7}"
            f" {'95% CI (nm)':>12}  kinds / warnings"
        )
        for s in res.samples:
            extra = ",".join(s.kinds) + ("  " + ";".join(s.warnings) if s.warnings else "")
            fail = f" ({s.n_failed} failed)" if s.n_failed else ""
            ci = f"± {_fmt(s.ci95_nm, 3)}" if s.ci95_nm is not None else "-"
            print(
                f"{s.sample:<20} {s.n:>3} {_fmt(s.mean_nm, 6):>11} {_fmt(s.std_nm, 3):>9}"
                f" {_fmt(s.cv_percent, 2):>7} {ci:>12}  {extra}{fail}"
            )
        for it in res.items:
            if it.error:
                print(f"  {it.path.name}: {it.error}")
    if a.output:
        from .export import plots
        from .export.project import save_project
        from .export.tables import write_csv, write_xlsx

        out = Path(a.output)
        if "csv" in a.formats:
            print(f"  wrote {write_csv(out / 'batch_scans.csv', res.scan_rows())}")
            print(f"  wrote {write_csv(out / 'batch_samples.csv', res.sample_rows())}")
        if "xlsx" in a.formats:
            notes = []
            for it in res.items:
                if it.result:
                    notes += [
                        {
                            "file": it.path.name,
                            "severity": n.severity,
                            "code": n.code,
                            "message": n.message,
                        }
                        for n in it.result.notices
                    ]
            sheets = {
                "Samples": res.sample_rows(),
                "Scans": res.scan_rows(),
                "Notices": notes,
                "Settings": [{"setting": k, "value": v} for k, v in st.to_dict().items()],
            }
            print(f"  wrote {write_xlsx(out / 'batch.xlsx', sheets)}")
        if "json" in a.formats:
            print(f"  wrote {save_project(res, out / 'batch.json', DEFAULT_INSTRUMENT)}")
        from .export import origin as O

        _origin(
            out / "batch",
            {"Samples": O.batch_worksheet(res), "Scans": O.batch_scans_worksheet(res)},
            a.formats,
        )
        _figures(
            plots.plot_batch, res, out / "batch_samples", a.formats, a.style, "", _fig_options(a)
        )
        if a.plots:
            for it in res.items:
                if it.result is not None:
                    _figures(
                        plots.plot_step,
                        it.result,
                        out / "scans" / f"{it.path.stem}_step",
                        [f for f in a.formats if f in FIG_FORMATS] or ["png"],
                        a.style,
                        it.path.name,
                        _fig_options(a),
                    )
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from .export.project import verify

    code = 0
    for f in a.projects:
        rep = verify(f, rtol=a.rtol)
        print(rep.text())
        code = code or (0 if rep.ok else 1)
    return code


def cmd_synth(a: argparse.Namespace) -> int:
    from .synthetic import demo_set

    files = demo_set(a.out, seed=a.seed)
    print(f"wrote {len(files)} synthetic scans and truth.json to {a.out}")
    return 0


def cmd_substrates(a: argparse.Namespace) -> int:
    for s in SUBSTRATES.values():
        flag = "" if s.verified else "  [UNVERIFIED]"
        print(f"{s.key:<13} {s.name:<28} M = E/(1-ν) = {s.biaxial_modulus_gpa:7.2f} GPa{flag}")
        print(f"{'':<13} source: {s.source}")
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    try:
        from .gui import run
    except ImportError as exc:  # pragma: no cover - depends on the installation
        print(
            f"the desktop application needs PySide6 and pyqtgraph ({exc});"
            ' install with: pip install "profilometry-analyzer[gui]"',
            file=sys.stderr,
        )
        return 2
    return int(run([sys.argv[0], *a.files]))


def cmd_update_check(a: argparse.Namespace) -> int:
    from .update_check import UpdateCheckError, check_for_update, describe

    try:
        info = check_for_update()
    except UpdateCheckError as exc:
        print(f"update check failed: {exc}", file=sys.stderr)
        return 1
    print(describe(info))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="profilometry-analyzer",
        description="Step height, film thickness, roughness and film stress from stylus"
        " profilometer line scans (Dektak-compatible CSV/TXT/xlsx exports).",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="read a file and show columns, units and scan parameters")
    p.add_argument("file")
    _add_reader(p)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("level", help="remove tilt (and bow) and write the levelled profile")
    p.add_argument("file")
    _add_reader(p)
    p.add_argument("--method", choices=("huber", "ransac", "lsq", "none"), default="huber")
    p.add_argument("--level-regions", type=_pairs, help="reference regions a:b,c:d (µm)")
    p.add_argument("--separate-levels", action="store_true")
    p.add_argument("--bow", type=int, default=1)
    p.add_argument("--tip-radius", type=float)
    p.add_argument("--no-despike", action="store_true")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_level)

    p = sub.add_parser("step", help="step height / film thickness of one or more scans")
    p.add_argument("files", nargs="+")
    _add_reader(p)
    _add_step(p)
    _add_output(p)
    p.set_defaults(func=cmd_step)

    p = sub.add_parser("roughness", help="ISO 21920-2 roughness parameters")
    p.add_argument("file")
    _add_reader(p)
    p.add_argument("--lc", type=float, help="λc cut-off in µm (default: setting class from Ra)")
    p.add_argument("--ls", type=float, help="λs cut-off in µm (default: class value; 0 = none)")
    p.add_argument("--sections", type=int, default=5, help="nominal number of sections")
    p.add_argument("--form", type=int, default=1, help="form removal polynomial order")
    p.add_argument("--range", type=_pair, help="evaluate only x range a:b (µm), e.g. one plateau")
    p.add_argument("--crop", type=_pair, help="crop the profile before filtering")
    p.add_argument("--tip-radius", type=float)
    p.add_argument("--cone-angle", type=float, help="stylus cone angle in degrees (default 60)")
    p.add_argument("--no-despike", action="store_true")
    _add_output(p)
    p.set_defaults(func=cmd_roughness)

    p = sub.add_parser("stress", help="film stress from curvature (Stoney)")
    p.add_argument("--post", required=True, help="scan after deposition")
    p.add_argument("--pre", help="scan before deposition (same line)")
    _add_reader(p)
    p.add_argument("--substrate", default="si100", choices=sorted(SUBSTRATES))
    p.add_argument("--modulus", type=float, help="biaxial modulus E/(1-ν) in GPa (overrides)")
    p.add_argument("--youngs", type=float, help="substrate Young's modulus in GPa (with --poisson)")
    p.add_argument("--poisson", type=float, help="substrate Poisson ratio (with --youngs)")
    p.add_argument("--hs", type=float, help="substrate thickness in µm")
    p.add_argument("--hf", type=float, required=True, help="film thickness in nm")
    p.add_argument("--range", type=_pair, help="fit only x range a:b (µm)")
    p.add_argument("--backside", action="store_true", help="scans taken on the back side")
    p.add_argument("--no-despike", action="store_true")
    _add_output(p)
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("batch", help="every scan of a folder, mean ± s.d. per sample")
    p.add_argument("inputs", nargs="+", help="a folder or files")
    _add_reader(p)
    _add_step(p)
    p.add_argument(
        "--group", help="sample grouping: regex with (?P<sample>...), 'folder' or 'none'"
    )
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--plots", action="store_true", help="also a figure per scan")
    _add_output(p, default="csv,xlsx,json,png")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("verify", help="repeat the analysis of project files and compare")
    p.add_argument("projects", nargs="+")
    p.add_argument("--rtol", type=float, default=1e-6)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("synth", help="write synthetic scans with known values")
    p.add_argument("--out", default="synthetic_scans")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("substrates", help="list substrate moduli and their sources")
    p.set_defaults(func=cmd_substrates)

    p = sub.add_parser("gui", help="start the desktop application")
    p.add_argument("files", nargs="*")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("update-check", help="check GitHub for a newer release")
    p.set_defaults(func=cmd_update_check)
    return ap


_PROBE_TEXT = "λc λs µm ± σ Å"


def _make_streams_encodable() -> None:
    """Keep printing when stdout or stderr uses a narrow code page.

    A piped or redirected stream on Windows uses the locale code page (for example
    cp1252), which cannot encode the λ of the roughness report; ``print`` then raises
    ``UnicodeEncodeError``, a ``ValueError``, and the command failed.  Such a stream is
    switched to UTF-8; a console keeps its encoding and replaces what it cannot show.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if stream is None or reconfigure is None:
            continue
        try:
            _PROBE_TEXT.encode(stream.encoding or "ascii")
            continue
        except (LookupError, UnicodeEncodeError):
            pass
        try:
            if stream.isatty():
                reconfigure(errors="replace")
            else:
                reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover - stream already in use
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _make_streams_encodable()
    ap = build_parser()
    a = ap.parse_args(list(argv) if argv is not None else None)
    try:
        return int(a.func(a))
    except (ReadError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
