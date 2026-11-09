from __future__ import annotations

import json
from pathlib import Path

import pytest

from profilometry_analyzer import __version__
from profilometry_analyzer.analysis import StepSettings, analyze_step
from profilometry_analyzer.batch import run_batch, sample_name
from profilometry_analyzer.cli import main
from profilometry_analyzer.export.project import load_project, save_project, verify
from profilometry_analyzer.export.tables import write_csv, write_xlsx
from profilometry_analyzer.readers import read_profile
from profilometry_analyzer.synthetic import demo_set
from profilometry_analyzer.update_check import UpdateCheckError, check_for_update, is_newer


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    demo_set(out)
    return out


@pytest.mark.parametrize(
    ("stem", "sample"),
    [
        ("filmA_1", "filmA"),
        ("filmA-pt2", "filmA"),
        ("filmA (3)", "filmA"),
        ("film2", "film2"),
        ("film2_pos3", "film2"),
        ("S1 #4", "S1"),
        ("plain", "plain"),
    ],
)
def test_sample_names(stem: str, sample: str) -> None:
    assert sample_name(f"{stem}.csv") == sample


def test_custom_grouping() -> None:
    assert sample_name("run7_waferB_x3.csv", r"^run\d+_(?P<sample>wafer\w)_") == "waferB"
    assert sample_name("d/filmA_1.csv", "folder") == "d"
    assert sample_name("filmA_1.csv", "none") == "filmA_1"


def test_batch_means(demo: Path) -> None:
    res = run_batch(demo)
    by = {s.sample: s for s in res.samples}
    truth = json.loads((demo / "truth.json").read_text())
    for s, expect in (("filmA", 85.0), ("filmB", 152.0), ("filmC", 310.0)):
        assert by[s].n == 3
        assert by[s].mean_nm == pytest.approx(expect, abs=0.2)
        assert by[s].std_nm == pytest.approx(0.6, abs=0.15)  # truth: 0.6 nm spacing
    assert by["scratch"].mean_nm == pytest.approx(truth["scratch_1.csv"], abs=0.5)
    assert "groove" in by["scratch"].kinds


def test_batch_reports_failed_files(tmp_path: Path) -> None:
    (tmp_path / "bad_1.csv").write_text("nothing here\n")
    res = run_batch(tmp_path)
    assert res.items[0].error and res.samples[0].n_failed == 1


def test_project_roundtrip_and_verify(demo: Path, tmp_path: Path) -> None:
    res = analyze_step(read_profile(demo / "scratch_1.csv"), StepSettings(bow_order=2))
    pj = save_project(res, tmp_path / "p.json")
    data = load_project(pj)
    assert data["version"] == __version__ and data["analysis"] == "step"
    assert data["inputs"][0]["sha256"] and data["settings"]["bow_order"] == 2
    rep = verify(pj)
    assert rep.ok, rep.text()
    # a changed stored number is found
    data["result"]["key_numbers"]["value_nm"] += 1.0
    pj.write_text(json.dumps(data))
    rep = verify(pj)
    assert not rep.ok and rep.mismatches[0]["key"] == "value_nm"


def test_verify_detects_changed_input(tmp_path: Path, edge_csv: Path) -> None:
    res = analyze_step(read_profile(edge_csv))
    pj = save_project(res, tmp_path / "e.json")
    edge_csv.write_text(edge_csv.read_text() + "\n")
    rep = verify(pj)
    assert not rep.ok and rep.file_problems


def test_tables(tmp_path: Path) -> None:
    rows = [{"a": 1.5, "b": None, "c": [1, 2]}, {"a": float("nan"), "d": "x"}]
    p = write_csv(tmp_path / "t.csv", rows)
    text = p.read_text(encoding="utf-8-sig").splitlines()
    assert text[0] == "a,b,c,d" and text[1].startswith("1.5,,")
    x = write_xlsx(tmp_path / "t.xlsx", {"One": rows, "Two": []})
    from openpyxl import load_workbook

    wb = load_workbook(x)
    assert wb.sheetnames == ["One", "Two"]


# ------------------------------------------------------------------ command line
def test_cli_commands(demo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = str(demo / "filmB_2.csv")
    assert main(["info", f]) == 0
    assert "stylus_force_mg: 3.0" in capsys.readouterr().out
    out = tmp_path / "o"
    assert main(["level", f, "-o", str(out)]) == 0
    assert (out / "filmB_2_levelled.csv").is_file()
    assert main(["step", f, "-o", str(out), "--formats", "csv,xlsx,json,png,svg"]) == 0
    text = capsys.readouterr().out
    assert "step = 152" in text
    for suffix in ("_step.csv", "_step.xlsx", "_step.json", "_step.png", "_step.svg"):
        assert (out / f"filmB_2{suffix}").is_file(), suffix
    assert main(["verify", str(out / "filmB_2_step.json")]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["step", f, "--regions", "100:400,600:900", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["measurement"]["method"] == "regions"


def test_cli_roughness_stress_batch(
    demo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "o"
    r = str(demo / "roughness" / "sine_A12nm_40um.csv")
    assert main(["roughness", r, "--lc", "800", "--ls", "0", "-o", str(out)]) == 0
    assert "Ra  = 7.6" in capsys.readouterr().out
    pre, post = str(demo / "stress" / "wafer_pre.csv"), str(demo / "stress" / "wafer_post.csv")
    assert (
        main(["stress", "--pre", pre, "--post", post, "--hs", "525", "--hf", "500", "-o", str(out)])
        == 0
    )
    assert "tensile" in capsys.readouterr().out
    assert main(["batch", str(demo), "-o", str(out / "b"), "--plots"]) == 0
    text = capsys.readouterr().out
    assert "filmA" in text and (out / "b" / "batch.xlsx").is_file()
    assert (out / "b" / "scans" / "filmA_1_step.png").is_file()
    assert (
        main(
            [
                "verify",
                str(out / "b" / "batch.json"),
                str(out / "sine_A12nm_40um_roughness.json"),
                str(out / "wafer_post_stress.json"),
            ]
        )
        == 0
    )
    assert main(["substrates"]) == 0
    assert "UNVERIFIED" in capsys.readouterr().out


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", str(tmp_path / "missing.csv")]) == 1
    (tmp_path / "x.opdx").write_bytes(b"\0")
    assert main(["step", str(tmp_path / "x.opdx")]) == 1
    assert "OPDx" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["step", "a.csv", "--crop", "nonsense"])


def test_cli_synth(tmp_path: Path) -> None:
    assert main(["synth", "--out", str(tmp_path / "s")]) == 0
    assert (tmp_path / "s" / "truth.json").is_file()


# ------------------------------------------------------------------ update check
def test_update_check_offline_and_compare() -> None:
    assert is_newer("v0.2.0", "0.1.0") and not is_newer("0.1.0", "0.1.0")
    assert not is_newer("0.1.0rc1", "0.1.0")

    def fake(url: str, timeout: float) -> dict:
        assert "jongsu-king/profilometry-analyzer" in url
        return {
            "tag_name": "v9.0.0",
            "assets": [
                {
                    "name": "Profilometry-Analyzer-9.0.0-macOS-arm64.dmg",
                    "browser_download_url": "https://x/y.dmg",
                }
            ],
        }

    info = check_for_update("0.1.0", fetch=fake)
    assert info.is_newer and info.latest == "9.0.0"

    def offline(url: str, timeout: float) -> dict:
        raise UpdateCheckError("offline")

    with pytest.raises(UpdateCheckError):
        check_for_update("0.1.0", fetch=offline)
