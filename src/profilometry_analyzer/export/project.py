"""Project files (JSON) that reproduce a result, and ``verify``.

A project file records the program version, every input file (path, size, SHA-256 and
the reader options), the complete settings, the result and its notices.  ``verify``
reads the inputs again, checks that they are byte-identical, repeats the analysis with
the stored settings and compares the key numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..analysis import (
    RoughnessAnalysis,
    StepResult,
    StepSettings,
    StressAnalysis,
    analyze_roughness,
    analyze_step,
    analyze_stress,
)
from ..batch import BatchResult, run_batch
from ..core.roughness import RoughnessSettings
from ..core.stress import StressSettings
from ..instrument import InstrumentSpec
from ..notices import _plain
from ..profile import Profile
from ..readers import read_profile

FORMAT = "profilometry-analyzer project"
FORMAT_VERSION = 1
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_record(profile: Profile) -> dict[str, Any]:
    if not profile.path:
        return {"path": None, "name": profile.name, "note": "not read from a file"}
    p = Path(profile.path)
    return {
        "path": str(p.resolve()),
        "name": p.name,
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "sha256": sha256(p) if p.is_file() else None,
        "reader_options": dict(profile.reader.get("options", {})),
        "block_index": profile.reader.get("block_index"),
    }


def _envelope(
    analysis: str,
    inputs: list[dict[str, Any]],
    settings: dict[str, Any],
    result: dict[str, Any],
    instrument: InstrumentSpec | None,
) -> dict[str, Any]:
    return _plain(
        {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "program": "profilometry-analyzer",
            "version": __version__,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "platform": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "system": platform.system(),
            },
            "analysis": analysis,
            "inputs": inputs,
            "settings": settings,
            "instrument": instrument.to_dict() if instrument else None,
            "result": result,
        }
    )


def project_dict(obj: Any, instrument: InstrumentSpec | None = None) -> dict[str, Any]:
    """The project dictionary of a step, roughness, stress or batch result."""
    if isinstance(obj, StepResult):
        return _envelope(
            "step", [input_record(obj.source)], obj.settings.to_dict(), obj.to_dict(), instrument
        )
    if isinstance(obj, RoughnessAnalysis):
        settings = {
            **obj.settings.to_dict(),
            "despike": obj.despiked,
            "crop_um": _crop_of(obj.profile),
        }
        return _envelope("roughness", [input_record(obj.source)], settings, obj.to_dict(), None)
    if isinstance(obj, StressAnalysis):
        inputs = [input_record(s) for s in obj.sources]
        settings = {**obj.settings.to_dict(), "has_pre": obj.pre is not None}
        return _envelope("stress", inputs, settings, obj.to_dict(), None)
    if isinstance(obj, BatchResult):
        inputs = []
        for it in obj.items:
            rec: dict[str, Any] = (
                input_record(it.result.source) if it.result else {"path": str(it.path)}
            )
            rec["sample"] = it.sample
            rec["error"] = it.error
            inputs.append(rec)
        result = {
            "scans": obj.scan_rows(),
            "samples": obj.sample_rows(),
            "key_numbers": obj.key_numbers(),
        }
        settings = {**obj.settings.to_dict(), "grouping": obj.grouping}
        return _envelope("batch", inputs, settings, result, instrument)
    raise TypeError(f"cannot store {type(obj).__name__} in a project file")


def _crop_of(profile: Profile) -> list[float] | None:
    for h in profile.history:
        if h.get("step") == "crop":
            return [h["x_start_um"], h["x_stop_um"]]
    return None


def save_project(obj: Any, path: str | Path, instrument: InstrumentSpec | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = project_dict(obj, instrument)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    return p


def load_project(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} file")
    return data


# ---------------------------------------------------------------- verify
@dataclass
class VerifyReport:
    ok: bool
    analysis: str
    compared: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    file_problems: list[str] = field(default_factory=list)
    version_stored: str = ""
    version_now: str = __version__

    def text(self) -> str:
        lines = [
            f"verify {self.analysis}: {'OK' if self.ok else 'FAILED'} "
            f"({self.compared} numbers compared; stored with {self.version_stored},"
            f" now {self.version_now})"
        ]
        lines += [f"  file: {p}" for p in self.file_problems]
        for m in self.mismatches:
            lines.append(f"  {m['key']}: stored {m['stored']!r}, now {m['now']!r}")
        return "\n".join(lines)


def _resolve(rec: dict[str, Any], project_path: Path) -> Path | None:
    for cand in (rec.get("path"), rec.get("name")):
        if not cand:
            continue
        p = Path(cand)
        if p.is_file():
            return p
        q = project_path.parent / p.name
        if q.is_file():
            return q
    return None


def _same(a: Any, b: Any, rtol: float, atol: float) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return True
        return abs(fa - fb) <= atol + rtol * max(abs(fa), abs(fb))
    return bool(a == b)


def compare(stored: dict[str, Any], now: dict[str, Any], rtol: float, atol: float) -> list[dict]:
    out = []
    for k in sorted(set(stored) | set(now)):
        a, b = stored.get(k), now.get(k)
        if not _same(a, b, rtol, atol):
            out.append({"key": k, "stored": a, "now": b})
    return out


def _read(rec: dict[str, Any], project_path: Path, problems: list[str]) -> Profile:
    p = _resolve(rec, project_path)
    if p is None:
        raise FileNotFoundError(f"input not found: {rec.get('path')}")
    if rec.get("sha256") and sha256(p) != rec["sha256"]:
        problems.append(f"{p} differs from the file that was analysed (SHA-256)")
    opts = dict(rec.get("reader_options") or {})
    if rec.get("block_index") is not None and "block" not in opts:
        opts["block"] = rec["block_index"]
    return read_profile(p, **opts)


def recompute(data: dict[str, Any], project_path: Path, problems: list[str]) -> dict[str, Any]:
    """Repeat the stored analysis; returns the new key numbers."""
    kind = data["analysis"]
    st = dict(data.get("settings") or {})
    inst = InstrumentSpec.from_dict(data.get("instrument"))
    if kind == "step":
        prof = _read(data["inputs"][0], project_path, problems)
        return analyze_step(prof, StepSettings.from_dict(st), inst).key_numbers()
    if kind == "roughness":
        prof = _read(data["inputs"][0], project_path, problems)
        crop = st.pop("crop_um", None)
        desp = bool(st.pop("despike", True))
        res = analyze_roughness(
            prof,
            RoughnessSettings.from_dict(st),
            despike_first=desp,
            crop_um=tuple(crop) if crop else None,
        )
        return res.key_numbers()
    if kind == "stress":
        has_pre = bool(st.pop("has_pre", False))
        profs = [_read(r, project_path, problems) for r in data["inputs"]]
        pre = profs[0] if has_pre else None
        return analyze_stress(pre, profs[-1], StressSettings.from_dict(st)).key_numbers()
    if kind == "batch":
        grouping = st.pop("grouping", None)
        paths = []
        for rec in data["inputs"]:
            p = _resolve(rec, project_path)
            if p is None:
                problems.append(f"input not found: {rec.get('path')}")
                continue
            if rec.get("sha256") and sha256(p) != rec["sha256"]:
                problems.append(f"{p} differs from the file that was analysed (SHA-256)")
            paths.append(p)
        opts = next((dict(r.get("reader_options") or {}) for r in data["inputs"]), {})
        batch = run_batch(
            paths,
            StepSettings.from_dict(st),
            grouping=grouping,
            reader_options=opts,
            instrument=inst,
        )
        return batch.key_numbers()
    raise ValueError(f"unknown analysis {kind!r}")


def verify(
    path: str | Path, rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL
) -> VerifyReport:
    p = Path(path)
    data = load_project(p)
    problems: list[str] = []
    stored = dict(data["result"].get("key_numbers") or {})
    try:
        now = _plain(recompute(data, p, problems))
    except (FileNotFoundError, ValueError) as exc:
        return VerifyReport(False, data["analysis"], 0, [], [str(exc)], str(data.get("version")))
    mism = compare(stored, now, rtol, atol)
    return VerifyReport(
        ok=not mism and not problems,
        analysis=data["analysis"],
        compared=len(stored),
        mismatches=mism,
        file_problems=problems,
        version_stored=str(data.get("version")),
    )
