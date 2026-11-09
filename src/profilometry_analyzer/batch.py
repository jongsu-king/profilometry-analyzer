"""Batch step-height analysis: every scan of a folder, grouped by sample.

The sample of a file comes from its name: a trailing scan index separated by ``_``,
``-``, ``.``, ``#`` or a space (optionally ``p``, ``pt``, ``pos``, ``point``, ``spot``,
``scan`` before the number), or in brackets, is removed - ``filmA_1``, ``filmA-pt2``,
``filmA (3)`` all belong to ``filmA``; ``film2`` stays ``film2``.  A regular expression
with a named group ``sample`` replaces the rule, ``folder`` groups by the parent folder
and ``none`` treats every file as its own sample.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import notices as N
from .analysis import StepResult, StepSettings, analyze_step
from .instrument import DEFAULT_INSTRUMENT, InstrumentSpec
from .readers import SUPPORTED_SUFFIXES, read_profile

DEFAULT_GROUP_PATTERN = (
    r"^(?P<sample>.+?)(?:[ _\-#.]+(?:p|pt|pos|point|spot|scan)?\d+|\s*\(\d+\))?$"
)


def sample_name(path: str | Path, grouping: str | None = None) -> str:
    p = Path(path)
    if grouping == "none":
        return p.stem
    if grouping == "folder":
        return p.parent.name or p.stem
    pattern = grouping or DEFAULT_GROUP_PATTERN
    m = re.match(pattern, p.stem, flags=re.IGNORECASE)
    if m and m.groupdict().get("sample"):
        return m.group("sample")
    return p.stem


def find_files(folder: str | Path, recursive: bool = False) -> list[Path]:
    root = Path(folder)
    it: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    files = [
        p
        for p in it
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    ]
    return sorted(files)


@dataclass
class BatchItem:
    path: Path
    sample: str
    result: StepResult | None = None
    error: str | None = None

    def row(self) -> dict[str, Any]:
        if self.result is None:
            return {
                "file": str(self.path),
                "sample": self.sample,
                "status": "failed",
                "error": self.error,
            }
        return {"sample": self.sample, **self.result.row(), "error": None}


@dataclass
class SampleSummary:
    sample: str
    n: int
    n_failed: int
    mean_nm: float | None
    std_nm: float | None
    sem_nm: float | None
    min_nm: float | None
    max_nm: float | None
    mean_u_nm: float | None
    kinds: list[str]
    warnings: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "n": self.n,
            "n_failed": self.n_failed,
            "thickness_mean_nm": self.mean_nm,
            "thickness_std_nm": self.std_nm,
            "thickness_sem_nm": self.sem_nm,
            "thickness_min_nm": self.min_nm,
            "thickness_max_nm": self.max_nm,
            "mean_single_scan_u_nm": self.mean_u_nm,
            "kinds": ";".join(self.kinds),
            "warnings": ";".join(self.warnings),
        }


@dataclass
class BatchResult:
    items: list[BatchItem]
    samples: list[SampleSummary]
    settings: StepSettings
    grouping: str | None
    folder: str | None = None

    def scan_rows(self) -> list[dict[str, Any]]:
        return [it.row() for it in self.items]

    def sample_rows(self) -> list[dict[str, Any]]:
        return [s.row() for s in self.samples]

    def key_numbers(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for it in self.items:
            if it.result is not None:
                out[f"{it.path.name}:thickness_nm"] = it.result.thickness_nm
                out[f"{it.path.name}:u_nm"] = it.result.measurement.u_nm
        for s in self.samples:
            out[f"sample {s.sample}:mean_nm"] = s.mean_nm
            out[f"sample {s.sample}:std_nm"] = s.std_nm
        return out


def summarize(items: list[BatchItem]) -> list[SampleSummary]:
    groups: dict[str, list[BatchItem]] = {}
    for it in items:
        groups.setdefault(it.sample, []).append(it)
    out: list[SampleSummary] = []
    for name, its in groups.items():
        ok = [i for i in its if i.result is not None and i.result.thickness_nm is not None]
        t = np.array([float(i.result.thickness_nm) for i in ok if i.result])  # type: ignore[arg-type]
        u = [i.result.measurement.u_nm for i in ok if i.result and i.result.measurement.u_nm]
        warn: list[str] = []
        for i in ok:
            assert i.result is not None
            warn += [n.code for n in i.result.notices if n.severity in (N.WARNING, N.ERROR)]
        kinds = sorted({i.result.measurement.kind for i in its if i.result is not None})
        out.append(
            SampleSummary(
                sample=name,
                n=int(t.size),
                n_failed=len(its) - len(ok),
                mean_nm=float(np.mean(t)) if t.size else None,
                std_nm=float(np.std(t, ddof=1)) if t.size > 1 else None,
                sem_nm=float(np.std(t, ddof=1) / np.sqrt(t.size)) if t.size > 1 else None,
                min_nm=float(np.min(t)) if t.size else None,
                max_nm=float(np.max(t)) if t.size else None,
                mean_u_nm=float(np.mean(u)) if u else None,
                kinds=kinds,
                warnings=list(dict.fromkeys(warn)),
            )
        )
    return out


def run_batch(
    files: Sequence[str | Path] | str | Path,
    settings: StepSettings | None = None,
    *,
    grouping: str | None = None,
    reader_options: dict[str, Any] | None = None,
    instrument: InstrumentSpec = DEFAULT_INSTRUMENT,
    recursive: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> BatchResult:
    """Analyse every file (or every supported file of a folder) with the same settings."""
    folder: str | None = None
    if isinstance(files, str | Path) and Path(files).is_dir():
        folder = str(files)
        paths = find_files(files, recursive=recursive)
    elif isinstance(files, str | Path):
        paths = [Path(files)]
    else:
        paths = [Path(f) for f in files]
    st = settings or StepSettings()
    opts = dict(reader_options or {})
    items: list[BatchItem] = []
    for k, path in enumerate(paths):
        if progress:
            progress(k, len(paths), path.name)
        item = BatchItem(path=path, sample=sample_name(path, grouping))
        try:
            prof = read_profile(path, **opts)
            item.result = analyze_step(prof, st, instrument)
        except (ValueError, OSError) as exc:
            item.error = f"{type(exc).__name__}: {exc}"
        items.append(item)
    return BatchResult(items, summarize(items), st, grouping, folder)
