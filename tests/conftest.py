from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from profilometry_analyzer.profile import Profile
from profilometry_analyzer.synthetic import SynthSpec, synthesize, write_csv


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def edge_csv(tmp_path: Path) -> Path:
    syn = synthesize(
        SynthSpec(
            kind="edge",
            height_nm=120.0,
            tilt_nm_per_um=0.05,
            noise_nm=0.5,
            spikes=2,
            tip_radius_um=2.0,
            seed=4,
        )
    )
    return write_csv(
        syn.profile,
        tmp_path / "sample_1.csv",
        metadata={"Stylus Force": "3 mg", "Scan Length": "1000 um", "Stylus Radius": "2 um"},
    )


def make(kind: str = "edge", **kw: object) -> tuple[Profile, dict]:
    syn = synthesize(SynthSpec(kind=kind, **kw))  # type: ignore[arg-type]
    return syn.profile, syn.truth


# ------------------------------------------------------------ clean process exit
_EXIT_STATUS = 0


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """End the test process without the interpreter's finalisation when Qt was used.

    Same safeguard as in xps-autofit-studio: PySide6/Shiboken objects collected while
    the interpreter shuts down have aborted the process on Linux / Python 3.11 after
    every test had passed ("Fatal Python error: bool_dealloc", exit code 134).  The GUI
    tests close and delete their windows while the interpreter is healthy; this hook is
    the last resort and runs after everything has been reported, so the process ends
    with pytest's own exit status.
    """
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        for widget in app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        app.processEvents()
        gc.collect()
        app.processEvents()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
