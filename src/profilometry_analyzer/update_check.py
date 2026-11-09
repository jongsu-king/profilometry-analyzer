"""Check GitHub Releases for a newer version.

The check is a single anonymous request to the GitHub REST API
(``/repos/<owner>/<repo>/releases/latest``), sends nothing but the package version
in the ``User-Agent`` header, and fails quietly when offline.  The desktop
application runs it at most once a day and can be told not to
(*Help > Check for updates at startup*, or the environment variable
``PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK=1``); the command line runs it only on request
(``profilometry-analyzer update-check``).
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import __version__

REPOSITORY = "jongsu-king/profilometry-analyzer"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_URL = f"https://github.com/{REPOSITORY}/releases"
ENV_DISABLE = "PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK"
DEFAULT_TIMEOUT = 4.0

Fetcher = Callable[[str, float], dict[str, Any]]


class UpdateCheckError(RuntimeError):
    """The release information could not be retrieved (offline, rate limit, ...)."""


def parse_version(text: str) -> tuple[int, ...]:
    """``"v0.4.1"`` -> ``(0, 4, 1)``; a pre-release suffix sorts below the release.

    ``"0.4.0rc1"`` -> ``(0, 4, 0, -1, 1)`` so that it compares lower than ``(0, 4, 0)``
    padded with zeros.  Unparsable strings raise ``ValueError``.
    """
    m = re.match(r"^v?(\d+(?:\.\d+)*)(?:[-.]?(a|b|rc|alpha|beta|dev)\.?(\d*))?$", text.strip())
    if not m:
        raise ValueError(f"not a version: {text!r}")
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) < 3:
        parts.append(0)
    if m.group(2):
        parts += [-1, int(m.group(3) or 0)]
    return tuple(parts)


def is_newer(latest: str, current: str) -> bool:
    try:
        a, b = parse_version(latest), parse_version(current)
    except ValueError:
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))


@dataclass
class UpdateInfo:
    current: str
    latest: str
    is_newer: bool
    url: str = RELEASES_URL
    published: str = ""
    notes: str = ""
    assets: dict[str, str] = field(default_factory=dict)
    """Asset file name -> download URL."""

    @property
    def platform_asset(self) -> tuple[str, str] | None:
        """``(name, url)`` of the installer for this operating system, if attached."""
        return pick_platform_asset(self.assets)

    def to_dict(self) -> dict[str, Any]:
        asset = self.platform_asset
        return {
            "current": self.current,
            "latest": self.latest,
            "is_newer": self.is_newer,
            "url": self.url,
            "published": self.published,
            "platform_asset": asset[0] if asset else None,
            "platform_asset_url": asset[1] if asset else None,
            "frozen": is_frozen(),
        }


def is_frozen() -> bool:
    """True inside a PyInstaller build (installer users), False for a pip install."""
    return bool(getattr(sys, "frozen", False))


def pick_platform_asset(assets: dict[str, str]) -> tuple[str, str] | None:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        patterns = ["-Windows-x64-Setup.exe", "-Windows-x64.zip"]
    elif system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
        patterns = [f"-macOS-{arch}.dmg", "-macOS-arm64.dmg"]
    else:
        patterns = ["_amd64.deb", "-Linux-x86_64.tar.gz"]
    for pat in patterns:
        for name, url in assets.items():
            if name.endswith(pat):
                return name, url
    return None


def fetch_github(url: str, timeout: float) -> dict[str, Any]:
    """Default fetcher: one GET with a short timeout; raises :class:`UpdateCheckError`."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"profilometry-analyzer/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise UpdateCheckError(f"GitHub answered {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise UpdateCheckError(f"cannot reach GitHub: {exc}") from exc
    if not isinstance(data, dict):
        raise UpdateCheckError("unexpected answer from GitHub")
    return data


def check_for_update(
    current: str = __version__,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Fetcher | None = None,
) -> UpdateInfo:
    """Compare ``current`` with the latest GitHub release.

    ``fetch`` replaces the network call (tests, other hosts); it must return the
    JSON of the ``releases/latest`` endpoint.
    """
    data = (fetch or fetch_github)(API_URL, timeout)
    tag = str(data.get("tag_name") or data.get("name") or "").strip()
    if not tag:
        raise UpdateCheckError("the latest release has no tag")
    latest = tag[1:] if tag.startswith("v") else tag
    assets: dict[str, str] = {}
    for a in data.get("assets") or []:
        name, url = a.get("name"), a.get("browser_download_url")
        if name and url:
            assets[str(name)] = str(url)
    return UpdateInfo(
        current=current,
        latest=latest,
        is_newer=is_newer(latest, current),
        url=str(data.get("html_url") or RELEASES_URL),
        published=str(data.get("published_at") or ""),
        notes=str(data.get("body") or ""),
        assets=assets,
    )


def disabled_by_environment() -> bool:
    """``PROFILOMETRY_ANALYZER_NO_UPDATE_CHECK=1`` or a packaging smoke test disables the check."""
    return os.environ.get(ENV_DISABLE, "").strip() not in ("", "0") or bool(
        os.environ.get("PROFILOMETRY_ANALYZER_SMOKE")
    )


def describe(info: UpdateInfo) -> str:
    """Plain-text summary for the command line and the log."""
    if not info.is_newer:
        return f"profilometry-analyzer {info.current} is up to date (latest release {info.latest})"
    lines = [
        f"profilometry-analyzer {info.latest} is available (you have {info.current})",
        info.url,
    ]
    asset = info.platform_asset
    if is_frozen() and asset:
        lines.append(f"installer for this computer: {asset[0]}")
        lines.append(asset[1])
    elif not is_frozen():
        lines.append('upgrade with:  pip install -U "profilometry-analyzer[gui]"')
    return "\n".join(lines)
