"""Entry point of the frozen command-line tool (same engine as the desktop app)."""

from __future__ import annotations

import sys

import frozen_env


def main() -> int:
    frozen_env.prepare()
    from profilometry_analyzer.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
