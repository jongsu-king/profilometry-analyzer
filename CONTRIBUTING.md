# Contributing

Bug reports with a project file are the most useful: run the analysis with
`--formats json` (or EXPORT in the application) and attach the `.json` and, if you can
share it, the exported scan. Real exported files are especially welcome - the CSV
layout of instrument software is only supported through generic rules until sample
files are available.

## Development

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,gui]"
ruff format src tests examples packaging && ruff check src tests examples packaging
mypy
QT_QPA_PLATFORM=offscreen pytest
```

Rules:

* New analysis behaviour needs a synthetic test with a known truth
  (`profilometry_analyzer.synthetic`) or an analytical check.
* Every number that depends on a standard, a data sheet or a paper cites its source in
  the code; values that could not be checked are marked as such.
* No GPL code (for example from Gwyddion) may be copied or translated into this
  project, which is MIT licensed.
* Results never change the input profile; each processing step adds a history entry.
* GUI tests close and delete their windows (`close()`, `deleteLater()`, `processEvents()`).
