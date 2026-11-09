## What this changes

<!-- One paragraph: the user-visible effect and why. -->

## Checks

- [ ] `ruff format --check src tests examples packaging && ruff check src tests examples packaging && mypy && pytest` pass locally
- [ ] New behaviour has a synthetic-data test with known truth or an analytical check
- [ ] A project file written by the new code passes `profilometry-analyzer verify`
- [ ] Reference values cite their source; unchecked values are marked
- [ ] No GPL code copied or translated
- [ ] `CHANGELOG.md` has an entry
