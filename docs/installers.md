# Installers and releases

The workflow `.github/workflows/installers.yml` runs on every `v*` tag (and by hand):

| platform | files | built with |
|---|---|---|
| Windows x64 | `Profilometry-Analyzer-<v>-Windows-x64-Setup.exe`, `…-Windows-x64.zip` | PyInstaller + Inno Setup |
| macOS arm64 | `Profilometry-Analyzer-<v>-macOS-arm64.dmg` | PyInstaller + `hdiutil` |
| Linux x64 | `profilometry-analyzer_<v>_amd64.deb`, `…-Linux-x86_64.tar.gz` | PyInstaller + `dpkg-deb` |

Each build contains the desktop application and the command-line tool
(`profilometry-analyzer`) sharing one runtime. Before packaging, the frozen CLI writes the
synthetic scans, analyses a groove and verifies the project file, and the GUI is started
headless (`PROFILOMETRY_ANALYZER_SMOKE=1`). The Windows installer is installed and
uninstalled silently, the deb is installed with apt, the DMG is mounted and the CLI run.
`SHA256SUMS.txt` is attached to the release.

The builds are not code-signed. macOS: right-click the app and choose *Open* the first
time (or `xattr -dr com.apple.quarantine "/Applications/Profilometry Analyzer.app"`).

## Release steps (maintainer)

1. Zenodo: switch the repository on at <https://zenodo.org/account/settings/github/>
   **before** publishing the release (Zenodo archives only releases published after the
   webhook exists).
2. PyPI: register a pending trusted publisher on pypi.org for project
   `profilometry-analyzer`, owner `jongsu-king`, repository `profilometry-analyzer`,
   workflow `publish.yml`, environment `pypi`; then set the repository variable
   `PYPI_TRUSTED_PUBLISHER=configured`. No token is stored anywhere.
3. Tag and release: `git tag -a v0.1.0 -m v0.1.0 && git push origin v0.1.0`, then
   `gh release create v0.1.0 --verify-tag`. The tag starts CI and the installers; the
   published release starts `publish.yml` and Zenodo.

## Local build

```bash
pip install -e ".[gui]" pyinstaller
pyinstaller packaging/profilometry_analyzer.spec --noconfirm
packaging/build_mac_dmg.sh 0.1.0 installers      # macOS
packaging/build_linux_deb.sh 0.1.0 installers    # Linux
```
