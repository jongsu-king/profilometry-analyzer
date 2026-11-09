#!/usr/bin/env bash
# Build a DMG from dist/Profilometry Analyzer.app (run after pyinstaller).
#   packaging/build_mac_dmg.sh <version> <output-dir>
set -euo pipefail
VERSION="${1:?version}"
OUT="${2:?output dir}"
APP="dist/Profilometry Analyzer.app"
ARCH="$(uname -m)"
NAME="Profilometry-Analyzer-${VERSION}-macOS-${ARCH}"
test -d "$APP" || { echo "missing $APP"; exit 1; }
mkdir -p "$OUT"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
cat > "$STAGE/READ ME.txt" <<EOF
Profilometry Analyzer ${VERSION}

Drag "Profilometry Analyzer" onto the Applications folder.

The application is not signed with an Apple Developer certificate. On first start
macOS shows "cannot be opened because the developer cannot be verified": right-click
(or Control-click) the app and choose Open, then Open again. Alternatively run
    xattr -dr com.apple.quarantine "/Applications/Profilometry Analyzer.app"
in Terminal once.

The command-line tool is inside the bundle:
    "/Applications/Profilometry Analyzer.app/Contents/MacOS/profilometry-analyzer" --help
EOF
hdiutil create -volname "Profilometry Analyzer" -srcfolder "$STAGE" -ov -format UDZO -quiet "$OUT/$NAME.dmg"
hdiutil verify -quiet "$OUT/$NAME.dmg"
rm -rf "$STAGE"
shasum -a 256 "$OUT/$NAME.dmg" | tee "$OUT/$NAME.dmg.sha256"
