#!/bin/bash
# Copies RayTracerGUI.app from this disk image to /Applications, removes the "downloaded from the internet" quarantine flag from the copy, and opens it.
# The app is signed ad hoc, not notarized by Apple, so macOS refuses to open a quarantined copy ("Apple cannot check it for malicious software").
# Removing the flag (`xattr -dr com.apple.quarantine`; not `xattr -cr`, which would also strip the code-signature attributes the bundle stores on some files) is the standard way past that for an app you trust.
#
# If macOS refuses to run THIS file as well, open Terminal and run:   bash "/Volumes/RayTracerGUI/Install RayTracerGUI.command"
# Set INSTALL_DIR to install somewhere other than /Applications.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
APP_SRC="$HERE/RayTracerGUI.app"
DEST_DIR="${INSTALL_DIR:-/Applications}"
APP_DST="$DEST_DIR/RayTracerGUI.app"

if [[ ! -d "$APP_SRC" ]]; then
	echo "Cannot find RayTracerGUI.app next to this script ($HERE)." >&2
	exit 1
fi
echo "Installing RayTracerGUI to $DEST_DIR ..."
if [[ -d "$APP_DST" ]]; then
	echo "Replacing the copy that is already there."
	rm -rf "$APP_DST"
fi
cp -R "$APP_SRC" "$APP_DST"
xattr -dr com.apple.quarantine "$APP_DST"
echo "Done. Opening RayTracerGUI ..."
if [[ -z "$NO_OPEN" ]]; then open "$APP_DST"; fi
