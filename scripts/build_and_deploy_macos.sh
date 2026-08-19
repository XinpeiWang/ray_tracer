#!/usr/bin/env bash
# Build and package the Ray Tracer for macOS: CPU renderer + CLI + Qt GUI,
# bundled into a signed-or-not RayTracerGUI.app and a distributable
# RayTracerGUI.dmg. Run this ON macOS - it is not usable from Windows.
#
# Mirrors scripts/build_and_deploy.ps1's job on Windows, but there is no GPU
# renderer to build here at all: gpu/optix/ and optix_renderer/ are
# CUDA/OptiX-only with no macOS equivalent (see README.md's "macOS
# (CPU-only)" section). This script only ever touches the root
# CMakeLists.txt (cpu_renderer, ray_tracer, scene_metadata) and
# qt_gui/RayTracerGUI.pro.
#
# IMPORTANT - external mesh assets ("Large Scenes" / most "Models" category
# scenes - anything with requires_files=true in scene_registry.h) are NOT
# bundled into the .app or .dmg by this script:
#   - Many are hundreds of MB to 1GB+ (Sponza, Bistro, San Miguel, Power
#     Plant, ...), which would balloon the installer for assets most users
#     won't render.
#   - Some carry non-commercial-only licenses (e.g. Power Plant) that make
#     redistributing them inside an installer questionable even if the size
#     were fine.
# Every scene that does NOT require external files (Basics/Materials/
# Lights/Cameras/Volumes/Geometry - the large majority of the registry,
# procedurally generated) works out of the box from the installed .app with
# no extra setup. To also render the external-asset scenes, copy this
# repo's models/ directory into the installed app bundle yourself:
#   cp -R /path/to/ray_tracer/models "/Applications/RayTracerGUI.app/Contents/MacOS/models"
# (that exact path - Contents/MacOS/ - matches where the app looks: see
# mainwindow.cpp's setWorkingDirectory(applicationDirPath()) and
# launcher/main.cpp's/gpu's kSearchPrefixes, whose first entry is "models/"
# relative to the CLI's own working directory).
#
# Usage:
#   ./scripts/build_and_deploy_macos.sh [--skip-dmg]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build_macos"
APP_NAME="RayTracerGUI"
DEPLOY_DIR="$REPO_ROOT/RayTracer_Package_macOS"
SKIP_DMG=0

for arg in "$@"; do
	case "$arg" in
		--skip-dmg) SKIP_DMG=1 ;;
		*) echo "Unknown argument: $arg" >&2; exit 1 ;;
	esac
done

command -v cmake >/dev/null || { echo "ERROR: cmake not found on PATH" >&2; exit 1; }
command -v qmake >/dev/null || { echo "ERROR: qmake not found on PATH - add Qt's bin dir, e.g. \$HOME/Qt/6.x.y/macos/bin" >&2; exit 1; }

echo "========================================"
echo "Ray Tracer - macOS build + package"
echo "========================================"

echo
echo "[1/5] Building cpu_renderer + ray_tracer CLI + scene_metadata (CMake)..."
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --config Release -j"$(sysctl -n hw.ncpu)"

CLI_BIN="$BUILD_DIR/ray_tracer"
SCENE_METADATA_LIB="$BUILD_DIR/scene_metadata.dylib"
[[ -f "$CLI_BIN" ]] || { echo "ERROR: $CLI_BIN not found after build" >&2; exit 1; }
[[ -f "$SCENE_METADATA_LIB" ]] || { echo "ERROR: $SCENE_METADATA_LIB not found after build" >&2; exit 1; }

echo
echo "[2/5] Building Qt GUI (qmake + make)..."
GUI_BUILD_DIR="$REPO_ROOT/qt_gui/build_macos"
mkdir -p "$GUI_BUILD_DIR"
( cd "$GUI_BUILD_DIR" && qmake ../RayTracerGUI.pro CONFIG+=release && make -j"$(sysctl -n hw.ncpu)" )

APP_BUNDLE="$GUI_BUILD_DIR/$APP_NAME.app"
[[ -d "$APP_BUNDLE" ]] || { echo "ERROR: $APP_BUNDLE not found after build" >&2; exit 1; }

echo
echo "[3/5] Copying ray_tracer CLI + scene_metadata.dylib into the app bundle..."
# Contents/MacOS/ specifically - QCoreApplication::applicationDirPath() for a
# bundled Mac app resolves there, and that is what both the GUI's subprocess
# working directory and scene_metadata_client.cpp's dlopen() call use to
# find these two files at runtime.
cp "$CLI_BIN" "$APP_BUNDLE/Contents/MacOS/ray_tracer"
cp "$SCENE_METADATA_LIB" "$APP_BUNDLE/Contents/MacOS/scene_metadata.dylib"
chmod +x "$APP_BUNDLE/Contents/MacOS/ray_tracer"

echo
echo "[4/5] Running macdeployqt to bundle Qt frameworks..."
QMAKE_PATH="$(command -v qmake)"
QT_BIN_DIR="$(dirname "$QMAKE_PATH")"
MACDEPLOYQT="$QT_BIN_DIR/macdeployqt"
[[ -x "$MACDEPLOYQT" ]] || { echo "ERROR: macdeployqt not found next to qmake at $MACDEPLOYQT" >&2; exit 1; }

if [[ "$SKIP_DMG" -eq 1 ]]; then
	"$MACDEPLOYQT" "$APP_BUNDLE"
else
	"$MACDEPLOYQT" "$APP_BUNDLE" -dmg
fi

echo
echo "[5/5] Collecting output..."
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
cp -R "$APP_BUNDLE" "$DEPLOY_DIR/"
if [[ "$SKIP_DMG" -eq 0 ]]; then
	DMG_PATH="$GUI_BUILD_DIR/$APP_NAME.dmg"
	if [[ -f "$DMG_PATH" ]]; then
		cp "$DMG_PATH" "$DEPLOY_DIR/"
		echo "DMG:  $DEPLOY_DIR/$APP_NAME.dmg"
	else
		echo "WARNING: macdeployqt -dmg did not produce $DMG_PATH - check its output above." >&2
	fi
fi

echo
echo "========================================"
echo "Done."
echo "App:  $DEPLOY_DIR/$APP_NAME.app"
echo "========================================"
echo "Reminder: scenes that need external mesh files (Sponza, Bistro, the"
echo "H-family large environments, most single-model scenes, ...) need"
echo "models/ copied into the installed app's Contents/MacOS/models/ - see"
echo "this script's own header comment for the exact command and why those"
echo "assets aren't bundled automatically."
