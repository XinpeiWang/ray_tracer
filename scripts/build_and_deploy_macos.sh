#!/usr/bin/env bash
# Build and package the Ray Tracer for macOS: CPU renderer + CLI + Qt GUI,
# bundled into a signed-or-not RayTracerGUI.app and a distributable
# RayTracerGUI.dmg. Run this ON macOS - it is not usable from Windows.
#
# Mirrors scripts/build_and_deploy.ps1's job on Windows. There is no
# CUDA/OptiX GPU renderer to build here - gpu/optix/ and optix_renderer/
# are CUDA/OptiX-only with no macOS equivalent - but this DOES build the
# real Metal GPU backend (gpu/metal/, docs/METAL_GPU_FEASIBILITY.md),
# fully integrated into ray_tracer's own --gpu dispatch since PR #84 -
# passing -DRT_BUILD_METAL=ON below is what makes that actually happen;
# without it, RT_HAVE_METAL is never defined and the packaged app is
# silently CPU-only regardless of what the GUI's own "Renderer: GPU"
# option looks like (a real, previously-shipped bug this comment used to
# describe as "no GPU at all on macOS" - stale ever since Metal support
# landed, and exactly the mistake that produced a "gpu-ui-preview"-named
# .dmg with no actual Metal support inside it). This script only ever
# touches the root CMakeLists.txt (cpu_renderer, ray_tracer,
# scene_metadata, metal_renderer) and qt_gui/RayTracerGUI.pro.
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
# Lights/Cameras/Volumes/Geometry/Textures - the large majority of the
# registry, procedurally generated) works out of the box from the installed .app with
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

# The single most important line in this whole script, found the hard way
# (section 113 in docs/METAL_GPU_FEASIBILITY.md): cpu_renderer/ray_tracer/
# scene_metadata (built via CMake below) and RayTracerGUI (built via qmake
# a few steps down) must come out the SAME architecture, or the GUI's own
# dlopen() of scene_metadata.dylib fails outright with "incompatible
# architecture" - a real failure a real user hit, invisible to every
# check this script ran before (it never launched the actual GUI binary
# and inspected its own dialog, only the CLI). An EARLIER version of this
# script forced CMake to `$(uname -m)` (the HOST's own native
# architecture) on the theory that only cmake itself could be the
# Rosetta-translated, wrong-architecture one - true on SOME machines, but
# backwards on this one: `qmake`/its whole Qt installation are the
# x86_64-only (non-universal) side here, and there is no native arm64 Qt
# install anywhere on this machine to point at instead (installing one is
# a real, separate, slower undertaking - not something this script does
# for you). Querying qmake's OWN binary architecture directly and using
# THAT for both builds - whichever direction the mismatch actually runs -
# is the only fix that works regardless of which build machine this runs
# on. `lipo -archs` on a Rosetta-translated x86_64 tool still reports
# "x86_64" correctly (lipo inspects the FILE's own architecture, not the
# architecture of the process executing lipo itself).
QT_TARGET_ARCH="$(lipo -archs "$(command -v qmake)" 2>/dev/null | awk '{print $1}')"
[[ -n "$QT_TARGET_ARCH" ]] || { echo "ERROR: could not determine qmake's own architecture via 'lipo -archs'" >&2; exit 1; }
if [[ "$QT_TARGET_ARCH" != "$(uname -m)" ]]; then
	echo "NOTE: this Qt install is $QT_TARGET_ARCH, not this Mac's native $(uname -m) - building"
	echo "      everything as $QT_TARGET_ARCH to match (runs translated/emulated, slower, but the"
	echo "      only way to keep the GUI and CLI/dylib loadable together without installing a"
	echo "      native-architecture Qt)."
fi

echo
echo "[1/5] Building cpu_renderer + ray_tracer CLI + scene_metadata (CMake)..."
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="$QT_TARGET_ARCH" -DRT_BUILD_METAL=ON
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

# RayTracerGUI.pro's own DESTDIR ($$PWD/../RayTracer_Package) is NOT
# inside $GUI_BUILD_DIR - it's unconditional (no macx{}/win32{} split) and
# points at the same RayTracer_Package/ folder the Windows packager uses,
# so the built .app lands there regardless of qmake's own build directory.
APP_BUNDLE="$REPO_ROOT/RayTracer_Package/$APP_NAME.app"
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

# Deliberately NEVER passed -dmg here, even when SKIP_DMG=0 - macdeployqt's
# own -dmg flag builds the .dmg from Contents/MacOS/'s CURRENT contents
# at the moment it runs, which is BEFORE the metal_poc.metal/models/images
# copy below. A real, previously-shipped bug found by mounting the actual
# .dmg this script produced (not just checking the loose .app folder,
# which - copied from $APP_BUNDLE in step 5, AFTER the asset copy below -
# looked correct while the real .dmg silently did not): the first "fixed"
# release .dmg (section 110) still had no Metal shader/demo assets in it
# at all. The dmg is now built explicitly via hdiutil, further below,
# once every asset this app needs is actually in place.
"$MACDEPLOYQT" "$APP_BUNDLE"

# metal_poc.mm compiles its own Metal shader from SOURCE at runtime (it has
# no offline .metallib step) - only when RT_BUILD_METAL=ON above actually
# defined RT_HAVE_METAL. Its own RT_METAL_SHADER_DIR fallback is a compile-
# time absolute path into THIS machine's own source tree, meaningless once
# ray_tracer is copied anywhere else - copying the real shader source next
# to the bundled CLI here is what makes metal_poc.mm's own "next to the
# running executable" lookup (section 110) find it on an install machine
# that never had this repo checked out at all. Deliberately done AFTER
# macdeployqt above, not alongside the ray_tracer/scene_metadata.dylib
# copy: macdeployqt otool-scans every file under Contents/MacOS/ looking
# for Mach-O binaries to rewrite/codesign, and a plain-text .metal source
# file there makes it print a real (if ultimately harmless) "Could not
# parse otool output" error - found by actually running this script after
# adding the copy, not assumed safe.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && grep -q "RT_BUILD_METAL:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
	cp "$REPO_ROOT/gpu/metal/metal_poc.metal" "$APP_BUNDLE/Contents/MacOS/metal_poc.metal"
fi

# The hardcoded-room demo scene's own small, ALWAYS-needed assets (unlike
# the large external scene meshes the reminder below deliberately skips) -
# models/suzanne.obj + models/spot.obj (~370KB combined) and
# images/earthmap.jpg (~940KB). Same RT_MODELS_DIR-points-at-the-build-
# machine problem as the Metal shader above (section 110): without these
# bundled here too, a genuinely different install machine would silently
# render every scene missing Suzanne/Spot/the back-wall texture, degraded
# but not crashing (loadObjMesh()/earthPixels both have a graceful
# fallback) - found the same way, by testing a relocated package rather
# than trusting the code alone.
mkdir -p "$APP_BUNDLE/Contents/MacOS/models" "$APP_BUNDLE/Contents/MacOS/images"
cp "$REPO_ROOT/models/suzanne.obj" "$APP_BUNDLE/Contents/MacOS/models/suzanne.obj"
cp "$REPO_ROOT/models/spot.obj" "$APP_BUNDLE/Contents/MacOS/models/spot.obj"
cp "$REPO_ROOT/images/earthmap.jpg" "$APP_BUNDLE/Contents/MacOS/images/earthmap.jpg"

echo
echo "[5/5] Collecting output..."
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"
cp -R "$APP_BUNDLE" "$DEPLOY_DIR/"
if [[ "$SKIP_DMG" -eq 0 ]]; then
	# Built explicitly via hdiutil now, from the FULLY-ASSEMBLED $APP_BUNDLE
	# (Qt frameworks + Metal shader + demo assets all already copied in
	# above) - not macdeployqt's own -dmg flag, which packages whatever is
	# in Contents/MacOS/ at the moment IT runs, too early for this script's
	# own later asset-copy steps (see the comment above the plain
	# macdeployqt call for the real bug this caused).
	DMG_PATH="$(dirname "$APP_BUNDLE")/$APP_NAME.dmg"
	rm -f "$DMG_PATH"
	hdiutil create -volname "$APP_NAME" -srcfolder "$APP_BUNDLE" -ov -format UDZO "$DMG_PATH" >/dev/null
	if [[ -f "$DMG_PATH" ]]; then
		cp "$DMG_PATH" "$DEPLOY_DIR/"
		echo "DMG:  $DEPLOY_DIR/$APP_NAME.dmg"
	else
		echo "WARNING: hdiutil did not produce $DMG_PATH - check its output above." >&2
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
