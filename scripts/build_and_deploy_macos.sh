#!/usr/bin/env bash
# Build and package the Ray Tracer for macOS: CPU renderer + CLI + Qt GUI,
# bundled into a signed-or-not RayTracerGUI.app and a distributable
# RayTracerGUI.dmg. Run this ON macOS - it is not usable from Windows.
#
# Mirrors scripts/build_and_deploy.ps1's job on Windows. There is no
# CUDA/OptiX GPU renderer to build here - gpu/optix/ and optix_renderer/
# are CUDA/OptiX-only with no macOS equivalent - but this DOES build the
# real Metal GPU backend (gpu/metal/, docs/history/METAL_GPU_FEASIBILITY.md),
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
# no extra setup. For the external-asset scenes the GUI offers a "Download
# missing files" button: it fetches them from their original sites into a
# per-user folder (~/Library/Application Support/Ray Tracer/user_assets, see
# RAY_TRACER_USER_ASSETS) that the renderer also searches, so it works from a
# read-only disk image too. Copying files by hand into the app's Contents/MacOS/
# (models/, pbrt_scenes/, ...) still works as well.
#
# Usage:
#   ./scripts/build_and_deploy_macos.sh [--arch native|arm64|x86_64|universal] [--skip-dmg]
#     --arch native     (default) this Mac's own CPU: arm64 on Apple silicon. Fastest to build and run.
#     --arch universal  arm64 + x86_64 in every binary: one dmg for every Mac (about twice the build time).
#     --arch x86_64     Intel build; on Apple silicon it runs under Rosetta.
#   Needs a Qt install that contains the chosen architecture(s) - an official Qt 6 install is universal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="RayTracerGUI"
DEPLOY_DIR="$REPO_ROOT/RayTracer_Package_macOS"
SKIP_DMG=0
ARCH_CHOICE=native

while [[ $# -gt 0 ]]; do
	case "$1" in
		--skip-dmg) SKIP_DMG=1 ;;
		--arch) [[ $# -ge 2 ]] || { echo "--arch needs a value" >&2; exit 1; }; ARCH_CHOICE="$2"; shift ;;
		--arch=*) ARCH_CHOICE="${1#--arch=}" ;;
		*) echo "Unknown argument: $1" >&2; exit 1 ;;
	esac
	shift
done

command -v cmake >/dev/null || { echo "ERROR: cmake not found on PATH" >&2; exit 1; }
command -v qmake >/dev/null || { echo "ERROR: qmake not found on PATH - add Qt's bin dir, e.g. \$HOME/Qt/6.x.y/macos/bin" >&2; exit 1; }

echo "========================================"
echo "Ray Tracer - macOS build + package"
echo "========================================"

# The single most important rule in this whole script, found the hard way (section 113 in docs/history/METAL_GPU_FEASIBILITY.md):
# cpu_renderer/ray_tracer/scene_metadata (built via CMake below) and RayTracerGUI (built via qmake a few steps down) must
# come out with the SAME architecture(s), or the GUI's own dlopen() of scene_metadata.dylib fails outright with "incompatible
# architecture". So ONE choice is made here and handed to both builds (see --arch in the usage above).
#
# "native" is decided from the CPU, not `uname -m`, which says x86_64 when this script itself runs under Rosetta
# (`arch -x86_64 bash ...`, which this script used to need). The Qt install must contain every requested architecture: an
# official Qt 6 macOS install is universal, and the check below says so clearly when one is not. (An older version of this
# script took the FIRST architecture `lipo -archs qmake` listed, which for a universal Qt is x86_64, and so built every release
# for Rosetta although the Qt install was universal all along.)
HOST_ARCH="$(uname -m)"
[[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == "1" ]] && HOST_ARCH=arm64
[[ "$ARCH_CHOICE" == "native" ]] && ARCH_CHOICE="$HOST_ARCH"
case "$ARCH_CHOICE" in
	arm64|x86_64) WANT_ARCHS="$ARCH_CHOICE" ;;
	universal)    WANT_ARCHS="arm64 x86_64" ;;
	*) echo "ERROR: --arch must be native, arm64, x86_64 or universal (got '$ARCH_CHOICE')" >&2; exit 1 ;;
esac
QT_ARCHS="$(lipo -archs "$(command -v qmake)" 2>/dev/null || true)"
[[ -n "$QT_ARCHS" ]] || { echo "ERROR: could not determine qmake's architectures via 'lipo -archs'" >&2; exit 1; }
for a in $WANT_ARCHS; do
	[[ " $QT_ARCHS " == *" $a "* ]] || { echo "ERROR: this Qt install ($(command -v qmake)) has only: $QT_ARCHS - it cannot build $a. Use --arch with one of those, or install a universal Qt 6 (e.g. through aqtinstall or the online installer)." >&2; exit 1; }
done
# CMake's list separator is ';', qmake's QMAKE_APPLE_DEVICE_ARCHS is space-separated.
CMAKE_ARCHS="${WANT_ARCHS// /;}"
# Build directories are per architecture choice, so a release build never shares objects with a development build of another one,
# and re-running the script is incremental.
BUILD_TAG="$ARCH_CHOICE"
BUILD_DIR="$REPO_ROOT/build_macos_$BUILD_TAG"

if [[ "$WANT_ARCHS" == "x86_64" && "$HOST_ARCH" == "arm64" ]]; then
	echo "NOTE: building x86_64 on an Apple-silicon Mac: the result runs under Rosetta (slower, and needs Rosetta installed)."
fi
echo "Architectures: $WANT_ARCHS (Qt provides: $QT_ARCHS)"

echo
echo "[1/5] Building cpu_renderer + ray_tracer CLI + scene_metadata (CMake)..."
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES="$CMAKE_ARCHS" -DRT_BUILD_METAL=ON
cmake --build "$BUILD_DIR" --config Release -j"$(sysctl -n hw.ncpu)"

CLI_BIN="$BUILD_DIR/ray_tracer"
SCENE_METADATA_LIB="$BUILD_DIR/scene_metadata.dylib"
[[ -f "$CLI_BIN" ]] || { echo "ERROR: $CLI_BIN not found after build" >&2; exit 1; }
[[ -f "$SCENE_METADATA_LIB" ]] || { echo "ERROR: $SCENE_METADATA_LIB not found after build" >&2; exit 1; }

echo
echo "[2/5] Building Qt GUI (qmake + make)..."
# Per-architecture Qt build directory, for the same reason as BUILD_DIR above: a qmake build dir that has seen
# objects of the other architecture fails to link ("found architecture x86_64, required arm64"), which used to
# mean wiping qt_gui/build_macos before every release. Separate dirs make a re-run INCREMENTAL (minutes -> ~1 min).
GUI_BUILD_DIR="$REPO_ROOT/qt_gui/build_macos_$BUILD_TAG"
mkdir -p "$GUI_BUILD_DIR"
# RayTracerGUI.pro's DESTDIR puts the .app in RayTracer_Package/ whichever architecture built it, and make would
# consider an existing .app of the OTHER architecture up to date. Drop it so the (cheap) link always reruns.
rm -rf "$REPO_ROOT/RayTracer_Package/$APP_NAME.app"
( cd "$GUI_BUILD_DIR" && qmake ../RayTracerGUI.pro CONFIG+=release "QMAKE_APPLE_DEVICE_ARCHS=$WANT_ARCHS" && make -j"$(sysctl -n hw.ncpu)" )

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
# Live Preview: the Metal-backed realtime_renderer.dylib the GUI dlopen()s (qt_gui/realtime_preview_session.cpp). Built
# only with RT_BUILD_METAL=ON; without it the GUI shows Live Preview greyed out.
if [[ -f "$BUILD_DIR/realtime_renderer.dylib" ]]; then
	cp "$BUILD_DIR/realtime_renderer.dylib" "$APP_BUNDLE/Contents/MacOS/realtime_renderer.dylib"
fi
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

# Qt's own strings (standard dialog buttons, QFontDialog, the Cut/Copy/Paste
# context menu) are translated by Qt's qtbase_<lang>.qm, which main.cpp loads
# for the languages this app ships. macdeployqt is not relied on to copy them -
# whether it does depends on the Qt build - so put them where main.cpp looks.
QT_TRANSLATIONS_DIR="$(qmake -query QT_INSTALL_TRANSLATIONS)"
mkdir -p "$APP_BUNDLE/Contents/Resources/translations"
for lang in es fr ja zh_CN; do
	if [[ -f "$QT_TRANSLATIONS_DIR/qtbase_$lang.qm" ]]; then
		cp "$QT_TRANSLATIONS_DIR/qtbase_$lang.qm" "$APP_BUNDLE/Contents/Resources/translations/"
	else
		echo "WARNING: qtbase_$lang.qm not found in $QT_TRANSLATIONS_DIR - Qt's own dialogs stay English in that language" >&2
	fi
done

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
#
# The shader source is now split across several metal_poc_*.metal files
# (gpu/metal/metal_poc_shader_files.h's own comment - the former single
# metal_poc.metal no longer exists) - every one of them has to be copied,
# not just one, since metal_poc.mm's own runtime loader concatenates them
# all from this SAME directory. Globbed rather than hardcoded file-by-file
# so a future file added to that list doesn't also need a matching edit
# here to actually ship.
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && grep -q "RT_BUILD_METAL:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
	cp "$REPO_ROOT"/gpu/metal/metal_poc_*.metal "$APP_BUNDLE/Contents/MacOS/"
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
# The Scene Builder's model library (Add > Model library...) lists what is in models/ beside the app: only the three small models whose terms are clear
# (Spot, Suzanne, the Utah teapot) go in the installer, with their thumbnails; the Stanford scans and the rest stay in a source checkout (docs/MODELS.md).
cp "$REPO_ROOT/models/teapot.obj" "$APP_BUNDLE/Contents/MacOS/models/teapot.obj"
mkdir -p "$APP_BUNDLE/Contents/MacOS/models/thumbnails"
for m in spot suzanne teapot; do cp "$REPO_ROOT/models/thumbnails/$m.png" "$APP_BUNDLE/Contents/MacOS/models/thumbnails/$m.png"; done
cp "$REPO_ROOT/images/earthmap.jpg" "$APP_BUNDLE/Contents/MacOS/images/earthmap.jpg"

# The pbrt scene collection (~5MB). Every pbrt-backed scene - nearly the whole
# registry now - loads from pbrt_scenes/*.pbrt, and the GUI runs the renderer
# with the app's MacOS folder as its working directory (qt_gui/mainwindow.cpp's
# setWorkingDirectory), so a pbrt_scenes/ folder beside the executable is the
# one search path that resolves on any install machine (src/shared/
# pbrt_discover.h). Without it the renderer finds no scene file and fails with
# "Scene is Empty (CPU)" (error 101).
cp -R "$REPO_ROOT/pbrt_scenes" "$APP_BUNDLE/Contents/MacOS/pbrt_scenes"

# The optional "Object from a photo" helper (docs/PHOTO_TO_SCENE.md): its Python tool and the script that sets up its Python environment (about 5 GB, made on
# the user's request from the Diagnostics tab). Inside the bundle, beside the executable, so they travel with the app when it is dragged out of the disk image.
# Copied after macdeployqt (like the Metal shaders above) since they are not Mach-O files.
mkdir -p "$APP_BUNDLE/Contents/MacOS/tools/photo_to_mesh" "$APP_BUNDLE/Contents/MacOS/scripts"
cp "$REPO_ROOT/tools/photo_to_mesh/photo_to_mesh.py" "$APP_BUNDLE/Contents/MacOS/tools/photo_to_mesh/photo_to_mesh.py"
cp "$REPO_ROOT/scripts/setup_photo_to_mesh.sh" "$APP_BUNDLE/Contents/MacOS/scripts/setup_photo_to_mesh.sh"
chmod +x "$APP_BUNDLE/Contents/MacOS/scripts/setup_photo_to_mesh.sh"

# Seal the bundle LAST. macdeployqt signs the app ad hoc, but everything copied in after it (the Metal shaders, translations, scenes, photo tool) is
# not in that seal, so `codesign --verify` fails with "a sealed resource is missing or invalid" - and a disk image downloaded through a browser or a chat
# app (which adds the quarantine flag) then opens with "RayTracerGUI is damaged and can't be opened". Signed again here, ad hoc, after the last copy,
# the bundle verifies. (It is still not notarized, so a first launch still needs right-click > Open; that is a different, normal prompt.)
echo "Sealing the bundle (ad hoc signature)..."
codesign --force --deep --sign - "$APP_BUNDLE" || { echo "ERROR: codesign failed" >&2; exit 1; }
codesign --verify --deep --strict "$APP_BUNDLE" || { echo "ERROR: the sealed bundle does not verify" >&2; exit 1; }

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
	# A staging folder with an "Applications" SYMLINK alongside the .app,
	# not the bare .app folder alone - matches the drag-to-Applications
	# convention virtually every macOS .dmg installer uses, and for a real
	# reason found the hard way (section 114): a mounted disk image is
	# READ-ONLY, so running the app straight from the mount (rather than
	# copying it out first) fails outright the moment it tries to write
	# anything - a real user hit exactly this ("filesystem error: in
	# create_directories... Read-only file system") trying to render from
	# `/Volumes/RayTracerGUI/...` directly. The symlink alone can't force
	# anyone to actually drag the icon over, but it's the same one visual
	# nudge every other macOS app relies on, and this one had nothing at
	# all - just a bare app sitting alone in the mounted window, nothing
	# suggesting it needed to move anywhere first.
	DMG_STAGING="$(dirname "$APP_BUNDLE")/dmg_staging"
	rm -rf "$DMG_STAGING"
	mkdir -p "$DMG_STAGING"
	cp -R "$APP_BUNDLE" "$DMG_STAGING/"
	ln -s /Applications "$DMG_STAGING/Applications"
	# A README and a one-click installer beside the app (not inside it, so the bundle's seal is untouched): the app is not notarized, so a copy that
	# arrived through a browser or WeChat needs its quarantine flag removed once (scripts/macos_dmg/).
	cp "$REPO_ROOT/scripts/macos_dmg/README.txt" "$DMG_STAGING/README.txt"
	cp "$REPO_ROOT/scripts/macos_dmg/Install RayTracerGUI.command" "$DMG_STAGING/Install RayTracerGUI.command"
	chmod +x "$DMG_STAGING/Install RayTracerGUI.command"
	hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_PATH" >/dev/null
	rm -rf "$DMG_STAGING"
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
echo "Architectures: $WANT_ARCHS."
echo "Reminder: scenes that need external files (Sponza, Bistro, the H-family"
echo "large environments, most single-model scenes, ...) are not bundled; the"
echo "app's \"Download missing files\" button fetches them. See this script's"
echo "header comment."
