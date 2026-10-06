#!/usr/bin/env bash
# Automated smoke test of the real macOS GUI (the project has no other GUI test): runs the built RayTracerGUI.app with
# RT_GUI_SELFTEST (qt_gui/mainwindow_selftest.cpp) in each mode, headless, and checks the result.
#   ui           - the Output Mode list contains an ENABLED "Live Preview (interactive)" item
#   livepreview  - selects it, starts Live Preview (Metal, realtime_renderer.dylib), lets it render, orbits the camera
#                  like a mouse drag, and requires frames to flow AND the picture to change
# Screenshots are of the app's own window (never the screen) and are written, with a text log, to the output directory.
#
# Usage: scripts/gui_selftest_macos.sh [path/to/RayTracerGUI.app] [output-dir]
#   defaults: RayTracer_Package/RayTracerGUI.app  and  a fresh temp directory (printed at the end)
#
# It runs with a throwaway HOME so the app never reads or writes ~/Pictures (macOS asks permission for that, and an
# unanswered prompt blocks startup invisibly; CFFIXED_USER_HOME makes QSettings/CFPreferences use it too, so the test never touches the
# real app preferences), and with Qt's `offscreen` platform plugin taken from the Qt install next to
# `qmake` (macdeployqt only bundles `cocoa`). Run it under `arch -x86_64` if the app is x86_64 and this shell is arm64.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${1:-$REPO_ROOT/RayTracer_Package/RayTracerGUI.app}"
OUT="${2:-$(mktemp -d /tmp/gui_selftest.XXXXXX)}"
[[ -d "$APP" ]] || { echo "ERROR: $APP not found" >&2; exit 2; }
APP="$(cd "$APP" && pwd)"      # absolute: the app is launched from "/" below
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
EXE="$APP/Contents/MacOS/RayTracerGUI"
[[ -x "$EXE" ]] || { echo "ERROR: $EXE not found - build the app first (scripts/build_and_deploy_macos.sh)" >&2; exit 2; }
command -v qmake >/dev/null || { echo "ERROR: qmake not on PATH (needed to find Qt's offscreen platform plugin)" >&2; exit 2; }
PLUGINS="$(cd "$(dirname "$(command -v qmake)")/../plugins/platforms" && pwd)"
[[ -f "$PLUGINS/libqoffscreen.dylib" ]] || { echo "ERROR: $PLUGINS/libqoffscreen.dylib not found" >&2; exit 2; }
FAKE_HOME="$(mktemp -d /tmp/gui_selftest_home.XXXXXX)"; mkdir -p "$FAKE_HOME/Pictures"
mkdir -p "$OUT"

ARCH_PREFIX=()
if [[ "$(file "$EXE" | grep -o 'x86_64\|arm64' | head -1)" == "x86_64" && "$(uname -m)" != "x86_64" ]]; then ARCH_PREFIX=(arch -x86_64); fi

run_mode() {   # mode, wait-seconds
	local mode="$1" wait_s="$2" prefix="$OUT/$1"
	rm -f "$prefix.txt" "$prefix.stdout"
	# cwd "/" on purpose: that is where a Finder/Dock launch starts (Live Preview runs inside this process, so it must find its
	# scene files without help from the working directory - the bug a launch from inside the bundle used to hide).
	( cd / && HOME="$FAKE_HOME" CFFIXED_USER_HOME="$FAKE_HOME" QT_QPA_PLATFORM=offscreen QT_QPA_PLATFORM_PLUGIN_PATH="$PLUGINS" \
	    RT_GUI_SELFTEST="$mode" RT_GUI_SELFTEST_OUT="$prefix" "${ARCH_PREFIX[@]}" "$EXE" > "$prefix.stdout" 2>&1 ) &
	local pid=$! waited=0
	while kill -0 "$pid" 2>/dev/null && (( waited < wait_s )); do sleep 1; waited=$((waited + 1)); done
	if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; echo "FAIL [$mode]: still running after ${wait_s}s (hung)"; return 1; fi
	wait "$pid"; local rc=$?
	if [[ $rc -ne 0 ]]; then echo "FAIL [$mode]: exit code $rc"; cat "$prefix.txt" 2>/dev/null; return 1; fi
	echo "PASS [$mode]"; return 0
}

status=0
run_mode ui 40 || status=1
grep -q 'Live Preview (interactive)" enabled=1' "$OUT/ui.txt" 2>/dev/null || { echo "FAIL [ui]: no enabled Live Preview item"; status=1; }
run_mode livepreview 60 || status=1
grep -E "frames=|picture change|RESULT" "$OUT/livepreview.txt" 2>/dev/null | sed 's/^/  /'
rm -rf "$FAKE_HOME"
defaults delete com.raytracer.RayTracerGUI-selftest >/dev/null 2>&1 || true   # the app's self-test settings domain (separate from the real one)
rm -f "$HOME/Library/Preferences/com.raytracer.RayTracerGUI-selftest.plist"
echo "logs and screenshots: $OUT"
exit $status
