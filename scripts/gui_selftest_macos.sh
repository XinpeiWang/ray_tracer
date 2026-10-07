#!/usr/bin/env bash
# macOS entry point for the GUI smoke test. The test itself is scripts/gui_selftest.py (the same one Windows uses); this wrapper keeps the
# old command working, runs it with Live Preview checks on (a Mac always has Metal), and picks a Python that works: /usr/bin/python3 first,
# since a stray python3 earlier on PATH (an old framework install) can be unusable.
#   scripts/gui_selftest_macos.sh [path/to/RayTracerGUI.app] [output-dir]
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py=python3; [[ -x /usr/bin/python3 ]] && py=/usr/bin/python3
args=(--live-preview)
[[ -n "${1:-}" ]] && app="$1" || app="$here/../RayTracer_Package_macOS/RayTracerGUI.app"
[[ -n "${2:-}" ]] && args+=(--out "$2")
exec "$py" "$here/gui_selftest.py" "$app" "${args[@]}"
