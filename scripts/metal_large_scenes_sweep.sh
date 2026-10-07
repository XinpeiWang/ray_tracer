#!/usr/bin/env bash
# Metal-only sanity render of the large third-party scenes (H1-H21): does each one load, render, and give a lit picture, how long, how many triangles.
#
# These scenes are not in a checkout - the GUI's "Download missing files" fetches them into a per-user folder - and a CPU reference render of
# them takes far too long for the CPU-vs-Metal parity sweep, so this is the check that they at least work on Metal. It renders each at 320x180,
# 8 spp, depth 4 through build_macos/metal_poc and prints one line per scene. A scene whose files are not downloaded is reported and skipped.
#
#   scripts/metal_large_scenes_sweep.sh [assets-dir]
#     assets-dir  where the downloads are (default: $RAY_TRACER_USER_ASSETS, else ~/Library/Application Support/Ray Tracer/user_assets)
#
# Prerequisites: build_macos built with RT_BUILD_METAL=ON (cmake -S . -B build_macos -G Ninja -DCMAKE_BUILD_TYPE=Release -DRT_BUILD_METAL=ON), and the
# scenes downloaded. The pictures are left in /tmp/metal_large_scenes/. Exit status is 1 if any scene failed to render (not for a dark picture:
# H13 and H19 are dark here because the GUI applies a per-scene exposure that this plain render does not).
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assets="${1:-${RAY_TRACER_USER_ASSETS:-$HOME/Library/Application Support/Ray Tracer/user_assets}}"
py=python3; [[ -x /usr/bin/python3 ]] && py=/usr/bin/python3
poc="$here/build_macos/metal_poc"
[[ -x "$poc" ]] || { echo "ERROR: $poc not found - build it first (see the header of this script)" >&2; exit 2; }
out=/tmp/metal_large_scenes; mkdir -p "$out"

scenes=(
  "H1 environment-sponza.pbrt" "H2 environment-bistro-exterior.pbrt" "H3 environment-rungholt.pbrt" "H4 environment-fireplace-room.pbrt"
  "H5 environment-san-miguel.pbrt" "H6 environment-sibenik-cathedral.pbrt" "H7 environment-breakfast-room.pbrt" "H8 environment-salle-de-bain.pbrt"
  "H9 environment-gallery.pbrt" "H10 environment-lost-empire.pbrt" "H11 environment-vokselia-spawn.pbrt" "H12 environment-power-plant.pbrt"
  "H13 contemporary-bathroom/contemporary-bathroom.pbrt" "H14 barcelona-pavilion/pavilion-day.pbrt" "H15 sssdragon/dragon_10.pbrt"
  "H16 ganesha/ganesha.pbrt" "H17 sportscar/sportscar-sky.pbrt" "H18 zero-day/frame25.pbrt" "H19 crown/crown.pbrt"
  "H20 villa/villa-daylight.pbrt" "H21 transparent-machines/frame542.pbrt")

status=0
cd "$here"
for s in "${scenes[@]}"; do
  id="${s%% *}"; file="${s#* }"
  # The scene file itself is in the checkout (H1-H12) or in the downloaded folder (H13-H21).
  if [[ ! -f "pbrt_scenes/$file" && ! -f "$assets/pbrt_scenes/$file" ]]; then echo "$id: skipped (not downloaded)"; continue; fi
  if [[ "$id" =~ ^H([1-9]|1[0-2])$ ]] && [[ ! -d "$assets/models" ]]; then echo "$id: skipped (not downloaded)"; continue; fi
  png="$out/$id.png"; rm -f "$png"
  start=$(date +%s)
  log="$(RAY_TRACER_USER_ASSETS="$assets" perl -e 'alarm 900; exec @ARGV' "$poc" 320 180 "$png" 8 4 aces "pbrt_scenes/$file" 2>&1)"; rc=$?
  secs=$(( $(date +%s) - start ))
  tris="$(grep -o 'Scene: [0-9]* triangles' <<<"$log" | tail -1)"
  note="$(grep -i 'FAILED\|could not\|warning' <<<"$log" | grep -vi 'degenerate' | head -1 | cut -c1-160)"
  pic="no picture"; [[ -f "$png" ]] && pic="$("$py" "$here/scripts/png_stats.py" "$png")"
  [[ $rc -ne 0 || ! -f "$png" ]] && status=1
  echo "$id rc=$rc ${secs}s ${tris:-no scene} | $pic | $note"
done
exit $status
