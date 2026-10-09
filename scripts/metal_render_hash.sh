#!/usr/bin/env bash
# Bit-exact regression check for the Metal renderer: renders every scene of the parity sweep (plus a few option variants) with a fixed
# --seed and records a hash of each float EXR.
#
#   scripts/metal_render_hash.sh capture before-refactor   # on the known-good code (about 2 minutes)
#   ... make the change, rebuild build_macos ...
#   scripts/metal_render_hash.sh compare  before-refactor   # exit code 1 if any picture changed
#
# This is the gate for a change that must not alter a picture: moving host-side code in the pbrt loader (scene frame, tessellation, material
# tables), refactoring the Metal host. Metal renders with a fixed seed are reproducible (the ctest metal_poc_seed_reproducible), so unchanged
# code gives identical files. It does NOT suit a change to the SHADER (moving code between shader functions changes float rounding even when the
# maths is the same): for that use the parity sweep and scripts/update_metal_golden.sh, comparing pictures statistically.
# Scenes that are not reproducible even run to run (D8, D12) are reported as "unstable" and never fail the comparison.
#
# scripts/render_baseline.py is the statistical, three-seed version for the CPU and OptiX; it needs numpy and has no Metal backend.
set -euo pipefail

MODE="${1:-}"; NAME="${2:-}"
[[ "$MODE" == capture || "$MODE" == compare ]] && [[ -n "$NAME" ]] || { echo "usage: $0 capture|compare <name> [path/to/ray_tracer]" >&2; exit 2; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="${3:-$REPO_ROOT/build_macos/ray_tracer}"
[[ -x "$EXE" ]] || { echo "ERROR: $EXE not found - build with -DRT_BUILD_METAL=ON first" >&2; exit 2; }
DIR="$REPO_ROOT/baselines/metal-hash-$NAME"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
cd "$REPO_ROOT"

render() {  # tag, extra ray_tracer args, scene id, environment assignments
	local tag="$1" extra="$2" id="$3" envs="$4" h=NOIMAGE
	# shellcheck disable=SC2086
	env $envs "$EXE" --gpu --seed 5 $extra --output "$OUT/$tag.exr" 96 8 8 "$id" >"$OUT/$tag.log" 2>&1 </dev/null || true
	[[ -s "$OUT/$tag.exr" ]] && h=$(shasum -a 256 "$OUT/$tag.exr" | cut -c1-16)
	rm -f "$OUT/$tag.exr"
	printf '%s\t%s\n' "$tag" "$h"
}

specs() {   # tag | extra args | scene id | environment assignments (a '|' separator: empty fields must survive `read`)
	for id in $(grep -v '^#' gpu/metal/parity_golden.txt | cut -d' ' -f1); do printf '%s||%s|\n' "$id" "$id"; done
	for id in A1 B1 C1; do
		printf '%s|%s|%s|%s\n' "$id.adaptive" "--adaptive" "$id" ""
		printf '%s|%s|%s|%s\n' "$id.clamp" "--maxcomponentvalue 2" "$id" ""
		printf '%s|%s|%s|%s\n' "$id.lockstep" "" "$id" "METAL_REGEN=0"
		printf '%s|%s|%s|%s\n' "$id.bands" "" "$id" "METAL_BANDS=4"
	done
	printf '%s|%s|%s|%s\n' "A1.crop" "--crop 10 10 60 60" "A1" ""
}
panel() { specs | while IFS='|' read -r tag extra id envs; do render "$tag" "$extra" "$id" "$envs"; done; }

if [[ "$MODE" == capture ]]; then
	mkdir -p "$DIR"
	# Three times: a scene that differs from itself is recorded as UNSTABLE and left out of later comparisons.
	panel | sort > "$DIR/run1.tsv"; panel | sort > "$DIR/run2.tsv"; panel | sort > "$DIR/run3.tsv"
	paste "$DIR/run1.tsv" "$DIR/run2.tsv" "$DIR/run3.tsv" | awk -F'\t' '{ if ($2 == $4 && $2 == $6) print $1 "\t" $2; else print $1 "\tUNSTABLE" }' > "$DIR/hashes.tsv"
	rm -f "$DIR/run1.tsv" "$DIR/run2.tsv" "$DIR/run3.tsv"
	echo "Captured $(wc -l < "$DIR/hashes.tsv" | tr -d ' ') renders ($(grep -c UNSTABLE "$DIR/hashes.tsv" || true) unstable) into ${DIR#"$REPO_ROOT"/}"
	exit 0
fi

[[ -f "$DIR/hashes.tsv" ]] || { echo "No baseline '$NAME' (looked in $DIR). Capture one first." >&2; exit 2; }
panel | sort > "$OUT/now.tsv"
sort "$DIR/hashes.tsv" > "$OUT/base.tsv"
changed=0; same=0; unstable=0
specs > "$OUT/specs.tsv"
while IFS=$'\t' read -r tag old new; do
	if [[ "$old" == UNSTABLE ]]; then unstable=$((unstable + 1)); continue; fi
	if [[ "$old" != "$new" ]]; then   # a scene that is occasionally non-reproducible gets a few more tries before it counts as changed
		IFS='|' read -r _ extra id envs < <(grep -m1 "^$tag|" "$OUT/specs.tsv")
		for _try in 1 2 3 4 5 6; do
			new=$(render "$tag" "$extra" "$id" "$envs" | cut -f2)
			[[ "$old" == "$new" ]] && break
		done
	fi
	if [[ "$old" == "$new" ]]; then same=$((same + 1)); else changed=$((changed + 1)); echo "  CHANGED  $tag (was $old, now $new)"; fi
done < <(join -t "$(printf '\t')" "$OUT/base.tsv" "$OUT/now.tsv")
echo "$same identical, $changed changed, $unstable unstable (not compared)"
[[ $changed -eq 0 ]]
