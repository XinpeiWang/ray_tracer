#!/usr/bin/env bash
# Renders models/thumbnails/<name>.png for every models/*.obj: the Scene Builder's model library shows these pictures. A 160 x 160 view from a
# front-right-above angle (Y up), a light grey-beige material, a pale sky and one sun, framed by each file's own bounding box.
# usage: scripts/make_model_thumbnails.sh [path/to/ray_tracer] [--gpu|--cpu]    (default: build_macos/ray_tracer, --gpu on a Mac)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RT="${1:-$REPO_ROOT/build_macos/ray_tracer}"
MODE="${2:---gpu}"
[[ -x "$RT" ]] || { echo "ERROR: $RT not found (build first)" >&2; exit 1; }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$REPO_ROOT/models/thumbnails"
cd "$REPO_ROOT"
for obj in models/*.obj; do
	name="$(basename "$obj" .obj)"
	# the bounding box from the vertex lines (awk: no Python or numpy needed)
	read -r cx cy cz size < <(awk '/^v /{for(i=0;i<3;i++){v=$(i+2); if(NR==1||!(i in lo)||v<lo[i])lo[i]=v; if(!(i in hi)||v>hi[i])hi[i]=v}} END{m=0; for(i=0;i<3;i++){d=hi[i]-lo[i]; if(d>m)m=d}; printf "%.9g %.9g %.9g %.9g\n",(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2,m}' "$obj")
	d="$(awk -v s="$size" 'BEGIN{printf "%.9g", 1.9*s}')"
	cat > "$TMP/$name.pbrt" <<PBRT
LookAt $(awk -v c="$cx" -v d="$d" 'BEGIN{printf "%.9g", c+0.6*d}') $(awk -v c="$cy" -v d="$d" 'BEGIN{printf "%.9g", c+0.35*d}') $(awk -v c="$cz" -v d="$d" 'BEGIN{printf "%.9g", c+0.8*d}')  $cx $cy $cz  0 1 0
Camera "perspective" "float fov" [ 35 ]
Film "rgb" "integer xresolution" [ 160 ] "integer yresolution" [ 160 ]
Sampler "halton" "integer pixelsamples" [ 24 ]
WorldBegin
LightSource "infinite" "rgb L" [ 0.55 0.6 0.7 ]
LightSource "distant" "point3 from" [ 1 2 1.5 ] "point3 to" [ 0 0 0 ] "rgb L" [ 3 3 3 ]
Material "diffuse" "rgb reflectance" [ 0.75 0.62 0.45 ]
Shape "plymesh" "string filename" [ "$REPO_ROOT/$obj" ]
PBRT
	"$RT" "$MODE" --seed 2 --output "$TMP/$name.ppm" 160 24 6 "$TMP/$name.pbrt" >/dev/null 2>&1
	cp "$TMP/$name.png" "models/thumbnails/$name.png"
	echo "models/thumbnails/$name.png"
done
