#!/bin/sh
# CTest driver for metal_poc_media_lights (see CMakeLists.txt): renders each scene of gpu/metal/test_scenes/media-*.pbrt (a homogeneous medium and a cloud, each lit by one
# distant light and by one point light, on black) with the CPU renderer and with `ray_tracer --gpu` (Metal), and runs metal_poc_media_lights_check on each pair.
# Prints MEDIA_LIGHTS_TEST_OK on success.
# usage: metal_poc_media_lights_test.sh <ray_tracer> <metal_poc_media_lights_check> <scratch-dir>
RT="$1"; CHK="$2"; DIR="$3"
status=0
for name in distant-homogeneous point-homogeneous distant-cloud point-cloud; do
  scene="gpu/metal/test_scenes/media-$name.pbrt"
  "$RT" --cpu --seed 3 --output "$DIR/media_${name}_cpu.exr" 48 64 12 "$scene" >/dev/null 2>&1 || { echo "$name: CPU render failed"; exit 1; }
  "$RT" --gpu --seed 3 --output "$DIR/media_${name}_metal.exr" 48 64 12 "$scene" >/dev/null 2>&1 || { echo "$name: Metal render failed"; exit 1; }
  "$CHK" "$name" "$DIR/media_${name}_cpu.exr" "$DIR/media_${name}_metal.exr" 5 || status=1
done
[ $status -eq 0 ] && echo MEDIA_LIGHTS_TEST_OK
exit $status
