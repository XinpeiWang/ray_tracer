$fp = "C:\Users\xinpe\source\repos\ray_tracer\src\shared\cameras.h"
$c = [System.IO.File]::ReadAllText($fp, [System.Text.Encoding]::UTF8)

# Fix 1a: scene-side paraxial ray offset: +T(1) metre -> +T(0.001) metre (= 1mm, matching pbrt-v4 +1mm)
$old = "bool ok = trace_lenses_from_scene(" + [char]13 + [char]10 + "                x, T(0), front_z + T(1),"
$new = "bool ok = trace_lenses_from_scene(" + [char]13 + [char]10 + "                x, T(0), front_z + T(0.001),"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); Write-Host "Fix1a OK" } else { Write-Host "Fix1a NOT FOUND" }

# Fix 1b: matching cardinal-point call
$old = "compute_cardinal_points(" + [char]13 + [char]10 + "                    x, T(0), front_z+T(1), T(0), T(0), T(-1),"
$new = "compute_cardinal_points(" + [char]13 + [char]10 + "                    x, T(0), front_z+T(0.001), T(0), T(0), T(-1),"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); Write-Host "Fix1b OK" } else { Write-Host "Fix1b NOT FOUND" }

# Fix 2a: film-side paraxial ray offset: -T(1) metre -> -T(0.001) metre (= -1mm)
$old = "T w = trace_lenses_from_film(" + [char]13 + [char]10 + "                x, T(0), orig_rear - T(1),"
$new = "T w = trace_lenses_from_film(" + [char]13 + [char]10 + "                x, T(0), orig_rear - T(0.001),"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); Write-Host "Fix2a OK" } else { Write-Host "Fix2a NOT FOUND" }

# Fix 2b: matching cardinal-point call
$old = "compute_cardinal_points(" + [char]13 + [char]10 + "                    x, T(0), orig_rear-T(1), T(0), T(0), T(1),"
$new = "compute_cardinal_points(" + [char]13 + [char]10 + "                    x, T(0), orig_rear-T(0.001), T(0), T(0), T(1),"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); Write-Host "Fix2b OK" } else { Write-Host "Fix2b NOT FOUND" }

# Fix 3: BoundExitPupil expand — match pbrt-v4 "2*Length(diagonal)/sqrt(nSamples)"
# pbrt-v4 diagonal of square side (projMax-projMin) = sqrt(2)*(projMax-projMin)
$old = "T expand = T(2)*(projMax-projMin)/std::sqrt(T(nSamples));"
$new = "// pbrt-v4: Expand(bounds, 2*Length(projRearBounds.Diagonal())/sqrt(nSamples))" + [char]13 + [char]10 + "            // Diagonal of the projection square = sqrt(2)*(projMax-projMin)." + [char]13 + [char]10 + "            T expand = T(2)*std::sqrt(T(2))*(projMax-projMin)/std::sqrt(T(nSamples));"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); Write-Host "Fix3 OK" } else { Write-Host "Fix3 NOT FOUND" }

[System.IO.File]::WriteAllText($fp, $c, [System.Text.Encoding]::UTF8)
Write-Host "All fixes written."
