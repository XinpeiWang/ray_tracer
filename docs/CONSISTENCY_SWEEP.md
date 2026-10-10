# Consistency sweep

A renderer can agree with another renderer and still be wrong, because both can share a mistake. It cannot disagree with *itself* about two descriptions of the same thing. `scripts/consistency_sweep.py` renders pairs of scenes that must look alike and reports the pairs that do not. It needs a built `ray_tracer` and python 3; nothing else (the EXR reader is in the script).

```
python3 scripts/consistency_sweep.py                       # CPU and Metal (--gpu), every material, every group
python3 scripts/consistency_sweep.py --backends gpu --only sky,scale -v
```

The scene is a floor, a ball of the material under test and a lamp, in a sky. Each pair is compared by the mean of its picture per colour channel (4% by default; `--tolerance`). Exit code 1 when a pair is off.

| group | the two descriptions | what it catches |
|---|---|---|
| `sky` | a constant sky `rgb L` against the same sky as a white picture | a shader that lights an object differently under an image sky (missing or wrong environment light sampling, MIS weights for a sample never taken) |
| `translate` | the whole scene moved | anything that depends on where the scene is (float precision, a scene frame that is not applied twice) |
| `scale` | the whole scene 8 times bigger (distances, radii, light powers; a subsurface mean free path and medium coefficients scale too) | a constant with a unit hiding in it (epsilons, step sizes, a missing `1/scale`) |
| `mirror` | the whole scene mirrored left to right | a handedness bug (tangent frames, normals, winding) |
| `lightform` | a point light against a small emissive sphere of the same intensity | the two light kinds disagreeing about power |
| `size` | the same scene at two picture sizes | a pixel-size dependency |
| `spp` | 16 against 128 samples a pixel | bias that grows or shrinks with the sample count |

The glass materials (`dielectric`, `thindielectric`, `roughdielectric`) skip `lightform` (a point light cannot be seen in a perfect specular surface) and the two fireflier ones get a looser tolerance (8%) in the groups that compare noisy means. Each material is a ball that fills the frame in the `sky` group, so the whole picture is the material.

## What it found (October 2026)

* **Coated materials under an image sky** read 6% dark: `coateddiffuse` and `coatedconductor` did no environment light sampling, but the escape of their continuation ray was weighted as if they did. Both now sample the picture (and the pbrt image light) and weight by the walk's own `f` and `pdf`.
* **Fog and clouds under an image sky**: fog read 30% dark and a cloud 8%, for the same reason (a scatter in a medium samples area, point and distant lights and the constant sky, never a picture). A ray that leaves a medium scatter now takes the whole sky (`PathState::fromMediumScatter`); noisier than light sampling would be, but right. Sampling the picture from inside a medium would reduce the noise and is not done yet.
* **`subsurface` with `reflectance` and `mfp`** (pbrt-v4's fourth way to give a subsurface material) was silently replaced by the default coefficients, so the mean free path had no effect at all. The flattener now inverts the reflectance through the beam-diffusion table (`SubsurfaceFromDiffuse`), the way pbrt does.
* Earlier by hand, the same method: subsurface exit normal, sphere-clipped media, distant lights in media.

## In ctest

`metal_poc_consistency` runs the Metal half (about half a minute; tolerance 6%, the pictures are small). The CPU half is slower and is run by hand.

## Adding a case

Add a material to `MATERIALS` (a string, or a function of the scene when it must scale with it), and a name to `SPECULAR` / `NOISY` if it needs them. A new group is a method on `Sweep` that returns pairs of `Scene`s.
