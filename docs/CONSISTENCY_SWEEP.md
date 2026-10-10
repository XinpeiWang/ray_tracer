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
| `skyrotate` | a picture sky that is brighter on one side, with the whole scene and the sky turned 90 and 200 degrees | a sky orientation that the lookup and the light sampling disagree on |
| `shapeform` | the ball written as a radius 2 sphere under `Scale 0.5`, under an arbitrary `Rotate`, with every sphere parameter spelled out, and as an `ObjectInstance`; the floor with the other diagonal, as a bilinear patch with four indices, and as one without indices | a shape or transform path that draws something else for the same geometry |
| `lights` | a spot light with a hard wide cone against a point light; a distant light against a point light 1000 units away | the light kinds disagreeing about power or direction |
| `texture` | a diffuse reflectance as an inline colour, a constant texture, an image (`encoding "linear"`), a checkerboard of two equal colours, a mix of two equal colours; roughness and coated reflectance as textures | a texture path that reads a different value than the plain one |
| `skyscale` | a constant sky at 0.4 against a white picture at `scale 0.4` | the `scale` of a picture sky |
| `backends` | the same scene on the CPU and on the GPU, under a constant and a picture sky (only when both are asked for) | anything the two renderers disagree about; glass uses 512 samples, a caustic from a small lamp can move a 64-sample mean by 9% |
| `size` | the same scene at two picture sizes | a pixel-size dependency |
| `spp` | 16 against 128 samples a pixel | bias that grows or shrinks with the sample count |

The glass materials (`dielectric`, `thindielectric`, `roughdielectric`) skip `lightform` (a point light cannot be seen in a perfect specular surface) and the two fireflier ones get a looser tolerance (8%) in the groups that compare noisy means. Each material is a ball that fills the frame in the `sky` group, so the whole picture is the material.

## What it found (October 2026)

* **Coated materials under an image sky** read 6% dark: `coateddiffuse` and `coatedconductor` did no environment light sampling, but the escape of their continuation ray was weighted as if they did. Both now sample the picture (and the pbrt image light) and weight by the walk's own `f` and `pdf`.
* **Fog and clouds under an image sky**: fog read 30% dark and a cloud 8%, for the same reason (a scatter in a medium samples area, point and distant lights and the constant sky, never a picture). A ray that leaves a medium scatter now takes the whole sky (`PathState::fromMediumScatter`); noisier than light sampling would be, but right. Sampling the picture from inside a medium would reduce the noise and is not done yet.
* **`subsurface` with `reflectance` and `mfp`** (pbrt-v4's fourth way to give a subsurface material) was silently replaced by the default coefficients, so the mean free path had no effect at all. The flattener now inverts the reflectance through the beam-diffusion table (`SubsurfaceFromDiffuse`), the way pbrt does.
* **A ball written as an `ObjectInstance` was missing on Metal** (every instanced sphere was skipped, not only a stretched one): a sphere under a rotation, mirror and uniform scale is still a sphere, and is now placed as one (an instanced emissive, grid- or cloud-bounded or clipped sphere is still skipped, with a warning).
* **A `bilinearmesh` with `integer indices`** (four of them, one patch) was read in the order the points were listed, giving a twisted patch on every backend (13% darker floor). The indices pick the corners now; more than four mean several patches, which the loader still does not build.
* **An image texture with `encoding "linear"` (or `gamma`, or `invert`) was decoded as sRGB on Metal** (12% darker than the CPU); the Metal loader now redoes the curve.
* **A `mix` texture of plain colours drew as the default grey on Metal**: the flattener leaves the blend as the flat colour for a backend with no mix texture.
* Earlier by hand, the same method: subsurface exit normal, sphere-clipped media, distant lights in media.

## In ctest

`metal_poc_consistency` runs the Metal half (about a minute; tolerance 6%, the pictures are small). `python3 scripts/consistency_sweep.py` runs the CPU half and the CPU-against-GPU group as well (about 70 seconds in all).

## Adding a case

Add a material to `MATERIALS` (a string, or a function of the scene when it must scale with it), and a name to `SPECULAR` / `NOISY` if it needs them. A new group is a method on `Sweep` that returns pairs of `Scene`s.
