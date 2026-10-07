# Custom scenes

Any `.pbrt` file placed directly in this folder appears in the renderer as a
scene, in the CLI (by its file name: `ray_tracer.exe --cpu 400 64 8 my-scene`, or by
the file's path) and in the GUI's **Custom Scenes** tab, with no rebuild.

Point the renderer at a different folder with the `RAY_TRACER_PBRT_DIR`
environment variable — useful for a scene collection you keep outside this
repository:

```bash
RAY_TRACER_PBRT_DIR=D:/pbrt-v4-scenes/killeroo ./ray_tracer.exe --cpu 400 64 8 killeroo
```

## What is read, and when

Listing a scene reads only the part of the file before `WorldBegin` — the
camera, film and sampler. Geometry, `Include`d files and `.ply` meshes are read
the first time that scene is actually rendered, so a folder of large scenes
costs nothing to browse.

That is also why a scene's reported performance is "Unknown": nothing in a
pbrt header says whether the world behind it holds three triangles or ten
million.

## Names and ids

A scene is named by its file name without the extension, lower-cased with hyphens:
`My Scene.pbrt` is `my-scene`. That name is the scene's durable key: it is what the
command line, the recent-renders list and the thumbnail cache use, and it does not
change when you add or remove other files. A scene from a collection in its own
sub-folder is prefixed with that folder (`killeroo/frame25.pbrt` is `killeroo-frame25`);
if two files would get the same name the later one gets `-2`.

Each scene also has a short id (`K37`), numbered in filename order after the built-in
ones, and the id is accepted anywhere a name is. But adding a file that sorts earlier
shifts the ids of the ones after it, so scripts should use the name.

## Supported subset

The parser covers a large and growing subset of pbrt-v4 — rather than keep an
itemized list in sync here (it has outgrown that once already), the
authoritative list is the directive dispatch in `src/shared/pbrt_scene.h`.
As of this writing that includes: `LookAt`/`Translate`/`Rotate`/`Scale`/
`Transform`, `AttributeBegin`/`End`, `ObjectBegin`/`ObjectInstance`
(instancing), `ReverseOrientation`, `Accelerator`, `CoordinateSystem`,
`ColorSpace`, `Camera`, `Film`, `Sampler`, `Material`, `MakeNamedMaterial`/
`NamedMaterial`, `Texture`, `MakeNamedMedium`/`MediumInterface` (homogeneous,
cloud, nanovdb and rgbgrid media), `LightSource` (point, spot, distant,
infinite, projection, goniometric) and `AreaLightSource`, `Shape`
(`trianglemesh`, `plymesh`, `sphere`, `cylinder`, `cone`, `disk`,
`paraboloid`, `curve`, `bilinearmesh`), animated transforms (motion blur),
and `Include`. The `measured` material (a real, importance-sampled pbrt-v4
`MeasuredBxDF` loaded from a `.bsdf` tensor file) is also supported — see
`pbrt_scenes/measured-brdf-showroom.pbrt` for an example.

Anything the parser does not understand is skipped with a warning on stderr
rather than failing the load, so a scene using an unsupported feature still
renders — without that feature. Unsupported constructs are worth reading the warnings
for: a missing displacement map or medium can change a render substantially.
One known structural gap: there is no `.obj` mesh ingestion at all (only
inline `trianglemesh` and external `.ply`), so a scene that references raw
`.obj` files won't load as-is.

## Authoring conventions

Not enforced by the parser, but followed by every scene this project authors
itself (as opposed to a downloaded multi-file bundle like
`barcelona-pavilion/` or `sportscar/`):

- **Licensing header**: a self-authored scene ends its header comment with
  `# Original, not derived from any distributed scene, so it carries no
  licence constraints.` A scene migrated from a native C++ demo scene opens
  with a short provenance note instead (which native scene/function it
  replaces, and whether it's a byte-for-byte port or a disclosed fidelity
  improvement) — see `scene_registry_data.h`'s matching comment for the same
  scene for the full rationale.
- **Sibling binary assets resolve scene-directory-first**: a bare filename
  in `"string filename" [ "foo.bsdf" ]` (or `.exr`, `.ply`, `.bmp`, `.nvdb`)
  resolves against the `.pbrt` file's own directory first, then as given —
  the same rule `Include`d files use for their own nested includes (see
  `src/shared/pbrt_load.h`'s `resolveExistingPath()`). This is why every
  scene here references its sibling assets by bare filename with no path.

## CPU and GPU

These scenes render on both backends, from the same parsed scene — pass
`--gpu` or `--cpu`.

The GPU is the weaker of the two here, in one specific way: it can only sample
area lights that are spheres or **parallelograms**. pbrt writes area lights as
triangle meshes, and the loader rejoins triangle pairs into quads to cover the
usual case, but a light that is a general quadrilateral, a triangle fan, or a
genuine single triangle cannot be converted. Those still emit when a ray hits
them directly, so the image is darker and noisier rather than wrong — and the
GPU build prints exactly how many lights it could not convert. If you see that
warning, use `--cpu` for that scene.

## Getting scenes

The pbrt-v4 scene collection lives at
<https://github.com/mmp/pbrt-v4-scenes>. Individual scenes there carry their
own licences — several are non-commercial or no-derivatives, so check before
redistributing anything you download.
