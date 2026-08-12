# User scenes

Any `.pbrt` file placed directly in this folder appears in the renderer as a
scene, in the CLI (`--scene <id>`) and in the GUI's **User Scenes** tab, with
no rebuild.

Point the renderer at a different folder with the `RAY_TRACER_PBRT_DIR`
environment variable — useful for a scene collection you keep outside this
repository:

```bash
RAY_TRACER_PBRT_DIR=D:/pbrt-v4-scenes/killeroo ./ray_tracer.exe --scene 65
```

## What is read, and when

Listing a scene reads only the part of the file before `WorldBegin` — the
camera, film and sampler. Geometry, `Include`d files and `.ply` meshes are read
the first time that scene is actually rendered, so a folder of large scenes
costs nothing to browse.

That is also why a scene's reported performance is "Unknown": nothing in a
pbrt header says whether the world behind it holds three triangles or ten
million.

## Ids

Loaded scenes are numbered after the built-in ones, in filename order. Adding a
file that sorts earlier therefore shifts the ids of the ones after it — if you
script against scene numbers, prefix your filenames (`10-crown.pbrt`) so the
order is yours to control.

## Supported subset

The parser covers pbrt-v4's core: `LookAt`/`Translate`/`Rotate`/`Scale`/
`Transform`, `AttributeBegin`/`End`, `Camera`, `Film`, `Sampler`, `Material`,
`MakeNamedMaterial`/`NamedMaterial`, `Texture`, `AreaLightSource`, `Shape`
(`trianglemesh`, `plymesh`, `sphere`), and `Include`.

### Known issue: `Material "conductor"` renders black

A conductor with no explicit `reflectance` or `k` currently renders as a pure
black surface, at any roughness. Verified by rendering the same sphere at
roughness 0.1 and 0.9 — both are absolutely black, which rules out the
plausible innocent explanation (a mirror sphere in an open-fronted box really
does reflect mostly darkness). The colour defaulting in
`src/shared/pbrt_flatten.h` looks correct on inspection, so the fault is
somewhere further along; it has not yet been traced.

Until it is fixed, prefer `coateddiffuse` or an explicit
`"rgb reflectance"` for metallic-looking surfaces. This is why the bundled
`example-cornell.pbrt` uses a diffuse sphere where a metal one would be the
natural choice.

Anything the parser does not understand is skipped with a warning on stderr
rather than failing the load, so a scene using an unsupported feature still
renders — without that feature. Unsupported constructs are worth reading the warnings
for: a missing displacement map or medium can change a render substantially.

These scenes render on the CPU only. No GPU path consumes them yet.

## Getting scenes

The pbrt-v4 scene collection lives at
<https://github.com/mmp/pbrt-v4-scenes>. Individual scenes there carry their
own licences — several are non-commercial or no-derivatives, so check before
redistributing anything you download.
