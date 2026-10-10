# The model library

The Scene Builder's **Add > Model library...** lists the meshes in the `models/` folder next to the app, with a picture of each. Adding one puts it on the floor under
the middle of the view, about 1.6 units across, as one undo step; **Scale** in its properties resizes it. Any other `.obj` or `.ply` file can be added with **Choose another
file...** (or Add > Mesh), and then keeps its own size, so a big scan may need a smaller Scale.

What is in the folder depends on where the app comes from:

* **The installers** (macOS dmg, Windows package) carry three small models whose terms are clear: Spot, Suzanne and the Utah teapot.
* **A source checkout** has all 22 meshes in `models/` (about 66 MB), and the library shows every one it finds.

To add your own to the library's folder, put the `.obj` there; only the models named in `src/shared/model_library.h` appear in the library, so add a line to its catalog (and a
picture `models/thumbnails/<name>.png`, made by `scripts/make_model_thumbnails.sh`) to list a new one. Anything else is still available through Choose another file.

## Where the models come from

Each model keeps the licence and attribution terms of its source. **This table was not checked file by file**: it says what is known about each source, and the right
thing to do before publishing pictures or scenes made with a model, or shipping it in a product, is to read the source's own terms.

| Model | Source | What the source asks | In the installers |
|---|---|---|---|
| Spot (cow) | Keenan Crane's 3D model repository | CC0 (public domain dedication) | yes |
| Suzanne | Blender (the Blender Foundation distributes it freely with Blender) | none that we know of | yes |
| Utah teapot | Martin Newell, University of Utah, 1975 | widely treated as public domain | yes |
| Stanford bunny, Armadillo, Happy Buddha, Lucy, Dragon (XYZ RGB) | The Stanford 3D Scanning Repository (graphics.stanford.edu/data/3Dscanrep) | acknowledge the source; the page says some models need permission for commercial use | no |
| Cow, Horse, Ogre, Beast, Homer, Cheburashka, Beetle (two), Nefertiti, Max Planck, Bimba, Igea, Fandisk, Rocker arm | The common-3d-test-models collection (github.com/alecjacobson/common-3d-test-models), which gathers meshes from several places (Stanford, AIM@SHAPE, Keenan Crane, others) | its README names each model's origin; some allow research use only or non-commercial use | no |

The thumbnails in `models/thumbnails/` are renders made for this project (`scripts/make_model_thumbnails.sh`); they carry no terms of their own beyond the model's.
`THIRD_PARTY_NOTICES.md` says the same in short.
