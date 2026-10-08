#!/usr/bin/env python3
"""photo_to_mesh.py - turn ONE photo of an object into a textured 3D mesh the renderer can load.

This is an optional helper for the Scene Builder's "Import from photo" button. It runs entirely on your own
computer (no upload): the open TripoSR model (MIT licence) guesses the object's 3D shape from the picture.
The guess is plausible, not accurate - the back of the object is invented and fine detail is soft.

Output, in --out:
    mesh.obj      the mesh: Y up, standing on y = 0, centred on x and z, about 2 units tall
    texture.png   the surface colour (the Scene Builder uses it as the mesh's picture)
    result.json   {"mesh", "texture", "size": [x, y, z], "triangles"}

While it runs it prints lines "PROGRESS <percent> <message>" so a caller can show a progress bar.

Needs the Python environment made by scripts/setup_photo_to_mesh.ps1 (PyTorch, TripoSR, rembg, xatlas, scikit-image).
"""
import argparse
import json
import os
import sys
import time


def progress(percent, message):
    print("PROGRESS %d %s" % (percent, message), flush=True)


def fail(message, code=2):
    print("ERROR " + message, flush=True)
    sys.exit(code)


def find_triposr(arg):
    """The folder holding TripoSR's `tsr` package: --triposr-dir, $RT_TRIPOSR_DIR, or next to the venv."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (arg, os.environ.get("RT_TRIPOSR_DIR"), os.path.join(os.path.dirname(sys.prefix), "TripoSR"),
              os.path.join(here, "TripoSR")):
        if c and os.path.isdir(os.path.join(c, "tsr")):
            return c
    return None


def patch_marching_cubes(np, torch):
    """TripoSR calls torchmcubes (a compiled extension that is a pain to build on Windows). scikit-image does the same job."""
    import types
    from skimage import measure

    def marching_cubes(vol, thresh):
        v, f, _n, _val = measure.marching_cubes(vol.detach().cpu().numpy(), float(thresh))
        # torchmcubes lists a vertex as (x, y, z) = (last index, ..., first index); TripoSR undoes that with [2, 1, 0].
        return torch.from_numpy(np.ascontiguousarray(v[:, ::-1])).float(), torch.from_numpy(f.astype(np.int64))

    stub = types.ModuleType("torchmcubes")
    stub.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = stub


def bake_vertex_colours(np, uvs, indices, colours, size):
    """Paint per-vertex colours into a size x size texture along each triangle (by barycentric sampling), then spread
    the colour into the empty texels around the charts so bilinear filtering never blends in black. uvs are in [0, 1]."""
    from scipy import ndimage

    tex = np.zeros((size, size, 3), np.float32)
    filled = np.zeros((size, size), bool)
    px = uvs * (size - 1)
    tri = indices
    a, b, c = px[tri[:, 0]], px[tri[:, 1]], px[tri[:, 2]]
    longest = np.maximum.reduce([np.linalg.norm(a - b, axis=1), np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1)])
    steps = np.clip(np.ceil(longest * 1.5).astype(np.int32), 1, 256)  # two samples per texel along the longest edge, or better
    for n in np.unique(steps):
        sel = np.nonzero(steps == n)[0]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        keep = (i + j) <= n
        w1, w2 = (i[keep] / n).astype(np.float32), (j[keep] / n).astype(np.float32)
        w0 = 1.0 - w1 - w2
        # chunk so a huge group cannot exhaust memory
        for s in range(0, len(sel), 20000):
            t = sel[s:s + 20000]
            p = a[t, None, :] * w0[None, :, None] + b[t, None, :] * w1[None, :, None] + c[t, None, :] * w2[None, :, None]
            col = (colours[tri[t, 0]][:, None, :] * w0[None, :, None] + colours[tri[t, 1]][:, None, :] * w1[None, :, None]
                   + colours[tri[t, 2]][:, None, :] * w2[None, :, None])
            x = np.clip(np.rint(p[..., 0]).astype(np.int64), 0, size - 1).ravel()
            y = np.clip(np.rint(p[..., 1]).astype(np.int64), 0, size - 1).ravel()
            tex[y, x] = col.reshape(-1, 3)
            filled[y, x] = True
    # Spread into the gaps: every empty texel takes the colour of its nearest painted one.
    nearest = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    return tex[nearest[0], nearest[1]]


def main():
    ap = argparse.ArgumentParser(description="One photo -> a textured mesh (TripoSR, runs locally).")
    ap.add_argument("image", help="a photo of one object")
    ap.add_argument("--out", required=True, help="folder for mesh.obj, texture.png and result.json")
    ap.add_argument("--resolution", type=int, default=256, help="marching-cubes grid; higher = finer mesh, slower (default 256)")
    ap.add_argument("--texture-size", type=int, default=1024, help="texture width and height in pixels (default 1024)")
    ap.add_argument("--no-remove-bg", action="store_true", help="the photo already has a plain background and the object fills most of it")
    ap.add_argument("--foreground-ratio", type=float, default=0.85)
    ap.add_argument("--triposr-dir", help="folder containing TripoSR's tsr package")
    ap.add_argument("--device", default="auto", help="cuda:0, cpu or auto (default)")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        fail("the photo %s was not found." % args.image)
    os.makedirs(args.out, exist_ok=True)
    started = time.time()

    progress(2, "Loading the libraries")
    try:
        import numpy as np
        import torch
        import trimesh
        import xatlas
        from PIL import Image
    except ImportError as e:
        fail("a Python package is missing (%s). Run scripts/setup_photo_to_mesh.ps1 first." % e)
    triposr = find_triposr(args.triposr_dir)
    if not triposr:
        fail("TripoSR was not found. Run scripts/setup_photo_to_mesh.ps1, or pass --triposr-dir.")
    sys.path.insert(0, triposr)
    patch_marching_cubes(np, torch)
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    if device == "cpu":
        print("NOTE no graphics card was found for PyTorch, so this runs on the processor and takes several minutes.", flush=True)

    progress(8, "Loading the model (the first run downloads about 1.7 GB)")
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)

    progress(25, "Preparing the photo")
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(args.image))  # a phone photo is often stored sideways with a rotation flag
    img.thumbnail((1024, 1024))  # the model looks at 512 pixels anyway; a phone photo's 12 megapixels only slow the background removal down
    if args.no_remove_bg:
        prepared = img.convert("RGB")
    else:
        if img.mode in ("RGBA", "LA") and np.asarray(img.convert("RGBA"))[:, :, 3].min() < 250:
            rgba = img.convert("RGBA")  # a cut-out already: its transparency is the mask
        else:
            import rembg
            rgba = remove_background(img, rembg.new_session())
        rgba = resize_foreground(rgba, args.foreground_ratio)
        arr = np.array(rgba).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5  # grey background, as the model expects
        prepared = Image.fromarray((arr * 255.0).astype(np.uint8))
    prepared.save(os.path.join(args.out, "input_prepared.png"))

    progress(35, "Guessing the 3D shape")
    with torch.no_grad():
        codes = model([prepared], device=device)

    progress(60, "Building the mesh")
    mesh = model.extract_mesh(codes, True, resolution=args.resolution)[0]  # trimesh with vertex colours
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
    if len(faces) == 0:
        fail("the model found no object in the photo. Try a clearer photo of one object on a plain background.")

    # TripoSR's output has Z up and the front of the object facing +X. The renderer is Y up and the Scene Builder's camera looks from +Z:
    # (x, y, z) -> (y, z, x) is a turn (not a mirror image) that stands the object up with its front facing +Z.
    verts = verts[:, [1, 2, 0]]
    verts -= [(verts[:, 0].min() + verts[:, 0].max()) / 2, verts[:, 1].min(), (verts[:, 2].min() + verts[:, 2].max()) / 2]
    height = verts[:, 1].max()
    verts *= 2.0 / max(height, 1e-9)
    # Outward-facing triangles: a negative enclosed volume means the winding came out inside-out.
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:
        faces = faces[:, ::-1]

    progress(72, "Unwrapping the surface")
    vmapping, indices, uvs = xatlas.parametrize(verts.astype(np.float32), faces.astype(np.uint32))
    progress(82, "Painting the texture")
    tex = bake_vertex_colours(np, np.asarray(uvs, dtype=np.float32), np.asarray(indices, dtype=np.int64), colours[vmapping], args.texture_size)
    # In pbrt, v = 0 is the bottom of the image; PNG rows run top to bottom.
    Image.fromarray((np.clip(tex, 0, 1) * 255.0 + 0.5).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(args.out, "texture.png"))

    progress(95, "Writing the files")
    out_verts = verts[vmapping]
    normals = trimesh.Trimesh(vertices=out_verts, faces=np.asarray(indices), process=False).vertex_normals
    mesh_path = os.path.join(args.out, "mesh.obj")
    with open(mesh_path, "w", newline="\n") as f:
        f.write("# made by photo_to_mesh.py (TripoSR); Y up, standing on y = 0\n")
        for p in out_verts:
            f.write("v %.6f %.6f %.6f\n" % (p[0], p[1], p[2]))
        for t in np.asarray(uvs):
            f.write("vt %.6f %.6f\n" % (t[0], t[1]))
        for n in normals:
            f.write("vn %.6f %.6f %.6f\n" % (n[0], n[1], n[2]))
        for tri in np.asarray(indices):
            a, b, c = (int(i) + 1 for i in tri)
            f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (a, a, a, b, b, b, c, c, c))
    size = (out_verts.max(axis=0) - out_verts.min(axis=0)).tolist()
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump({"mesh": "mesh.obj", "texture": "texture.png", "size": size, "triangles": int(len(indices)), "seconds": round(time.time() - started, 1)}, f)
    progress(100, "Done in %.0f s" % (time.time() - started))


if __name__ == "__main__":
    main()
