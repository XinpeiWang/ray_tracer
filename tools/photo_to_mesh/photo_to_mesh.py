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

Needs the Python environment made by scripts/setup_photo_to_mesh.ps1 on Windows or scripts/setup_photo_to_mesh.sh on macOS (PyTorch, TripoSR, rembg, xatlas, scikit-image).
"""
import argparse
import json
import os
import sys
import time

# On Apple silicon PyTorch runs on the GPU through MPS; an operator MPS lacks then runs on the processor instead of failing. Must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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


def orient_mesh(np, verts, faces):
    """Puts TripoSR's mesh where the Scene Builder wants it: Y up, standing on y = 0, centred on x and z, 2 units tall, front towards +Z,
    triangles facing outward. TripoSR's output has Z up and the object's front facing +X; (x, y, z) -> (y, z, x) is a turn (not a mirror
    image) that stands it up with its front facing +Z. verts is (n, 3) floats, faces (m, 3) ints; returns new arrays."""
    verts = np.asarray(verts, dtype=np.float64)[:, [1, 2, 0]]
    verts = verts - [(verts[:, 0].min() + verts[:, 0].max()) / 2, verts[:, 1].min(), (verts[:, 2].min() + verts[:, 2].max()) / 2]
    verts = verts * (2.0 / max(verts[:, 1].max(), 1e-9))
    faces = np.asarray(faces, dtype=np.int64)
    # A negative enclosed volume means the winding came out inside-out.
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    if np.einsum("ij,ij->", v0, np.cross(v1, v2)) < 0:
        faces = np.ascontiguousarray(faces[:, ::-1])
    return verts, faces


def check_environment(triposr_arg):
    """Prints one "Key: value" fact per line about everything the helper needs, for the Diagnostics tab. The wording is what the
    tab colours by: "present" / "available" are good, "missing" / "not available" / "not usable" are problems."""
    import platform
    from importlib import metadata

    def fact(key, value):
        print("%s: %s" % (key, value), flush=True)

    def size(path):
        return "%.1f GB" % (os.path.getsize(path) / 1e9) if os.path.getsize(path) > 5e8 else "%d MB" % (os.path.getsize(path) // 1000000)

    fact("Python", "%s (%s)" % (platform.python_version(), sys.executable))
    for name, dist in (("PyTorch", "torch"), ("Transformers", "transformers"), ("rembg", "rembg"), ("xatlas", "xatlas"),
                       ("scikit-image", "scikit-image"), ("SciPy", "scipy"), ("trimesh", "trimesh"), ("NumPy", "numpy"),
                       ("Pillow", "Pillow"), ("einops", "einops"), ("OmegaConf", "omegaconf"),
                       ("Hugging Face Hub", "huggingface-hub"), ("ONNX Runtime", "onnxruntime")):
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            fact(name, "missing (not installed; run scripts/setup_photo_to_mesh.ps1 or .sh)")
            continue
        if dist == "transformers" and int(version.split(".")[0]) >= 5:
            fact(name, "%s not usable (TripoSR's model needs a version below 5)" % version)
        else:
            fact(name, "%s present" % version)
    try:
        metadata.version("torch")
    except metadata.PackageNotFoundError:
        fact("Graphics Card for PyTorch", "unknown (PyTorch is not installed)")
    else:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                fact("Graphics Card for PyTorch", "available (%s, %.0f GB, CUDA %s)" % (props.name, props.total_memory / 2**30, torch.version.cuda))
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                fact("Graphics Card for PyTorch", "available (Apple Metal; not tested with this helper)")
            else:
                fact("Graphics Card for PyTorch", "not available (a photo runs on the processor and takes several minutes)")
        except Exception as e:  # a broken install can fail in many ways
            fact("PyTorch Import", "not usable (%s)" % str(e).splitlines()[0][:150])
    triposr = find_triposr(triposr_arg)
    fact("TripoSR Code", "present (%s)" % triposr if triposr else "missing (run scripts/setup_photo_to_mesh.ps1 or .sh)")
    try:
        from huggingface_hub import try_to_load_from_cache
        ckpt = try_to_load_from_cache("stabilityai/TripoSR", "model.ckpt")
        if isinstance(ckpt, str) and os.path.isfile(ckpt):
            fact("TripoSR Weights", "present (%s, cached)" % size(ckpt))
        else:
            fact("TripoSR Weights", "missing (about 1.7 GB, downloaded the first time a photo is converted)")
    except ImportError:
        fact("TripoSR Weights", "unknown (huggingface-hub is not installed)")
    except Exception as e:
        fact("TripoSR Weights", "not usable (%s)" % str(e).splitlines()[0][:150])
    # rembg keeps models in ~/.rembg/models/<name>/<name>.onnx (older versions: ~/.u2net/<name>.onnx)
    candidates = [os.path.join(os.path.expanduser("~"), ".rembg", "models", "u2net", "u2net.onnx"),
                  os.path.join(os.environ.get("U2NET_HOME", os.path.join(os.path.expanduser("~"), ".u2net")), "u2net.onnx")]
    u2net = next((c for c in candidates if os.path.isfile(c)), candidates[0])
    if os.path.isfile(u2net):
        fact("Background Remover Model (U2-Net)", "present (%s, cached)" % size(u2net))
    else:
        fact("Background Remover Model (U2-Net)", "missing (about 176 MB, downloaded the first time a photo needs its background removed)")


def main():
    ap = argparse.ArgumentParser(description="One photo -> a textured mesh (TripoSR, runs locally).")
    ap.add_argument("image", nargs="?", help="a photo of one object")
    ap.add_argument("--out", help="folder for mesh.obj, texture.png and result.json")
    ap.add_argument("--check", action="store_true", help="report what this environment has (for the GUI's Diagnostics tab) and exit")
    ap.add_argument("--resolution", type=int, default=256, help="marching-cubes grid; higher = finer mesh, slower (default 256)")
    ap.add_argument("--texture-size", type=int, default=1024, help="texture width and height in pixels (default 1024)")
    ap.add_argument("--no-remove-bg", action="store_true", help="the photo already has a plain background and the object fills most of it")
    ap.add_argument("--foreground-ratio", type=float, default=0.85)
    ap.add_argument("--triposr-dir", help="folder containing TripoSR's tsr package")
    ap.add_argument("--device", default="auto", help="cuda:0, cpu or auto (default)")
    args = ap.parse_args()

    if args.check:
        check_environment(args.triposr_dir)
        return
    if not args.image or not args.out:
        ap.error("a photo and --out are needed (or use --check)")
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
        fail("a Python package is missing (%s). Run scripts/setup_photo_to_mesh.ps1 (Windows) or .sh (macOS) first." % e)
    triposr = find_triposr(args.triposr_dir)
    if not triposr:
        fail("TripoSR was not found. Run scripts/setup_photo_to_mesh.ps1 (Windows) or .sh (macOS), or pass --triposr-dir.")
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
            rgba = remove_background(img, rembg.new_session("u2net"))  # U2-Net: Apache-2.0, 176 MB (rembg's own default is a 1 GB non-commercial model)
        rgba = resize_foreground(rgba, args.foreground_ratio)
        arr = np.array(rgba).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5  # grey background, as the model expects
        prepared = Image.fromarray((arr * 255.0).astype(np.uint8))

    def run_model(dev):
        progress(35, "Guessing the 3D shape")
        with torch.no_grad():
            codes = model([prepared], device=dev)
        progress(60, "Building the mesh")
        return model.extract_mesh(codes, True, resolution=args.resolution)[0]  # trimesh with vertex colours

    try:
        mesh = run_model(device)
    except Exception as e:  # Apple's GPU backend (MPS) cannot run every operation this model uses: the processor always can
        if device != "mps":
            raise
        print("NOTE the Apple GPU could not run the model (%s); running on the processor instead, which takes several minutes." % str(e)[:200], flush=True)
        device = "cpu"
        model.to(device)
        mesh = run_model(device)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colours = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
    if len(faces) == 0:
        fail("the model found no object in the photo. Try a clearer photo of one object on a plain background.")

    verts, faces = orient_mesh(np, verts, faces)

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
