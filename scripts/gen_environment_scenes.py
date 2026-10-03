"""Generates pbrt_scenes/environment-*.pbrt (the H1-H12 environment scenes) from models/*.obj + .mtl.

Usage:  python scripts/gen_environment_scenes.py H1 H4 ...   (run from anywhere; needs Pillow)

Ports load_obj_mtl()'s material rules (src/TheRestOfYourLife/mesh_mtl.h) to pbrt directives, one
Material + Shape per .mtl material, using the "file.obj#material" group path (ply_mesh.h).
"""
import os, sys, re, math

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(REPO, "models")
OUT = os.path.join(REPO, "pbrt_scenes")

try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    Image = None

SCENES = {
    "H1": dict(slug="environment-sponza", obj="sponza.obj", tex="sponza_textures", off=(60.52, 126.44, 38.69), scale=1.0,
               sky=(1.3, 1.56, 1.9), fb=(0.80, 0.74, 0.62), cam=(70, (-800, 300, 0), (800, 300, 0)), spp=150,
               title="Crytek Sponza", legacy=62),
    "H2": dict(slug="environment-bistro-exterior", obj="bistro_exterior.obj", tex="bistro_textures", off=(-1526.37, 472.62, -267.01),
               scale=1.0, sky=(0.55, 0.72, 0.95), fb=(0.75, 0.62, 0.50), cam=(60, (1500, 700, 1700), (4000, 700, 2000)), spp=150,
               title="Amazon Lumberyard Bistro (exterior)", legacy=63),
    "H3": dict(slug="environment-rungholt", obj="rungholt.obj", tex="", off=(0, 0, 0), scale=1.0,
               sky=(0.55, 0.72, 0.95), fb=(0.62, 0.48, 0.34), cam=(45, (400, 300, 400), (0, 40, 0)), spp=150,
               title="Rungholt", legacy=64),
    "H4": dict(slug="environment-fireplace-room", obj="fireplace_room.obj", tex="fireplace_room_textures", off=(-2.305, 0.003, 1.518),
               scale=1.0, sky=(0.6, 0.75, 0.95), fb=(0.55, 0.45, 0.35), cam=(55, (-2.0, 1.6, -1.5), (0, 1.3, 0)), spp=150,
               title="Fireplace Room", legacy=73),
    "H5": dict(slug="environment-san-miguel", obj="san_miguel.obj", tex="san_miguel_textures", off=(-12.25, 0.463, -1.4475),
               scale=1.0, sky=(1.4, 1.68, 2.0), fb=(0.75, 0.65, 0.55), cam=(45, (10, 3, 5), (0, 3, 0)), spp=150,
               title="San Miguel", legacy=74),
    "H6": dict(slug="environment-sibenik-cathedral", obj="sibenik_cathedral.obj", tex="sibenik_cathedral_textures", off=(0.0, 15.3123, 0.0),
               scale=1.0, sky=(4.5, 4.8, 5.2), fb=(0.72, 0.71, 0.65), cam=(60, (-15, 1.7, 0), (15, 5, 0)), spp=400,
               title="Sibenik Cathedral", legacy=75),
    "H7": dict(slug="environment-breakfast-room", obj="breakfast_room.obj", tex="breakfast_room_textures", off=(0.54, 1.42, -2.67),
               scale=1.0, sky=(1.1, 1.2, 1.35), fb=(0.6, 0.55, 0.5), cam=(70, (-3.0, 1.5, 3.0), (2.5, 1.3, 0)), spp=300,
               title="Breakfast Room", legacy=76),
    "H8": dict(slug="environment-salle-de-bain", obj="salle_de_bain.obj", tex="salle_de_bain_textures", off=(0.08, -0.03, 0.39),
               scale=1.0, sky=(0.4, 0.45, 0.5), fb=(0.85, 0.85, 0.85), cam=(50, (10, 15, -5), (-10, 12, 5)), spp=150,
               title="Salle de Bain", legacy=77),
    "H9": dict(slug="environment-gallery", obj="gallery.obj", tex="gallery_textures", off=(0.60, -0.06, 1.33),
               scale=1.0, sky=(3.0, 3.2, 3.6), fb=(0.6, 0.55, 0.45), cam=(55, (0, 2.2, -5), (0, 2.2, 0)), spp=300,
               title="Gallery", legacy=78),
    "H10": dict(slug="environment-lost-empire", obj="lost_empire.obj", tex="lost_empire_textures", off=(-0.51, 0.0, -0.56),
                scale=1.0, sky=(0.5, 0.6, 0.8), fb=(0.6, 0.6, 0.6), cam=(55, (0, 60, 100), (0, 10, 0)), spp=150,
                title="Lost Empire", legacy=79),
    "H11": dict(slug="environment-vokselia-spawn", obj="vokselia_spawn.obj", tex="vokselia_spawn_textures", off=(0, 0, 0),
                scale=1.0, sky=(0.5, 0.6, 0.8), fb=(0.6, 0.6, 0.6), cam=(40, (4.5, 0.9, 4.5), (0, 0.25, 0)), spp=100,
                title="Vokselia Spawn", legacy=80),
    "H12": dict(slug="environment-power-plant", obj="powerplant.obj", tex="", off=(-40.267, 0.0, -26.8903),
                scale=0.0004, sky=(0.5, 0.6, 0.8), fb=(0.6, 0.6, 0.6), cam=(40, (130, 85, 130), (-55, 40, -35)), spp=150,
                title="Power Plant", legacy=81),
}


def num(x):
    s = ("%.9g" % x)
    return s


def parse_mtl(path):
    """newmtl -> {key: [tokens...]} following mtl_parse.h's `ss >> tok` + per-key rules."""
    mats = {}
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            t = line.split()
            if not t:
                continue
            k = t[0]
            if k == "newmtl":
                cur = t[1] if len(t) > 1 else None
                if cur is not None:
                    mats.setdefault(cur, {})
            elif cur:
                mats[cur].setdefault("_raw", {})[k] = line.split(None, 1)[1].rstrip("\r\n") if len(t) > 1 else ""
    return mats


def floats(s, n=3):
    v = []
    for tok in s.split()[:n]:
        try:
            v.append(float(tok))
        except ValueError:
            v.append(0.0)
    while len(v) < n:
        v.append(0.0)
    return v


def extract_texture_path(rest):
    """mtl_parse::extract_texture_path - skips leading -opt <numeric args...> groups."""
    p = rest
    pos = len(p) - len(p.lstrip(" \t"))
    while pos < len(p) and p[pos] == "-":
        m = re.compile(r"[ \t]+").search(p, pos)
        if not m:
            pos = len(p)
            break
        pos = m.end()
        while pos < len(p):
            m2 = re.compile(r"[ \t]+").search(p, pos)
            end = m2.start() if m2 else len(p)
            tok = p[pos:end]
            try:
                float(tok)
            except ValueError:
                break
            pos = m2.end() if m2 else len(p)
    return p[pos:].strip(" \t\r\n")


def resolve_texture(rel, texdir):
    p = rel.replace("\\", "/")
    pos = 0
    while True:
        if p.startswith("../", pos):
            pos += 3
        elif p.startswith("./", pos):
            pos += 2
        else:
            break
    return texdir + "/" + p[pos:]


def has_real_tf(tf):
    return any(abs(c - 1.0) > 0.05 for c in tf)


def load_ok(path):
    if not os.path.isfile(path):
        return False
    if Image is None:
        return True
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


_gray_cache = {}


def is_grayscale(path):
    """is_grayscale_image(): 8x8 grid sample, max channel diff <= 10."""
    if path in _gray_cache:
        return _gray_cache[path]
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        px = im.load()
        mx = 0
        for sy in range(8):
            y = (sy * h) // 8
            for sx in range(8):
                x = (sx * w) // 8
                r, g, b = px[x, y]
                mx = max(mx, abs(r - g), abs(g - b), abs(r - b))
    _gray_cache[path] = mx <= 10
    return _gray_cache[path]


def bbox_diag(path):
    """Length of the OBJ's axis-aligned bounding-box diagonal, in the file's own units."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with open(path, "rb") as f:
        for raw in f:
            if raw[:2] == b"v ":
                try:
                    x, y, z = (float(c) for c in raw[2:].split()[:3])
                except ValueError:
                    continue
                for i, c in enumerate((x, y, z)):
                    lo[i] = min(lo[i], c)
                    hi[i] = max(hi[i], c)
    return math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(3)))


def scan_obj(path):
    """Streams the OBJ once: (mtllib, ordered usemtl names that own >=1 face, faces-before-first-usemtl?)."""
    mtllib = ""
    used = []
    seen = set()
    cur = ""          # "" == before any usemtl
    stray = False
    with open(path, "rb") as f:
        for raw in f:
            if raw[:2] == b"f ":
                if cur == "":
                    stray = True
                elif cur not in seen:
                    seen.add(cur)
                    used.append(cur)
            elif raw[:7] == b"usemtl ":
                cur = raw[7:].decode("utf-8", "replace").strip()
            elif raw[:7] == b"mtllib " and not mtllib:
                mtllib = raw[7:].decode("utf-8", "replace").strip()
    return mtllib, used, stray


def generate(sid, cfg):
    obj_path = os.path.join(MODELS, cfg["obj"])
    mtllib, used, stray = scan_obj(obj_path)
    mtl_path = os.path.join(MODELS, mtllib) if mtllib else ""
    mats = parse_mtl(mtl_path) if mtl_path and os.path.isfile(mtl_path) else {}
    # load_obj_mtl: fall back to "<obj>.mtl" when the mtllib has no Kd lines at all
    if not any("Kd" in m.get("_raw", {}) for m in mats.values()):
        alt = os.path.join(MODELS, os.path.splitext(cfg["obj"])[0] + ".mtl")
        if os.path.isfile(alt):
            mats = parse_mtl(alt)
            mtl_path = alt
    texdir = cfg["tex"]
    tex_root = os.path.join(MODELS, texdir) if texdir else None

    def raw(name, key):
        return mats.get(name, {}).get("_raw", {}).get(key)

    def first_raw(name, *keys):
        for k in keys:
            v = raw(name, k)
            if v is not None:
                return v
        return None

    tex_decl = {}      # (kind, rel path) -> texture name
    decls = []

    def texture(kind, relpath):
        key = (kind, relpath)
        if key not in tex_decl:
            nm = "%s%d" % ({"spectrum": "c", "float": "f"}[kind], len(tex_decl))
            tex_decl[key] = nm
            decls.append('Texture "%s" "%s" "imagemap" "string filename" [ "../models/%s" ]' % (nm, kind, relpath))
        return tex_decl[key]

    # A grayscale map_Bump is a height map with no stated unit, but pbrt's bump mapping reads it as a
    # displacement in world units (scale 1 would be a one-unit relief: wildly strong normals). Scale it to
    # a relief of 0.03% of the model's size - about a centimetre on a 40 m building - which is what the
    # native loader's nearest-neighbour lookup effectively produced. Normal maps need no scale.
    relief_state = {}
    scaled_decl = {}

    def relief_scaled(tex_name):
        if tex_name not in scaled_decl:
            if "relief" not in relief_state:
                relief_state["relief"] = 0.0003 * bbox_diag(obj_path)
            nm = tex_name + "s"
            decls.append('Texture "%s" "float" "scale" "float scale" [ %s ] "texture tex" [ "%s" ]'
                         % (nm, num(relief_state["relief"]), tex_name))
            scaled_decl[tex_name] = nm
        return scaled_decl[tex_name]

    stats = {"emissive": 0, "glass": 0, "metal": 0, "textured": 0, "flat": 0, "fallback": 0, "bump": 0, "normal": 0, "alpha": 0}
    blocks = []
    for name in used + ([""] if stray else []):
        lines = []
        emissive = None
        ke = raw(name, "Ke")
        if ke is not None:
            kev = floats(ke)
            if kev[0] > 1e-6 or kev[1] > 1e-6 or kev[2] > 1e-6:
                emissive = kev
        base = None      # the Material line
        kd_raw = raw(name, "Kd")
        kd = floats(kd_raw) if kd_raw is not None else None
        map_kd = None
        if texdir:
            r = raw(name, "map_Kd")
            if r is not None:
                p = extract_texture_path(r)
                if p:
                    map_kd = resolve_texture(p, texdir)
        ke_tex = None
        if texdir and emissive is None:
            r = raw(name, "map_Ke")
            if r is not None:
                p = extract_texture_path(r)
                if p:
                    ke_tex = resolve_texture(p, texdir)
        spec_illum = int(floats(raw(name, "illum") or "-1", 1)[0]) if raw(name, "illum") is not None else -1
        ks = floats(raw(name, "Ks")) if raw(name, "Ks") is not None else [0, 0, 0]
        ns = floats(raw(name, "Ns"), 1)[0] if raw(name, "Ns") is not None else 0.0
        ni = floats(raw(name, "Ni"), 1)[0] if raw(name, "Ni") is not None else -1.0
        tf = floats(raw(name, "Tf")) if raw(name, "Tf") is not None else [1, 1, 1]
        area = None
        emissive_mat = False
        if emissive is not None:
            area = 'AreaLightSource "diffuse" "rgb L" [ %s ]' % " ".join(num(c) for c in emissive)
            base = 'Material "diffuse" "rgb reflectance" [ 0 0 0 ]'
            emissive_mat = True
            stats["emissive"] += 1
        elif ke_tex and load_ok(os.path.join(MODELS, ke_tex)):
            # map_Ke with no scalar Ke (Gallery): textured emission, not registered as a sampled light
            area = 'AreaLightSource "diffuse" "string filename" [ "../models/%s" ]' % ke_tex
            base = 'Material "diffuse" "rgb reflectance" [ 0 0 0 ]'
            emissive_mat = True
            stats["emissive"] += 1
        else:
            tfs = " ".join(num(c) for c in tf)
            if spec_illum == 7 or (spec_illum in (4, 6) and has_real_tf(tf) and map_kd is None):
                eta = ni if ni > 0 else 1.5
                extra = (' "rgb tf" [ %s ]' % tfs) if has_real_tf(tf) else ""
                base = 'Material "dielectric" "float eta" [ %s ]%s' % (num(eta), extra)
                stats["glass"] += 1
            elif spec_illum in (2, 3) and max(ks) > 0.02:
                kdv = kd if kd is not None else [1, 1, 1]
                rough = math.sqrt(2.0 / (ns + 2.0))
                base = 'Material "conductor" "rgb reflectance" [ %s ] "float roughness" [ %s ]' % (
                    " ".join(num(c) for c in kdv), num(rough))
                stats["metal"] += 1
            elif map_kd and load_ok(os.path.join(MODELS, map_kd)):
                base = 'Material "diffuse" "texture reflectance" [ "%s" ]' % texture("spectrum", map_kd)
                stats["textured"] += 1
            elif kd is not None:
                base = 'Material "diffuse" "rgb reflectance" [ %s ]' % " ".join(num(c) for c in kd)
                stats["flat"] += 1
            else:
                base = 'Material "diffuse" "rgb reflectance" [ %s ]' % " ".join(num(c) for c in cfg["fb"])
                stats["fallback"] += 1
        # map_Bump wraps the resolved base material (not emissive ones)
        if texdir and not emissive_mat:
            bump = first_raw(name, "map_Bump", "map_bump", "bump")
            if bump is not None:
                bp = bump.strip(" \t\r\n")
                if bp:
                    rel = resolve_texture(bp, texdir)
                    full = os.path.join(MODELS, rel)
                    if load_ok(full):
                        gray = is_grayscale(full)
                        disp = texture("float" if gray else "spectrum", rel)
                        if gray:
                            disp = relief_scaled(disp)
                        base += ' "texture displacement" [ "%s" ]' % disp
                        stats["bump" if gray else "normal"] += 1
        alpha_clause = ""
        if texdir:
            a = first_raw(name, "map_d", "map_D")
            if a is not None:
                ap = a.strip(" \t\r\n")
                if ap:
                    rel = resolve_texture(ap, texdir)
                    if load_ok(os.path.join(MODELS, rel)):
                        alpha_clause = ' "texture alpha" [ "%s" ]' % texture("float", rel)
                        stats["alpha"] += 1
        blocks.append((name, area, base, alpha_clause))

    sky = cfg["sky"]
    fov, eye, at = cfg["cam"]
    off = cfg["off"]
    out = []
    out.append("# %s.pbrt -- migration of the native \"%s %s\" environment scene (scenes_mesh_gallery.h's build_*()\n"
               "# + its GPU twin) to a pbrt-backed scene. Generated by scripts/gen_environment_scenes.py - rerun it\n"
               "# rather than hand-editing: it ports load_obj_mtl()'s per-material rules (mesh_mtl.h) to explicit pbrt directives."
               % (cfg["slug"], sid, cfg["title"]))
    out.append("#")
    out.append("# The mesh is models/%s, named once per .mtl material as Shape \"plymesh\" \"file.obj#material\"" % cfg["obj"])
    out.append("# (ply_mesh.h's OBJ group paths), so the multi-megabyte asset stays where it is. Per material: Ke -> an")
    out.append("# area light; illum 7 (or illum 4/6 with a real Tf and no map_Kd) -> dielectric (\"rgb tf\" is the .mtl")
    out.append("# transmission filter); illum 2/3 with a real Ks -> conductor; map_Kd -> a diffuse imagemap; else Kd;")
    out.append("# map_Bump -> a displacement texture (a float imagemap for a grayscale bump, a spectrum one for a")
    out.append("# tangent-space normal map); map_d -> the Shape's alpha cutout. Faces whose material the .mtl does not")
    out.append("# define use the native scene's fallback colour.")
    out.append("#")
    out.append("# Materials: " + ", ".join("%s=%d" % kv for kv in stats.items() if kv[1]) + " (of %d shapes)" % len(blocks))
    out.append("# Requires models/%s%s (registry requires_files=true)." % (cfg["obj"], (" and models/%s/" % texdir) if texdir else ""))
    out.append("")
    out.append("LookAt  %s    %s    0 1 0" % (" ".join(num(c) for c in eye), " ".join(num(c) for c in at)))
    out.append('Camera "perspective" "float fov" [ %s ]' % num(fov))
    out.append("")
    out.append('Film "rgb" "integer xresolution" [ 600 ] "integer yresolution" [ 600 ]')
    out.append('Sampler "halton" "integer pixelsamples" [ %d ]' % cfg["spp"])
    out.append("")
    out.append("WorldBegin")
    out.append("")
    out.append("# Flat sky = the native scene's constant sky_light colour.")
    out.append('LightSource "infinite" "rgb L" [ %s ]' % " ".join(num(c) for c in sky))
    out.append("")
    if decls:
        out.extend(decls)
        out.append("")
    out.append("# The native loader applied x*scale + offset: Translate then Scale.")
    out.append("AttributeBegin")
    out.append("  Translate %s" % " ".join(num(c) for c in off))
    if cfg["scale"] != 1.0:
        out.append("  Scale %s %s %s" % ((num(cfg["scale"]),) * 3))
    for name, area, base, alpha_clause in blocks:
        out.append("")
        out.append("  AttributeBegin")
        out.append("    # %s" % (name if name else "(faces before the first usemtl)"))
        if area:
            out.append("    " + area)
        out.append("    " + base)
        out.append('    Shape "plymesh" "string filename" [ "../models/%s#%s" ]%s' % (cfg["obj"], name, alpha_clause))
        out.append("  AttributeEnd")
    out.append("AttributeEnd")
    out.append("")
    path = os.path.join(OUT, cfg["slug"] + ".pbrt")
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(sid, cfg["slug"], len(blocks), "shapes", {k: v for k, v in stats.items() if v}, "stray" if stray else "")


if __name__ == "__main__":
    for sid in sys.argv[1:]:
        generate(sid, SCENES[sid])
