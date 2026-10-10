#!/usr/bin/env python3
"""Consistency sweep: the same scene written two ways must give the same picture.

A renderer can agree with another renderer and still be wrong, because both can share a mistake. It cannot disagree with ITSELF about two descriptions of the same
thing. This renders pairs of scenes that must look alike, on each backend, and reports the pairs that do not:

  sky         a constant sky ("rgb L") against the same sky as a white picture, for every material
  translate   the whole scene moved by an arbitrary offset
  scale       the whole scene made 8 times bigger (distances, radii, light intensities)
  mirror      the whole scene mirrored left to right
  lightform   a point light against a small emissive sphere of the same intensity
  size        the same scene at two picture sizes
  spp         the same scene at 16 and 128 samples a pixel

The means are compared, per colour channel, within a tolerance. Needs a built `ray_tracer` (build_macos/ray_tracer by default; --gpu is Metal on a Mac) and nothing else:
the EXR reader is in this file.

  python3 scripts/consistency_sweep.py [--exe PATH] [--backends cpu,gpu] [--only sky,translate,...] [--tolerance 4] [-v]

Exit code 1 when a pair is off by more than the tolerance. See docs/CONSISTENCY_SWEEP.md.
"""
import argparse
import os
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

_LINES = {0: 1, 2: 1, 3: 16}


def read_exr_means(path):
    """Mean of each of R, G, B of a scanline OpenEXR file (none, ZIPS or ZIP compression, float or half channels are not needed: ray_tracer writes float)."""
    b = open(path, "rb").read()
    assert b[:4] == b"v/1\x01", "not an OpenEXR file"
    i = 8
    attrs = {}
    while b[i] != 0:
        j = b.index(b"\0", i); name = b[i:j].decode(); i = j + 1
        j = b.index(b"\0", i); i = j + 1
        size = struct.unpack("<I", b[i:i + 4])[0]; i += 4
        attrs[name] = b[i:i + size]; i += size
    pos = i + 1
    ch = attrs["channels"]; names = []; k = 0
    while ch[k] != 0:
        j = ch.index(b"\0", k); names.append(ch[k:j].decode()); k = j + 1 + 16
    comp = attrs["compression"][0]; lpb = _LINES[comp]
    x0, y0, x1, y1 = struct.unpack("<4i", attrs["dataWindow"])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    nb = (H + lpb - 1) // lpb
    offs = struct.unpack("<%dQ" % nb, b[pos:pos + 8 * nb])
    sums = {n: 0.0 for n in names}
    for off in offs:
        y, size = struct.unpack("<ii", b[off:off + 8]); data = b[off + 8:off + 8 + size]
        rows = min(lpb, H - (y - y0)); raw = rows * len(names) * W * 4
        if comp != 0 and size < raw:
            data = zlib.decompress(data)
            a = bytearray(data)
            for t in range(1, len(a)):
                a[t] = (a[t - 1] + a[t] - 128) & 0xFF
            half = (len(a) + 1) // 2
            d = bytearray(len(a)); d[0::2] = a[:half]; d[1::2] = a[half:]; data = bytes(d)
        vals = struct.unpack("<%df" % (rows * len(names) * W), data)
        p = 0
        for _ in range(rows):
            for n in names:
                sums[n] += sum(vals[p:p + W]); p += W
    return [sums[c] / (W * H) for c in ("R", "G", "B")]


# ---- the scene: a floor, a ball of the material under test and a lamp, in a sky --------------------------------------------------------------------------------

MATERIALS = {
    "diffuse": 'Material "diffuse" "rgb reflectance" [ 0.6 0.4 0.3 ]',
    "conductor": 'Material "conductor" "rgb reflectance" [ 0.9 0.6 0.4 ] "float roughness" [ 0.1 ]',
    "dielectric": 'Material "dielectric" "float eta" [ 1.5 ]',
    "roughdielectric": 'Material "dielectric" "float eta" [ 1.5 ] "float roughness" [ 0.2 ]',
    "thindielectric": 'Material "thindielectric" "float eta" [ 1.5 ]',
    "coateddiffuse": 'Material "coateddiffuse" "rgb reflectance" [ 0.5 0.3 0.2 ] "float roughness" [ 0.1 ]',
    "coatedconductor": 'Material "coatedconductor" "rgb reflectance" [ 0.9 0.7 0.5 ] "float interface.roughness" [ 0.05 ] "float conductor.roughness" [ 0.1 ]',
    "diffusetransmission": 'Material "diffusetransmission" "rgb reflectance" [ 0.5 0.4 0.3 ] "rgb transmittance" [ 0.3 0.3 0.3 ]',
    # the mean free path is a length: scaling the scene scales it too (or the ball is a different ball)
    "subsurface": lambda sc: 'Material "subsurface" "rgb mfp" [ %g %g %g ] "rgb reflectance" [ 0.8 0.8 0.8 ] "float eta" [ 1.33 ] "float roughness" [ 0 ]' % ((0.4 * sc.scale,) * 3),
    # participating media: a sphere of "interface" material holding the medium (sigma is per unit length, so it scales with 1/scale)
    "fog": lambda sc: 'Material "interface"\n  MediumInterface "m" ""',
    "cloud": lambda sc: 'Material "interface"\n  MediumInterface "puffcloud" ""',
}

# materials whose light paths go through a specular chain a point light cannot reach (a point light cannot be hit by a random walk through glass, so the caustic a sphere
# lamp makes is not there): the point-against-sphere-lamp pair is not a fair one for them
SPECULAR = {"dielectric", "roughdielectric", "thindielectric"}
NOISY = {"dielectric", "thindielectric"}   # fireflies: a pair needs a looser tolerance


def medium_prelude(sc):
    """MakeNamedMedium for the media 'materials', placed in the scene's own (scaled, moved, mirrored) frame."""
    s_ = sc.scale
    if sc.material == "fog":
        return 'MakeNamedMedium "m" "string type" [ "homogeneous" ] "rgb sigma_a" [ %g %g %g ] "rgb sigma_s" [ %g %g %g ] "float g" [ 0.3 ]' % ((0.05 / s_,) * 3 + (0.6 / s_,) * 3)
    if sc.material == "cloud":
        # a cloud whose box (2.4 across) is bigger than its interface sphere (radius 1): the sphere clips it
        c = sc.p(0, 0, 0) if sc.fill else sc.p(0.5, 1.0, 0)
        return ('AttributeBegin\n  Translate %s\n  Scale %g %g %g\n  Translate -0.5 -0.5 -0.5\n  MakeNamedMedium "puffcloud" "string type" [ "cloud" ] "rgb sigma_a" [ 0 0 0 ] "rgb sigma_s" [ %g %g %g ] '
                '"float g" [ 0.3 ] "float density" [ 1 ] "float wispiness" [ 1 ] "float frequency" [ 4 ]\nAttributeEnd') % (sc.f(c), 2.4 * s_, 2.4 * s_, 2.4 * s_, 0.5 / s_, 0.5 / s_, 0.5 / s_)
    return ""


class Scene:
    """A scene description with the knobs the sweep turns: offset, scale, mirror, sky form, lamp form, size, samples."""

    def __init__(self, material="diffuse", sky="constant", lamp="area", offset=(0, 0, 0), scale=1.0, mirror=False, size=48, spp=64, fill=False, depth=8, sky_png=None):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def p(self, x, y, z):
        """A point of the base scene, mirrored, scaled and moved."""
        if self.mirror:
            x = -x
        s = self.scale
        return (x * s + self.offset[0], y * s + self.offset[1], z * s + self.offset[2])

    @staticmethod
    def f(v):
        return " ".join("%.9g" % c for c in v)

    def text(self):
        s, p, f = self.scale, self.p, self.f
        eye, look = (p(0, 0, 5) if self.fill else p(0.5, 1.5, 6.0)), (p(0, 0, 0) if self.fill else p(0.2, 0.6, 0))
        fov = 10 if self.fill else 38
        out = ["LookAt %s  %s  0 1 0" % (f(eye), f(look)), 'Camera "perspective" "float fov" [ %g ]' % fov,
               'Film "rgb" "integer xresolution" [ %d ] "integer yresolution" [ %d ]' % (self.size, self.size),
               'Sampler "halton" "integer pixelsamples" [ %d ]' % self.spp, 'Integrator "volpath" "integer maxdepth" [ %d ]' % self.depth, "WorldBegin"]
        if self.sky == "constant":
            out.append('LightSource "infinite" "rgb L" [ %s ]' % ("1 1 1" if self.fill else "0.25 0.25 0.3"))
        else:
            out.append('LightSource "infinite" "string filename" [ "%s" ]' % self.sky_png)
        if not self.fill:
            # a floor: two triangles; mirrored they wind the other way, which a diffuse floor does not care about
            q = [p(-6, 0, -6), p(6, 0, -6), p(6, 0, 6), p(-6, 0, 6)]
            if self.mirror:
                q = [q[1], q[0], q[3], q[2]]   # the mirrored floor keeps its winding (its normal stays up), so the pair differs only by the mirror
            out.append('AttributeBegin\n  Material "diffuse" "rgb reflectance" [ 0.7 0.7 0.7 ]\n  Shape "trianglemesh" "integer indices" [ 0 2 1 0 3 2 ] "point3 P" [ %s ]\nAttributeEnd' % " ".join(f(v) for v in q))
            lamp_at, power = p(-1.2, 3.0, 1.0), 40.0
            if self.lamp == "area":
                out.append('AttributeBegin\n  Translate %s\n  AreaLightSource "diffuse" "rgb L" [ %s ]\n  Shape "sphere" "float radius" [ %g ]\nAttributeEnd' % (f(lamp_at), f((power / (3.14159265 * 0.04),) * 3), 0.2 * s))
            else:
                out.append('LightSource "point" "point3 from" [ %s ] "rgb I" [ %s ]' % (f(lamp_at), f((power * s * s,) * 3)))
        ball = p(0, 0, 0) if self.fill else p(0.5, 1.0, 0)
        material = MATERIALS[self.material]
        if callable(material):
            prelude = medium_prelude(self)
            if prelude:
                out.append(prelude)
            material = material(self)
        out.append('AttributeBegin\n  Translate %s\n  %s\n  Shape "sphere" "float radius" [ %g ]\nAttributeEnd' % (f(ball), material, 1.0 * s))
        return "\n".join(out) + "\n"


def make_white_png(path):
    w, h = 64, 32
    raw = b"".join(b"\0" + bytes([255, 255, 255] * w) for _ in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Sweep:
    def __init__(self, exe, backends, workdir, verbose):
        self.exe, self.backends, self.dir, self.verbose = exe, backends, workdir, verbose
        self.white = os.path.join(workdir, "white.png")
        make_white_png(self.white)
        self.n = 0
        self.cache = {}

    def render(self, backend, scene, seed=3):
        key = (backend, scene.text(), scene.size, scene.spp, seed)
        if key in self.cache:
            return self.cache[key]
        self.n += 1
        pbrt = os.path.join(self.dir, "s%d.pbrt" % self.n)
        out = os.path.join(self.dir, "s%d.exr" % self.n)
        open(pbrt, "w").write(scene.text())
        r = subprocess.run([self.exe, "--" + backend, "--seed", str(seed), "--output", out, str(scene.size), str(scene.spp), str(scene.depth), pbrt], capture_output=True, text=True, cwd=REPO)
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("%s render failed: %s" % (backend, (r.stdout + r.stderr)[-300:]))
        self.cache[key] = read_exr_means(out)
        return self.cache[key]

    def compare(self, label, backend, a, b, tolerance, seed_b=3):
        ma, mb = self.render(backend, a), self.render(backend, b, seed_b)
        worst = max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(ma, mb))
        ok = worst * 100.0 <= tolerance
        if not ok or self.verbose:
            print("  %-4s %-34s %s  %.4f vs %.4f  off by %.1f%%" % (backend, label, "ok   " if ok else "WRONG", ma[1], mb[1], worst * 100.0))
        return ok


def run(args):
    workdir = tempfile.mkdtemp(prefix="consistency_sweep.")
    sw = Sweep(args.exe, args.backends, workdir, args.verbose)
    only = set(args.only.split(",")) if args.only else None
    tol = args.tolerance
    failures = []
    total = 0

    def noise_tol(name):
        return max(tol, 8.0) if name in NOISY else tol

    def check(group, label, backend, a, b, tolerance=None, seed_b=3):
        nonlocal total
        if only and group not in only:
            return
        total += 1
        try:
            if not sw.compare(label, backend, a, b, tolerance if tolerance is not None else tol, seed_b):
                failures.append((group, backend, label))
        except RuntimeError as e:
            print("  %-4s %-34s FAILED TO RENDER: %s" % (backend, label, e))
            failures.append((group, backend, label))

    for backend in args.backends:
        print("[%s]" % backend)
        for name in MATERIALS:
            # sky: a constant against the same sky as a picture (a ball that fills the frame, so the whole picture is the material)
            check("sky", "sky constant vs picture, %s" % name, backend,
                  Scene(material=name, sky="constant", fill=True, size=32, spp=64, depth=30), Scene(material=name, sky="picture", sky_png=sw.white, fill=True, size=32, spp=64, depth=30), tolerance=noise_tol(name))
            base = Scene(material=name)
            check("translate", "translate, %s" % name, backend, base, Scene(material=name, offset=(37.0, -11.0, 5.0)))
            check("scale", "scale x8, %s" % name, backend, base, Scene(material=name, scale=8.0))
            check("mirror", "mirror, %s" % name, backend, base, Scene(material=name, mirror=True), tolerance=noise_tol(name))
            check("size", "size 32 vs 64, %s" % name, backend, Scene(material=name, size=32), Scene(material=name, size=64), tolerance=noise_tol(name))
            check("spp", "spp 16 vs 128, %s" % name, backend, Scene(material=name, spp=16, size=64), Scene(material=name, spp=128, size=64), tolerance=max(noise_tol(name), 6.0))
            if name not in SPECULAR:
                check("lightform", "point vs sphere lamp, %s" % name, backend, Scene(material=name, lamp="point"), Scene(material=name, lamp="area"), tolerance=max(tol, 6.0))
    print("\n%d pairs, %d off by more than %g%%" % (total, len(failures), tol))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--exe", default=os.path.join(REPO, "build_macos", "ray_tracer"))
    ap.add_argument("--backends", default="cpu,gpu")
    ap.add_argument("--only", default="")
    ap.add_argument("--tolerance", type=float, default=4.0, help="percent, per colour channel")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    args.backends = [b for b in args.backends.split(",") if b]
    if not os.path.exists(args.exe):
        sys.exit("ERROR: %s not found - build it first" % args.exe)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
