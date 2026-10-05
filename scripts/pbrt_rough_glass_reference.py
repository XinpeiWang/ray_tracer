"""Independent path-level reference for a rough/smooth dielectric sphere under a sphere lamp.

Pure unidirectional BSDF-sampling path tracer (no NEE, no MIS), float64 numpy, with pbrt-v4's DielectricBxDF
(src/pbrt/bxdfs.cpp: Sample_f, radiance-mode eta^2 scaling) and TrowbridgeReitz (util/scattering.h) written out from the
pbrt source - nothing from this repository's BxDF code. Scene (same as pbrt_scenes/rough-glass-lamp.pbrt):
  camera LookAt (0,2,8) -> (0,1,0) up (0,1,0), perspective fov 40 (shorter axis), 4:3
  glass sphere  center (0,1.5,0) r 1.5, eta 1.5, roughness R (remapped: alpha = sqrt(R))
  lamp sphere   center (0,5,-2)  r 1,   L = 15, diffuse reflectance 0 (absorbs), emits on its outer side
  black background
Output: the mean image value after the renderer's own transform (--tonemap none --exposure E): per pixel clip(E*L),
sRGB encode, 8 bit, decode to linear, divide by E - the quantity to compare with the mean of a real render decoded the same
way. Use an exposure at which nothing clips (0.05 for the lamp): the renderer clips AFTER its film filter, this script
after a box average, so any clipped pixel makes the two differ.

Why this exists: it is the independent check that decided which side was biased in the rough-dielectric-under-a-lamp
discrepancy (CPU and GPU both over-counted NEE at the vertices where a ray leaves glass; see docs/PBRT_SUPPORT.md).
It agrees with the renderers to 0.3% on smooth glass, and its sampler agrees with pbrt's own f() by quadrature to 0.1%.

    python scripts/pbrt_rough_glass_reference.py 0.05 0.25 0.6                 # lamp scene, 1000 spp
    python scripts/pbrt_rough_glass_reference.py 0.25 --sky 0.7333 --exposure 0.5   # sky furnace
Needs only numpy. A few minutes per roughness value at the default size.
"""
import sys, math
import numpy as np

PI = math.pi
EYE = np.array([0.0, 2.0, 8.0]); AT = np.array([0.0, 1.0, 0.0]); UP = np.array([0.0, 1.0, 0.0])
GLASS_C = np.array([0.0, 1.5, 0.0]); GLASS_R = 1.5
LAMP_C = np.array([0.0, 5.0, -2.0]); LAMP_R = 1.0; LAMP_L = 15.0
ETA = 1.5
SKY = 0.0   # uniform sky radiance; > 0 replaces the lamp with an environment (furnace mode), set by --sky


def norm(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(a * b, axis=-1)


# ---- pbrt-v4 FrDielectric / Refract ---------------------------------------------------------------------------------
def fr_dielectric(cos_i, eta):
    cos_i = np.clip(cos_i, -1.0, 1.0)
    eta = np.where(cos_i < 0, 1.0 / eta, eta)
    cos_i = np.abs(cos_i)
    sin2_i = 1.0 - cos_i ** 2
    sin2_t = sin2_i / eta ** 2
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
    return np.where(tir, 1.0, (r_parl ** 2 + r_perp ** 2) / 2.0)


def refract(wi, n, eta):
    """pbrt Refract: returns (valid, wt, etap)."""
    cos_i = dot(n, wi)
    flip = cos_i < 0
    eta = np.where(flip, 1.0 / eta, eta)
    cos_i = np.abs(cos_i)
    n = np.where(flip[:, None], -n, n)
    sin2_i = np.maximum(0.0, 1.0 - cos_i ** 2)
    sin2_t = sin2_i / eta ** 2
    valid = sin2_t < 1.0
    cos_t = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))
    wt = -wi / eta[:, None] + ((cos_i / eta - cos_t)[:, None]) * n
    return valid, wt, eta


# ---- pbrt-v4 TrowbridgeReitzDistribution (isotropic) ----------------------------------------------------------------
def tan2_theta(w):
    cos2 = w[:, 2] ** 2
    sin2 = np.maximum(0.0, 1.0 - cos2)
    return sin2 / np.maximum(cos2, 1e-30)


def D(wm, alpha):
    t2 = tan2_theta(wm)
    cos4 = wm[:, 2] ** 4
    e = t2 / alpha ** 2
    return 1.0 / (PI * alpha ** 2 * np.maximum(cos4, 1e-30) * (1.0 + e) ** 2)


def Lambda(w, alpha):
    t2 = tan2_theta(w)
    return (np.sqrt(1.0 + alpha ** 2 * t2) - 1.0) / 2.0


def G1(w, alpha):
    return 1.0 / (1.0 + Lambda(w, alpha))


def G(wo, wi, alpha):
    return 1.0 / (1.0 + Lambda(wo, alpha) + Lambda(wi, alpha))


def D_visible(w, wm, alpha):
    return G1(w, alpha) / np.maximum(np.abs(w[:, 2]), 1e-30) * D(wm, alpha) * np.abs(dot(w, wm))


def sample_wm(w, u, alpha):
    wh = norm(np.stack([alpha * w[:, 0], alpha * w[:, 1], w[:, 2]], axis=-1))
    wh = np.where((wh[:, 2] < 0)[:, None], -wh, wh)
    z = np.array([0.0, 0.0, 1.0])
    t1 = np.cross(np.broadcast_to(z, wh.shape), wh)
    l = np.linalg.norm(t1, axis=-1, keepdims=True)
    t1 = np.where((wh[:, 2] < 0.99999)[:, None], t1 / np.maximum(l, 1e-30), np.array([1.0, 0.0, 0.0]))
    t2 = np.cross(wh, t1)
    r = np.sqrt(u[:, 0]); th = 2 * PI * u[:, 1]
    px = r * np.cos(th); py = r * np.sin(th)
    h = np.sqrt(1.0 - px ** 2)
    s = (1.0 + wh[:, 2]) / 2.0
    py = (1.0 - s) * h + s * py
    pz = np.sqrt(np.maximum(0.0, 1.0 - px ** 2 - py ** 2))
    nh = px[:, None] * t1 + py[:, None] * t2 + pz[:, None] * wh
    return norm(np.stack([alpha * nh[:, 0], alpha * nh[:, 1], np.maximum(1e-6, nh[:, 2])], axis=-1))


# ---- pbrt-v4 DielectricBxDF::Sample_f (Radiance mode); returns (valid, wi_local, weight = f |cos wi| / pdf) -----------
def sample_dielectric(wo, eta, alpha, rng):
    n = wo.shape[0]
    uc = rng.random(n)
    if alpha < 1e-3:  # EffectivelySmooth
        R = fr_dielectric(wo[:, 2], eta); T = 1 - R
        refl = uc < R
        wi_r = np.stack([-wo[:, 0], -wo[:, 1], wo[:, 2]], axis=-1)
        valid_t, wi_t, etap = refract(wo, np.broadcast_to(np.array([0.0, 0.0, 1.0]), wo.shape), np.full(n, eta))
        wi = np.where(refl[:, None], wi_r, wi_t)
        valid = np.where(refl, True, valid_t)
        weight = np.where(refl, 1.0, 1.0 / etap ** 2)  # f|cos|/pdf: R/pr = 1 ; T/(pt etap^2) = 1/etap^2
        return valid, wi, weight
    u = rng.random((n, 2))
    wm = sample_wm(wo, u, alpha)
    R = fr_dielectric(dot(wo, wm), eta); T = 1 - R
    refl = uc < R
    # reflection
    wi_r = -wo + 2.0 * dot(wo, wm)[:, None] * wm
    same_r = wo[:, 2] * wi_r[:, 2] > 0
    pdf_r = D_visible(wo, wm, alpha) / (4 * np.abs(dot(wo, wm))) * R
    f_r = D(wm, alpha) * G(wo, wi_r, alpha) * R / np.abs(4 * wi_r[:, 2] * wo[:, 2])
    w_r = np.where(pdf_r > 0, f_r * np.abs(wi_r[:, 2]) / np.maximum(pdf_r, 1e-30), 0.0)
    # transmission
    valid_t, wi_t, etap = refract(wo, wm, np.full(n, eta))
    same_t = wo[:, 2] * wi_t[:, 2] > 0
    denom = (dot(wi_t, wm) + dot(wo, wm) / etap) ** 2
    dwm_dwi = np.abs(dot(wi_t, wm)) / np.maximum(denom, 1e-30)
    pdf_t = D_visible(wo, wm, alpha) * dwm_dwi * T
    ft = T * D(wm, alpha) * G(wo, wi_t, alpha) * np.abs(dot(wi_t, wm) * dot(wo, wm)) /         np.maximum(np.abs(wi_t[:, 2] * wo[:, 2]) * denom, 1e-30)
    ft = ft / etap ** 2  # radiance-mode non-symmetry scaling
    w_t = np.where(pdf_t > 0, ft * np.abs(wi_t[:, 2]) / np.maximum(pdf_t, 1e-30), 0.0)
    wi = np.where(refl[:, None], wi_r, wi_t)
    valid = np.where(refl, same_r, valid_t & ~same_t & (np.abs(wi_t[:, 2]) > 0))
    weight = np.where(refl, w_r, w_t)
    return valid, wi, weight


def frame(nrm):
    a = np.where((np.abs(nrm[:, 0]) > 0.9)[:, None], np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    t = norm(np.cross(a, nrm)); b = np.cross(nrm, t)
    return t, b


def hit_sphere(o, d, c, r):
    oc = o - c
    b = dot(oc, d); cc = dot(oc, oc) - r * r
    disc = b * b - cc
    ok = disc > 0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t0 = -b - sq; t1 = -b + sq
    t = np.where(t0 > 1e-7, t0, np.where(t1 > 1e-7, t1, np.inf))
    return np.where(ok, t, np.inf)


def render(width, height, spp, roughness, max_rays, seed=1):
    rng = np.random.default_rng(seed)
    alpha = math.sqrt(roughness) if roughness > 0 else 0.0
    dirv = norm((AT - EYE)[None, :])[0]
    right = norm(np.cross(UP[None, :], dirv[None, :]))[0]
    newup = np.cross(dirv, right)
    aspect = width / height
    tanh = math.tan(math.radians(40.0) / 2.0)
    img = np.zeros((height, width))
    for y in range(height):
        # one row at a time, spp rays per pixel
        px = np.repeat(np.arange(width), spp)
        n = px.size
        sx = ((px + rng.random(n)) / width * 2 - 1) * aspect * tanh
        sy = (1 - (y + rng.random(n)) / height * 2) * tanh
        d = norm(right[None, :] * sx[:, None] + newup[None, :] * sy[:, None] + dirv[None, :])
        o = np.broadcast_to(EYE, d.shape).copy()
        beta = np.ones(n)
        L = np.zeros(n)
        idx = np.arange(n)
        for bounce in range(max_rays):
            if idx.size == 0:
                break
            tg = hit_sphere(o, d, GLASS_C, GLASS_R)
            tl = hit_sphere(o, d, LAMP_C, LAMP_R)
            if SKY > 0: tl = np.full_like(tl, np.inf)
            lamp = tl < tg
            miss = ~np.isfinite(np.minimum(tg, tl))
            if SKY > 0 and miss.any():
                np.add.at(L, idx[miss], beta[miss] * SKY)
            # lamp hit: emission (outer side only) then absorbed
            if lamp.any():
                p = o[lamp] + tl[lamp][:, None] * d[lamp]
                nl = norm(p - LAMP_C)
                emit = dot(nl, -d[lamp]) > 0
                np.add.at(L, idx[lamp], np.where(emit, beta[lamp] * LAMP_L, 0.0))
            gl = (~lamp) & (~miss)
            keep = np.zeros(idx.size, bool)
            if gl.any():
                sel = np.nonzero(gl)[0]
                p = o[sel] + tg[sel][:, None] * d[sel]
                nrm = norm(p - GLASS_C)
                t, b = frame(nrm)
                wo_w = -d[sel]
                wo = np.stack([dot(wo_w, t), dot(wo_w, b), dot(wo_w, nrm)], axis=-1)
                valid, wi, w = sample_dielectric(wo, ETA, alpha, rng)
                wi_w = wi[:, 0:1] * t + wi[:, 1:2] * b + wi[:, 2:3] * nrm
                wi_w = norm(wi_w)
                ok = valid & np.isfinite(w) & (w > 0)
                side = np.where(dot(wi_w, nrm) > 0, 1.0, -1.0)
                o_new = p + nrm * (side[:, None] * 1e-6)
                beta_sel = beta[sel] * w
                # write back survivors
                surv = sel[ok]
                o[surv] = o_new[ok]; d[surv] = wi_w[ok]; beta[surv] = beta_sel[ok]
                keep[surv] = True
            o = o[keep]; d = d[keep]; beta = beta[keep]; idx = idx[keep]
        # pixel average
        np.add.at(img[y], np.arange(width), 0.0)
        img[y] = np.bincount(np.repeat(np.arange(width), spp), weights=L, minlength=width) / spp
    return img


def srgb_lut():
    return np.array([(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4) for c in (i / 255 for i in range(256))])


def renderer_style_mean(img, expo):
    v = np.clip(img * expo, 0.0, 1.0)
    enc = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(v, 1 / 2.4) - 0.055)
    q = np.clip(np.rint(enc * 255), 0, 255).astype(int)
    return float(srgb_lut()[q].mean() / expo)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roughness", type=float, nargs="+", help="pbrt dielectric 'roughness' values (0 = smooth)")
    ap.add_argument("--size", type=int, default=96, help="square image size (the renderer's --width run is square)")
    ap.add_argument("--spp", type=int, default=1000)
    ap.add_argument("--max-rays", type=int, default=16, help="rays per path, like the renderer's depth argument")
    ap.add_argument("--exposure", type=float, default=0.05,
                    help="renderer-style output transform; keep it low enough that nothing clips (lamp L=15)")
    ap.add_argument("--sky", type=float, default=0.0, help="furnace mode: uniform sky radiance instead of the lamp")
    a = ap.parse_args()
    SKY = a.sky
    for r in a.roughness:
        img = render(a.size, a.size, a.spp, r, a.max_rays)
        print(f"roughness {r}: reference mean {renderer_style_mean(img, a.exposure):.5f}"
              f"  (raw radiance mean {img.mean():.5f})", flush=True)
