// metal_poc_bssrdf.metal
// The tabulated BSSRDF's radial profile (pbrt-v4's TabulatedBSSRDF: Sr, its sampling pdf, and sampling a radius), ported from
// gpu/optix/optix_bssrdf.h, which is itself the device version of src/shared/bssrdf.h. Used by shadeSubsurface() (metal_poc_kernel_subsurface.metal).
//
// A table lives in the shared float buffer (rgbGridData) at an element offset the material carries; its layout is
//   [0] n_rho   [1] n_radius   (small integers stored as floats)
//   rho_samples[n_rho]  radius_samples[n_radius]  profile[n_rho * n_radius]  profile_cdf[n_rho * n_radius]
// (row i = rho_samples[i], column j = radius_samples[j]); the host writes it in MetalPocApp::mapPbrtMaterial. Every function takes a pointer to
// the table's first float. sigma_t and rho are the material's per colour channel; lengths are in the (rescaled) Metal scene's units, and the
// table itself is in optical units (radius * sigma_t), so the rescale needs no change here.

struct BssrdfTableView {
    int nRho;
    int nRadius;
    device const float* rhoSamples;
    device const float* radiusSamples;
    device const float* profile;
    device const float* profileCdf;
};

inline BssrdfTableView bssrdfTableAt(device const float* t) {
    BssrdfTableView v;
    v.nRho = int(t[0]);
    v.nRadius = int(t[1]);
    v.rhoSamples = t + 2;
    v.radiusSamples = v.rhoSamples + v.nRho;
    v.profile = v.radiusSamples + v.nRadius;
    v.profileCdf = v.profile + v.nRho * v.nRadius;
    return v;
}

// pbrt-v4's FindInterval over a sorted node array: the largest index i in [0, n - 2] with arr[i] <= x.
inline int bssrdfFindIntervalLe(device const float* arr, int n, float x) {
    int size = n - 2, first = 1;
    while (size > 0) {
        const int halfSize = size >> 1;
        const int middle = first + halfSize;
        if (arr[middle] <= x) { first = middle + 1; size -= halfSize + 1; }
        else                  { size = halfSize; }
    }
    return clamp(first - 1, 0, n - 2);
}

// pbrt-v4's CatmullRomWeights: the four spline weights at x over `nodes` and the index of the first of the four nodes. False outside the nodes.
inline bool bssrdfCatmullRomWeights(device const float* nodes, int n, float x, thread int& offset, thread float* weights) {
    if (!(x >= nodes[0] && x <= nodes[n - 1])) return false;
    const int idx = bssrdfFindIntervalLe(nodes, n, x);
    offset = idx - 1;
    const float x0 = nodes[idx], x1 = nodes[idx + 1];
    const float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
    weights[1] = 2.0 * t3 - 3.0 * t2 + 1.0;
    weights[2] = -2.0 * t3 + 3.0 * t2;
    if (idx > 0) {
        const float w0 = (t3 - 2.0 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        const float w0 = t3 - 2.0 * t2 + t;
        weights[0] = 0.0;
        weights[1] -= w0;
        weights[2] += w0;
    }
    if (idx + 2 < n) {
        const float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        const float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0;
    }
    return true;
}

// Sr(r) for one colour channel: the profile interpolated over (rho, r * sigma_t), divided by 2 pi r_optical, times sigma_t^2.
inline float bssrdfSr(BssrdfTableView tab, float sigmaT, float rho, float r) {
    const float rOpt = r * sigmaT;
    int rhoOff, radOff;
    float rhoW[4], radW[4];
    if (!bssrdfCatmullRomWeights(tab.rhoSamples, tab.nRho, rho, rhoOff, rhoW)) return 0.0;
    if (!bssrdfCatmullRomWeights(tab.radiusSamples, tab.nRadius, rOpt, radOff, radW)) return 0.0;
    float val = 0.0;
    for (int j = 0; j < 4; ++j) {
        if (rhoW[j] == 0.0) continue;
        const int ri = rhoOff + j;
        if (ri < 0 || ri >= tab.nRho) continue;
        for (int k = 0; k < 4; ++k) {
            if (radW[k] == 0.0) continue;
            const int rj = radOff + k;
            if (rj < 0 || rj >= tab.nRadius) continue;
            val += rhoW[j] * radW[k] * tab.profile[ri * tab.nRadius + rj];
        }
    }
    if (rOpt != 0.0) val /= (2.0 * M_PI_F * rOpt);
    val = max(0.0, val);
    return val * sigmaT * sigmaT;
}

// The pdf Sr is sampled with: Sr divided by the effective albedo of the interpolated row.
inline float bssrdfPdfSr(BssrdfTableView tab, float sigmaT, float rho, float r) {
    const float rOpt = r * sigmaT;
    int rhoOff, radOff;
    float rhoW[4], radW[4];
    if (!bssrdfCatmullRomWeights(tab.rhoSamples, tab.nRho, rho, rhoOff, rhoW)) return 0.0;
    if (!bssrdfCatmullRomWeights(tab.radiusSamples, tab.nRadius, rOpt, radOff, radW)) return 0.0;
    float val = 0.0, rhoEffInterp = 0.0;
    for (int j = 0; j < 4; ++j) {
        if (rhoW[j] == 0.0) continue;
        const int ri = rhoOff + j;
        if (ri < 0 || ri >= tab.nRho) continue;
        rhoEffInterp += rhoW[j] * tab.profileCdf[ri * tab.nRadius + (tab.nRadius - 1)];
        for (int k = 0; k < 4; ++k) {
            if (radW[k] == 0.0) continue;
            const int rj = radOff + k;
            if (rj < 0 || rj >= tab.nRadius) continue;
            val += rhoW[j] * radW[k] * tab.profile[ri * tab.nRadius + rj];
        }
    }
    if (rOpt != 0.0) val /= (2.0 * M_PI_F * rOpt);
    if (rhoEffInterp <= 0.0) return 0.0;
    return max(0.0, val * sigmaT * sigmaT / rhoEffInterp);
}

// One column of a (rho-interpolated) 2D table: sum_k weights[k] * arr[(offset + k) * n2 + col].
inline float bssrdfInterpRho(device const float* arr, int col, int offset, thread const float* weights, int n1, int n2) {
    float v = 0.0;
    for (int k = 0; k < 4; ++k) {
        const int ri = offset + k;
        if (weights[k] != 0.0 && ri >= 0 && ri < n1) v += arr[ri * n2 + col] * weights[k];
    }
    return v;
}

// The cubic Hermite spline's CDF and value at t in [0, 1], minus u (the root of this is the sampled position).
inline void bssrdfHermiteCdfEval(float t, float f0, float d0, float f1, float d1, float u, thread float& fhatMinusU, thread float& fhat) {
    const float t2 = t * t, t3 = t2 * t, t4 = t3 * t;
    const float Fhat = f0 * t + 0.5 * d0 * t2
        + ((-2.0 * d0 - d1) * (1.0 / 3.0) + f1 - f0) * t3
        + (0.25 * (d0 + d1) + 0.5 * (f0 - f1)) * t4;
    fhat = f0 + d0 * t
        + (-2.0 * d0 - d1 + 3.0 * (f1 - f0)) * t2
        + (d0 + d1 + 2.0 * (f0 - f1)) * t3;
    fhatMinusU = Fhat - u;
}

// pbrt-v4's SampleCatmullRom2D's Newton-bisection: the t in [0, 1] where the spline's CDF reaches u.
inline float bssrdfCatmullRomNewtonBisection(float f0, float d0, float f1, float d1, float u) {
    float x0 = 0.0, x1 = 1.0;
    float Fmu0, fp0, Fmu1, fp1;
    bssrdfHermiteCdfEval(x0, f0, d0, f1, d1, u, Fmu0, fp0);
    bssrdfHermiteCdfEval(x1, f0, d0, f1, d1, u, Fmu1, fp1);
    if (abs(Fmu0) < 1e-6) return x0;
    if (abs(Fmu1) < 1e-6) return x1;
    const bool negStart = (Fmu0 < 0.0);
    float xMid = x0 + (x1 - x0) * (-Fmu0) / (Fmu1 - Fmu0);
    for (int iter = 0; iter < 64; ++iter) {
        if (!(x0 < xMid && xMid < x1)) xMid = 0.5 * (x0 + x1);
        float Fmid, dfMid;
        bssrdfHermiteCdfEval(xMid, f0, d0, f1, d1, u, Fmid, dfMid);
        if (abs(Fmid) < 1e-6 || x1 - x0 < 1e-6) break;
        if ((Fmid < 0.0) == negStart) x0 = xMid; else x1 = xMid;
        xMid -= Fmid / dfMid;
    }
    return xMid;
}

// Samples an optical radius from the profile at albedo rho (pbrt-v4's SampleCatmullRom2D over the table), returns it divided by sigma_t, or a
// negative number when the albedo or the sample is out of range.
inline float bssrdfSampleSr(BssrdfTableView tab, float sigmaT, float rho, float u) {
    if (sigmaT <= 0.0) return -1.0;
    int offset;
    float weights[4];
    if (!bssrdfCatmullRomWeights(tab.rhoSamples, tab.nRho, rho, offset, weights)) return 0.0;
    const int n1 = tab.nRho, n2 = tab.nRadius;
    const float maximum = bssrdfInterpRho(tab.profileCdf, n2 - 1, offset, weights, n1, n2);
    if (maximum <= 0.0) return 0.0;
    u *= maximum;

    int size = n2 - 2, first = 1;
    while (size > 0) {
        const int halfSize = size >> 1;
        const int middle = first + halfSize;
        if (bssrdfInterpRho(tab.profileCdf, middle, offset, weights, n1, n2) <= u) { first = middle + 1; size -= halfSize + 1; }
        else                                                                        { size = halfSize; }
    }
    const int idx = clamp(first - 1, 0, n2 - 2);

    const float f0 = bssrdfInterpRho(tab.profile, idx, offset, weights, n1, n2);
    const float f1 = bssrdfInterpRho(tab.profile, idx + 1, offset, weights, n1, n2);
    const float x0 = tab.radiusSamples[idx], x1 = tab.radiusSamples[idx + 1];
    const float w = x1 - x0;
    const float d0 = (idx > 0)
        ? w * (f1 - bssrdfInterpRho(tab.profile, idx - 1, offset, weights, n1, n2)) / (x1 - tab.radiusSamples[idx - 1])
        : (f1 - f0);
    const float d1 = (idx + 2 < n2)
        ? w * (bssrdfInterpRho(tab.profile, idx + 2, offset, weights, n1, n2) - f0) / (tab.radiusSamples[idx + 2] - x0)
        : (f1 - f0);
    const float uu = (u - bssrdfInterpRho(tab.profileCdf, idx, offset, weights, n1, n2)) / w;

    const float t = bssrdfCatmullRomNewtonBisection(f0, d0, f1, d1, uu);
    return (x0 + w * t) / sigmaT;
}
