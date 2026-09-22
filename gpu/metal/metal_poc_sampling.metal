inline bool sampleRealisticCameraRay(constant Uniforms& uniforms,
                                      device const LensElement* lensElements,
                                      device const ExitPupilBounds* exitPupilBounds,
                                      float u, float v, thread uint& rngState,
                                      float3 su, float3 sv, float3 sw, float3 originWorld,
                                      thread float3& outOrigin, thread float3& outDirection,
                                      thread float& outWeight) {
    outWeight = 0.0;
    if (uniforms.numLensElements == 0u || uniforms.numExitPupilBounds == 0u) return false;

    // NO leading negation on pfx (unlike pbrt-v4/CUDA's own
    // `pfx = -sample.pFilm_x`) - a real sign fix found via a mirrored
    // first render, not assumed: the CUDA reference's own `su` is
    // `RealisticCamera<T>::world_right()`, built from CPU's ALT-camera
    // path (`cameras.h::make_look_at()`'s own `cross(up,forward)`,
    // section 150's own D6 finding), but THIS function is deliberately
    // passed this loader's own `cameraRight` (`cross(forward,up)`,
    // matching CPU's PRIMARY camera instead - see this function's own
    // declaration comment for why). Dropping the negation here is
    // algebraically identical to negating `su`'s own final contribution
    // below (pfx enters the whole downstream trace linearly, only ever
    // multiplied by `su` at the very end) - the same "negate the right-
    // vector term" shape D6/D7's own fixes already used, just applied
    // at the INPUT instead of the output since this trace has many
    // intermediate steps between the two.
    float pfx = (2.0 * u - 1.0) * uniforms.filmHalfX;
    float pfy = (2.0 * v - 1.0) * uniforms.filmHalfY;

    // sample_exit_pupil
    float rFilm = sqrt(pfx * pfx + pfy * pfy);
    float filmDiag = 2.0 * sqrt(uniforms.filmHalfX * uniforms.filmHalfX + uniforms.filmHalfY * uniforms.filmHalfY);
    int sz = int(uniforms.numExitPupilBounds);
    int rIndex = int(rFilm / (filmDiag * 0.5) * float(sz));
    if (rIndex >= sz) rIndex = sz - 1;
    if (rIndex < 0) rIndex = 0;
    ExitPupilBounds b = exitPupilBounds[rIndex];
    if (b.degenerate != 0u) return false;

    float area = (b.xMax - b.xMin) * (b.yMax - b.yMin);
    if (area <= 0.0) return false;
    float ppdf = 1.0 / area;

    float u0 = randFloat(rngState), u1 = randFloat(rngState);
    float lx = b.xMin + u0 * (b.xMax - b.xMin);
    float ly = b.yMin + u1 * (b.yMax - b.yMin);

    float sinTheta = (rFilm > 0.0) ? pfy / rFilm : 0.0;
    float cosTheta0 = (rFilm > 0.0) ? pfx / rFilm : 1.0;
    float ppx = cosTheta0 * lx - sinTheta * ly;
    float ppy = sinTheta * lx + cosTheta0 * ly;
    float ppz = uniforms.lensRearZ;

    float rdx = ppx - pfx, rdy = ppy - pfy, rdz = ppz;
    float rLen = sqrt(rdx * rdx + rdy * rdy + rdz * rdz);

    // trace_lenses_from_film: camera space (film z=0, +z toward scene) ->
    // lens space (z flipped): loz=-oz, ldz=-dz.
    float lox = pfx, loy = pfy, loz = 0.0;
    float ldx = rdx, ldy = rdy, ldz = -rdz;
    float elementZ = 0.0;

    for (int i = int(uniforms.numLensElements) - 1; i >= 0; --i) {
        LensElement el = lensElements[i];
        elementZ -= el.thickness;
        bool isStop = (el.curvatureRadius == 0.0);
        float t, nx = 0.0, ny = 0.0, nz = 0.0;

        if (isStop) {
            if (ldz == 0.0) return false;
            t = (elementZ - loz) / ldz;
            if (t < 0.0) return false;
        } else {
            float zCenter = elementZ + el.curvatureRadius;
            float cox = lox, coy = loy, coz = loz - zCenter;
            float A = ldx * ldx + ldy * ldy + ldz * ldz;
            float B = 2.0 * (ldx * cox + ldy * coy + ldz * coz);
            float C = cox * cox + coy * coy + coz * coz - el.curvatureRadius * el.curvatureRadius;
            float disc = B * B - 4.0 * A * C;
            if (disc < 0.0) return false;
            float sq = sqrt(disc);
            float q = (B < 0.0) ? -0.5 * (B - sq) : -0.5 * (B + sq);
            float t0 = q / A;
            float t1 = C / q;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            bool useCloserT = (ldz > 0.0) != (el.curvatureRadius < 0.0);
            t = useCloserT ? min(t0, t1) : max(t0, t1);
            if (t < 0.0) return false;
            float hx0 = lox + t * ldx, hy0 = loy + t * ldy, hz0 = loz + t * ldz;
            nx = hx0; ny = hy0; nz = hz0 - zCenter;
            float nlen = sqrt(nx * nx + ny * ny + nz * nz);
            if (nlen == 0.0) return false;
            nx /= nlen; ny /= nlen; nz /= nlen;
            if (ldx * nx + ldy * ny + ldz * nz > 0.0) { nx = -nx; ny = -ny; nz = -nz; }
        }

        float hx = lox + t * ldx, hy = loy + t * ldy, hz = loz + t * ldz;
        if (hx * hx + hy * hy > el.apertureRadius * el.apertureRadius) return false;
        lox = hx; loy = hy; loz = hz;

        if (!isStop) {
            float etaI = (el.eta == 0.0) ? 1.0 : el.eta;
            float etaT = (i > 0 && lensElements[i - 1].eta != 0.0) ? lensElements[i - 1].eta : 1.0;
            float len = sqrt(ldx * ldx + ldy * ldy + ldz * ldz);
            float dxn = ldx / len, dyn = ldy / len, dzn = ldz / len;
            float eta = etaI / etaT;
            float cosI = -(dxn * nx + dyn * ny + dzn * nz);
            float sin2T = eta * eta * max(0.0, 1.0 - cosI * cosI);
            if (sin2T >= 1.0) return false;
            float cosT = sqrt(1.0 - sin2T);
            ldx = eta * dxn + (eta * cosI - cosT) * nx;
            ldy = eta * dyn + (eta * cosI - cosT) * ny;
            ldz = eta * dzn + (eta * cosI - cosT) * nz;
        }
    }

    float lensOutOx = lox, lensOutOy = loy, lensOutOz = -loz;
    float lensOutDx = ldx, lensOutDy = ldy, lensOutDz = -ldz;

    float cosThetaW = (rLen > 0.0) ? abs(rdz / rLen) : 0.0;
    float lrz = uniforms.lensRearZ;
    if (lrz <= 0.0) return false;
    float w = (cosThetaW * cosThetaW * cosThetaW * cosThetaW) / (ppdf * lrz * lrz);

    outOrigin = originWorld + lensOutOx * su + lensOutOy * sv + lensOutOz * sw;
    outDirection = normalize(lensOutDx * su + lensOutDy * sv + lensOutDz * sw);
    outWeight = w;
    return true;
}



// Uniform sample on a unit disk (r = sqrt(u1) for area-uniform density,
// not r = u1 - the same sqrt used for the hemisphere sample's own radius
// below, same reason: linear r would bunch samples toward the centre).
// Used only by the thin-lens depth-of-field sample in the kernel below -
// a camera aperture is a flat disk, not a hemisphere, so this is its own
// small helper rather than reusing cosineSampleHemisphere's.
inline float2 sampleUnitDisk(thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

// A regular-polygon aperture instead of a circular one - real camera
// lenses focus light through a finite number of physical aperture
// blades, not a perfect circle, which is exactly why out-of-focus
// highlights ("bokeh") in a real photo read as hexagons/pentagons/etc.
// rather than perfectly round discs; `sampleUnitDisk()` above is the
// idealized circular-aperture limit (infinite blades), a real
// approximation this POC's own DOF used unconditionally through step 19.
// Samples uniformly by picking one of `sides` equal triangular wedges
// (origin - vertex_k - vertex_{k+1}) uniformly at random, then a point
// within that wedge via the standard sqrt-for-uniform-triangle-area
// trick - the textbook regular-polygon sampling construction, not an
// approximation of one.
inline float2 samplePolygonAperture(uint sides, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float u3 = randFloat(rngState);
    uint blade = min(uint(u1 * float(sides)), sides - 1);
    float angleStep = 2.0 * M_PI_F / float(sides);
    float2 vertexA = float2(cos(angleStep * float(blade)), sin(angleStep * float(blade)));
    float2 vertexB = float2(cos(angleStep * float(blade + 1)), sin(angleStep * float(blade + 1)));
    float s = sqrt(u2);
    return s * ((1.0 - u3) * vertexA + u3 * vertexB);
}

inline float3 cosineSampleHemisphere(float3 normal, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u1));

    // Build an orthonormal basis around `normal` (Duff et al.'s branchless
    // construction) - same "pick any tangent frame, only the normal
    // matters" approach this project's own onb.h uses for the CPU path.
    float sign = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    float3 tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);

    return normalize(x * tangent + y * bitangent + z * normal);
}

// UNIFORM (not cosine-weighted) hemisphere sampling - materialType 14's
// own velvet material needs this: Ashikhmin & Shirley's own velvet BRDF
// (see shadeVelvet's own comment) is sampled uniformly in the reference
// this was ported from (Blender Cycles' own bsdf_ashikhmin_velvet.h),
// the same "don't bother importance-sampling a niche lobe's own oddly-
// shaped distribution, plain uniform/cosine sampling is simpler and
// still unbiased, just higher-variance" simplification this POC's own
// Oren-Nayar material (materialType 13) already makes too. Same Duff et
// al. branchless ONB construction as cosineSampleHemisphere() above -
// only the (x,y,z) distribution differs (z = u1 directly, not sqrt(u1),
// the standard uniform-over-solid-angle construction).
inline float3 sampleUniformHemisphere(float3 normal, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float z = u1;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float theta = 2.0 * M_PI_F * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);

    float sign = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    float3 tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);

    return normalize(x * tangent + y * bitangent + z * normal);
}

// Barycentric-interpolated shading normal - replaces the earlier flat
// per-triangle face normal (cross product of two edges, same value at
// every point of a triangle) with a real per-vertex-normal blend, using
// the hit's own barycentric coordinates (triangle_data tag on the
// intersector/intersection_result is what makes those available at all).
// For the hand-authored room quads, the three corner normals stored in
// `normals` are all identical (the quad's own flat face normal, written
// that way host-side) so this reduces to exactly the old flat-shading
// behaviour there - only a real mesh with genuinely different per-vertex
// normals (Suzanne's own `vn` data) sees a different, smoothly-varying
// result. Same interpolation `.obj`/pbrt-v4/this project's own CPU
// triangle.h use for a shading normal, not an approximation of it.
inline float3 shadingNormalFor(uint primId, float2 barycentric, device const packed_float3* normals) {
    float3 n0 = float3(normals[primId * 3 + 0]);
    float3 n1 = float3(normals[primId * 3 + 1]);
    float3 n2 = float3(normals[primId * 3 + 2]);
    float w0 = 1.0 - barycentric.x - barycentric.y;
    return normalize(w0 * n0 + barycentric.x * n1 + barycentric.y * n2);
}

// Same barycentric-blend idea as shadingNormalFor(), for texture
// coordinates instead of normals - a UV buffer parallel to
// vertices/normals, same per-triangle-corner indexing. Every non-textured
// primitive's three corners carry (0,0) (see addQuad()'s/loadObjMesh()'s
// host-side default), which interpolates to (0,0) too - harmless, since
// only materialType == 3 ever reads it.
inline float2 texCoordFor(uint primId, float2 barycentric, device const packed_float2* uvs) {
    float2 uv0 = uvs[primId * 3 + 0];
    float2 uv1 = uvs[primId * 3 + 1];
    float2 uv2 = uvs[primId * 3 + 2];
    float w0 = 1.0 - barycentric.x - barycentric.y;
    return w0 * uv0 + barycentric.x * uv1 + barycentric.y * uv2;
}

// A per-triangle tangent (constant across the triangle, same "flat is
// fine here" reasoning addQuad()'s own flat face normal already relies
// on - every quad this POC hand-authors is planar with a single UV
// gradient, not a smoothly-varying mesh surface): the standard position/
// UV partial-derivative construction (solve for the UV-space basis that
// maps to world-space edge1/edge2), same technique pbrt-v4's own
// triangle tangent setup uses. `primId`-indexed the same flat, 3-per-
// triangle way vertices/uvs already are - reuses those two buffers
// directly, no new per-triangle data needed host-side.
inline float3 tangentFor(uint primId, device const packed_float3* verts, device const packed_float2* uvs) {
    float3 p0 = float3(verts[primId * 3 + 0]);
    float3 p1 = float3(verts[primId * 3 + 1]);
    float3 p2 = float3(verts[primId * 3 + 2]);
    float2 uv0 = uvs[primId * 3 + 0];
    float2 uv1 = uvs[primId * 3 + 1];
    float2 uv2 = uvs[primId * 3 + 2];
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    // A degenerate UV mapping (det == 0, e.g. every corner sharing (0,0) -
    // every non-materialType-7 primitive's own default UVs) has no real
    // tangent to solve for; returning SOME unit vector rather than NaN
    // keeps this safe to call unconditionally, even though only
    // materialType == 7 ever actually uses the result.
    if (abs(det) < 1e-10) {
        return normalize(edge1);
    }
    float f = 1.0 / det;
    float3 tangent = f * (duv2.y * edge1 - duv1.y * edge2);
    return normalize(tangent);
}

// Procedural "egg carton" bump map - an analytic height field h(u,v)
// instead of a sampled normal-map texture (no new image asset needed for
// this POC to demonstrate genuine tangent-space shading-normal
// perturbation): standard bump-mapping math, perturbing the normal by
// the height field's own partial derivatives along the tangent/
// bitangent axes (`normal - dh/du * tangent - dh/dv * bitangent`,
// renormalized) rather than actually displacing geometry - the textbook
// distinction between bump mapping (shading only, what this is) and
// real displacement mapping (which this is NOT). `strength` scales the
// derivative term directly - 0.0 (every material type other than 7)
// exactly reproduces the unperturbed normal, a true no-op, not just a
// visually-close approximation of one.
inline float3 proceduralBumpNormal(float3 normal, float3 tangent, float2 uv, float strength) {
    float3 bitangent = cross(normal, tangent);
    const float freqU = 4.0;
    const float freqV = 4.0;
    float au = uv.x * 2.0 * M_PI_F * freqU;
    float av = uv.y * 2.0 * M_PI_F * freqV;
    // `strength` scales the slope DIRECTLY (capped at 1 by cos/sin, so
    // `strength` itself is the max tilt magnitude along each tangent
    // axis) rather than also carrying a 2*pi*freq amplitude factor - a
    // literal height-field derivative would include that factor, but even
    // at this modest freqU/freqV it inflates to ~25x, drowning out the
    // unit normal entirely regardless of how small `strength` is. Tuned
    // as a slope, not a physical height, the same "whatever reads well"
    // spirit checkerColor()'s own tile-B-darkening fraction already uses
    // instead of a physically-derived constant. freqU/freqV == 4 (a few
    // bumps across this panel's own 0-1 UV span) rather than something
    // higher-frequency - a bump's own spatial period needs to stay well
    // above this scene's pixel footprint per UV unit, or it aliases into
    // per-pixel noise indistinguishable from Monte Carlo grain instead of
    // a visible bump shape (a real mistake this PR's own first attempt at
    // this material made, caught by inspecting a raw shading-normal
    // visualization render, not assumed away).
    float dhdu = strength * cos(au) * sin(av);
    float dhdv = strength * sin(au) * cos(av);
    float3 bumped = normal - (dhdu * tangent + dhdv * bitangent);
    return normalize(bumped);
}

// Standard equirectangular direction-to-UV mapping (longitude from
// atan2, latitude from asin) - a genuinely different way of sampling
// `earthTexture` than texCoordFor()'s own per-vertex-UV lookup above:
// this one has no notion of a surface or a mesh at all, just a ray
// DIRECTION, the way a real environment/IBL map is sampled for a miss
// ray (or, in a fuller renderer, for image-based lighting on rough
// surfaces too - not implemented here, this POC only uses it for the
// miss/"sky" case).
inline float2 equirectangularUV(float3 dir) {
    float u = atan2(dir.z, dir.x) * (1.0 / (2.0 * M_PI_F)) + 0.5;
    float v = asin(clamp(dir.y, -1.0, 1.0)) * (1.0 / M_PI_F) + 0.5;
    return float2(u, v);
}

// --- Environment-map importance sampling (phase 2) --------------------
// Device-side counterpart to gpu/metal/metal_poc_host_math.h's own
// EnvDistribution2D/findCdfInterval/sampleEnvDistribution2D/
// pdfEnvDistribution2D (section 69, phase 1) - same CDF-slope-as-pdf
// convention, same piecewise-constant-bucket binary search, just
// reading `device const float*` buffers instead of a `std::vector`, and
// folding the equirectangular direction<->UV conversion (equirectangularUV()
// above, inverted here) and its own sin(theta) solid-angle Jacobian in
// directly, since a device-side caller wants a world DIRECTION and a
// solid-angle pdf, not an image-space (u,v) and an image-space density -
// that conversion has nowhere else to live.

inline int findCdfIntervalDevice(device const float* cdf, int n, float u) {
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= u) lo = mid; else hi = mid;
    }
    return (lo < n - 1) ? lo : (n - 1);
}

// Draws a world direction from the environment map's own importance
// distribution and returns its solid-angle pdf - mirrors
// sampleEnvDistribution2D()'s own image-space sampling exactly, then
// inverts equirectangularUV() (phi = 2*pi*(u-0.5), lambda = pi*(v-0.5),
// dir = (cos(lambda)*cos(phi), sin(lambda), cos(lambda)*sin(phi))) and
// applies the equirectangular Jacobian (dOmega = 2*pi^2*cos(lambda)
// du*dv, and cos(lambda) == sin(pi*v) - see docs section 71's own
// derivation) to convert the image-space pdf into the solid-angle one
// every other light-sampling strategy in this shader already returns.
inline float3 sampleEnvironmentDirection(device const float* marginalCDF, device const float* conditionalCDF,
                                          int width, int height, float u1, float u2,
                                          thread float& pdfSolidAngle) {
    int row = findCdfIntervalDevice(marginalCDF, height, u1);
    float rowLo = marginalCDF[row], rowHi = marginalCDF[row + 1];
    float rowSpan = max(rowHi - rowLo, 1e-9);
    float dv = (u1 - rowLo) / rowSpan;
    float v = (float(row) + dv) / float(height);
    float rowPdf = rowSpan * float(height);

    device const float* condRow = conditionalCDF + row * (width + 1);
    int col = findCdfIntervalDevice(condRow, width, u2);
    float colLo = condRow[col], colHi = condRow[col + 1];
    float colSpan = max(colHi - colLo, 1e-9);
    float du = (u2 - colLo) / colSpan;
    float u = (float(col) + du) / float(width);
    float colPdf = colSpan * float(width);

    float pdfImage = rowPdf * colPdf;
    float sinTheta = max(sin(M_PI_F * v), 1e-6);
    pdfSolidAngle = pdfImage / (2.0 * M_PI_F * M_PI_F * sinTheta);

    float phi = 2.0 * M_PI_F * (u - 0.5);
    float lambda = M_PI_F * (v - 0.5);
    float cosLambda = cos(lambda);
    return float3(cosLambda * cos(phi), sin(lambda), cosLambda * sin(phi));
}

// Evaluates the SAME solid-angle pdf at an arbitrary world direction -
// what a BSDF-sampled ray that escaped toward some direction needs for
// its own MIS weight against this strategy (the miss-path's own
// contribution, see primaryRayKernel's own comment on this).
inline float pdfEnvironmentDirection(device const float* marginalCDF, device const float* conditionalCDF,
                                      int width, int height, float3 dir) {
    float2 uv = equirectangularUV(dir);
    int row = clamp(int(uv.y * float(height)), 0, height - 1);
    int col = clamp(int(uv.x * float(width)), 0, width - 1);
    float rowPdf = (marginalCDF[row + 1] - marginalCDF[row]) * float(height);
    device const float* condRow = conditionalCDF + row * (width + 1);
    float colPdf = (condRow[col + 1] - condRow[col]) * float(width);
    float pdfImage = rowPdf * colPdf;
    float sinTheta = max(sin(M_PI_F * uv.y), 1e-6);
    return pdfImage / (2.0 * M_PI_F * M_PI_F * sinTheta);
}

// A PROCEDURAL texture (materialType 6) - analytic, computed directly
// from the hit's own UV, no image/sampler involved at all, unlike
// materialType 3's earthTexture lookup or step 18's equirectangularUV()
// (both still ultimately a texture2d::sample() call). `scale` tiles are
// per UV unit; alternating tiles pick `colorA`/`colorB` based on the
// parity of floor(u*scale)+floor(v*scale) - the textbook checkerboard
// construction, same one this project's own CPU checker_texture.h uses.
inline float3 checkerColor(float2 uv, float scale, float3 colorA, float3 colorB) {
    float2 tile = floor(uv * scale);
    float parity = fmod(tile.x + tile.y, 2.0);
    return (abs(parity) < 0.5) ? colorA : colorB;
}

// materialType 16's own albedo function - the REAL 3D world-space
// checkerboard this project's own CPU renderer's checker_texture (src/
// TheRestOfYourLife/texture.h) actually implements: parity of
// floor(p.x/scale)+floor(p.y/scale)+floor(p.z/scale), keyed purely on the
// hit's own world-space POSITION, not a UV coordinate at all - unlike
// checkerColor() above (materialType 6), which needs a triangle's own
// interpolated UV and so only ever fires on a triangle (see that
// function's own comment). Needing no UV is exactly why this one CAN run
// on a sphere hit (checkerColor() above cannot - metal_poc.metal's
// sphere-intersection path computes no UV at all, the documented reason
// category-G's own mesh gallery ground uses a flat quad instead of a
// checker sphere, section 117) - this closes that gap for scenes that
// only ever needed the REAL 3D book-checker in the first place, section
// 121, docs/METAL_GPU_FEASIBILITY.md.
inline float3 checker3DColor(float3 p, float scale, float3 colorA, float3 colorB) {
    float3 cell = floor(p / scale);
    float parity = fmod(abs(cell.x) + abs(cell.y) + abs(cell.z), 2.0);
    return (parity < 0.5) ? colorA : colorB;
}

// materialType 17's own noise field - a direct port of this project's
// own CPU/GPU-shared `src/shared/noise.h` (pbrt-v4's Noise()/
// Turbulence(), CPU_GPU-tagged - already NVCC/CUDA-portable, but MSL
// itself can't #include that header directly, so this is a genuine
// re-transcription of the SAME fixed permutation table and formulas,
// not a from-scratch reimplementation). Section 124, docs/
// METAL_GPU_FEASIBILITY.md. `kNoisePerm` is pbrt-v4's own fixed table
// (noise.cpp) - identical values, do not reorder.
constant int kNoisePerm[512] = {
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
    140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
    247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
     57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
     74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
     60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
     65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
    200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
     52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
    207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
    119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
    129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
    218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
     81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,
    106,157,184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,
    205, 93,222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,
    156,180,
    // second copy (identical to first 256 entries, starting at index 256)
    151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,
      7,225,140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,
      6,148,247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35,
     11, 32, 57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168,
     68,175, 74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,
    229,122, 60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,
    143, 54, 65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89,
     18,169,200,196,135,130,116,188,159, 86,164,100,109,198,173,186,
      3, 64, 52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82,
     85,212,207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,
    170,213,119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,
    172,  9,129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,
    112,104,218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,
    162,241, 81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199
};

// pbrt-v4's own Grad(): maps a lattice-point hash to one of 12 gradient
// directions - direct port of noise.h's own noise_detail::Grad<T>().
inline float noiseGrad(int x, int y, int z, float dx, float dy, float dz) {
    int h = kNoisePerm[kNoisePerm[kNoisePerm[x & 255] + (y & 255)] + (z & 255)];
    h &= 15;
    float u = (h < 8 || h == 12 || h == 13) ? dx : dy;
    float v = (h < 4 || h == 12 || h == 13) ? dy : dz;
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

// pbrt-v4's own NoiseWeight(): quintic C2 smoothstep (6t^5-15t^4+10t^3) -
// direct port, replacing Book-3's older cubic (only C1, visible seams).
inline float noiseWeight(float t) {
    float t3 = t * t * t, t4 = t3 * t, t5 = t4 * t;
    return 6.0 * t5 - 15.0 * t4 + 10.0 * t3;
}

// pbrt-v4's own Noise(x,y,z) - trilinear-interpolated gradient noise in
// [-1,1], direct port of noise.h's own perlin_noise<T>().
inline float perlinNoise3D(float3 p) {
    const float wrap = float(1 << 30);
    p = fmod(p, wrap);
    int3 i = int3(floor(p));
    float3 d = p - float3(i);
    int ix = i.x & 255, iy = i.y & 255, iz = i.z & 255;

    float w000 = noiseGrad(ix,   iy,   iz,   d.x,       d.y,       d.z);
    float w100 = noiseGrad(ix+1, iy,   iz,   d.x - 1.0, d.y,       d.z);
    float w010 = noiseGrad(ix,   iy+1, iz,   d.x,       d.y - 1.0, d.z);
    float w110 = noiseGrad(ix+1, iy+1, iz,   d.x - 1.0, d.y - 1.0, d.z);
    float w001 = noiseGrad(ix,   iy,   iz+1, d.x,       d.y,       d.z - 1.0);
    float w101 = noiseGrad(ix+1, iy,   iz+1, d.x - 1.0, d.y,       d.z - 1.0);
    float w011 = noiseGrad(ix,   iy+1, iz+1, d.x,       d.y - 1.0, d.z - 1.0);
    float w111 = noiseGrad(ix+1, iy+1, iz+1, d.x - 1.0, d.y - 1.0, d.z - 1.0);

    float wx = noiseWeight(d.x), wy = noiseWeight(d.y), wz = noiseWeight(d.z);
    float x00 = mix(w000, w100, wx);
    float x10 = mix(w010, w110, wx);
    float x01 = mix(w001, w101, wx);
    float x11 = mix(w011, w111, wx);
    float y0 = mix(x00, x10, wy);
    float y1 = mix(x01, x11, wy);
    return mix(y0, y1, wz);
}

// pbrt-v4's own Turbulence(), no-antialiasing overload (this project's
// own `turbulence_simple<T>()`, noise.h) - sum of |noise| across
// `maxOctaves`, each octave at 1.99x the previous frequency and
// `omega`x the previous amplitude. Used by materialType 17's own marble
// pattern, matching CPU's `noise_texture`/`perlin::turb()` exactly
// (depth 7, omega 0.5 - see that class's own comment).
inline float turbulenceSimple(float3 p, float omega, int maxOctaves) {
    float sum = 0.0, lambda = 1.0, o = 1.0;
    for (int i = 0; i < maxOctaves; ++i) {
        sum += o * abs(perlinNoise3D(p * lambda));
        lambda *= 1.99;
        o *= omega;
    }
    return sum;
}

// The REAL (unpolarized, real-valued-IOR) Fresnel dielectric
// reflectance - ported directly from this project's own CPU renderer
// (src/shared/fresnel.h's own FrDielectric(), mirroring pbrt-v4's
// scattering.h exactly), NOT Schlick's approximation. An earlier
// version of this comment claimed Schlick's approximation was "the same
// one... this project's own CPU dielectric material use[s]" for this
// exact reflect-vs-refract decision - checked while reviewing this
// exact code and found to be WRONG: this project's own `dielectric`
// material (src/TheRestOfYourLife/material_simple.h) uses
// `DielectricBxDF`, which itself calls FrDielectric, not Schlick - a
// documentation inaccuracy as much as a missed accuracy opportunity.
// `cosThetaI` here is ALREADY guaranteed non-negative by construction
// (computed via `facingNormal`, which always faces the incoming ray -
// see the call site), so the `< 0` flip branch below is dead code for
// how this is actually invoked here, kept anyway for a faithful,
// recognizable port rather than a call-site-specific simplification.
inline float frDielectric(float cosThetaI, float eta) {
    cosThetaI = clamp(cosThetaI, -1.0, 1.0);
    if (cosThetaI < 0.0) {
        eta = 1.0 / eta;
        cosThetaI = -cosThetaI;
    }
    float sin2ThetaI = 1.0 - cosThetaI * cosThetaI;
    float sin2ThetaT = sin2ThetaI / (eta * eta);
    if (sin2ThetaT >= 1.0) {
        return 1.0; // Total internal reflection.
    }
    float cosThetaT = sqrt(max(0.0, 1.0 - sin2ThetaT));
    float rParl = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rPerp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return (rParl * rParl + rPerp * rPerp) / 2.0;
}

// Beer-Lambert colour absorption for a dielectric (materialType 2/5) -
// real tinted glass absorbs light proportionally to how far it travels
// THROUGH the medium (a thick paperweight reads far more saturated than
// a thin windowpane of the same glass), not by a flat per-bounce
// multiply the way every earlier version of this POC's dielectric
// branches applied `albedo`. Only fires when `!frontFace` (this hit is
// on the surface's own BACKFACE, i.e. the ray is exiting, not entering) -
// for a CONVEX primitive (true of every dielectric shape this POC has,
// the sphere), the segment just travelled (the previous bounce's own
// entry point to this exit hit) was entirely inside the medium, so
// `result.distance` at the EXIT hit is exactly the in-medium path
// length Beer's law needs, no separate distance-tracking state required.
// A concave dielectric could re-enter/exit multiple times without this
// simple per-hit check catching every segment correctly - not handled,
// same "document the assumption, don't silently rely on it" approach
// this POC's other simplifications use.
//
// `mat.color` is reinterpreted here as a per-unit-distance absorption
// COEFFICIENT, not the flat reflectance/tint every other material reads
// it as - {0,0,0} means zero absorption (exp(-0*dist) == 1 exactly, a
// true no-op reproducing perfectly clear glass), not "black," the
// opposite of what {0,0,0} would mean as a reflectance colour elsewhere
// in this same struct.
inline void applyBeerLambertAbsorption(thread float3& throughput, packed_float3 absorption, bool frontFace, float distance) {
    if (!frontFace) {
        throughput *= exp(-float3(absorption) * distance);
    }
}

// ---------------------------------------------------------------------------
// GGX / Trowbridge-Reitz microfacet distribution + height-correlated Smith
// masking-shadowing - the standard model materialType == 4 (rough
// conductor) uses below, same formulation pbrt-v4's own
// TrowbridgeReitzDistribution implements. Genuinely ANISOTROPIC (alphaX,
// alphaY, not a single scalar alpha) - all four take full LOCAL-frame
// vectors (tangent/bitangent/normal components), not just a `NdotX`
// scalar, since the anisotropic case needs the vector's azimuthal
// (tangent/bitangent) components too, not only its angle to the normal.
// Passing alphaX == alphaY reduces every one of these EXACTLY to the
// isotropic formulas this POC used through step 22 (verified
// algebraically, not just assumed) - not a separate code path, the same
// formula degenerating correctly at its own isotropic boundary case,
// same spirit as the Henyey-Greenstein phase function's own g==0 case
// (step 19). Cross-checked against Blender Cycles' own anisotropic GGX
// implementation (`bsdf_aniso_D`/`bsdf_aniso_lambda` in
// intern/cycles/kernel/closure/bsdf_microfacet.h) before being committed
// here, not derived from first principles alone this time.
// ---------------------------------------------------------------------------
inline float ggxD(float3 hLocal, float alphaX, float alphaY) {
    float3 hr = float3(hLocal.x / alphaX, hLocal.y / alphaY, hLocal.z);
    float lenSq = max(dot(hr, hr), 1e-12);
    return (1.0 / M_PI_F) / max(alphaX * alphaY * lenSq * lenSq, 1e-12);
}

// Smith's Lambda function (anisotropic GGX closed form) - how much of a
// microfacet's neighbourhood is masked/shadowed as seen from direction
// `wLocal`, folded into G1/G below rather than used standalone.
inline float ggxLambda(float3 wLocal, float alphaX, float alphaY) {
    float wz2 = max(wLocal.z * wLocal.z, 1e-12);
    float sqrAlphaTanN = (alphaX * alphaX * wLocal.x * wLocal.x + alphaY * alphaY * wLocal.y * wLocal.y) / wz2;
    return 0.5 * (sqrt(1.0 + sqrAlphaTanN) - 1.0);
}

inline float ggxG1(float3 wLocal, float alphaX, float alphaY) {
    return 1.0 / (1.0 + ggxLambda(wLocal, alphaX, alphaY));
}

// Height-correlated Smith masking-shadowing for a full reflection lobe
// (both the view and light direction masked/shadowed jointly, not treated
// as independent) - the same correlated form pbrt-v4 uses, less energy
// loss at grazing angles than a naive G1(wo)*G1(wi) product.
inline float ggxG(float3 woLocal, float3 wiLocal, float alphaX, float alphaY) {
    return 1.0 / (1.0 + ggxLambda(woLocal, alphaX, alphaY) + ggxLambda(wiLocal, alphaX, alphaY));
}

// Schlick's Fresnel approximation for a CONDUCTOR: F0 (reflectance at
// normal incidence) is itself an RGB colour here, not derived from a
// scalar IOR the way frDielectric()'s own real-valued formula is - a
// metal's complex refractive index (real eta + imaginary k, wavelength-
// dependent) is what actually produces that colour, and this whole curve
// is approximated from its own F0 value rather than solving the full
// complex-Fresnel equations, unlike this POC's own dielectric material,
// which now uses the exact real-valued formula (frDielectric()) instead
// of Schlick's own approximation of it.
inline float3 fresnelSchlickConductor(float cosTheta, float3 F0) {
    float t = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (float3(1.0) - F0) * t;
}

// frComplex() - the REAL complex-valued Fresnel reflectance for a
// conductor interface (pbrt-v4's own FrComplex, src/pbrt/util/
// scattering.h; ported from this POC's own reference copy at
// src/shared/fresnel.h). Unlike fresnelSchlickConductor()'s single-F0
// curve (which can only ever interpolate towards white at grazing
// angles), a genuine complex index of refraction (eta + i*k, both
// wavelength/channel-dependent) reproduces the real per-channel colour
// SHIFT actual metals show at grazing incidence - the same kind of
// upgrade frDielectric() (section 55) already made for this POC's
// dielectric materials over Schlick's own approximation of THAT curve.
// All complex arithmetic expanded manually (no complex<> type on
// Metal), identical to the reference's own GPU-compatible expansion.
inline float frComplex(float cosThetaI, float etaR, float etaK) {
    cosThetaI = clamp(cosThetaI, 0.0, 1.0);
    float sin2I = 1.0 - cosThetaI * cosThetaI;

    // Complex Snell's law: sin2T = sin2I / (etaR + i*etaK)^2
    float denomR = etaR * etaR - etaK * etaK;
    float denomI = 2.0 * etaR * etaK;
    float denomSq = denomR * denomR + denomI * denomI;
    float sin2TR = sin2I * denomR / denomSq;
    float sin2TI = -sin2I * denomI / denomSq;

    // cosT = sqrt(1 - sin2T) via the standard complex sqrt formula.
    float cR = 1.0 - sin2TR;
    float cI = -sin2TI;
    float mag = sqrt(cR * cR + cI * cI);
    float cosTR = sqrt(max(0.0, (mag + cR) * 0.5));
    float cosTI = (cI >= 0.0 ? 1.0 : -1.0) * sqrt(max(0.0, (mag - cR) * 0.5));

    // r_parl = (eta*cosI - cosT) / (eta*cosI + cosT), eta = etaR + i*etaK.
    float ecR = etaR * cosThetaI - cosTR;
    float ecI = etaK * cosThetaI - cosTI;
    float edR = etaR * cosThetaI + cosTR;
    float edI = etaK * cosThetaI + cosTI;
    float edSq = edR * edR + edI * edI;
    float rpR = (ecR * edR + ecI * edI) / edSq;
    float rpI = (ecI * edR - ecR * edI) / edSq;
    float rParlSq = rpR * rpR + rpI * rpI;

    // r_perp = (cosI - eta*cosT) / (cosI + eta*cosT).
    float etcR = etaR * cosTR - etaK * cosTI;
    float etcI = etaR * cosTI + etaK * cosTR;
    float ncR = cosThetaI - etcR;
    float ncI = -etcI;
    float ndR = cosThetaI + etcR;
    float ndI = etcI;
    float ndSq = ndR * ndR + ndI * ndI;
    float rsR = (ncR * ndR + ncI * ndI) / ndSq;
    float rsI = (ncI * ndR - ncR * ndI) / ndSq;
    float rPerpSq = rsR * rsR + rsI * rsI;

    return (rParlSq + rPerpSq) * 0.5;
}

// Evaluates frComplex() independently per RGB channel - the real
// per-channel complex Fresnel this POC's GGX conductor material
// (materialType 4/9) now uses in place of fresnelSchlickConductor().
inline float3 frComplexRGB(float cosThetaI, float3 eta, float3 k) {
    return float3(frComplex(cosThetaI, eta.x, k.x),
                  frComplex(cosThetaI, eta.y, k.y),
                  frComplex(cosThetaI, eta.z, k.z));
}

// Builds an orthonormal (tangent, bitangent) frame around `n` - same Duff
// et al. construction cosineSampleHemisphere uses inline, factored out
// here since GGX sampling needs to move both the outgoing direction and
// the sampled half-vector between world space and this local frame
// explicitly (unlike cosineSampleHemisphere, which only ever produces a
// world-space result and never needs the frame itself back).
inline void buildOnb(float3 n, thread float3& tangent, thread float3& bitangent) {
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    bitangent = float3(b, sign + n.y * n.y * a, -n.y);
}

// Unlike buildOnb() above (an ARBITRARY orthonormal frame - fine for an
// isotropic BRDF/phase function, which is rotationally symmetric around
// the normal so any tangent choice gives an identical result), an
// ANISOTROPIC material's highlight orientation depends on which
// direction the tangent actually points - an arbitrary, discontinuously-
// varying tangent (buildOnb()'s own choice depends on the normal's sign
// bit) would make the anisotropy direction jump around incoherently
// across a curved surface instead of reading as a single consistent
// "brushed" direction. This projects a FIXED world-space reference axis
// onto the tangent plane instead (Gram-Schmidt: bitangent = normalize
// (cross(normal, ref)), tangent = cross(bitangent, normal)) - the same
// construction Blender Cycles' own make_orthonormals_tangent() uses,
// given a real per-vertex tangent there; this POC's analytic sphere has
// no per-vertex tangent data to begin with, so a fixed world axis
// (world-up, falling back to world-X exactly at the poles where up is
// parallel to the normal and the projection would be degenerate) is the
// simplest thing that gives a consistent "lines of longitude" brushed-
// metal pattern instead of an arbitrary one.
// Branchless orthonormal basis from a unit normal - Duff, Burgess,
// Christensen, Hery, Kensler, Liani, Villemin, "Building an Orthonormal
// Basis, Revisited" (JCGT 2017), ported from this POC's own reference
// copy at src/shared/microfacet.h's BuildArbitraryTangentFrame(). Fixes
// a real bug this function used to have: the earlier "switch to a
// different world axis when `normal` gets too close to the reference
// direction" construction (`if (abs(dot(refDir, normal)) > 0.999)
// refDir = ...`) has a HARD DISCONTINUITY exactly at that 0.999
// threshold - as `normal` sweeps across it (e.g. anywhere on a sphere
// whose surface normal passes near world +/-Y), the chosen tangent/
// bitangent axes pop to a completely different orientation with no
// continuous transition. Invisible for an ISOTROPIC GGX lobe
// (rotationally symmetric in the tangent plane, so the frame's own
// orientation never affects the result) - a real, visible seam for a
// genuinely ANISOTROPIC one (materialType 4's own brushed-metal
// sphere, alphaX != alphaY), which is the only caller of this function.
// This formulation (using copysign rather than a manual branch) has no
// singularity anywhere on the unit sphere, unlike the one it replaces.
inline void buildAnisotropicOnb(float3 normal, thread float3& tangent, thread float3& bitangent) {
    float sign = copysign(1.0, normal.z);
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
}

// Henyey-Greenstein phase function - the standard analytic model for
// directional (not just isotropic) volume scattering, same one pbrt-v4's
// own HGPhaseFunction implements. `cosTheta` here is dot(wo, wi) in the
// SAME `wo` convention the GGX conductor code above already uses (wo
// points back toward where the ray came from, i.e. `-rayDir`) - under
// that convention, g > 0 peaking at cosTheta == -1 (wi antiparallel to
// wo, i.e. wi roughly EQUAL to the ray's own original travel direction)
// is exactly "forward scattering," matching the physical convention;
// getting this sign backwards is the single easiest mistake to make with
// this formula, so it's called out explicitly rather than left to be
// inferred from the algebra alone.
inline float henyeyGreensteinPhase(float cosTheta, float g) {
    float denom = 1.0 + g * g + 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * M_PI_F * denom * sqrt(max(denom, 1e-6)));
}

// Samples a direction from the HG phase function's own distribution
// relative to `wo` (same convention as the evaluation function above -
// the local frame's own Z axis IS wo, via buildOnb(), so the returned
// direction's dot product with wo equals the sampled `cosTheta` by
// construction, consistent with what henyeyGreensteinPhase() expects to
// be called with for MIS/NEE against this same sample). At g == 0 this
// reduces to a uniform-over-the-sphere DISTRIBUTION - a rotationally-
// invariant distribution is identical whether sampled relative to world
// axes or relative to an arbitrary local frame like `wo` - not a
// separate code path that happens to agree, the same formula
// degenerating correctly at its own
// boundary case.
inline float3 sampleHenyeyGreenstein(float3 wo, float g, thread uint& rngState) {
    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u1;
    } else {
        float sqrTerm = (1.0 - g * g) / (1.0 + g - 2.0 * g * u1);
        cosTheta = -1.0 / (2.0 * g) * (1.0 + g * g - sqrTerm * sqrTerm);
    }
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * M_PI_F * u2;
    float3 tangent, bitangent;
    buildOnb(wo, tangent, bitangent);
    return sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * wo;
}

// E2/section 178: 5-octave-FBm-only cloud density at a MEDIUM-space
// point, [0,1]-clamped - direct port of gpu/optix/optix_intersection_
// sphere.h's own gpu_cloud_density() (the GPU-backend variant every GPU
// backend uses; deliberately without CloudMedium<T>::compute_density()'s
// own CPU-only wispiness perturbation - see that function's comment,
// src/shared/cloud_medium.h). Reuses materialType 17's own
// perlinNoise3D()/kNoisePerm (this file, above) rather than a second
// copy of the same fixed permutation table - the fixed table + Noise()
// formula is exactly the same pbrt-v4 primitive either caller needs.
// `my` (medium-space Y) drives the altitude falloff: pbrt-v4's own
// convention is medium-y=0 is the cloud's dense base, medium-y=1 is
// thinned to nothing.
inline float gpuCloudDensity(GpuCloudMedium cloud, float mx, float my, float mz) {
    float3 pp = cloud.frequency * float3(mx, my, mz);
    float d = 0.0;
    float omega = 0.5, lambda = 1.0;
    for (int oct = 0; oct < 5; ++oct) {
        d += omega * perlinNoise3D(pp * lambda);
        omega *= 0.5;
        lambda *= 1.99;
    }
    d = clamp((1.0 - my) * 4.5 * cloud.density * d, 0.0, 1.0);
    float extra = 2.0 * max(0.0, 0.5 - my);
    return clamp(d + extra, 0.0, 1.0);
}

// world_to_medium_pt() - direct port of CloudMedium<T>::world_to_medium_pt()
// (src/shared/cloud_medium.h). `worldToMediumMat` is row-major, matching
// that function's own `mat[0]*wx + mat[1]*wy + mat[2]*wz + translate[0]`
// convention exactly.
inline float3 worldToMediumPoint(GpuCloudMedium cloud, float3 p) {
    float mx = cloud.worldToMediumMat[0]*p.x + cloud.worldToMediumMat[1]*p.y + cloud.worldToMediumMat[2]*p.z + cloud.worldToMediumTranslate[0];
    float my = cloud.worldToMediumMat[3]*p.x + cloud.worldToMediumMat[4]*p.y + cloud.worldToMediumMat[5]*p.z + cloud.worldToMediumTranslate[1];
    float mz = cloud.worldToMediumMat[6]*p.x + cloud.worldToMediumMat[7]*p.y + cloud.worldToMediumMat[8]*p.z + cloud.worldToMediumTranslate[2];
    return float3(mx, my, mz);
}

// Ray/AABB slab test against the medium's own `boundsMin`/`boundsMax`, in
// MEDIUM space, given the ray ALREADY transformed there (`mo`/`md` -
// direction transformed by the matrix only, no translation, matching
// CloudMedium<T>::sample_ray()'s own convention). Because the world-to-
// medium map is affine, world_point(t) = ray_o + t*ray_d maps to
// medium_point(t) = mo + t*md for the SAME t - so the tMin/tMax this
// returns are valid directly as world-space ray parameters along the
// ORIGINAL (world-space) ray direction, exactly like
// CloudMedium<T>::sample_ray()'s own returned segment. Returns false
// (ray misses the box entirely, or is degenerate) via `outTMin >
// outTMax`.
inline bool cloudAabbSlabIntersect(GpuCloudMedium cloud, float3 mo, float3 md,
                                    thread float& outTMin, thread float& outTMax) {
    float tMin = -1e30, tMax = 1e30;
    float moArr[3] = { mo.x, mo.y, mo.z };
    float mdArr[3] = { md.x, md.y, md.z };
    for (int i = 0; i < 3; ++i) {
        float bmin = cloud.boundsMin[i], bmax = cloud.boundsMax[i];
        if (abs(mdArr[i]) < 1e-12) {
            if (moArr[i] < bmin || moArr[i] > bmax) { outTMin = 1.0; outTMax = 0.0; return false; }
        } else {
            float invD = 1.0 / mdArr[i];
            float t0 = (bmin - moArr[i]) * invD;
            float t1 = (bmax - moArr[i]) * invD;
            if (t0 > t1) { float tmp = t0; t0 = t1; t1 = tmp; }
            tMin = max(tMin, t0);
            tMax = min(tMax, t1);
        }
    }
    outTMin = tMin;
    outTMax = tMax;
    return tMin <= tMax;
}

// Samples a half-vector from the GGX distribution of VISIBLE normals
// (Heitz 2018, "Sampling the GGX Distribution of Visible Normals"), given
// the outgoing direction `woLocal` already in the local (Z-up == shading
// normal) frame. Dramatically lower variance than importance-sampling
// D(h) directly, especially near grazing angles - the same algorithm
// pbrt-v4's TrowbridgeReitzDistribution::Sample_wm implements, chosen
// here for the same reason: it's what makes a rough-conductor path
// tracer converge in a reasonable sample count instead of needing far
// more samples to beat down grazing-angle noise.
// Picks one light from `lights`, power-proportionally (see AreaLight's own
// `pmf`/`aliasProb`/`aliasIndex` comment - replaces this POC's old uniform
// 1/lightCount picking), and samples a uniform point on its parallelogram -
// the "pick a light, then a point on it" step every NEE call site
// (Lambertian and GGX conductor both) shares verbatim; only what happens
// with the sampled point differs per BSDF. The picked light's own `pmf` is
// returned (not a flat 1/lightCount constant anymore, so it can no longer
// be folded in as a plain scalar the way the old comment here described) -
// every caller multiplies `ls.pmf` into its own area-to-solid-angle pdf
// expression instead.
struct LightSample {
    float3 point;
    float3 normal;
    float3 emission;
    float area;
    float pmf;
    // Mirrors AreaLight::twoSided below (section 104) - copied out of
    // the picked light here so every NEE call site's own cosLight check
    // can read it without a second lights[] lookup.
    float twoSided;
};

inline LightSample sampleAreaLight(device const AreaLight* lights, uint lightCount, thread uint& rngState,
                                    texture2d<float, access::sample> pbrtAreaLightTexture, sampler textureSampler) {
    // max(lightCount, 1u) guards the `- 1` below from underflowing (uint
    // wraps to 0xFFFFFFFF, not -1) if this were ever called on a 0-light
    // scene - not reachable with this POC's own hardcoded 2-light scene,
    // but every NEE call site calls this unconditionally with no count
    // check of its own, so this needs to be safe on its own terms.
    uint lastIdx = max(lightCount, 1u) - 1;
    // Vose alias-table sample (see PowerLightSampler::sample() in
    // src/shared/power_light_sampler_scaffold.h, ported verbatim): map u
    // into [0, lightCount), split into a slot index and its own
    // fractional remainder, then either accept that slot or fall through
    // to its precomputed alias - O(1) regardless of how skewed the
    // per-light probabilities are, unlike a running-sum/binary-search CDF
    // walk over `lights`.
    float scaled = randFloat(rngState) * float(lightCount);
    uint slot = min(uint(scaled), lastIdx);
    float frac = scaled - float(slot);
    uint idx = (frac < lights[slot].aliasProb) ? slot : lights[slot].aliasIndex;
    idx = min(idx, lastIdx);
    AreaLight light = lights[idx];
    float3 edgeU = float3(light.edgeU);
    float3 edgeV = float3(light.edgeV);
    float2 u = float2(randFloat(rngState), randFloat(rngState));
    LightSample result;
    result.point = float3(light.center) - 0.5 * edgeU - 0.5 * edgeV + u.x * edgeU + u.y * edgeV;
    result.normal = float3(light.normal);
    // Patterned emission (see AreaLight's own comment): the SAME (u.x,
    // u.y) this NEE sample point was just built from doubles as the
    // pattern's own UV coordinate, no separate UV needed. `patternScale
    // <= 0.0` (every light before this one) skips this entirely,
    // reproducing flat `light.emission` exactly.
    result.emission = (light.useTexture > 0.0)
        ? pbrtAreaLightTexture.sample(textureSampler, u).rgb * float3(light.emission)
        : (light.patternScale > 0.0)
            ? checkerColor(u, light.patternScale, float3(light.emission), float3(light.emission) * light.patternTileB)
            : float3(light.emission);
    result.area = light.area;
    result.pmf = light.pmf;
    result.twoSided = light.twoSided;
    return result;
}

// Generalized to anisotropic alphaX/alphaY (Heitz 2018's own Section
// 3.2/3.4 stretch-and-unstretch steps, using alphaX/alphaY on their
// respective axes instead of one shared alpha) - alphaX == alphaY
// reduces this exactly to the isotropic version this POC used through
// step 22, cross-checked against Blender Cycles' own
// `microfacet_ggx_sample_vndf` before being committed here.
inline float3 sampleGGXVNDF(float3 woLocal, float alphaX, float alphaY, thread uint& rngState) {
    float3 Vh = normalize(float3(alphaX * woLocal.x, alphaY * woLocal.y, woLocal.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : float3(1.0, 0.0, 0.0);
    float3 T2 = cross(Vh, T1);

    float u1 = randFloat(rngState);
    float u2 = randFloat(rngState);
    float r = sqrt(u1);
    float phi = 2.0 * M_PI_F * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = float3(alphaX * Nh.x, alphaY * Nh.y, max(0.0, Nh.z));
    return normalize(Ne);
}

// GGX multi-scatter energy compensation (phase 2 - see
// metal_poc_host_math.h's own buildGGXEnergyTable() comment for the full
// "why"). Bilinear lookup into the E(roughness, mu) table built once at
// host startup, mirroring sampleGGXEnergyTable()'s own host-side
// interpolation exactly (same index math, same clamping) rather than
// texture2d::sample() - this is plain float data with no image-file/
// sRGB concerns, and a `device const float*` buffer already matches the
// layout buildGGXEnergyTable() itself produces with no repacking needed.
inline float sampleGGXEnergyTableDevice(device const float* E, uint roughRes, uint muRes,
                                         float roughness, float mu) {
    float rf = roughness * float(roughRes) - 0.5;
    float mf = mu * float(muRes) - 0.5;
    int r0 = int(floor(rf)), m0 = int(floor(mf));
    float rt = rf - float(r0), mt = mf - float(m0);
    int r1 = r0 + 1, m1 = m0 + 1;
    r0 = clamp(r0, 0, int(roughRes) - 1);
    r1 = clamp(r1, 0, int(roughRes) - 1);
    m0 = clamp(m0, 0, int(muRes) - 1);
    m1 = clamp(m1, 0, int(muRes) - 1);
    float e00 = E[r0 * int(muRes) + m0];
    float e10 = E[r1 * int(muRes) + m0];
    float e01 = E[r0 * int(muRes) + m1];
    float e11 = E[r1 * int(muRes) + m1];
    float e0 = e00 + (e10 - e00) * rt;
    float e1 = e01 + (e11 - e01) * rt;
    return e0 + (e1 - e0) * mt;
}

