// metal_poc_kernel_subsurface.metal
// pbrt-v4's SubsurfaceMaterial (METAL_MAT_SUBSURFACE): a smooth dielectric boundary over a tabulated BSSRDF. A ray that refracts in does not
// walk through the object; it leaves the surface at another point found by a "probe walk" (pbrt-v4's TabulatedBSSRDF::Sample / Pdf), and there
// scatters by Sw, the NormalizedFresnel BSDF. This is a port of OptiX's bssrdf_probe_walk() (gpu/optix/optix_device_scatter_helpers.h) and the
// MaterialType::Subsurface case of shade_material(), which in turn follow the CPU's camera::sample_bssrdf_exit() (src/TheRestOfYourLife/
// camera_path.h); the radial profile itself is in metal_poc_bssrdf.metal.
//
// The material (TriangleMaterial, written by MetalPocApp::mapPbrtMaterial): color = sigma_a and transmitColor = sigma_s per colour channel, both
// per Metal-scene unit (the pbrt values divided by the scene scale), ior = eta, roughness = the NormalizedFresnel constant c of the exit BSDF,
// conductorEta.x = the table's element offset in the shared float buffer (an int stored bit for bit).

// "The same material" for the probe walk: pbrt compares the material objects; here, the parameters that define it.
inline bool sameSubsurfaceMaterial(TriangleMaterial a, TriangleMaterial b) {
    return b.materialType == METAL_MAT_SUBSURFACE && as_type<int>(a.conductorEta.x) == as_type<int>(b.conductorEta.x) &&
           a.ior == b.ior && all(float3(a.color) == float3(b.color)) && all(float3(a.transmitColor) == float3(b.transmitColor));
}

// Finds the exit point: samples a probe segment around p0 (a colour channel, a radius from that channel's profile, an angle, one of three
// axes), collects every hit of the same material along it, keeps one uniformly, and returns its position, its normal (facing the probe ray),
// Sp = Sr(distance) per channel, and the combined sampling pdf (the three axes and three channels, as pbrt does) and the selection
// probability. False when no exit point is found, which ends the path.
inline bool subsurfaceProbeWalk(thread const KernelRes& R, thread PathState& P, thread BounceState& B, float3 p0, float3 axis0,
                                thread float3& outPos, thread float3& outNormal, thread float3& outSp, thread float& outPdf, thread float& outSampleProb) {
    KERNEL_RES_ALIASES(R)
    constexpr int kMaxProbeSteps = 100;
    const TriangleMaterial surface = B.mat;
    const BssrdfTableView tab = bssrdfTableAt(rgbGridData + as_type<int>(surface.conductorEta.x));
    const float3 sigmaA = float3(surface.color), sigmaS = float3(surface.transmitColor);
    const float3 sigmaT = sigmaA + sigmaS;
    float3 rho = float3(0.0);
    for (int c = 0; c < 3; ++c) rho[c] = (sigmaT[c] > 0.0) ? (sigmaS[c] / sigmaT[c]) : 0.0;

    const float3 axis = normalize(axis0);
    const float3 t1 = (abs(axis.x) > 0.9) ? normalize(cross(float3(0, 1, 0), axis)) : normalize(cross(float3(1, 0, 0), axis));
    const float3 t2 = cross(axis, t1);

    const int channel = min(2, int(randFloat(P.rngState) * 3.0));
    const float r = bssrdfSampleSr(tab, sigmaT[channel], rho[channel], randFloat(P.rngState));
    if (r < 0.0) return false;
    const float rMax = bssrdfSampleSr(tab, sigmaT[channel], rho[channel], 0.999);
    if (rMax <= 0.0 || r >= rMax) return false;

    const float phi = 2.0 * M_PI_F * randFloat(P.rngState);
    const float halfLen = sqrt(max(0.0, rMax * rMax - r * r));

    const float uAxis = randFloat(P.rngState);
    float3 probeAxis, basisA, basisB;
    if (uAxis < 0.5)       { probeAxis = axis; basisA = t1;   basisB = t2; }
    else if (uAxis < 0.75) { probeAxis = t1;   basisA = t2;   basisB = axis; }
    else                   { probeAxis = t2;   basisA = axis; basisB = t1; }

    const float3 pTarget = p0 + r * (cos(phi) * basisA + sin(phi) * basisB);
    const float3 pStart = pTarget - halfLen * probeAxis;
    const float3 pEnd = pTarget + halfLen * probeAxis;
    float3 segDir = pEnd - pStart;
    const float segLen = length(segDir);
    if (segLen < 1e-10) return false;
    segDir /= segLen;

    float3 chosenPos = float3(0.0), chosenNormal = float3(0.0);
    int candidateCount = 0;
    float3 base = pStart;
    float remaining = segLen;
    for (int iter = 0; iter < kMaxProbeSteps && remaining > 1e-6; ++iter) {
        ray probeRay;
        probeRay.origin = base;
        probeRay.direction = segDir;
        probeRay.min_distance = 0.0f;
        probeRay.max_distance = remaining;
        SpherePayload probePayload{P.shutterT, false, -1};
        intersection_result<instancing, triangle_data> hit = isect.intersect(probeRay, accelStructure, functionTable, probePayload);
        if (hit.type == intersection_type::none) break;
        // Position, normal and material of the hit, by the same code a path vertex uses.
        PathState probeP = P;
        BounceState probeB;
        probeP.rayOrigin = base;
        probeP.rayDir = segDir;
        probeB.result = hit;
        resolveHit(R, probeP, probeB);
        if (sameSubsurfaceMaterial(surface, probeB.mat)) {
            ++candidateCount;
            if (randFloat(P.rngState) < 1.0 / float(candidateCount)) {
                chosenPos = probeB.hitPoint;
                chosenNormal = probeB.facingNormal;
            }
        }
        const float step = hit.distance + 1e-4;
        base += step * segDir;
        remaining -= step;
    }
    if (candidateCount == 0) return false;
    const float sampleProb = 1.0 / float(candidateCount);

    const float3 d = chosenPos - p0;
    const float dist = length(d);
    const float3 sp = float3(bssrdfSr(tab, sigmaT[0], rho[0], dist), bssrdfSr(tab, sigmaT[1], rho[1], dist), bssrdfSr(tab, sigmaT[2], rho[2], dist));

    // The pdf of having found this point: the sum over the three probe axes (probability 1/2, 1/4, 1/4) and the three colour channels.
    const float3 exitN = normalize(chosenNormal);
    const float dN = dot(d, axis), dT1 = dot(d, t1), dT2 = dot(d, t2);
    const float rProjAxis = sqrt(max(0.0, dT1 * dT1 + dT2 * dT2));
    const float rProjT1 = sqrt(max(0.0, dT2 * dT2 + dN * dN));
    const float rProjT2 = sqrt(max(0.0, dN * dN + dT1 * dT1));
    const float cosAxis = abs(dot(exitN, axis)), cosT1 = abs(dot(exitN, t1)), cosT2 = abs(dot(exitN, t2));
    float pdf = 0.0;
    for (int c = 0; c < 3; ++c) {
        pdf += 0.5 * bssrdfPdfSr(tab, sigmaT[c], rho[c], rProjAxis) * cosAxis
             + 0.25 * bssrdfPdfSr(tab, sigmaT[c], rho[c], rProjT1) * cosT1
             + 0.25 * bssrdfPdfSr(tab, sigmaT[c], rho[c], rProjT2) * cosT2;
    }
    pdf /= 3.0;
    if (pdf <= 0.0) return false;

    outPos = chosenPos;
    outNormal = exitN;
    outSp = sp;
    outPdf = pdf;
    outSampleProb = sampleProb;
    return true;
}

// The whole material at one hit: the smooth dielectric interface (reflect, or refract in); on refraction the probe walk, then the exit point
// shaded by NormalizedFresnel (Sw) with the path weighted by Sp / (selection probability * pdf). Returns false when the path ends.
inline bool shadeSubsurface(thread const KernelRes& R, thread PathState& P, thread BounceState& B) {
    KERNEL_RES_ALIASES(R)
    PATH_STATE_ALIASES(P)
    BOUNCE_STATE_ALIASES(B)
    // The interface is a plain DielectricBxDF(eta): no absorption (the medium's is in the BSSRDF).
    TriangleMaterial entryMat = mat;
    entryMat.color = float3(0.0);
    if (!shadeDielectric(entryMat, hitPoint, normal, facingNormal, frontFace, result.distance, rayDir, rayOrigin, throughput, specularBounce, rngState)) return false;
    if (dot(rayDir, facingNormal) >= 0.0) return true;   // reflected off the interface: an ordinary specular bounce

    float3 exitPos, exitNormal, sp;
    float pdf, sampleProb;
    if (!subsurfaceProbeWalk(R, P, B, hitPoint, facingNormal, exitPos, exitNormal, sp, pdf, sampleProb)) return false;
    throughput *= sp / (sampleProb * pdf);

    // Sw at the exit point: the same NormalizedFresnel the material of that name uses (ior = eta, roughness = c), with its own direct light
    // already seeing the weighted throughput.
    TriangleMaterial exitMat = mat;
    exitMat.materialType = METAL_MAT_NORMALIZED_FRESNEL;
    return shadeNormalizedFresnel(exitMat, exitPos, exitNormal, uniforms,
                          lights, pointLights, directionalLights, projectionLights, goniometricLights,
                          envMarginalCDF, envConditionalCDF, uniforms.envMapWidth, uniforms.envMapHeight,
                          pbrtEnvMarginalCDF, pbrtEnvConditionalCDF, uniforms.pbrtEnvMapWidth, uniforms.pbrtEnvMapHeight,
                          earthTexture, pbrtEnvTexture, goniometricTexture, pbrtGoniometricTexture, pbrtProjectionTexture, pbrtAreaLightTexture, textureSampler,
                          isect, accelStructure, functionTable,
                          rayDir, rayOrigin, throughput, radiance, bsdfPdf, specularBounce, rngState);
}
