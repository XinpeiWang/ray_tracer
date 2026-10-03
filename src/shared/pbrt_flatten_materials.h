#pragma once
// pbrt_flatten_materials.h -- flatten()'s materials phase
//
// Part of pbrt_flatten.h: included by it at the point flatten() is defined, and not usable on
// its own (it needs FlatScene, the flatten_detail helpers and the pbrt_scene types declared
// above that point). Each function is one phase of flatten(), moved here verbatim so no
// single function is thousands of lines long.

namespace flatten_detail {

// Mirrors scene.materials into out.materials 1:1 (same order, no skipped entries), resolving
// textures, mix/named references and every per-kind parameter.
inline void flattenMaterials(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- materials -------------------------------------------------------
	// Named materials, by name -> index in scene.materials. Built BEFORE the
	// main loop below (rather than incrementally during it) because a "mix"
	// material's "materials" parameter can name one declared LATER in the
	// file - pbrt itself resolves MixMaterial's sub-materials against the
	// full scene, not just what came before the mix directive - and because
	// flatten() mirrors scene.materials into out.materials 1:1 (same order,
	// no skipped entries), an index found here is already the right index
	// into out.materials too.
	std::map<std::string, int> namedMaterialIndex;
	for (std::size_t i = 0; i < scene.materials.size(); ++i)
		if (!scene.materials[i].name.empty())
			namedMaterialIndex[scene.materials[i].name] = static_cast<int>(i);

	for (const pbrt_scene::MaterialDecl &mdSrc : scene.materials) {
		// Mutable copy: any "texture"-typed param bound to a Texture of
		// class "constant" (pbrt-v4's literal-value texture - a scene
		// author's indirection for a colour reused across several
		// materials) gets rewritten IN PLACE to its real numeric value
		// below, before anything else in this loop iteration reads it. That
		// means every reader further down - the reflectance-vs-k precedence
		// just below, the conductor eta/k complex-IOR resolution, roughness,
		// transmittance, mix amount, etc - sees real numbers exactly like a
		// plain numeric param would, with no per-parameter special-casing
		// needed anywhere else in this function. p.strings is left alone,
		// so a param whose rewrite doesn't apply (no "value" found, or the
		// texture isn't "constant") still correctly reads as texture-bound
		// for the generic "not supported" warning below.
		pbrt_scene::MaterialDecl md = mdSrc;
		for (pbrt_scene::Param &p : md.params.items) {
			if (p.type != "texture" || p.strings.empty()) continue;
			const pbrt_scene::TextureDecl *constTex = findTexture(scene, p.strings[0]);
			if (!constTex || constTex->cls != "constant") continue;
			const pbrt_scene::Param *valueP = constTex->params.find("value");
			if (!valueP) continue;
			if (valueP->numbers.size() >= 3) {
				p.type = "rgb";
				p.numbers = {valueP->numbers[0], valueP->numbers[1], valueP->numbers[2]};
			} else if (valueP->numbers.size() == 1) {
				// "float value" form - broadcast to RGB (3 identical
				// numbers), same convention pbrt-v4 itself uses feeding a
				// float texture into an RGB parameter. Must actually
				// produce 3 numbers, not 1: getVec3() (pbrt_scene.h)
				// requires numbers.size() >= 3 and silently returns its
				// caller's default otherwise - a getFloat() consumer reads
				// numbers[0] regardless of size, so broadcasting to 3 stays
				// correct for both kinds of downstream reader.
				p.type = "rgb";
				const double v = valueP->numbers[0];
				p.numbers = {v, v, v};
			}
		}

		Material m;
		m.pbrtType = md.type;
		m.kind = materialKindFor(md.type);
		if (m.kind == MaterialKind::Unsupported) {
			warn("material type '" + md.type + "' is not supported; "
				 "it will fall back to a diffuse approximation");
		}
		// pbrt spells the base colour differently per material: reflectance for
		// diffuse, but conductors are described by their complex IOR and use
		// k/eta instead. Take whichever is present, defaulting to mid grey.
		const pbrt_scene::Vec3 def{m.color[0], m.color[1], m.color[2]};
		pbrt_scene::Vec3 c = md.params.getVec3("reflectance", def);
		if (!md.params.find("reflectance")) c = md.params.getVec3("k", c);
		m.color[0] = c.x; m.color[1] = c.y; m.color[2] = c.z;

		// A parameter bound to a Texture is silently worth a warning: the
		// material still renders, but with a constant colour where the scene
		// asked for a pattern. pbrt's ganesha (a CoatedDiffuse statue whose
		// reflectance is a bare imagemap) and barcelona-pavilion (many
		// CoatedDiffuse surfaces, mostly reflectance bound to a "scale"
		// texture wrapping an imagemap) are the cases that showed this -
		// without a word about it the render just looks like a shading bug.
		// The cases actually wired up below (each field's own comment has
		// the detail): "reflectance" bound to a bare "imagemap" on a Diffuse,
		// CoatedDiffuse, OR DiffuseTransmission material, optionally further
		// wrapped in a "scale" texture for all three kinds now
		// (Material::textureFilename/textureScale), "transmittance" bound
		// to a bare imagemap on DiffuseTransmission ONLY, likewise
		// optionally scale-wrapped (transmittanceTextureFilename/
		// transmittanceTextureScale), "reflectance" bound to a
		// flat-colour "checkerboard"/"fbm"/"marble"/"mix" texture on a
		// Diffuse OR CoatedDiffuse material (hasCheckerReflectance etc. -
		// no bundled scene binds any of these to a DiffuseTransmission
		// reflectance, so this scope stays narrower than the imagemap case),
		// "roughness" bound to a bare imagemap on Dielectric ONLY
		// (roughnessTextureFilename - no bundled scene needs this, added for
		// texture-parity with reflectance/transmittance above), and
		// "displacement" bound to an imagemap (optionally wrapped in a
		// "scale" texture) on ANY material kind (displacementTextureFilename).
		// Every other texture binding (other parameters/kinds, "mix" bound
		// directly to reflectance without a wrapping scale, or a
		// checkerboard/mix whose OWN tex1/tex2 are themselves texture
		// references) still just warns.
		for (const pbrt_scene::Param &p : md.params.items) {
			if (p.type != "texture") continue;
			if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse ||
				 m.kind == MaterialKind::DiffuseTransmission) &&
				p.name == "reflectance" && !p.strings.empty()) {
				const pbrt_scene::TextureDecl *tex = findTexture(scene, p.strings[0]);
				// "reflectance" bound to a "scale"-class Texture wrapping an
				// imagemap (barcelona-pavilion's own dominant pattern, e.g.
				// materials.pbrt's "concrete-kd": Texture "concrete-kd"
				// "scale" "texture tex" ["concrete-kd-img"]) - same unwrap as
				// "displacement" below, applied here too so reflectance gets
				// the same real-scene coverage rather than falling through
				// to the generic warning for every scale-wrapped case.
				// Diffuse/CoatedDiffuse/DiffuseTransmission all apply
				// m.textureScale for reflectance now (scaled_texture on CPU,
				// MaterialData::emissionScale reused on GPU) - checkerboard/
				// fbm/marble/mix stay Diffuse/CoatedDiffuse-only below (no
				// bundled scene binds any of those to a DiffuseTransmission
				// reflectance), so this stays narrower than the bare-imagemap
				// case just below, which all three kinds already handle
				// correctly. No kind-check needed here beyond the enclosing
				// `if` above (already exactly these three kinds) - unlike the
				// checkerboard/fbm/marble/mix branches below, which DO still
				// need their own narrower gate.
				// Unwrapped into a SEPARATE variable (imgTex), not reassigned
				// into `tex` itself: the checkerboard/fbm/marble/mix branches
				// below must still see the ORIGINAL (possibly "scale") decl
				// and correctly fall through to the generic warning for e.g.
				// a scale-wrapped checkerboard - unwrapping `tex` in place
				// would silently drop that scale factor and treat it as a
				// bare checkerboard instead.
				const pbrt_scene::TextureDecl *imgTex = tex;
				double texScale = 1.0;
				if (imgTex && imgTex->cls == "scale") {
					texScale = imgTex->params.getFloat("scale", 1.0);
					const pbrt_scene::Param *inner = imgTex->params.find("tex");
					imgTex = (inner && inner->type == "texture" && !inner->strings.empty())
						? findTexture(scene, inner->strings[0]) : nullptr;
				}
				if (imgTex && imgTex->cls == "imagemap") {
					const std::string filename = imgTex->params.getString("filename", "");
					if (!filename.empty()) {
						m.textureFilename = filename;
						m.textureScale = texScale;
						// imageMapOptionsFor() (pbrt_cpu_builder.h) and GPU's
						// own call sites read these 3 loose fields by name
						// (kept as-is rather than switched to a
						// TextureDecodeOptions field, to avoid touching every
						// existing reader) - resolveTextureDecodeOptions() is
						// still the single resolution point feeding them,
						// same as transmittanceTextureOptions/
						// roughnessTextureOptions below use directly.
						{
							const TextureDecodeOptions opts = resolveTextureDecodeOptions(*imgTex, warn);
							m.textureGamma = opts.gamma;
							m.textureWrap = opts.wrap;
							m.textureWrapIndex = opts.wrapIndex;
							m.textureInvert = opts.invert;
						}
						continue;   // resolved to an image, not a "not supported" warning
					}
				}
				// Shared by checkerboard/mix below: resolves a "texture"-typed
				// tex1/tex2 param to a bare imagemap's filename, ONE level
				// only - returns empty if the referenced Texture isn't a
				// bare "imagemap" (itself nested further, or a different
				// class), which the caller treats as "can't resolve, fall
				// through to the generic warning" (see hasCheckerReflectance/
				// hasMixReflectance's own comments).
				auto resolveNestedImagemap = [&](const pbrt_scene::Param *p) -> std::string {
					if (!p || p->strings.empty()) return std::string();
					// Last-declaration-wins, matching every other texture-
					// name lookup in this loop (e.g. the "reflectance" tex
					// lookup above) - pbrt's own redeclare-by-name-overrides
					// convention. Keep walking rather than returning on the
					// first match, or a scene that redeclares a Texture name
					// (imagemap first, something else later) would silently
					// resolve to the STALE earlier declaration instead of
					// correctly falling through to the generic warning.
					const pbrt_scene::TextureDecl *found = findTexture(scene, p->strings[0]);
					if (found && found->cls == "imagemap")
						return found->params.getString("filename", "");
					return std::string();
				};
				// Second level: resolves a "texture"-typed tex1/tex2 param to
				// ANOTHER checkerboard/mix Texture - see NestedProceduralTexture's
				// own comment for the exact (deliberately bounded) scope. Returns
				// a struct with an empty `kind` if the reference isn't itself a
				// checkerboard/mix (a different class, or unresolved), which the
				// caller treats as "can't resolve at this level either, fall
				// through to the generic warning" - same contract as
				// resolveNestedImagemap above.
				auto resolveNestedProcedural = [&](const pbrt_scene::Param *p) -> NestedProceduralTexture {
					NestedProceduralTexture out2;
					if (!p || p->strings.empty()) return out2;
					const pbrt_scene::TextureDecl *found = findTexture(scene, p->strings[0]);
					if (!found || (found->cls != "checkerboard" && found->cls != "mix"))
						return out2;
					const pbrt_scene::Param *t1p = found->params.find("tex1");
					const pbrt_scene::Param *t2p = found->params.find("tex2");
					const bool t1Nested = t1p && t1p->type == "texture";
					const bool t2Nested = t2p && t2p->type == "texture";
					const std::string t1Img = t1Nested ? resolveNestedImagemap(t1p) : std::string();
					const std::string t2Img = t2Nested ? resolveNestedImagemap(t2p) : std::string();
					// This level's own tex1/tex2 may only be a flat literal or a
					// bare imagemap - not nested further (the bound this struct's
					// own comment documents). A nested-but-unresolvable slot here
					// fails the WHOLE second level, same "all or nothing per
					// texture" rule the top level already enforces.
					if ((t1Nested && t1Img.empty()) || (t2Nested && t2Img.empty()))
						return out2;
					out2.kind = found->cls;
					if (t1Nested) out2.tex1Filename = t1Img;
					else {
						const pbrt_scene::Vec3 c1 = found->params.getVec3(
							"tex1", found->cls == "checkerboard" ? pbrt_scene::Vec3{1.0, 1.0, 1.0}
																  : pbrt_scene::Vec3{0.0, 0.0, 0.0});
						out2.color1[0] = c1.x; out2.color1[1] = c1.y; out2.color1[2] = c1.z;
					}
					if (t2Nested) out2.tex2Filename = t2Img;
					else {
						const pbrt_scene::Vec3 c2 = found->params.getVec3(
							"tex2", found->cls == "checkerboard" ? pbrt_scene::Vec3{0.0, 0.0, 0.0}
																  : pbrt_scene::Vec3{1.0, 1.0, 1.0});
						out2.color2[0] = c2.x; out2.color2[1] = c2.y; out2.color2[2] = c2.z;
					}
					if (found->cls == "checkerboard") {
						out2.uscale = found->params.getFloat("uscale", 1.0);
						out2.vscale = found->params.getFloat("vscale", 1.0);
					} else {
						const pbrt_scene::Param *amtp = found->params.find("amount");
						if (amtp && amtp->type == "texture") {
							out2.amountFilename = resolveNestedImagemap(amtp);
							if (out2.amountFilename.empty()) { out2.kind.clear(); return out2; }
						} else {
							out2.amount = found->params.getFloat("amount", 0.5);
						}
					}
					return out2;
				};
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "checkerboard") {
					const pbrt_scene::Param *tex1p = tex->params.find("tex1");
					const pbrt_scene::Param *tex2p = tex->params.find("tex2");
					const bool tex1IsNested = tex1p && tex1p->type == "texture";
					const bool tex2IsNested = tex2p && tex2p->type == "texture";
					const std::string tex1Img = tex1IsNested ? resolveNestedImagemap(tex1p) : std::string();
					const std::string tex2Img = tex2IsNested ? resolveNestedImagemap(tex2p) : std::string();
					// A nested slot that isn't a bare imagemap gets one more
					// try at the second level (another checkerboard/mix)
					// before giving up - see resolveNestedProcedural's own
					// comment for that level's own (narrower) scope.
					const NestedProceduralTexture tex1Proc =
						(tex1IsNested && tex1Img.empty()) ? resolveNestedProcedural(tex1p) : NestedProceduralTexture{};
					const NestedProceduralTexture tex2Proc =
						(tex2IsNested && tex2Img.empty()) ? resolveNestedProcedural(tex2p) : NestedProceduralTexture{};
					const bool tex1Resolved = !tex1IsNested || !tex1Img.empty() || !tex1Proc.kind.empty();
					const bool tex2Resolved = !tex2IsNested || !tex2Img.empty() || !tex2Proc.kind.empty();
					if (tex1Resolved && tex2Resolved) {
						if (tex1IsNested && !tex1Img.empty()) {
							m.checkerTex1Filename = tex1Img;
						} else if (tex1IsNested) {
							m.checkerTex1Nested = tex1Proc;
							// GPU has no representation for a nested procedural
							// sub-texture (TextureData's tex1ImageIdx/tex2ImageIdx
							// are image-only - see pbrt_gpu_builder.h's
							// resolveProceduralReflectanceTexture()) - leave
							// checkerColor1 as a flat average-of-the-nested-
							// pattern stand-in for GPU (real GPU-side warning
							// lives in scene_builder.cpp, matching every other
							// CPU-real/GPU-approximated gap's own convention).
							nestedProceduralAverageColor(tex1Proc, m.checkerColor1);
						} else {
							const pbrt_scene::Vec3 c1 = tex->params.getVec3("tex1", {1.0, 1.0, 1.0});
							m.checkerColor1[0] = c1.x; m.checkerColor1[1] = c1.y; m.checkerColor1[2] = c1.z;
						}
						if (tex2IsNested && !tex2Img.empty()) {
							m.checkerTex2Filename = tex2Img;
						} else if (tex2IsNested) {
							m.checkerTex2Nested = tex2Proc;
							nestedProceduralAverageColor(tex2Proc, m.checkerColor2);
						} else {
							const pbrt_scene::Vec3 c2 = tex->params.getVec3("tex2", {0.0, 0.0, 0.0});
							m.checkerColor2[0] = c2.x; m.checkerColor2[1] = c2.y; m.checkerColor2[2] = c2.z;
						}
						m.checkerUScale = tex->params.getFloat("uscale", 1.0);
						m.checkerVScale = tex->params.getFloat("vscale", 1.0);
						// "integer dimension" [3] - pbrt-v4's real 3D world-space
						// checker variant (Material::checkerIs3D's own comment).
						// worldToTexture is the CTM's inverse at Texture-
						// declaration time; inverseAffine() fails only for a
						// singular (zero-scale-on-some-axis) CTM, which would
						// make the checker degenerate on any renderer anyway -
						// falls back to identity (whole-world one cell) rather
						// than propagating a failure, matching this loader's
						// own "render something sensible, don't abort the
						// load" convention for a malformed-but-present feature.
						if (tex->params.getInt("dimension", 2) == 3) {
							m.checkerIs3D = true;
							pbrt_scene::Matrix4 worldToTexture;
							if (!tex->xform.inverseAffine(worldToTexture))
								worldToTexture = pbrt_scene::Matrix4::identity();
							for (int i = 0; i < 16; ++i) m.checkerWorldToTexture[i] = worldToTexture.m[i];
						}
						m.hasCheckerReflectance = true;
						continue;   // resolved to a procedural checker, not a "not supported" warning
					}
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "fbm") {
					m.fbmOctaves = tex->params.getInt("octaves", 8);
					m.fbmRoughness = tex->params.getFloat("roughness", 0.5);
					m.hasFbmReflectance = true;
					continue;   // resolved to procedural fbm noise, not a "not supported" warning
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "marble") {
					m.marbleOctaves = tex->params.getInt("octaves", 8);
					m.marbleRoughness = tex->params.getFloat("roughness", 0.5);
					m.marbleScale = tex->params.getFloat("scale", 1.0);
					m.marbleVariation = tex->params.getFloat("variation", 0.2);
					m.hasMarbleReflectance = true;
					continue;   // resolved to procedural marble, not a "not supported" warning
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "mix") {
					// See hasMixReflectance's own comment - tex1/tex2/amount
					// all support up to two levels (bare imagemap, or a
					// further checkerboard/mix via resolveNestedProcedural);
					// a nested slot that resolves at neither level still
					// falls through to the generic warning below.
					const pbrt_scene::Param *tex1p = tex->params.find("tex1");
					const pbrt_scene::Param *tex2p = tex->params.find("tex2");
					const pbrt_scene::Param *amountp = tex->params.find("amount");
					const bool tex1IsNested = tex1p && tex1p->type == "texture";
					const bool tex2IsNested = tex2p && tex2p->type == "texture";
					const bool amountIsNested = amountp && amountp->type == "texture";
					const std::string tex1Img = tex1IsNested ? resolveNestedImagemap(tex1p) : std::string();
					const std::string tex2Img = tex2IsNested ? resolveNestedImagemap(tex2p) : std::string();
					const std::string amountImg = amountIsNested ? resolveNestedImagemap(amountp) : std::string();
					const NestedProceduralTexture tex1Proc =
						(tex1IsNested && tex1Img.empty()) ? resolveNestedProcedural(tex1p) : NestedProceduralTexture{};
					const NestedProceduralTexture tex2Proc =
						(tex2IsNested && tex2Img.empty()) ? resolveNestedProcedural(tex2p) : NestedProceduralTexture{};
					const NestedProceduralTexture amountProc =
						(amountIsNested && amountImg.empty()) ? resolveNestedProcedural(amountp) : NestedProceduralTexture{};
					const bool tex1Resolved = !tex1IsNested || !tex1Img.empty() || !tex1Proc.kind.empty();
					const bool tex2Resolved = !tex2IsNested || !tex2Img.empty() || !tex2Proc.kind.empty();
					const bool amountResolved = !amountIsNested || !amountImg.empty() || !amountProc.kind.empty();
					if (tex1Resolved && tex2Resolved && amountResolved) {
						if (tex1IsNested && !tex1Img.empty()) {
							m.mixTex1Filename = tex1Img;
						} else if (tex1IsNested) {
							m.mixTex1Nested = tex1Proc;
							// See the checkerboard case's identical comment above -
							// same GPU-approximation, same reason.
							nestedProceduralAverageColor(tex1Proc, m.mixColor1);
						} else {
							const pbrt_scene::Vec3 c1 = tex->params.getVec3("tex1", {0.0, 0.0, 0.0});
							m.mixColor1[0] = c1.x; m.mixColor1[1] = c1.y; m.mixColor1[2] = c1.z;
						}
						if (tex2IsNested && !tex2Img.empty()) {
							m.mixTex2Filename = tex2Img;
						} else if (tex2IsNested) {
							m.mixTex2Nested = tex2Proc;
							nestedProceduralAverageColor(tex2Proc, m.mixColor2);
						} else {
							const pbrt_scene::Vec3 c2 = tex->params.getVec3("tex2", {1.0, 1.0, 1.0});
							m.mixColor2[0] = c2.x; m.mixColor2[1] = c2.y; m.mixColor2[2] = c2.z;
						}
						if (amountIsNested && !amountImg.empty()) {
							m.mixAmountTextureFilename = amountImg;
						} else if (amountIsNested) {
							m.mixAmountNested = amountProc;
							// See tex1/tex2's own identical GPU-approximation
							// comment above - same reason, scalar instead of
							// colour (nestedProceduralAverageScalar's own
							// comment).
							m.mixAmount = nestedProceduralAverageScalar(amountProc);
						} else {
							m.mixAmount = tex->params.getFloat("amount", 0.5);
						}
						m.hasMixReflectance = true;
						continue;   // resolved to a procedural mix, not a "not supported" warning
					}
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "windy") {
					// Parameterless in real pbrt-v4 - nothing to read.
					m.hasWindyReflectance = true;
					continue;   // resolved to procedural windy noise, not a "not supported" warning
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "wrinkled") {
					m.wrinkledOctaves = tex->params.getInt("octaves", 8);
					m.wrinkledRoughness = tex->params.getFloat("roughness", 0.5);
					m.hasWrinkledReflectance = true;
					continue;   // resolved to procedural wrinkled turbulence, not a "not supported" warning
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "dots") {
					// Same one-level-nested-bare-imagemap support for
					// inside/outside as checkerboard's own tex1/tex2 above.
					const pbrt_scene::Param *insideP = tex->params.find("inside");
					const pbrt_scene::Param *outsideP = tex->params.find("outside");
					const bool insideIsNested = insideP && insideP->type == "texture";
					const bool outsideIsNested = outsideP && outsideP->type == "texture";
					const std::string insideImg = insideIsNested ? resolveNestedImagemap(insideP) : std::string();
					const std::string outsideImg = outsideIsNested ? resolveNestedImagemap(outsideP) : std::string();
					if ((!insideIsNested || !insideImg.empty()) && (!outsideIsNested || !outsideImg.empty())) {
						if (insideIsNested) {
							m.dotsInsideTexFilename = insideImg;
						} else {
							const pbrt_scene::Vec3 c1 = tex->params.getVec3("inside", {1.0, 1.0, 1.0});
							m.dotsInsideColor[0] = c1.x; m.dotsInsideColor[1] = c1.y; m.dotsInsideColor[2] = c1.z;
						}
						if (outsideIsNested) {
							m.dotsOutsideTexFilename = outsideImg;
						} else {
							const pbrt_scene::Vec3 c2 = tex->params.getVec3("outside", {0.0, 0.0, 0.0});
							m.dotsOutsideColor[0] = c2.x; m.dotsOutsideColor[1] = c2.y; m.dotsOutsideColor[2] = c2.z;
						}
						m.hasDotsReflectance = true;
						continue;   // resolved to a procedural dots pattern, not a "not supported" warning
					}
				}
				if ((m.kind == MaterialKind::Diffuse || m.kind == MaterialKind::CoatedDiffuse) &&
					tex && tex->cls == "bilerp") {
					const pbrt_scene::Vec3 v00 = tex->params.getVec3("v00", {0.0, 0.0, 0.0});
					const pbrt_scene::Vec3 v01 = tex->params.getVec3("v01", {1.0, 1.0, 1.0});
					const pbrt_scene::Vec3 v10 = tex->params.getVec3("v10", {0.0, 0.0, 0.0});
					const pbrt_scene::Vec3 v11 = tex->params.getVec3("v11", {1.0, 1.0, 1.0});
					m.bilerpV00[0] = v00.x; m.bilerpV00[1] = v00.y; m.bilerpV00[2] = v00.z;
					m.bilerpV01[0] = v01.x; m.bilerpV01[1] = v01.y; m.bilerpV01[2] = v01.z;
					m.bilerpV10[0] = v10.x; m.bilerpV10[1] = v10.y; m.bilerpV10[2] = v10.z;
					m.bilerpV11[0] = v11.x; m.bilerpV11[1] = v11.y; m.bilerpV11[2] = v11.z;
					m.hasBilerpReflectance = true;
					continue;   // resolved to a procedural bilerp blend, not a "not supported" warning
				}
			}
			// DiffuseTransmission's own "transmittance" - same "scale"-wrap
			// support as "reflectance" just above (Material::
			// transmittanceTextureScale's own comment), still no procedural
			// (checkerboard/fbm/marble/mix) support - no bundled scene binds
			// any of those to a DiffuseTransmission transmittance either.
			if (m.kind == MaterialKind::DiffuseTransmission &&
				p.name == "transmittance" && !p.strings.empty()) {
				// No separate `tex`-vs-`imgTex` split needed here (unlike
				// reflectance's own block above): transmittance has no
				// checkerboard/fbm/marble/mix fallback that needs the
				// original (pre-unwrap) declaration, so unwrapping straight
				// into `imgTex` is safe.
				const pbrt_scene::TextureDecl *imgTex = findTexture(scene, p.strings[0]);
				double texScale = 1.0;
				if (imgTex && imgTex->cls == "scale") {
					texScale = imgTex->params.getFloat("scale", 1.0);
					const pbrt_scene::Param *inner = imgTex->params.find("tex");
					imgTex = (inner && inner->type == "texture" && !inner->strings.empty())
						? findTexture(scene, inner->strings[0]) : nullptr;
				}
				if (imgTex && imgTex->cls == "imagemap") {
					const std::string filename = imgTex->params.getString("filename", "");
					if (!filename.empty()) {
						m.transmittanceTextureFilename = filename;
						m.transmittanceTextureScale = texScale;
						m.transmittanceTextureOptions = resolveTextureDecodeOptions(*imgTex, warn);
						continue;   // resolved to an image, not a "not supported" warning
					}
				}
			}
			// Dielectric's own "roughness" - bare imagemap only (see
			// Material::roughnessTextureFilename's own comment); no
			// "scale"-wrap or procedural (checkerboard/fbm/marble/mix)
			// support, same scope narrowing as transmittance above.
			if (m.kind == MaterialKind::Dielectric &&
				p.name == "roughness" && !p.strings.empty()) {
				const pbrt_scene::TextureDecl *tex = findTexture(scene, p.strings[0]);
				if (tex && tex->cls == "imagemap") {
					const std::string filename = tex->params.getString("filename", "");
					if (!filename.empty()) {
						m.roughnessTextureFilename = filename;
						m.roughnessTextureOptions = resolveTextureDecodeOptions(*tex, warn);
						continue;   // resolved to an image, not a "not supported" warning
					}
				}
			}
			// "displacement" (bump mapping) - not gated on `kind`, since real
			// scenes bind it on coateddiffuse/dielectric/etc, not just
			// Diffuse. Handles both forms real scenes use (barcelona-
			// pavilion): a direct "imagemap" texture, or one wrapped in a
			// "scale"-class texture (its own "texture tex" naming the real
			// imagemap, its "float scale" becoming displacementScale).
			if (p.name == "displacement" && !p.strings.empty()) {
				const pbrt_scene::TextureDecl *tex = findTexture(scene, p.strings[0]);
				double scale = 1.0;
				if (tex && tex->cls == "scale") {
					scale = tex->params.getFloat("scale", 1.0);
					const pbrt_scene::Param *inner = tex->params.find("tex");
					tex = (inner && inner->type == "texture" && !inner->strings.empty())
						? findTexture(scene, inner->strings[0]) : nullptr;
				}
				if (tex && tex->cls == "imagemap") {
					const std::string filename = tex->params.getString("filename", "");
					if (!filename.empty()) {
						m.displacementTextureFilename = filename;
						m.displacementScale = scale;
						warnIfImagemapOptionsIgnored(*tex, "displacement", warn);
						continue;   // resolved to an image, not a "not supported" warning
					}
				}
			}
			warn("material '" + md.type + "' binds its '" + p.name +
				 "' to a texture, which is not supported; a constant colour is "
				 "used instead");
		}

		// "roughness" is pbrt's isotropic spelling; a scene that instead gives
		// separate "uroughness"/"vroughness" (its anisotropic spelling) has no
		// isotropic value to find, and without this fallback chain silently
		// renders as a perfect mirror (roughness 0) regardless of what either
		// one said - a much larger visual difference than the isotropic
		// approximation this settles for (using whichever of the two is
		// present; a scene with both must average them by hand, but that is
		// rare in practice and this format has no field for true anisotropy).
		m.roughness = md.params.getFloat("roughness",
							md.params.getFloat("uroughness",
								md.params.getFloat("vroughness", 0.0)));
		// True anisotropic roughness (Round 6 Phase 3) - each axis
		// independently falls back to "roughness" when its own name is
		// absent, matching pbrt-v4's own per-axis default (real pbrt-v4
		// material parsing: uroughness/vroughness each individually
		// default to the material's "roughness" parameter, not to 0).
		m.roughness_u = md.params.getFloat("uroughness", md.params.getFloat("roughness", 0.0));
		m.roughness_v = md.params.getFloat("vroughness", md.params.getFloat("roughness", 0.0));
		m.remapRoughness = md.params.getBool("remaproughness", true);
		// "eta" is pbrt's name for index of refraction on dielectrics.
		//
		// CoatedConductor is a real exception to this: it ALSO conventionally
		// spells its base conductor's own complex IOR "eta" (rgb- or
		// spectrum-typed), a completely different concept from the material's
		// own coat IOR read here. ParamList::find()/getFloat() match by name
		// only, never type, so an explicit "rgb eta" conductor base collides
		// with this generic read: getFloat() only checks `!numbers.empty()`,
		// and an "rgb eta" Param's numbers are its 3 real RGB values, so
		// getFloat("eta", ...) would silently return conductorEta.r as if it
		// were the coat's own IOR - confirmed to break CoatedConductor with
		// an explicit "rgb eta" base (the coat's own IOR ends up < 1, a
		// physically-degenerate dielectric). A "spectrum eta" conductor base
		// happens to collide harmlessly instead (its payload lives in
		// `strings`, so `numbers` is empty and this still falls through to
		// "ior"/1.5) - which is exactly why this went unnoticed until an
		// explicit-rgb-eta CoatedConductor scene hit it. Scoped to
		// CoatedConductor specifically (not e.g. Dielectric/Subsurface/Hair,
		// which have no second "eta" concept to collide with) because a
		// resolved-constant-texture "eta" legitimately arrives here with
		// Param::type=="rgb" too (the pre-pass above rewrites a "texture eta"
		// bound to a constant texture in place, changing its type - see
		// DielectricConstantTextureEtaResolvesWithoutASpuriousGlassWarning,
		// pbrt_flatten_tests.cpp) - a blanket "must be float-typed" check
		// would wrongly reject that real, already-correctly-resolved case.
		if (m.kind == MaterialKind::CoatedConductor) {
			const pbrt_scene::Param* etaFloatP = md.params.find("eta");
			if (!etaFloatP || etaFloatP->type != "float") etaFloatP = nullptr;
			if (!etaFloatP) {
				const pbrt_scene::Param* iorFloatP = md.params.find("ior");
				if (iorFloatP && iorFloatP->type == "float") etaFloatP = iorFloatP;
			}
			m.ior = (etaFloatP && !etaFloatP->numbers.empty()) ? etaFloatP->numbers[0] : 1.5;
		} else {
			m.ior = md.params.getFloat("eta", md.params.getFloat("ior", 1.5));
		}
		// MaterialKind::Interface (pbrt's Material "none"/"") ignores m.ior
		// entirely - see interface_material's own comment
		// (material_simple.h) for why it's a real pass-through, not a
		// near-1-eta Dielectric.

		// Dielectric/ThinDielectric/CoatedDiffuse: pbrt-v4 conventionally
		// describes a dielectric's IOR via "spectrum eta" bound to a NAMED
		// glass ("glass-BK7" etc.), not the plain float getFloat() above can
		// read - getFloat() only inspects a Param's `numbers`, and a named-
		// spectrum Param has none, so this previously fell through to the
		// generic 1.5 default with NO warning at all (worse than this
		// loader's usual "unrecognized -> fallback + warn" convention,
		// since 1.5 happens to look plausible for glass and gives no signal
		// anything was ignored). Gated on the Param's own declared type
		// ("spectrum"), not just "does eta have a string at all" - a
		// "texture eta" binding (pbrt-v4's other legal string-valued form
		// for eta) also stores its payload in `strings`. An UNRESOLVED
		// texture reference keeps type=="texture" (correctly excluded); a
		// texture that resolves via the constant-texture rewrite pre-pass
		// above (this loop's very first block) gets its type rewritten to
		// "rgb" (that pass deliberately leaves `strings` populated even
		// after rewriting, for ITS OWN "still texture-bound" bookkeeping
		// elsewhere) - also not "spectrum". Gating on type=="spectrum" here
		// correctly excludes both texture forms without special-casing
		// either.
		if (m.kind == MaterialKind::Dielectric || m.kind == MaterialKind::ThinDielectric ||
			m.kind == MaterialKind::CoatedDiffuse) {
			if (const pbrt_scene::Param *etaP = md.params.find("eta");
				etaP && etaP->type == "spectrum") {
				if (!etaP->strings.empty()) {
					const std::string &etaName = etaP->strings[0];
					const std::string glass = glassElementFromSpectrumName(etaName);
					if (!glass.empty()) {
						if (const GlassPreset *preset = FindGlassPreset(glass.c_str())) {
							m.ior = preset->nd;
						} else {
							warn("material '" + md.type + "' names an unrecognized glass "
								 "\"" + etaName + "\" for \"spectrum eta\"; eta=" +
								 std::to_string(m.ior) + " is used instead");
						}
					} else {
						warn("material '" + md.type + "' binds \"eta\" to the named spectrum "
							 "\"" + etaName + "\", which is not a recognized glass preset; eta=" +
							 std::to_string(m.ior) + " is used instead");
					}
				} else if (!etaP->numbers.empty()) {
					// Inline piecewise-linear spectral data ("spectrum eta"
					// [ 400 1.5168 700 1.5142 ... ], real pbrt-v4 syntax for
					// a custom spectrum with no named preset) - resolving
					// this properly would need a real spectral integration
					// this loader doesn't have; the generic getFloat() read
					// above already misread numbers[0] (a WAVELENGTH, e.g.
					// 400) as if it were the IOR itself, which is worse than
					// the plain default - reset to the ior-or-1.5 fallback
					// and warn instead of rendering with a nonsensical eta.
					warn("material '" + md.type + "' gives \"spectrum eta\" as inline "
						 "wavelength/value data, which is not supported; eta=" +
						 std::to_string(md.params.getFloat("ior", 1.5)) + " is used instead");
					m.ior = md.params.getFloat("ior", 1.5);
				}
			}
		}

		// Principled only (this loader's own non-standard material - see
		// MaterialKind::Principled's own comment): "metallic"/"clearcoat"/
		// "clearcoatroughness" plain float params, same defaults as
		// `principled`'s own CPU convenience constructor. Scoped to this one
		// kind so an unrelated material's own unrecognized float param
		// doesn't silently do nothing here instead of hitting whatever
		// generic "not supported" handling it would otherwise get.
		if (m.kind == MaterialKind::Principled) {
			m.metallic = md.params.getFloat("metallic", 0.0);
			m.clearcoat = md.params.getFloat("clearcoat", 0.0);
			m.clearcoatRoughness = md.params.getFloat("clearcoatroughness", 0.1);
		}

		// Dielectric only - "abbenumber", a plain float (see Material::
		// abbeNumber's own comment for why this isn't real pbrt-v4 syntax).
		// Scoped to this one kind for the same reason as Principled's own
		// scalar params just above.
		if (m.kind == MaterialKind::Dielectric) {
			m.abbeNumber = md.params.getFloat("abbenumber", 0.0);
			const pbrt_scene::Vec3 tf = md.params.getVec3("tf", pbrt_scene::Vec3{1.0, 1.0, 1.0});
			m.transmissionFilter[0] = tf.x; m.transmissionFilter[1] = tf.y; m.transmissionFilter[2] = tf.z;
			// Only the smooth, non-dispersive glass applies it (both builders) - say so
			// rather than silently rendering clear glass for a rough/dispersive one.
			if ((tf.x != 1.0 || tf.y != 1.0 || tf.z != 1.0) &&
				(m.roughness_u > 0.0 || m.roughness_v > 0.0 || m.abbeNumber > 0.0))
				warn("Material \"dielectric\" \"rgb tf\" is ignored on a rough or dispersive "
					 "dielectric (only smooth, non-dispersive glass applies the transmission filter)");
		}

		// Conductor OR CoatedConductor: pbrt describes a conductor's complex
		// IOR via "spectrum eta"/"spectrum k" bound to a NAMED spectrum
		// ("metal-Ag-eta"/"metal-Ag-k", etc.), not the plain floats/RGB
		// getFloat()/getVec3() above can read - resolve the common named-
		// metal case against this codebase's own RGB-approximated conductor
		// table instead of always falling back to the metal/fuzz-mirror
		// approximation (getString() only inspects the param's `strings`
		// vector, so this is safe to call even when "eta"/"k" turn out to be
		// an explicit RGB value instead - it just won't find one there).
		// CoatedConductor previously never ran this at all (Conductor-only
		// gate) - even "metal-Ag-eta" was silently ignored, always falling
		// back to reflectanceToConductorK()'s approximation regardless of
		// what the scene actually asked for.
		if (m.kind == MaterialKind::Conductor || m.kind == MaterialKind::CoatedConductor) {
			std::string elem = conductorElementFromSpectrumName(md.params.getString("eta", ""));
			if (elem.empty()) elem = conductorElementFromSpectrumName(md.params.getString("k", ""));
			if (const ConductorPreset* preset = elem.empty() ? nullptr : FindConductorPreset(elem.c_str())) {
				m.hasConductorPreset = true;
				m.conductorEta[0] = preset->eta_r; m.conductorEta[1] = preset->eta_g; m.conductorEta[2] = preset->eta_b;
				m.conductorK[0]   = preset->k_r;   m.conductorK[1]   = preset->k_g;   m.conductorK[2]   = preset->k_b;
			} else if (const pbrt_scene::Param* etaP = md.params.find("eta"); etaP && etaP->numbers.size() >= 3 &&
					   md.params.find("k") && md.params.find("k")->numbers.size() >= 3) {
				// An explicit "rgb eta"/"rgb k" (not a named spectrum, or
				// pbrt_flatten.h wouldn't have reached here) - this
				// codebase's own real GGX conductor BxDF (bxdfs_conductor.h,
				// gpu/optix/optix_types.h's eta_c/k_c) is already a plain
				// 3-float RGB model matching ConductorPreset's own shape
				// exactly, unlike pbrt-v4's real per-wavelength spectral
				// upsample for this same case (RGBUnboundedSpectrum) - so
				// this just reads the 3 numbers directly, no spectral
				// machinery needed. Requires BOTH eta and k as real RGB
				// triples (not just one) to activate the real model, since a
				// scene giving only one has no pbrt-v4-documented default
				// for CoatedConductor to fall back to the way plain
				// Conductor's own Cu-default branch below does.
				const pbrt_scene::Vec3 eta = md.params.getVec3("eta", {0.0, 0.0, 0.0});
				const pbrt_scene::Vec3 k   = md.params.getVec3("k",   {0.0, 0.0, 0.0});
				m.hasConductorPreset = true;
				m.conductorEta[0] = eta.x; m.conductorEta[1] = eta.y; m.conductorEta[2] = eta.z;
				m.conductorK[0]   = k.x;   m.conductorK[1]   = k.y;   m.conductorK[2]   = k.z;
			} else if (m.kind == MaterialKind::Conductor &&
					   !md.params.find("eta") && !md.params.find("k") && !md.params.find("reflectance")) {
				// pbrt-v4's real default (materials.cpp's ConductorMaterial::
				// Create: "if (!reflectance) { if (!eta) eta = Cu-eta; if
				// (!k) k = Cu-k; }") when a scene gives NONE of eta/k/
				// reflectance at all: real copper, not a neutral/generic
				// reflector - this codebase's own bundled scenes
				// (example-cornell.pbrt, infinite-light.pbrt,
				// instanced-spheres.pbrt, realistic-camera.pbrt,
				// spherical-camera.pbrt) all just write
				// Material "conductor" "float roughness" [x] and expect
				// real pbrt-v4's shiny-copper look, not the grey fuzz-mirror
				// this fell back to before. An explicit (even if
				// unrecognized-as-a-named-spectrum) eta/k/reflectance still
				// falls through to that fuzz-mirror approximation below,
				// unchanged - this only replaces the "gave nothing at all"
				// case. Conductor ONLY: pbrt-v4 documents no equivalent
				// "nothing given" default for CoatedConductorMaterial, so
				// this doesn't extend to it - CoatedConductor's own
				// reflectanceToConductorK() approximation stays the
				// fallback for that kind's "nothing given" case, unchanged.
				if (const ConductorPreset* cu = FindConductorPreset("Cu")) {
					m.hasConductorPreset = true;
					m.conductorEta[0] = cu->eta_r; m.conductorEta[1] = cu->eta_g; m.conductorEta[2] = cu->eta_b;
					m.conductorK[0]   = cu->k_r;   m.conductorK[1]   = cu->k_g;   m.conductorK[2]   = cu->k_b;
				}
			}
		}

		// DiffuseTransmission only, but harmless to read unconditionally: no
		// other material kind has a "transmittance" parameter to collide with.
		const pbrt_scene::Vec3 defT{m.transmittance[0], m.transmittance[1], m.transmittance[2]};
		const pbrt_scene::Vec3 t = md.params.getVec3("transmittance", defT);
		m.transmittance[0] = t.x; m.transmittance[1] = t.y; m.transmittance[2] = t.z;

		// Subsurface: resolve sigma_a/sigma_s the way pbrt-v4's own
		// SubsurfaceMaterial::Create does (materials.cpp) - "4 mutually
		// exclusive ways to specify the subsurface properties", of which this
		// supports the first three (named preset, explicit sigma_a+sigma_s,
		// and "nothing specified" defaults). The fourth (reflectance+mfp,
		// inverted through the diffusion table via SubsurfaceFromDiffuse)
		// would need a BSSRDFTable built here at PARSE time just to invert
		// one number, for a form no scene in this loader's corpus uses; it
		// warns and falls back to the default coefficients instead.
		if (m.kind == MaterialKind::Subsurface) {
			const double scale = md.params.getFloat("scale", 1.0);
			double g = md.params.getFloat("g", 0.0);
			double sigA[3] = {m.sigma_a[0], m.sigma_a[1], m.sigma_a[2]};
			double sigS[3] = {m.sigma_s[0], m.sigma_s[1], m.sigma_s[2]};

			const std::string name = md.params.getString("name", "");
			const bool hasSigmaA = md.params.find("sigma_a") != nullptr;
			const bool hasSigmaS = md.params.find("sigma_s") != nullptr;

			if (!name.empty()) {
				if (const SubsurfacePreset *preset = subsurfacePresetFor(name)) {
					for (int c = 0; c < 3; ++c) {
						sigA[c] = preset->sigmaA[c];
						sigS[c] = preset->sigmaPrimeS[c];
					}
					// pbrt-v4: "Enforce g=0 (the database specifies reduced
					// scattering coefficients)".
					if (g != 0.0)
						warn("material 'subsurface' ignores \"g\" when \"name\" "
							 "selects a measured scattering preset (matches pbrt-v4)");
					g = 0.0;
				} else {
					warn("material 'subsurface' names unknown scattering preset '" +
						 name + "'; using the default coefficients instead");
				}
			} else if (hasSigmaA && hasSigmaS) {
				const pbrt_scene::Vec3 a =
					md.params.getVec3("sigma_a", pbrt_scene::Vec3{sigA[0], sigA[1], sigA[2]});
				const pbrt_scene::Vec3 s =
					md.params.getVec3("sigma_s", pbrt_scene::Vec3{sigS[0], sigS[1], sigS[2]});
				sigA[0] = a.x; sigA[1] = a.y; sigA[2] = a.z;
				sigS[0] = s.x; sigS[1] = s.y; sigS[2] = s.z;
			} else if (hasSigmaA != hasSigmaS) {
				warn("material 'subsurface' gives only one of \"sigma_a\"/\"sigma_s\"; "
					 "both are required together, so the default coefficients are used instead");
			} else if (md.params.find("reflectance")) {
				warn("material 'subsurface' gives \"reflectance\" without \"sigma_a\"/"
					 "\"sigma_s\"; this loader does not invert reflectance+mfp into "
					 "scattering coefficients, so the default coefficients are used instead");
			}
			// else: nothing specified at all -- m.sigma_a/sigma_s's own
			// defaults (already pbrt-v4's "nothing specified" preset) stand.

			for (int c = 0; c < 3; ++c) {
				m.sigma_a[c] = sigA[c] * scale;
				m.sigma_s[c] = sigS[c] * scale;
			}
			m.g = g;

			// pbrt-v4's SubsurfaceMaterial defaults eta to 1.33 (skin/water-
			// like), not the 1.5 (glass-like) default the generic "eta"/"ior"
			// read above already applied for every material kind - redo it
			// here with the right default when the scene gave neither.
			if (!md.params.find("eta") && !md.params.find("ior")) {
				m.ior = 1.33;
			} else if (const pbrt_scene::Param *etaP = md.params.find("eta");
					   etaP && etaP->type == "spectrum") {
				// Real pbrt-v4 SubsurfaceMaterial reads eta as a plain float
				// only (materials.cpp: GetOneFloat) - no spectrum support at
				// all, unlike Dielectric/ThinDielectric/CoatedDiffuse above.
				// A "spectrum eta" here (named glass or inline data) has no
				// usable float value either, so the generic getFloat() read
				// above landed on the generic 1.5 (glass) default - treat it
				// the same as "nothing given" instead, and say so, rather
				// than silently keeping a default this material kind never
				// actually uses.
				warn("material 'subsurface' binds \"eta\" to a spectrum, which "
					 "is not supported (pbrt-v4's own SubsurfaceMaterial reads "
					 "eta as a plain float only); the default eta (1.33) is "
					 "used instead");
				m.ior = 1.33;
			}
		}

		// Hair: resolve sigma_a the way pbrt-v4's own HairMaterial::Create
		// does (materials.cpp) - "sigma_a" wins if present; else
		// "reflectance"/"color"; else "eumelanin"/"pheomelanin"; else the
		// default brown preset (eumelanin=1.3, pheomelanin=0).
		if (m.kind == MaterialKind::Hair) {
			// beta_n/alpha/eta read BEFORE sigma_a resolution below - the
			// "reflectance" branch's SigmaAFromReflectance needs beta_n as an
			// input, so it has to already be known by the time that branch
			// runs.
			m.betaM = md.params.getFloat("beta_m", m.betaM);
			m.betaN = md.params.getFloat("beta_n", m.betaN);
			m.alphaDeg = md.params.getFloat("alpha", m.alphaDeg);
			// pbrt-v4's HairMaterial defaults eta to 1.55, not the 1.5
			// (glass-like) default the generic "eta"/"ior" read above
			// already applied for every material kind.
			if (!md.params.find("eta") && !md.params.find("ior")) {
				m.ior = 1.55;
			} else if (const pbrt_scene::Param *etaP = md.params.find("eta");
					   etaP && etaP->type == "spectrum") {
				// Same reasoning as Subsurface's own spectrum-eta check
				// above: real pbrt-v4 HairMaterial reads eta via a float
				// texture only (materials.cpp: GetFloatTexture) - no
				// spectrum support - so a "spectrum eta" has no usable
				// value here either.
				warn("material 'hair' binds \"eta\" to a spectrum, which is "
					 "not supported (pbrt-v4's own HairMaterial reads eta as "
					 "a float only); the default eta (1.55) is used instead");
				m.ior = 1.55;
			}

			const bool hasSigmaA = md.params.find("sigma_a") != nullptr;
			const bool hasReflectance = md.params.find("reflectance") != nullptr ||
										 md.params.find("color") != nullptr;
			const bool hasEumelanin = md.params.find("eumelanin") != nullptr;
			const bool hasPheomelanin = md.params.find("pheomelanin") != nullptr;

			// pbrt-v4 HairBxDF::SigmaAFromConcentration (bxdfs.cpp) - a
			// closed-form RGB fit (eumelanin/pheomelanin absorption spectra
			// pre-integrated against sRGB), not a real per-wavelength
			// spectral upsample. Shared by the eumelanin/pheomelanin branch
			// below and the "nothing specified" default (pbrt-v4's own
			// SigmaAFromConcentration(1.3, 0.) fallback).
			const auto sigmaAFromConcentration = [&](double ce, double cp) {
				m.sigma_a[0] = ce * 0.419 + cp * 0.187;
				m.sigma_a[1] = ce * 0.697 + cp * 0.4;
				m.sigma_a[2] = ce * 1.37  + cp * 1.05;
			};

			// pbrt-v4 HairBxDF::SigmaAFromReflectance (bxdfs.cpp) - ALSO a
			// closed-form per-channel formula (log of the channel divided by
			// a degree-5 polynomial in beta_n, then squared), not an
			// iterative fit - an earlier version of this comment mischaracterized
			// it as iterative and skipped it for that reason; it belongs
			// beside SigmaAFromConcentration above as an equally cheap,
			// equally closed-form option.
			const auto sigmaAFromReflectanceChannel = [](double c, double bn) {
				c = c < 1e-4 ? 1e-4 : (c > 1.0 - 1e-4 ? 1.0 - 1e-4 : c);
				const double bn2 = bn * bn, bn3 = bn2 * bn, bn4 = bn3 * bn, bn5 = bn4 * bn;
				const double denom = 5.969 - 0.215 * bn + 2.532 * bn2
					- 10.73 * bn3 + 5.574 * bn4 + 0.245 * bn5;
				const double x = std::log(c) / denom;
				return x * x;
			};

			if (hasSigmaA) {
				if (hasReflectance || hasEumelanin || hasPheomelanin)
					warn("material 'hair' gives \"sigma_a\" together with "
						 "\"reflectance\"/\"eumelanin\"/\"pheomelanin\"; "
						 "\"sigma_a\" wins (matches pbrt-v4)");
				const pbrt_scene::Vec3 a = md.params.getVec3("sigma_a",
					pbrt_scene::Vec3{m.sigma_a[0], m.sigma_a[1], m.sigma_a[2]});
				m.sigma_a[0] = a.x; m.sigma_a[1] = a.y; m.sigma_a[2] = a.z;
			} else if (hasReflectance) {
				if (hasEumelanin || hasPheomelanin)
					warn("material 'hair' gives \"reflectance\"/\"color\" together "
						 "with \"eumelanin\"/\"pheomelanin\"; \"reflectance\" wins "
						 "(matches pbrt-v4)");
				const pbrt_scene::Vec3 def{0.5, 0.3, 0.2};
				pbrt_scene::Vec3 c = md.params.getVec3("reflectance", def);
				if (!md.params.find("reflectance")) c = md.params.getVec3("color", def);
				m.sigma_a[0] = sigmaAFromReflectanceChannel(c.x, m.betaN);
				m.sigma_a[1] = sigmaAFromReflectanceChannel(c.y, m.betaN);
				m.sigma_a[2] = sigmaAFromReflectanceChannel(c.z, m.betaN);
			} else if (hasEumelanin || hasPheomelanin) {
				const double ce = std::max(0.0, md.params.getFloat("eumelanin", 0.0));
				const double cp = std::max(0.0, md.params.getFloat("pheomelanin", 0.0));
				sigmaAFromConcentration(ce, cp);
			} else {
				sigmaAFromConcentration(1.3, 0.0);
			}
		}

		// Measured: just the as-written filename (see Material::
		// measuredFilename's own comment for why resolution/loading happens
		// later, in pbrt_load.h). pbrt-v4's own MeasuredMaterial has no
		// other parameters worth reading - reflectance/roughness/eta above
		// don't apply to a tabulated BRDF.
		if (m.kind == MaterialKind::Measured) {
			m.measuredFilename = md.params.getString("filename", "");
			if (m.measuredFilename.empty()) {
				warn("material 'measured' has no \"filename\"; "
					 "it will fall back to a diffuse approximation");
			}
		}

		// Mix: resolve "materials" (two names) against namedMaterialIndex,
		// built above. A "mix" that cannot be resolved is downgraded to
		// Unsupported here rather than left half-populated - the generic
		// diffuse fallback below the main loop already exists and is exactly
		// what an unresolvable mix should do, so reusing it (instead of a
		// separate broken-mix code path in pbrt_cpu_builder.h) means that
		// file never has to check mixMaterialA/B are valid before indexing
		// with them (see those fields' own comment).
		if (m.kind == MaterialKind::Mix) {
			const pbrt_scene::Param *mats = md.params.find("materials");
			if (!mats || mats->strings.size() < 2) {
				warn("material 'mix' needs two names in \"materials\"; "
					 "it will fall back to a diffuse approximation");
				m.kind = MaterialKind::Unsupported;
			} else {
				const auto itA = namedMaterialIndex.find(mats->strings[0]);
				const auto itB = namedMaterialIndex.find(mats->strings[1]);
				if (itA == namedMaterialIndex.end() || itB == namedMaterialIndex.end()) {
					warn("material 'mix' names an unknown material in \"materials\"; "
						 "it will fall back to a diffuse approximation");
					m.kind = MaterialKind::Unsupported;
				} else {
					m.mixMaterialA = itA->second;
					m.mixMaterialB = itB->second;
					// pbrt-v4's own default; "amount" may instead be bound to
					// a texture, which the generic texture-binding warning
					// above (the `for (const pbrt_scene::Param &p : ...)`
					// loop near the top of this material's handling) already
					// reports - this just reads the constant fallback either
					// way, same as every other texture-eligible parameter in
					// this function.
					m.mixWeight = md.params.getFloat("amount", 0.5);
				}
			}
		}

		out.materials.push_back(m);
	}
}

} // namespace flatten_detail
