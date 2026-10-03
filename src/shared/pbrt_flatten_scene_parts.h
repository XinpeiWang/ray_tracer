#pragma once
// pbrt_flatten_scene_parts.h -- flatten()'s media, light and camera phases
//
// Part of pbrt_flatten.h: included by it at the point flatten() is defined, and not usable on
// its own (it needs FlatScene, the flatten_detail helpers and the pbrt_scene types declared
// above that point). Each function is one phase of flatten(), moved here verbatim so no
// single function is thousands of lines long.

namespace flatten_detail {

// MakeNamedMedium -> out.media (1:1).
inline void flattenMedia(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- media -------------------------------------------------------------
	// MakeNamedMedium, mirrored 1:1 into out.media (same convention as
	// materials above) - ShapeDecl::insideMedium already resolved names to
	// indices during parsing (see pbrt_scene.h's MediumInterface dispatch),
	// so this is a straight per-channel copy, not a second name lookup.
	for (const pbrt_scene::MediumDecl &md : scene.media) {
		Medium medium;
		const bool isCloud = (md.type == "cloud");
		// isRgbGrid is checked 3 separate times below - the flat-"Le"-field
		// skip (this loop, ~10 lines down), the cloud/uniformgrid-only
		// "Le is meaningless" warning condition, and that warning's own
		// hardcoded "homogeneous\"/\"rgbgrid\"" supported-types string -
		// each answers a genuinely different question (does this type use
		// the flat field at all / should THIS type's flat value warn if
		// nonzero / what to tell the user is actually supported), so a
		// code-review finding on this round's own rgbgrid-Le change flagged
		// the duplication risk but a single shared boolean would be a false
		// unification: forcing these into one flag would either be wrong
		// for one of the 3 questions or add an indirection with no real
		// payoff. Kept as 3 explicit sites instead, each now cross-
		// referencing the others by name so a future 4th emissive medium
		// type can't update one without being pointed at the rest.
		const bool isRgbGrid = (md.type == "rgbgrid");
		const bool isUniformGrid = (md.type == "uniformgrid");
		const bool isNanoVdb = (md.type == "nanovdb");
		if (!isCloud && !isRgbGrid && !isUniformGrid && !isNanoVdb && md.type != "homogeneous") {
			warn("medium type '" + md.type + "' is not supported; "
				 "treated as homogeneous with its given sigma_a/sigma_s");
		}
		medium.type = (isCloud || isRgbGrid || isUniformGrid || isNanoVdb) ? md.type : "homogeneous";

		// sigma_a/sigma_s/scale/g: for homogeneous, these ARE the medium
		// (a flat RGB colour each). For rgbgrid, these same param NAMES
		// instead mean a flat array of one RGB triple per voxel (see
		// below) - the plain Vec3 read here still runs for that case too
		// (harmless; nothing reads medium.sigma_a/sigma_s for rgbgrid) so
		// this stays one shared block rather than three divergent copies.
		const pbrt_scene::Vec3 defA{medium.sigma_a[0], medium.sigma_a[1], medium.sigma_a[2]};
		const pbrt_scene::Vec3 defS{medium.sigma_s[0], medium.sigma_s[1], medium.sigma_s[2]};
		pbrt_scene::Vec3 sa = md.params.getVec3("sigma_a", defA);
		pbrt_scene::Vec3 ss = md.params.getVec3("sigma_s", defS);
		const double scale = md.params.getFloat("scale", 1.0);
		medium.sigma_a[0] = sa.x * scale; medium.sigma_a[1] = sa.y * scale; medium.sigma_a[2] = sa.z * scale;
		medium.sigma_s[0] = ss.x * scale; medium.sigma_s[1] = ss.y * scale; medium.sigma_s[2] = ss.z * scale;
		medium.g = md.params.getFloat("g", 0.0);

		// "rgb Le"/"float Lescale" - see Medium::Le's own comment on why
		// this round is homogeneous-only, and Medium::Le_r/g/b's own
		// comment for why "rgbgrid" is now ALSO supported, but through a
		// separate, array-shaped field, not this one. This is site 1 of 3
		// where isRgbGrid is checked for this reason (see isRgbGrid's own
		// declaration comment above for the other 2). Skipped entirely for
		// isRgbGrid (handled below instead) - reading it here too would
		// misinterpret that type's own per-voxel "Le" array as a flat
		// colour (see Medium::Le_r's own comment for why). Read
		// unconditionally for the remaining types (harmless for cloud/
		// uniformgrid, matching this loop's own "one shared block"
		// convention for sigma_a/sigma_s above) but only ever
		// nonzero-and-warned for those two, never silently dropped. Uses
		// resolveEmissionColor() (this same file, used by every light's
		// own "L"/"I"), not a plain getVec3 - pbrt-v4 also accepts
		// "blackbody Le" [<kelvin>] (a single number, real Kelvin-to-RGB
		// conversion) for physically-plausible fire/plasma color, which a
		// plain getVec3 (needs >=3 numbers) would silently read as
		// "absent" and default to zero with no diagnostic. Uses the
		// default sRGB working color space (RGBColorSpace::sRGB(),
		// resolveEmissionColor()'s own default) rather than a per-medium
		// ColorSpace capture the way LightDecl::colorSpaceName gives every
		// light - MediumDecl has no equivalent field, and no bundled scene
		// combines a non-default ColorSpace directive with an emissive
		// medium; a real scope cut, not an oversight.
		if (!isRgbGrid) {
			const pbrt_scene::Vec3 le = resolveEmissionColor(md.params, "Le", pbrt_scene::Vec3{0,0,0});
			const double leScale = md.params.getFloat("Lescale", 1.0);
			medium.Le[0] = le.x * leScale; medium.Le[1] = le.y * leScale; medium.Le[2] = le.z * leScale;
		}

		if (isCloud || isRgbGrid || isUniformGrid) {
			// Medium-space bounds (pbrt-v4's own "p0"/"p1", default unit
			// cube - matches CloudMedium::Create/RGBGridMedium::Create's
			// real defaults).
			const pbrt_scene::Vec3 p0 = md.params.getVec3("p0", pbrt_scene::Vec3{0,0,0});
			const pbrt_scene::Vec3 p1 = md.params.getVec3("p1", pbrt_scene::Vec3{1,1,1});
			medium.p0[0]=p0.x; medium.p0[1]=p0.y; medium.p0[2]=p0.z;
			medium.p1[0]=p1.x; medium.p1[1]=p1.y; medium.p1[2]=p1.z;

			// World -> medium-space transform: invert the CTM captured at
			// MakeNamedMedium declaration time (world -> medium is what
			// CloudMedium/rgb_grid_medium_hittable's own sample-time math
			// wants - see each one's own world_to_medium_pt/world_to_medium
			// comment).
			pbrt_scene::Matrix4 worldToMedium;
			if (md.xform.inverseAffine(worldToMedium)) {
				splitAffine(worldToMedium, medium.toMediumMat, medium.toMediumTranslate);
			} else {
				warn("medium '" + md.name + "' has a singular transform "
					 "(zero scale on some axis); it will not render correctly");
			}

			// World-space AABB: the medium-space box [p0,p1] under the
			// medium's own declared CTM (world_from_medium - the INVERSE of
			// the transform just computed above).
			const double p0arr[3] = {p0.x, p0.y, p0.z};
			const double p1arr[3] = {p1.x, p1.y, p1.z};
			aabbOfTransformedBox(md.xform, p0arr, p1arr, medium.worldMin, medium.worldMax);
		}

		// "Le"/"Lescale" (Medium::Le's own comment) - homogeneous only this
		// round; cloud/uniformgrid's real pbrt-v4 emission is a per-voxel
		// grid, not this flat colour, so it's silently meaningless for them
		// - warn rather than let a scene author believe it did something.
		// rgbgrid is deliberately EXCLUDED here now (Medium::Le_r's own
		// comment; this is site 2 of 3, see isRgbGrid's own declaration
		// comment above) - it gets its own real per-voxel "Le" support
		// below, so this check would otherwise fire a misleading "dropped"
		// warning for a scene whose emission this loader now actually
		// honors. The warning message just below is site 3 of 3.
		// Grouped with the sigma_a-dropped warnings just below (same
		// "parameter X is meaningless for type Y" shape), not folded into
		// the AABB-computation block above.
		if ((isCloud || isUniformGrid || isNanoVdb) && isNonzeroRGB(medium.Le)) {
			warn("medium '" + md.name + "' (\"" + md.type + "\") has a nonzero \"Le\", "
				 "but only \"homogeneous\"/\"rgbgrid\" media support emission in this loader; "
				 "the emission is dropped");
		}
		// "nanovdb"'s own real emission mechanism is a SEPARATE named grid
		// ("string temperaturename", real blackbody emission from that
		// grid's values) - not the flat "Le" the check just above already
		// covers. IMPLEMENTED (see Medium::nanovdbTemperatureGridName's own
		// comment) - parsed in the isNanoVdb block below, alongside
		// nanovdbFilename/nanovdbGridName.

		if (isCloud) {
			// pbrt-v4's real per-param defaults (media.cpp's
			// CloudMedium::Create) - see Medium::density's own comment for
			// why these differ from this codebase's own E2 showcase scene.
			medium.density = md.params.getFloat("density", 1.0);
			medium.wispiness = md.params.getFloat("wispiness", 1.0);
			medium.frequency = md.params.getFloat("frequency", 5.0);
			// cloud_medium_hittable.h forces sigma_a to 0 (pure scattering) -
			// same convention constant_medium already uses, see that
			// header's own comment for why - so a scene that gave a real
			// absorption coefficient silently loses it. Worth a warning:
			// unlike most of this loader's approximations, this one can't
			// be inferred from the render (a too-bright cloud looks like a
			// lighting choice, not a dropped parameter).
			if (isNonzeroRGB(medium.sigma_a)) {
				warn("cloud medium '" + md.name + "' has a nonzero sigma_a; "
					 "cloud media only model scattering (sigma_a is forced to 0)");
			}
		}

		if (isRgbGrid || isUniformGrid) {
			medium.nx = md.params.getInt("nx", 1);
			medium.ny = md.params.getInt("ny", 1);
			medium.nz = md.params.getInt("nz", 1);
		}
		const std::size_t voxels = static_cast<std::size_t>(medium.nx)
			* static_cast<std::size_t>(medium.ny) * static_cast<std::size_t>(medium.nz);

		if (isRgbGrid) {
			// "rgb sigma_a"/"rgb sigma_s": a flat array of one RGB triple
			// PER VOXEL (length 3*nx*ny*nz), NOT the single flat colour the
			// generic Vec3 read above assumed - de-interleave into
			// per-channel vectors here, matching RGBGridMediumData::build()'s
			// own (sa_r,sa_g,sa_b,...) parameter shape. Either array is
			// optional (empty vector = "channel group absent" - see
			// RGBGridMediumData::build()'s own comment); a present array
			// whose length doesn't match voxels*3 is dropped with a warning
			// rather than read out of bounds or silently truncated/padded.
			const auto deinterleave = [&](const char *name, std::vector<double> &r,
										   std::vector<double> &g, std::vector<double> &b) {
				const pbrt_scene::Param *p = md.params.find(name);
				if (!p || p->numbers.empty()) return;
				if (p->numbers.size() != voxels * 3) {
					warn("medium '" + md.name + "'s \"" + name + "\" array has "
						 + std::to_string(p->numbers.size()) + " numbers, expected "
						 + std::to_string(voxels * 3) + " (3 * nx*ny*nz); ignored");
					return;
				}
				r.resize(voxels); g.resize(voxels); b.resize(voxels);
				for (std::size_t i = 0; i < voxels; ++i) {
					r[i] = p->numbers[i*3+0]; g[i] = p->numbers[i*3+1]; b[i] = p->numbers[i*3+2];
				}
			};
			deinterleave("sigma_a", medium.sigma_a_r, medium.sigma_a_g, medium.sigma_a_b);
			deinterleave("sigma_s", medium.sigma_s_r, medium.sigma_s_g, medium.sigma_s_b);

			// "rgb Le"/"float Lescale" - real per-voxel emission, pbrt-v4's
			// own RGBGridMedium::LeGrid/LeScale - see Medium::Le_r's own
			// comment for why this is parsed separately from the
			// homogeneous-only flat "Le" above. Lescale is NOT baked into
			// the array here (unlike homogeneous's own medium.Le, which
			// bakes its scale in at parse time) - RGBGridMediumData<T>::
			// sample_point() applies Le_scale at sample time itself
			// (`Le_out[c] = Le_scale * le;`), matching how sigma_scale
			// (this same struct's "scale" param, applied inside
			// RGBGridMediumData::build() too) already works for sigma_a/
			// sigma_s. Defaults to 0.0 (matches RGBGridMediumData::
			// Le_scale's own default and pbrt-v4's real "Lescale" default
			// of 1 - but see is_emissive()'s own `Le_scale > 0` gate: a
			// scene that sets a real "Le" array but never touches
			// "Lescale" at all should still glow, so the DEFAULT here is
			// 1.0 when omitted, matching pbrt-v4's spec exactly; only an
			// explicit "Lescale" of 0 (or omitting "Le" entirely, leaving
			// Le_r/g/b empty) turns emission off).
			// "blackbody Le" (a single Kelvin temperature) is a
			// homogeneous-only convenience - rgbgrid's own real "Le" is
			// always a per-voxel RGB array (one triple per voxel), which a
			// single temperature can't express. Checked explicitly (rather
			// than falling through to deinterleave()'s own generic
			// wrong-length warning below) since a code-review pass found
			// that generic message actively misleading here - it reads
			// "array has 1 numbers, expected N" which implies "supply more
			// numbers", not "this syntax isn't supported for rgbgrid".
			const pbrt_scene::Param *leParam = md.params.find("Le");
			if (leParam && leParam->type == "blackbody") {
				warn("medium '" + md.name + "' (\"rgbgrid\") has a \"blackbody Le\", "
					 "but rgbgrid media only support a per-voxel \"rgb Le\" array, not "
					 "a single blackbody temperature; the emission is dropped");
			} else {
				deinterleave("Le", medium.Le_r, medium.Le_g, medium.Le_b);
			}
			medium.Le_scale = md.params.getFloat("Lescale", 1.0);
			// "Lescale" set but no (valid) "Le" array ended up populated -
			// silently has no effect otherwise (RGBGridMediumData::
			// is_emissive() correctly stays false), which a code-review
			// pass found gives no hint to a scene author who forgot/
			// mistyped "Le" - warn, matching homogeneous Le's own "nonzero
			// param with no effect" diagnostic precedent above.
			if (md.params.find("Lescale") &&
				medium.Le_r.empty() && medium.Le_g.empty() && medium.Le_b.empty()) {
				warn("medium '" + md.name + "' (\"rgbgrid\") sets \"Lescale\" but has no "
					 "valid \"rgb Le\" array; \"Lescale\" has no effect without one");
			}
		}

		if (isUniformGrid) {
			// "float density": pbrt-v4's own REQUIRED flat scalar array,
			// exactly one value per voxel (length nx*ny*nz) - not a colour,
			// not the "cloud"-only scalar `density` field above. sigma_a/
			// sigma_s (the generic Vec3 read at the top of this loop) supply
			// the base coefficients each voxel's density multiplies (see
			// GridMediumData<T>::sample_sigma()). A missing or wrong-length
			// array is dropped with a warning (leaves gridDensity empty),
			// matching rgbgrid's own "wrong length -> warn and ignore rather
			// than read out of bounds" convention.
			const pbrt_scene::Param *p = md.params.find("density");
			if (!p || p->numbers.empty()) {
				warn("uniformgrid medium '" + md.name + "' has no \"float density\" "
					 "array (pbrt-v4 requires one); treated as empty (invisible)");
			} else if (p->numbers.size() != voxels) {
				warn("medium '" + md.name + "'s \"density\" array has "
					 + std::to_string(p->numbers.size()) + " numbers, expected "
					 + std::to_string(voxels) + " (nx*ny*nz); ignored");
			} else {
				medium.gridDensity = p->numbers;
			}
			// grid_medium_hittable.h forces sigma_a to 0 (pure scattering) -
			// same convention/reason as cloud above (see that block's own
			// comment) - so a scene that gave a real absorption coefficient
			// silently loses it.
			if (isNonzeroRGB(medium.sigma_a)) {
				warn("uniformgrid medium '" + md.name + "' has a nonzero sigma_a; "
					 "uniformgrid media only model scattering (sigma_a is forced to 0)");
			}
		}

		if (isNanoVdb) {
			// "string filename" - REQUIRED (Medium::nanovdbFilename's own
			// comment); left as written here, resolved to a real filesystem
			// path by pbrt_load.h afterward. An empty/missing filename is
			// caught there (same "empty means unresolved, warn and skip"
			// convention as every other filename field), not here - this
			// loader has no filesystem access to confirm the file even
			// exists yet at flatten() time.
			medium.nanovdbFilename = md.params.getString("filename", "");
			if (medium.nanovdbFilename.empty()) {
				warn("nanovdb medium '" + md.name + "' has no \"string filename\" "
					 "(pbrt-v4 requires one); treated as empty (invisible)");
			}
			medium.nanovdbGridName = md.params.getString("gridname", "density");
			// See Medium::nanovdbTemperatureGridName's own comment.
			medium.nanovdbTemperatureGridName = md.params.getString("temperaturename", "");
			medium.nanovdbLeScale = md.params.getFloat("Lescale", 1.0);
			// See Medium::nanovdbXform's own comment for why this is passed
			// through unbaked instead of pre-composing world<->medium here
			// the way every other medium type's p0/p1 block above does.
			for (int i = 0; i < 16; ++i) medium.nanovdbXform[i] = md.xform.m[i];
			// pbrt_cpu_builder.h's nanovdb path forces sigma_a to 0 (pure
			// scattering) UNLESS a "temperaturename" grid is present (see
			// Medium::nanovdbTemperatureGridName's own comment - real
			// blackbody emission needs a real, nonzero sigma_a to be
			// anything other than a physical no-op) - same convention/
			// reason as uniformgrid just above for the plain-scattering
			// case.
			if (isNonzeroRGB(medium.sigma_a) && medium.nanovdbTemperatureGridName.empty()) {
				warn("nanovdb medium '" + md.name + "' has a nonzero sigma_a; "
					 "this loader's nanovdb media only model scattering (sigma_a is "
					 "forced to 0) unless a \"temperaturename\" grid is also given");
			}
		}

		out.media.push_back(medium);
	}
}

// LightSource directives: the infinite light and the punctual lights.
inline void flattenLights(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- lights that are not area lights ---------------------------------
	// "infinite" (the environment/sky light) is carried through below - it is
	// usually a scene's main illumination, so dropping it silently produces a
	// render that looks exactly like a shading bug and sends you hunting
	// through the BSDF code instead. point/spot/distant/goniometric/
	// projection are carried through too, into FlatScene::punctualLights -
	// see PunctualLight's own comment for why this is a bridging job rather
	// than new rendering math (this codebase's own C2-C6 showcase scenes
	// already exercise every one of these five on both backends). Anything
	// else genuinely unknown still gets the warn-and-drop this whole block
	// used to give every non-infinite kind.
	//
	// Only the LAST "infinite" LightSource in the scene wins if there is more
	// than one - real pbrt scenes have at most one (this codebase has never
	// seen otherwise among the scenes it loads), and picking the last one
	// matches how pbrt's own graphics-state model would apply them (each
	// later directive's effect is visible, not merged). Punctual lights have
	// no such restriction - a scene can genuinely want three spotlights - so
	// every one of those is kept, not just the last.
	for (const pbrt_scene::LightDecl &ld : scene.lights) {
		if (ld.type == "infinite") {
			out.infiniteLight.present = true;
			const pbrt_scene::Vec3 L = resolveEmissionColor(ld.params, "L", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
			out.infiniteLight.L[0] = L.x; out.infiniteLight.L[1] = L.y; out.infiniteLight.L[2] = L.z;
			out.infiniteLight.scale = ld.params.getFloat("scale", 1.0);
			out.infiniteLight.imageFile = ld.params.getString("filename", "");
			out.infiniteLight.xform = ld.xform;
			// "point3 portal[4]" (pbrt-v4's windowed infinite light) - 4
			// corner points, 12 numbers total. Transformed into world/
			// render space here (matching every other point-valued light
			// param in this loop) since PortalImageInfiniteLightData itself
			// applies no further transform - see InfiniteLight::portal's
			// own comment. Reset unconditionally first, like every other field above -
			out.infiniteLight.hasPortal = false;
			if (const pbrt_scene::Param *portal = ld.params.find("portal")) {
				if (portal->numbers.size() >= 12) {
					out.infiniteLight.hasPortal = true;
					for (int i = 0; i < 4; ++i) {
						transformPoint(ld.xform,
							portal->numbers[i*3+0], portal->numbers[i*3+1], portal->numbers[i*3+2],
							&out.infiniteLight.portal[i*3]);
					}
				}
			}
			continue;
		}

		if (ld.type == "point") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Point;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			transformPoint(ld.xform, from.x, from.y, from.z, pl.pos);
			const pbrt_scene::Vec3 I = resolveEmissionColor(ld.params, "I", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			// pbrt-v4's "power" (total emitted Phi, watts): an isotropic point
			// source radiates over the full 4*pi sphere, so I = Phi/(4*pi) -
			// matching PointLight::Create exactly, up to the RGB-luminance
			// stand-in relativeLuminance() uses for its spectral normalization
			// (see that function's own comment). Multiplies into `scale`
			// rather than replacing it, so a scene giving both "power" and
			// "scale" gets both, the same composition pbrt-v4 itself performs.
			// Guarded on phi > 0, matching pbrt-v4's own PointLight::Create
			// (`if (phi_v > 0)`) - a zero or negative "power" is treated as
			// "not given" rather than literally zeroing or negating the
			// light, which a bare multiply would otherwise do.
			if (const pbrt_scene::Param *p = ld.params.find("power")) {
				if (!p->numbers.empty() && p->numbers[0] > 0.0) {
					const double phi = p->numbers[0];
					pl.scale *= (phi / (4.0 * 3.14159265358979323846)) / relativeLuminance(pl.intensity);
				}
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "spot") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Spot;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			const pbrt_scene::Vec3 to = ld.params.getVec3("to", pbrt_scene::Vec3{0, 0, 1});
			double worldFrom[3], worldTo[3];
			transformPoint(ld.xform, from.x, from.y, from.z, worldFrom);
			transformPoint(ld.xform, to.x, to.y, to.z, worldTo);
			for (int c = 0; c < 3; ++c) pl.pos[c] = worldFrom[c];
			// The cone axis - the direction the spot is AIMED, from `from`
			// toward `to` - both already CTM-transformed, so subtracting
			// cancels the translation and leaves exactly the CTM's rotation
			// applied to pbrt's own local (to - from) (matches pbrt-v4
			// SpotLight::Create's w = Normalize(to - from) exactly, just
			// composed with the CTM by transforming the endpoints instead of
			// building pbrt's own dirToZ rotation matrix - unnecessary here
			// since SpotLightData only ever needs the resulting axis, not a
			// full light-space frame).
			for (int c = 0; c < 3; ++c) pl.dir[c] = worldTo[c] - worldFrom[c];
			normalizeOrDefault(pl.dir, 0, -1, 0);
			const pbrt_scene::Vec3 I = resolveEmissionColor(ld.params, "I", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			pl.coneAngleDeg = ld.params.getFloat("coneangle", 30.0);
			// pbrt-v4's own SpotLight::Create passes coneangle - conedeltaangle
			// through UNCLAMPED (see SpotLight's cosFalloffStart/cosFalloffEnd
			// construction) - a conedeltaangle larger than coneangle yields a
			// negative falloffStartAngleDeg, which is not malformed: cos() is
			// even, so cos(-10deg) == cos(10deg), meaning a negative start angle
			// still produces a real, meaningful full-intensity core (just one
			// that extends slightly past the geometric cone axis) rather than
			// collapsing to cosFalloffStart=1.0 (no core at all), which is what
			// clamping to 0 here used to produce.
			const double delta = ld.params.getFloat("conedeltaangle", 5.0);
			pl.falloffStartAngleDeg = pl.coneAngleDeg - delta;
			// pbrt-v4's "power" (total emitted Phi, watts): SpotLight::Create's
			// own closed-form integral of the cone's falloff region, treating
			// it as contributing half its solid angle on average -
			// Phi = 2*pi*I*((1-cosFalloffStart) + (cosFalloffStart-cosFalloffEnd)/2)
			// - inverted for I here exactly like the point-light case just
			// above (same relativeLuminance() stand-in, same "multiplies into
			// scale" composition with an explicit "scale" param, same phi > 0
			// guard matching pbrt-v4's own SpotLight::Create).
			if (const pbrt_scene::Param *p = ld.params.find("power")) {
				if (!p->numbers.empty() && p->numbers[0] > 0.0) {
					const double phi = p->numbers[0];
					const double cosFalloffStart = std::cos(pl.falloffStartAngleDeg * 3.14159265358979323846 / 180.0);
					const double cosFalloffEnd = std::cos(pl.coneAngleDeg * 3.14159265358979323846 / 180.0);
					const double kE = 2.0 * 3.14159265358979323846 *
						((1.0 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) * 0.5);
					if (std::fabs(kE) > 1e-12) {
						pl.scale *= (phi / kE) / relativeLuminance(pl.intensity);
					} else {
						// A degenerate cone (e.g. "coneangle" 0) has no solid
						// angle to spread Phi over - same class of problem as
						// the area-light zero-area case below, so it gets the
						// same treatment: warn and leave "power" unconverted
						// rather than silently dropping it with no diagnostic.
						warn("spot light \"power\" given but the cone is degenerate "
							 "(coneangle/conedeltaangle leave no solid angle to spread "
							 "it over); \"scale\"/I used as given instead");
					}
				}
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "distant") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Distant;
			const pbrt_scene::Vec3 from = ld.params.getVec3("from", pbrt_scene::Vec3{0, 0, 0});
			const pbrt_scene::Vec3 to = ld.params.getVec3("to", pbrt_scene::Vec3{0, 0, 1});
			double worldFrom[3], worldTo[3];
			transformPoint(ld.xform, from.x, from.y, from.z, worldFrom);
			transformPoint(ld.xform, to.x, to.y, to.z, worldTo);
			// dir must be `wi` - the direction FROM ANY SHADING POINT TOWARD
			// THE LIGHT (see PunctualLight::dir's own comment) - i.e. the
			// OPPOSITE of the direction sunlight travels, which runs from
			// `from` toward `to`. pbrt-v4 DistantLight::Create builds this
			// same vector (w = Normalize(from - to)) for exactly this reason
			// - its own SampleLi later returns renderFromLight(local +Z),
			// and its local +Z axis is constructed to already equal that
			// (from - to) direction, so this is the same result reached
			// without needing pbrt's intermediate rotation-to-Z matrix.
			for (int c = 0; c < 3; ++c) pl.dir[c] = worldFrom[c] - worldTo[c];
			normalizeOrDefault(pl.dir, 0, 1, 0);
			const pbrt_scene::Vec3 L = resolveEmissionColor(ld.params, "L", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
			pl.intensity[0] = L.x; pl.intensity[1] = L.y; pl.intensity[2] = L.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			// pbrt-v4 itself has no "power" for a distant light - unlike every
			// other punctual kind, its irradiance doesn't fall off with
			// distance, so there is no finite total Phi to solve for without
			// an arbitrary reference area (real pbrt-v4 instead reads a
			// separate "illuminance" parameter, in lux, which this loader
			// does not parse either - out of scope here). Warn rather than
			// silently ignoring a "power" a scene author expected to matter.
			if (ld.params.find("power")) {
				warn("distant light \"power\" is not supported (pbrt-v4 has no "
					 "finite total power for a directional light); ignored");
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "goniometric") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Goniometric;
			transformPoint(ld.xform, 0.0, 0.0, 0.0, pl.pos);
			worldToLightRotation(ld.xform, pl.worldToLight);
			const pbrt_scene::Vec3 I = resolveEmissionColor(ld.params, "I", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
			pl.intensity[0] = I.x; pl.intensity[1] = I.y; pl.intensity[2] = I.z;
			pl.scale = ld.params.getFloat("scale", 1.0);
			const std::string file = ld.params.getString("filename", "");
			if (!file.empty()) {
				pl.hadImageFilename = true;
				pl.filename = file;
			}
			// pbrt-v4's real "power" for a goniometric light weighs the
			// profile image's own luminance distribution into the conversion
			// (GoniometricLight::Create) - out of scope here (see this
			// loader's matching scope decision for ReverseOrientation and
			// other narrowly-closed gaps this session); warn rather than
			// silently ignore it.
			if (ld.params.find("power")) {
				warn("goniometric light \"power\" is not supported (its real "
					 "pbrt-v4 conversion depends on the profile image); ignored");
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		if (ld.type == "projection") {
			PunctualLight pl;
			pl.kind = PunctualLightKind::Projection;
			transformPoint(ld.xform, 0.0, 0.0, 0.0, pl.pos);
			worldToLightRotation(ld.xform, pl.worldToLight);
			pl.scale = ld.params.getFloat("scale", 1.0);
			pl.fovDeg = ld.params.getFloat("fov", 90.0);
			const std::string file = ld.params.getString("filename", "");
			// pbrt-v4 requires "filename" (a projection light has no other
			// way to describe what it projects) - a scene that omits it is
			// itself malformed, worth a warning even though the light still
			// renders (as a uniform white beam) rather than being dropped.
			if (!file.empty()) {
				pl.hadImageFilename = true;
				pl.filename = file;
			} else {
				warn("light 'projection' has no \"filename\" (pbrt-v4 requires "
					 "one); emitted as a uniform white beam of the requested "
					 "shape/scale/aim instead of failing outright");
			}
			// Same scope decision as goniometric just above: real pbrt-v4
			// weighs the projected slide image's own luminance into the
			// conversion (ProjectionLight::Create) - warn instead of ignoring.
			if (ld.params.find("power")) {
				warn("projection light \"power\" is not supported (its real "
					 "pbrt-v4 conversion depends on the projected image); ignored");
			}
			out.punctualLights.push_back(pl);
			continue;
		}

		warn("light source '" + ld.type + "' is not supported and was dropped; "
			 "the scene will be darker than intended");
	}
}

// AreaLightSource directives -> out.areaLights.
inline void flattenAreaLights(const pbrt_scene::Scene &scene, FlatScene &out) {
	// ---- area lights -----------------------------------------------------
	for (const pbrt_scene::LightDecl &ld : scene.areaLights) {
		Emission e;
		// "blackbody L" declares a colour TEMPERATURE under the parameter
		// name L (resolveEmissionColor's own comment) - barcelona-pavilion's/
		// contemporary-bathroom's real night-lighting area lights use this.
		const pbrt_scene::Vec3 L = resolveEmissionColor(ld.params, "L", pbrt_scene::Vec3{1, 1, 1}, RGBColorSpaceFromName(ld.colorSpaceName));
		e.L[0] = L.x; e.L[1] = L.y; e.L[2] = L.z;
		e.scale = ld.params.getFloat("scale", 1.0);
		// Round 6 Phase 4: spatially-varying emission (an image mapped onto
		// the shape) and real two-sided emission - see Emission's own
		// comment on each field.
		e.filename = ld.params.getString("filename", "");
		e.twoSided = ld.params.getBool("twosided", false);
		if (const pbrt_scene::Param *p = ld.params.find("power")) {
			if (!p->numbers.empty()) {
				e.hasPower = true;
				e.power = p->numbers[0];
			}
		}
		out.areaLights.push_back(e);
	}
}

// Camera, film and screen window.
inline void flattenCamera(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- camera ----------------------------------------------------------
	out.camera = cameraFromWorldToCamera(scene.worldToCamera, scene.cameraLookAtDistance);
	{
		const double fov = scene.cameraFov();
		// pbrt's fov applies to the NARROWER image axis. Ours is always
		// vertical, so on a landscape frame they agree and on a portrait one
		// they do not - taking pbrt's number as vertical unconditionally
		// silently mis-frames every portrait scene.
		if (scene.xResolution >= scene.yResolution) {
			out.camera.vfov = fov;
		} else {
			const double aspect = (scene.yResolution > 0)
								  ? static_cast<double>(scene.xResolution) / scene.yResolution
								  : 1.0;
			const double halfRad = fov * 0.5 * 3.14159265358979323846 / 180.0;
			const double tanV = (aspect > 0.0) ? std::tan(halfRad) / aspect : std::tan(halfRad);
			out.camera.vfov = 2.0 * std::atan(tanV) * 180.0 / 3.14159265358979323846;
		}
		out.camera.aperture = scene.cameraParams.getFloat("lensradius", 0.0) * 2.0;
		// pbrt-v4 spells this "focaldistance" for perspective/orthographic but
		// "focusdistance" for realistic - reading both means a scene does not
		// silently keep the 1e6 sentinel just because it used the other
		// camera type's spelling.
		out.camera.focusDistance = scene.cameraParams.getFloat("focaldistance",
									   scene.cameraParams.getFloat("focusdistance",
										   out.camera.focusDistance));

		// Non-perspective cameras. lookfrom/lookat/up above already came from
		// the world-to-camera matrix, which every pbrt camera type shares -
		// only the type-specific parameters need reading here.
		out.camera.type = scene.cameraType;
		if (const pbrt_scene::Param *sw = scene.cameraParams.find("screenwindow")) {
			if (sw->numbers.size() >= 4) {
				out.camera.hasScreenWindow = true;
				out.camera.screenWindow[0] = sw->numbers[0];
				out.camera.screenWindow[1] = sw->numbers[1];
				out.camera.screenWindow[2] = sw->numbers[2];
				out.camera.screenWindow[3] = sw->numbers[3];
			} else {
				warn("a camera's \"screenwindow\" needs 4 numbers (xmin xmax ymin "
					 "ymax); ignored");
			}
		}
		if (out.camera.type == "spherical" || out.camera.type == "environment") {
			out.camera.sphericalMapping =
				scene.cameraParams.getString("mapping", out.camera.sphericalMapping);
		} else if (out.camera.type == "realistic") {
			out.camera.lensFile = scene.cameraParams.getString("lensfile", "");
			out.camera.apertureDiameterMM =
				scene.cameraParams.getFloat("aperturediameter", out.camera.apertureDiameterMM);
			out.camera.filmDiagonalMM =
				scene.cameraParams.getFloat("filmdiag", out.camera.filmDiagonalMM);
			if (out.camera.lensFile.empty()) {
				warn("a realistic camera has no \"lensfile\"; rendering with a "
					 "perspective camera instead");
				out.camera.type = "perspective";
			}
		}

		// Camera motion blur (pbrt-v4's real ActiveTransform "StartTime"/
		// "EndTime" idiom - see Camera::isAnimated's own comment). shutteropen/
		// shutterclose are read unconditionally (harmless when the camera
		// turns out not to be animated - CPU's own camera.h never consults
		// them unless camera_is_animated is set).
		out.camera.shutterOpen = scene.cameraParams.getFloat("shutteropen", out.camera.shutterOpen);
		out.camera.shutterClose = scene.cameraParams.getFloat("shutterclose", out.camera.shutterClose);
		out.camera.isAnimated = scene.cameraIsAnimated();
		if (out.camera.isAnimated) {
			const Camera endCam = cameraFromWorldToCamera(scene.worldToCameraEnd, scene.cameraLookAtDistanceEnd);
			for (int i = 0; i < 3; ++i) {
				out.camera.lookfrom1[i] = endCam.lookfrom[i];
				out.camera.lookat1[i] = endCam.lookat[i];
			}
			// TransformTimes's own distinct value has no effect (see
			// Camera::shutterOpen's own comment on why) - loud, not silent,
			// when a scene actually asked for one, so a scene author relying
			// on TransformTimes differing from shutteropen/shutterclose (an
			// unusual, but real pbrt-v4 combination) finds out rather than
			// silently getting shutteropen/shutterclose's own timing instead.
			if (scene.transformTimeStart != out.camera.shutterOpen ||
				scene.transformTimeEnd != out.camera.shutterClose) {
				warn("scene declares TransformTimes [" + std::to_string(scene.transformTimeStart) +
					 ", " + std::to_string(scene.transformTimeEnd) + "] distinct from the camera's own "
					 "shutteropen/shutterclose [" + std::to_string(out.camera.shutterOpen) + ", " +
					 std::to_string(out.camera.shutterClose) + "] - this renderer's camera motion blur "
					 "uses shutteropen/shutterclose for both the keyframe times and the shutter "
					 "sampling window, so TransformTimes's own distinct value has no effect");
			}
		}
	}
}

} // namespace flatten_detail
