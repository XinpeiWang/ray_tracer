#pragma once
// pbrt_flatten_shapes.h -- flatten()'s shape/instance phase and the resolution steps that follow it
//
// Part of pbrt_flatten.h: included by it at the point flatten() is defined, and not usable on
// its own (it needs FlatScene, the flatten_detail helpers and the pbrt_scene types declared
// above that point). Each function is one phase of flatten(), moved here verbatim so no
// single function is thousands of lines long.

namespace flatten_detail {

// Shape "sphere".
inline bool flattenSphereShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	const double r = shape.params.getFloat("radius", 1.0);
	Sphere s;
	transformPoint(xform, 0.0, 0.0, 0.0, s.center);

	// Partial-sphere clipping (pbrt-v4's real zmin/zmax/phimax) -
	// see Sphere's own field comments for why this needs a second,
	// object-space representation once any of these differ from a
	// full sphere. Defaults (-r/+r/360) exactly reproduce a full
	// sphere, so an unclipped scene never sets `clipped`. Computed
	// before the non-uniform-scale check below, since that warning's
	// wording depends on whether this sphere took the clipped path.
	//
	// zmin/zmax are reordered and both clamped to [-r,r], and phimax
	// clamped to [0,360], matching real pbrt-v4's own tolerant
	// Sphere::Create (Clamp(min(zmin,zmax),-r,r)/Clamp(max(...))/
	// Clamp(phimax,0,360)) - a scene with e.g. zmin/zmax swapped (a
	// plausible authoring or converter typo) is valid, rendering
	// input to real pbrt-v4, not something this loader should turn
	// into a silently-invisible sphere by feeding the clip test an
	// empty [zMin,zMax] range.
	const double zMinRaw = shape.params.getFloat("zmin", -r);
	const double zMaxRaw = shape.params.getFloat("zmax", r);
	const double zMinP = std::clamp(std::fmin(zMinRaw, zMaxRaw), -r, r);
	const double zMaxP = std::clamp(std::fmax(zMinRaw, zMaxRaw), -r, r);
	const double phiMaxP = std::clamp(shape.params.getFloat("phimax", 360.0), 0.0, 360.0);
	const double epsR = 1e-9 * std::fmax(1.0, r);
	if (std::fabs(zMinP - (-r)) > epsR || std::fabs(zMaxP - r) > epsR ||
		std::fabs(phiMaxP - 360.0) > 1e-9) {
		s.clipped = true;
		s.radiusLocal = r;
		s.zMin = zMinP;
		s.zMax = zMaxP;
		s.phiMaxDeg = phiMaxP;
		fromMatrix4(xform, s.xform);
	}

	double sc[3];
	axisScales(xform, sc);
	const double lo = std::fmin(sc[0], std::fmin(sc[1], sc[2]));
	const double hi = std::fmax(sc[0], std::fmax(sc[1], sc[2]));
	if (hi - lo > 1e-6 * std::fmax(1.0, hi)) {
		// CPU's clipped path (sphere_clipped_hittable.h) and GPU's
		// ClippedSphere path (pbrt_gpu_builder.h - carries the real
		// per-sphere o2w/w2o affine, not this baked center/radius)
		// both handle non-uniform scale EXACTLY for a clipped
		// sphere, so no warning is needed there. "Would be an
		// ellipsoid; emitted with its largest radius instead" only
		// applies to an UNCLIPPED sphere, where both backends still
		// bake down to this single center/s.radius value.
		if (!s.clipped) {
			warn("a sphere carries a non-uniform scale and would be an "
				 "ellipsoid; emitted with its largest radius instead");
		}
	}
	s.radius = r * hi;

	// Object motion blur (pbrt-v4's ActiveTransform "StartTime"/
	// "EndTime" around a Shape "sphere") - bake a second, end-time
	// centre the same way s.center was baked above, gated on real
	// inequality via Matrix4::differsFrom() (not just "ActiveTransform
	// appeared somewhere") - the same shared helper Scene::
	// cameraIsAnimated() uses, so the "real inequality" convention
	// can't drift between the two independently. Left at its default
	// (== s.center, i.e. no motion) for a
	// clipped sphere - that path carries its own unbaked xform and
	// was never wired for motion blur (see the ActiveTransform
	// directive's own comment). Computed regardless of s.clipped
	// (not just inside the !s.clipped branch) so the clipped case
	// can warn - matches this same loop's other warn()s for a
	// parseable-but-unsupported combination (non-uniform scale just
	// above, MediumInterface just below).
	const bool animated = xform.differsFrom(w.xformEnd);
	if (!s.clipped) {
		if (animated) {
			transformPoint(w.xformEnd, 0.0, 0.0, 0.0, s.center1);
		} else {
			s.center1[0] = s.center[0];
			s.center1[1] = s.center[1];
			s.center1[2] = s.center[2];
		}
	} else if (animated) {
		warn("a clipped sphere (zmin/zmax/phimax) has an ActiveTransform "
			 "StartTime/EndTime pair, but its own unbaked object-to-world "
			 "affine was never wired for motion blur - rendered as static, "
			 "at its StartTime position");
	}

	s.material = shape.materialIndex;
	s.areaLight = shape.areaLightIndex;
	// `medium` is populated unconditionally, regardless of clipping -
	// an unclipped sphere has no problem bounding a medium on either
	// backend, so both need the real index for that case.
	// cpuMediumUnsupported/the warn() below cover the clipped case:
	// an open shell (zmin/zmax/phimax cut it) can't bound a
	// participating medium correctly, on EITHER backend now that
	// GPU also renders a clipped sphere as a real open shell
	// (pbrt_gpu_builder.h's ClippedSphere branch - it resolves this
	// sphere's material via materialIndex(), never
	// mediumMaterialIndex(), for exactly this reason) rather than
	// the old always-full-closed-sphere approximation that used to
	// let GPU alone get away with keeping it.
	s.medium = shape.insideMedium;
	if (s.clipped && s.medium >= 0) {
		s.cpuMediumUnsupported = true;
		warn("a clipped sphere (zmin/zmax/phimax) has a MediumInterface, "
			 "but its open-shell boundary can't bound a participating "
			 "medium correctly - both CPU and GPU drop it, rendering "
			 "the clipped shell with its regular surface material "
			 "instead");
	}
	w.spheres->push_back(s);
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "disk".
inline bool flattenDiskShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	Disk d;
	d.height = shape.params.getFloat("height", 0.0);
	d.radius = shape.params.getFloat("radius", 1.0);
	d.innerRadius = shape.params.getFloat("innerradius", 0.0);
	d.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
	fromMatrix4(xform, d.xform);
	// Object motion blur - see Disk::xformEnd's own comment. Same
	// real-inequality gate as Sphere::center1's bake site.
	fromMatrix4(xform.differsFrom(w.xformEnd) ? w.xformEnd : xform, d.xformEnd);
	d.material = shape.materialIndex;
	d.areaLight = shape.areaLightIndex;
	d.medium = shape.insideMedium;
	w.disks->push_back(d);
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "cylinder".
inline bool flattenCylinderShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	Cylinder c;
	c.radius = shape.params.getFloat("radius", 1.0);
	c.zMin = shape.params.getFloat("zmin", -1.0);
	c.zMax = shape.params.getFloat("zmax", 1.0);
	c.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
	fromMatrix4(xform, c.xform);
	// See Disk's own ActiveTransform comment just above - identical.
	fromMatrix4(xform.differsFrom(w.xformEnd) ? w.xformEnd : xform, c.xformEnd);
	c.material = shape.materialIndex;
	c.areaLight = shape.areaLightIndex;
	c.medium = shape.insideMedium;
	w.cylinders->push_back(c);
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "cone".
inline bool flattenConeShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	Cone cn;
	cn.radius = shape.params.getFloat("radius", 1.0);
	cn.height = shape.params.getFloat("height", 1.0);
	cn.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
	fromMatrix4(xform, cn.xform);
	cn.material = shape.materialIndex;
	cn.areaLight = shape.areaLightIndex;
	cn.medium = shape.insideMedium;
	w.cones->push_back(cn);
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "paraboloid".
inline bool flattenParaboloidShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	Paraboloid pb;
	pb.radius = shape.params.getFloat("radius", 1.0);
	pb.zMin = shape.params.getFloat("zmin", 0.0);
	pb.zMax = shape.params.getFloat("zmax", 1.0);
	pb.phiMaxDeg = shape.params.getFloat("phimax", 360.0);
	fromMatrix4(xform, pb.xform);
	pb.material = shape.materialIndex;
	pb.areaLight = shape.areaLightIndex;
	pb.medium = shape.insideMedium;
	w.paraboloids->push_back(pb);
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "bilinearmesh".
inline bool flattenBilinearMeshShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	// Same medium-drop gap as trianglemesh/plymesh/loopsubdiv above -
	// BilinearPatch has no `medium` field either.
	if (shape.insideMedium >= 0) {
		warn("shape '" + shape.type + "' has a MediumInterface, but "
			 "this loader only supports an attached medium on an "
			 "unclipped sphere/disk/cylinder shape - the medium will "
			 "be dropped");
	}
	// Single-patch form only (see BilinearPatch's own comment) - a
	// bare "point3 P" with exactly 4 points, no "integer indices".
	// Anything else (the multi-patch form, or an instanced/object-
	// space one - see the null-bilinearPatches comment above) falls
	// through to the generic "shape not supported" warning below,
	// same as every other shape kind this file does not build.
	const pbrt_scene::Param *P = shape.params.find("P");
	// "integer indices" with exactly four entries is still one patch, its corners p00 p10 p01 p11 picked from the four points in that order (found by
	// scripts/consistency_sweep.py: a floor written that way was read in the order the points were listed, a twisted patch). More indices mean
	// several patches, which fall through to the warning below.
	const pbrt_scene::Param *indices = shape.params.find("indices");
	int corner[4] = {0, 1, 2, 3};
	bool singlePatch = true;
	if (indices) {
		singlePatch = indices->numbers.size() == 4;
		for (std::size_t i = 0; singlePatch && i < 4; ++i) {
			const double v = indices->numbers[i];
			singlePatch = v >= 0.0 && v < 4.0 && v == static_cast<int>(v);
			if (singlePatch) corner[i] = static_cast<int>(v);
		}
	}
	if (w.bilinearPatches && P && P->numbers.size() == 12 && singlePatch) {
		// OBJECT-space corners first (needed either way: the world
		// bake below always runs, and the animated case just below
		// needs these unbaked) - the ReverseOrientation swap is a
		// pure relabelling of which corner is "p10" vs "p01",
		// transform-independent, so applying it here (before the
		// bake) gives the identical result as applying it after.
		double objP[4][3];
		for (int i = 0; i < 4; ++i) {
			objP[i][0] = P->numbers[corner[i] * 3 + 0];
			objP[i][1] = P->numbers[corner[i] * 3 + 1];
			objP[i][2] = P->numbers[corner[i] * 3 + 2];
		}
		// pbrt-v4 ReverseOrientation, same reverseOrientation XOR
		// transformSwapsHandedness rule as the trianglemesh branch
		// below (see its own comment) - a bilinear patch's
		// dpdu(u,v)/dpdv(u,v) each depend on only ONE of its two
		// parametric axes (pbrt-v4's own p00/p10/p01/p11 corner
		// convention, matching this codebase's bp.p[0..3] layout),
		// so swapping corners p10/p01 (indices 1/2) transposes the
		// two axes and negates cross(dpdu,dpdv) - normal.length() -
		// everywhere on the patch, exactly mirroring the triangle
		// fix's own "swap two of the shape's stored points" trick.
		// No authored-normal or per-vertex-UV data to separately
		// negate/keep in sync here, unlike trianglemesh - a bilinear
		// patch carries neither.
		if (shape.reverseOrientation ^ matrixSwapsHandedness(xform)) {
			double tmp[3] = {objP[1][0], objP[1][1], objP[1][2]};
			objP[1][0] = objP[2][0]; objP[1][1] = objP[2][1]; objP[1][2] = objP[2][2];
			objP[2][0] = tmp[0]; objP[2][1] = tmp[1]; objP[2][2] = tmp[2];
		}

		BilinearPatch bp;
		for (int i = 0; i < 4; ++i)
			transformPoint(xform, objP[i][0], objP[i][1], objP[i][2], bp.p[i]);
		bp.material = shape.materialIndex;
		bp.areaLight = shape.areaLightIndex;

		// Real object motion blur (AnimatedBilinearPatch's own
		// comment) - same isTopLevelScene/real-inequality gate as
		// mesh motion blur, and the same emissive exclusion, warned
		// the identical way.
		const bool bpCouldAnimate =
			w.isTopLevelScene && xform.differsFrom(w.xformEnd);
		const bool bpAnimated = bpCouldAnimate && shape.areaLightIndex < 0;
		if (bpCouldAnimate && shape.areaLightIndex >= 0) {
			warn("an emissive '" + shape.type + "' has an ActiveTransform "
				 "\"StartTime\"/\"EndTime\" pair, but this loader's bilinear-"
				 "patch motion blur excludes emissive shapes (NEE sampling "
				 "needs enumerable world-space geometry); it will render "
				 "static at its StartTime pose instead");
		}
		if (bpAnimated) {
			bp.gpuOnlyStaticFallback = true;
			AnimatedBilinearPatch abp;
			for (int i = 0; i < 4; ++i) {
				abp.p[i][0] = objP[i][0]; abp.p[i][1] = objP[i][1]; abp.p[i][2] = objP[i][2];
			}
			abp.material = shape.materialIndex;
			fromMatrix4(xform, abp.xform);
			fromMatrix4(w.xformEnd, abp.xformEnd);
			out.animatedBilinearPatches.push_back(std::move(abp));
		}
		w.bilinearPatches->push_back(bp);
		return true;
	}
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "trianglemesh", "plymesh" and "loopsubdiv".
inline bool flattenTriangleMeshShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	// Real object motion blur (AnimatedTriangleMesh's own comment) -
	// gated on ShapeWork::isTopLevelScene (see that field's own
	// comment) - true only for a plain top-level scene shape, false
	// for both an ObjectInstance DEFINITION's own object-space
	// geometry (InstanceGroup::triangles has no xformEnd concept at
	// all and keeps baking to xform only - a genuinely rare "animated
	// mesh nested inside an instance definition" combination, scoped
	// out rather than adding motion blur to the instancing system
	// too) and an instanced EMISSIVE shape baked into world space
	// (excluded explicitly, not just via the separate
	// `shape.areaLightIndex < 0` check just below - see
	// isTopLevelScene's own comment for why relying on that check
	// alone, via a `triangles == &out.triangles` pointer-identity
	// gate, was a real fragility a code-review pass on this
	// feature's own commit caught and this field replaced).
	const bool meshCouldAnimate =
		w.isTopLevelScene && xform.differsFrom(w.xformEnd);
	// A MediumInterface on a mesh shape is silently dropped - Triangle
	// (unlike Sphere/Disk/Cylinder) has no `medium` field to carry it,
	// and neither CPU's addMediumIfPresent() nor GPU's sphere-only
	// MaterialType::Medium can attach one to a mesh. This used to be
	// loud (Material "none" fell to Unsupported and warned by name);
	// now that "none"/"" resolves to the real MaterialKind::Interface
	// with no warning of its own, a mesh-bounded medium boundary would
	// otherwise go silent - warn here instead, so the gap stays
	// visible even though real mesh-medium support isn't implemented.
	if (shape.insideMedium >= 0) {
		warn("shape '" + shape.type + "' has a MediumInterface, but "
			 "this loader only supports an attached medium on an "
			 "unclipped sphere/disk/cylinder shape - the medium will "
			 "be dropped");
	}
	// "texture alpha" - an alpha-cutout mask authored per-Shape (see
	// Material::alphaTextureFilename's own comment for why it lands
	// on the Shape's resolved material rather than a new per-
	// triangle field). Only the texture-bound form is handled here;
	// a literal "float alpha" constant has no bundled scene using it.
	// Every failure to resolve is warned about (mirroring the
	// Diffuse-"reflectance"-texture handling above) rather than
	// silently leaving the shape opaque with no diagnostic.
	if (const pbrt_scene::Param *pa = shape.params.find("alpha")) {
		if (shape.materialIndex < 0 ||
				static_cast<std::size_t>(shape.materialIndex) >= out.materials.size()) {
			warn("shape '" + shape.type + "' binds \"alpha\" but has no "
				 "resolved material to attach the cutout mask to; the "
				 "shape will render fully opaque");
		} else {
			const pbrt_scene::TextureDecl *tex = (pa->type == "texture" && !pa->strings.empty())
				? findTexture(scene, pa->strings[0]) : nullptr;
			const std::string filename = (tex && tex->cls == "imagemap")
				? tex->params.getString("filename", "") : std::string();
			if (filename.empty()) {
				warn("shape's \"alpha\" texture" +
					 (pa->strings.empty() ? std::string() : " '" + pa->strings[0] + "'") +
					 " could not be resolved to an imagemap; the shape "
					 "will render fully opaque");
			} else {
				Material &mat = out.materials[static_cast<std::size_t>(shape.materialIndex)];
				// A NamedMaterial shared by shapes with different
				// alpha masks (the one case Material::alphaTextureFilename's
				// own comment documents as unenforced) silently lets
				// the last shape processed win; warn about it here
				// since scene.materials[i].name (populated only for
				// NamedMaterial declarations) is already in scope.
				if (!mat.alphaTextureFilename.empty() && mat.alphaTextureFilename != filename &&
						!scene.materials[static_cast<std::size_t>(shape.materialIndex)].name.empty()) {
					warn("shape's \"alpha\" texture '" + filename + "' overwrites "
						 "material '" + scene.materials[static_cast<std::size_t>(shape.materialIndex)].name +
						 "'s already-assigned alpha mask '" + mat.alphaTextureFilename +
						 "' - this material is shared via NamedMaterial by multiple "
						 "shapes with different alpha masks; only the last one "
						 "processed will actually be used");
				}
				mat.alphaTextureFilename = filename;
				warnIfImagemapOptionsIgnored(*tex, "alpha", warn);
			}
		}
	}

	std::vector<double> P;
	std::vector<int> indices;
	std::vector<double> N;    // per-vertex, object space; empty = none
	std::vector<double> UV;   // per-vertex, (u,v) pairs; empty = none authored

	if (shape.type == "loopsubdiv") {
		// A subdivision surface is a control cage plus a refinement
		// rule, so it arrives looking exactly like a trianglemesh and
		// renders as a faceted lump if treated as one. loop_subdivide.h
		// is already in this project - itself a port of pbrt's own
		// loopsubdiv - so this is a refinement step, not a new feature.
		//
		// Refining in object space, before the CTM is applied, is safe
		// as well as convenient: Loop limit positions are affine
		// combinations of the control points, so subdividing and then
		// transforming gives the same surface as the reverse.
		const pbrt_scene::Param *pp = shape.params.find("P");
		const pbrt_scene::Param *pi = shape.params.find("indices");
		if (!pp || !pi) {
			warn("a loopsubdiv is missing its P or indices parameter; skipped");
			return true;
		}

		std::vector<std::array<double, 3>> cage(pp->numbers.size() / 3);
		for (std::size_t v = 0; v < cage.size(); ++v)
			cage[v] = {pp->numbers[v * 3], pp->numbers[v * 3 + 1],
					   pp->numbers[v * 3 + 2]};
		std::vector<int> cageIdx;
		cageIdx.reserve(pi->numbers.size());
		for (double d : pi->numbers) cageIdx.push_back(static_cast<int>(d));

		// Each level quadruples the triangle count, so an unbounded
		// value is a way to run out of memory rather than a way to get
		// a smoother surface. pbrt's own default is 3.
		int levels = shape.params.getInt("levels", 3);
		if (levels > kMaxSubdivLevels) {
			warn("loopsubdiv asks for " + std::to_string(levels) +
				 " levels; clamped to " + std::to_string(kMaxSubdivLevels) +
				 " (each level quadruples the triangle count)");
			levels = kMaxSubdivLevels;
		}
		if (levels < 0) levels = 0;

		const LoopSubdivResult<double> refined =
			loop_subdivide<double>(cage, cageIdx, levels);
		P.resize(refined.positions.size() * 3);
		for (std::size_t v = 0; v < refined.positions.size(); ++v) {
			P[v * 3]     = refined.positions[v][0];
			P[v * 3 + 1] = refined.positions[v][1];
			P[v * 3 + 2] = refined.positions[v][2];
		}
		indices = refined.indices;

		// The limit-surface normals are the entire reason to subdivide
		// rather than just tessellate. Discarding them renders the
		// refined mesh as visible facets - smoother facets than the
		// control cage, but facets.
		N.resize(refined.normals.size() * 3);
		for (std::size_t v = 0; v < refined.normals.size(); ++v) {
			N[v * 3]     = refined.normals[v][0];
			N[v * 3 + 1] = refined.normals[v][1];
			N[v * 3 + 2] = refined.normals[v][2];
		}
	} else if (shape.type == "trianglemesh") {
		const pbrt_scene::Param *pp = shape.params.find("P");
		const pbrt_scene::Param *pi = shape.params.find("indices");
		if (!pp || !pi) {
			warn("a trianglemesh is missing its P or indices parameter; skipped");
			return true;
		}
		P = pp->numbers;
		indices.reserve(pi->numbers.size());
		for (double d : pi->numbers) indices.push_back(static_cast<int>(d));
		// pbrt's own name for per-vertex shading normals.
		if (const pbrt_scene::Param *pn = shape.params.find("N")) N = pn->numbers;
		// pbrt's own name for per-vertex texture coordinates - confirmed
		// against pbrt-v4 source (shapes.cpp: GetPoint2fArray("uv")), no
		// "st" alias in this pbrt-v4 version's own TriangleMesh loader.
		if (const pbrt_scene::Param *puv = shape.params.find("uv")) UV = puv->numbers;
	} else {
		const std::string file = shape.params.getString("filename", "");
		if (file.empty()) { warn("a plymesh has no filename; skipped"); return true; }
		if (!meshes) {
			warn("plymesh '" + file + "' skipped: no mesh resolver supplied");
			return true;
		}
		std::vector<float> pos;
		std::vector<float> uvs;
		std::vector<float> nrm;
		if (!meshes(file, pos, indices, uvs, nrm)) {
			warn("plymesh '" + file + "' could not be read; skipped");
			out.missingFiles.push_back(file);
			return true;
		}
		P.assign(pos.begin(), pos.end());
		// Real per-vertex UV when the PLY file carried "u"/"v" (or
		// "s"/"t") vertex properties - see MeshResolver's own comment.
		// Fed into the same `UV` local the trianglemesh branch above
		// populates from its "uv" parameter, so the shared validation/
		// Triangle-construction code below (worldUV, t.uv[]/hasUVs)
		// needs no plymesh-specific handling.
		if (!uvs.empty()) UV.assign(uvs.begin(), uvs.end());
		// Real per-vertex shading normals when the file carried them (smooth
		// shading) - fed into the same `N` local the trianglemesh branch fills
		// from its "N" parameter, so the shared validation below applies.
		if (!nrm.empty()) N.assign(nrm.begin(), nrm.end());
	}

	const std::size_t vertexCount = P.size() / 3;
	if (indices.size() % 3 != 0)
		warn("a mesh has an index count that is not a multiple of 3; "
			 "the trailing indices are ignored");

	// Real object motion blur (AnimatedTriangleMesh's own comment,
	// meshCouldAnimate's own comment above) - excluded for an
	// emissive mesh (NEE needs enumerable world-space geometry);
	// falls back to a static, StartTime-only bake instead, same as
	// every mesh got before this feature existed, just now with an
	// explicit diagnostic instead of silent StartTime-only behavior.
	const bool meshAnimated = meshCouldAnimate && shape.areaLightIndex < 0;
	if (meshCouldAnimate && shape.areaLightIndex >= 0) {
		warn("an emissive '" + shape.type + "' has an ActiveTransform "
			 "\"StartTime\"/\"EndTime\" pair, but this loader's mesh motion "
			 "blur excludes emissive shapes (NEE sampling needs enumerable "
			 "world-space geometry); it will render static at its "
			 "StartTime pose");
	}

	// Transform once per vertex rather than once per index: a shared
	// vertex is referenced by several triangles, and transforming it
	// repeatedly is both slower and a source of tiny inconsistencies
	// between the same point on adjacent faces. Always computed in
	// WORLD space, animated or not - a meshAnimated shape still needs
	// this for its own GPU-fallback duplicate (Triangle::
	// gpuOnlyStaticFallback's own comment): GPU has no concept of
	// AnimatedTriangleMesh's separate object-space list at all, so
	// without a world-space bake here too the shape would silently
	// vanish from GPU renders instead of rendering static.
	std::vector<double> world(vertexCount * 3);
	for (std::size_t v = 0; v < vertexCount; ++v)
		transformPoint(xform, P[v * 3], P[v * 3 + 1], P[v * 3 + 2],
					   &world[v * 3]);

	// A normal list that does not cover every vertex cannot be indexed
	// safely by the face indices, so it is refused wholesale rather
	// than used for part of the mesh - half-smooth shading is a worse
	// artefact than none, and much harder to recognise.
	std::vector<double> worldN;
	if (!N.empty() && N.size() / 3 >= vertexCount) {
		worldN.resize(vertexCount * 3);
		for (std::size_t v = 0; v < vertexCount; ++v)
			transformNormal(xform, N[v * 3], N[v * 3 + 1], N[v * 3 + 2],
							&worldN[v * 3]);
	} else if (!N.empty()) {
		warn("a mesh supplied fewer normals than vertices; "
			 "they are ignored and it will render flat-shaded");
	}

	// pbrt-v4 negates every authored vertex normal when ReverseOrientation is set
	// (TriangleMesh's constructor, util/mesh.cpp:49-56, after the inverse-transpose
	// transform) - NOT the reverseOrientation-XOR-swapsHandedness rule the geometric
	// normal follows (that one is the winding swap below; a mirroring transform
	// already flips authored normals through transformNormal above). The negated
	// normal is what decides which side faces out and, for an emissive mesh, which
	// side emits.
	if (shape.reverseOrientation)
		for (double &x : worldN) x = -x;

	// meshAnimated only: the SAME vertices/normals, but left in
	// OBJECT space (no CTM applied) - what animated_transform_
	// instance.h's real per-ray motion-blur wrapper actually needs
	// (AnimatedTriangleMesh's own comment). `objP`/`objN` mirror
	// `world`/`worldN`'s exact shape (empty objN when worldN is
	// empty) so the per-face loop below can build both a world-space
	// and an object-space Triangle from the same index triple with
	// no extra branching on "does this mesh have normals".
	std::vector<double> objP, objN;
	if (meshAnimated) {
		objP = P;
		if (!worldN.empty()) {
			objN = N;
			if (shape.reverseOrientation)
				for (double &x : objN) x = -x;
		}
	}

	// UV: real per-vertex "point2 uv" when given (same "refused
	// wholesale rather than partially used" validation as N above -
	// no CTM transform, since a UV pair isn't a spatial coordinate).
	//
	// pbrt-v4's own real default when no "uv" is given is a fixed
	// (0,0)/(1,0)/(1,1) triple PER TRIANGLE CORNER, not shared across
	// faces the way an authored "uv" is - deliberately NOT
	// synthesized here. Doing so would give two triangles that share
	// a vertex position genuinely different UV at that shared
	// vertex (a seam pbrt-v4 itself has too, in this exact case),
	// which would silently inflate vertex-dedup counts for every
	// untextured mesh in this loader's corpus - a real cost paid by
	// scenes that never read UV at all. Left unset (hasUVs=false)
	// instead; both builders' own barycentric fallback (matching
	// triangle.h's existing `rec.u=b1,rec.v=b2` exactly - see
	// optix_intersection_triangle.h/wavefront_programs.cu's mirrored
	// fallback) already gives a non-degenerate, if not pbrt-v4-exact,
	// per-point UV when a texture ever actually reads it.
	std::vector<double> worldUV;
	if (!UV.empty() && UV.size() / 2 >= vertexCount) {
		worldUV = UV;
	} else if (!UV.empty()) {
		warn("a mesh supplied fewer uv pairs than vertices; they are ignored");
	}

	// pbrt-v4 ReverseOrientation: the GEOMETRIC normal ends up flipped when
	// reverseOrientation XOR transformSwapsHandedness (a "Scale -1 1 1"-style
	// mirroring transform already flips it once on its own, so ReverseOrientation
	// on top of one cancels back out). Achieved here as flatten-time data
	// manipulation: swapping the b/c vertex (and uv, to keep per-vertex
	// correspondence) order flips the sign of every backend's cross(e1,e2)-derived
	// geometric normal. AUTHORED vertex normals are flipped separately and by
	// reverseOrientation alone, above (worldN/objN) - both pieces together are
	// what pbrt-v4 does, and an authored normal decides the facing side.
	const bool flip = shape.reverseOrientation ^ matrixSwapsHandedness(xform);

	// meshAnimated: accumulated here instead of pushed straight into
	// *w.triangles (which, for the animated case, would otherwise
	// mean OBJECT-space triangles leaking into the world-space list
	// every other consumer of FlatScene::triangles assumes) - moved
	// into a single AnimatedTriangleMesh below once the face loop
	// finishes, carrying the shape's own xform/xformEnd along with it.
	std::vector<Triangle> animTris;

	bool reportedRange = false;
	std::size_t degenerate = 0;
	for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
		const int a = indices[i], b0 = indices[i + 1], c0 = indices[i + 2];
		const int b = flip ? c0 : b0;
		const int c = flip ? b0 : c0;
		if (a < 0 || b < 0 || c < 0
			|| static_cast<std::size_t>(a) >= vertexCount
			|| static_cast<std::size_t>(b) >= vertexCount
			|| static_cast<std::size_t>(c) >= vertexCount) {
			if (!reportedRange) {
				warn("a mesh has face indices outside its vertex list; "
					 "those faces are dropped");
				reportedRange = true;
			}
			continue;
		}
		// pbrt-v4 rejects zero-area triangles at intersection (shapes.cpp:172); dropping
		// them here also keeps an emissive sliver from carrying a NaN geometric normal
		// into light sampling. Repeated indices are the common cause.
		if (a == b || b == c || a == c) { ++degenerate; continue; }
		{
			const double *pa = &world[static_cast<std::size_t>(a) * 3];
			const double *pb = &world[static_cast<std::size_t>(b) * 3];
			const double *pc = &world[static_cast<std::size_t>(c) * 3];
			const double e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
			const double e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
			const double cx = e1[1] * e2[2] - e1[2] * e2[1];
			const double cy = e1[2] * e2[0] - e1[0] * e2[2];
			const double cz = e1[0] * e2[1] - e1[1] * e2[0];
			if (cx * cx + cy * cy + cz * cz == 0.0) { ++degenerate; continue; }
		}
		Triangle t;
		for (int k = 0; k < 3; ++k) {
			t.v[0 + k] = world[static_cast<std::size_t>(a) * 3 + k];
			t.v[3 + k] = world[static_cast<std::size_t>(b) * 3 + k];
			t.v[6 + k] = world[static_cast<std::size_t>(c) * 3 + k];
		}
		if (!worldN.empty()) {
			for (int k = 0; k < 3; ++k) {
				t.n[0 + k] = worldN[static_cast<std::size_t>(a) * 3 + k];
				t.n[3 + k] = worldN[static_cast<std::size_t>(b) * 3 + k];
				t.n[6 + k] = worldN[static_cast<std::size_t>(c) * 3 + k];
			}
			t.hasNormals = true;
		}
		if (!worldUV.empty()) {
			for (int k = 0; k < 2; ++k) {
				t.uv[0 + k] = worldUV[static_cast<std::size_t>(a) * 2 + k];
				t.uv[2 + k] = worldUV[static_cast<std::size_t>(b) * 2 + k];
				t.uv[4 + k] = worldUV[static_cast<std::size_t>(c) * 2 + k];
			}
			t.hasUVs = true;
		}
		t.material = shape.materialIndex;
		t.areaLight = shape.areaLightIndex;
		// meshAnimated: `t` above (world-space, StartTime pose) is
		// still pushed - it's GPU's own fallback duplicate (Triangle::
		// gpuOnlyStaticFallback's own comment), not the CPU-facing
		// triangle - so it's flagged rather than skipped.
		// pbrt_cpu_builder.h's emitGeometry() is what actually skips
		// building a CPU `triangle` hittable for a flagged entry.
		if (meshAnimated) t.gpuOnlyStaticFallback = true;
		w.triangles->push_back(t);

		if (meshAnimated) {
			// The real, OBJECT-space triangle animated_transform_
			// instance.h's wrapper needs (AnimatedTriangleMesh's own
			// comment) - same index triple, same material/areaLight,
			// built from objP/objN instead of world/worldN.
			Triangle tObj;
			for (int k = 0; k < 3; ++k) {
				tObj.v[0 + k] = objP[static_cast<std::size_t>(a) * 3 + k];
				tObj.v[3 + k] = objP[static_cast<std::size_t>(b) * 3 + k];
				tObj.v[6 + k] = objP[static_cast<std::size_t>(c) * 3 + k];
			}
			if (!objN.empty()) {
				for (int k = 0; k < 3; ++k) {
					tObj.n[0 + k] = objN[static_cast<std::size_t>(a) * 3 + k];
					tObj.n[3 + k] = objN[static_cast<std::size_t>(b) * 3 + k];
					tObj.n[6 + k] = objN[static_cast<std::size_t>(c) * 3 + k];
				}
				tObj.hasNormals = true;
			}
			if (!worldUV.empty()) {
				for (int k = 0; k < 2; ++k) {
					tObj.uv[0 + k] = worldUV[static_cast<std::size_t>(a) * 2 + k];
					tObj.uv[2 + k] = worldUV[static_cast<std::size_t>(b) * 2 + k];
					tObj.uv[4 + k] = worldUV[static_cast<std::size_t>(c) * 2 + k];
				}
				tObj.hasUVs = true;
			}
			tObj.material = shape.materialIndex;
			tObj.areaLight = shape.areaLightIndex;
			animTris.push_back(tObj);
		}
	}
	if (degenerate > 0)
		warn(std::to_string(degenerate) + " degenerate (zero-area or repeated-vertex) triangle(s) in a mesh were dropped");
	if (meshAnimated && !animTris.empty()) {
		AnimatedTriangleMesh atm;
		atm.triangles = std::move(animTris);
		fromMatrix4(xform, atm.xform);
		fromMatrix4(w.xformEnd, atm.xformEnd);
		out.animatedTriangleMeshes.push_back(std::move(atm));
	}
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Shape "curve".
inline bool flattenCurveShape(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes,
							 const flatten_detail::ShapeWork &w) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};
	const pbrt_scene::ShapeDecl &shape = *w.shape;
	const pbrt_scene::Matrix4 &xform = w.xform;

	const int degree = shape.params.getInt("degree", 3);
	const std::string basis = shape.params.getString("basis", "bezier");
	const pbrt_scene::Param *pp = shape.params.find("P");

	// Scope: cubic (degree 3) "bezier" (the original, still the
	// overwhelming common case), quadratic (degree 2) "bezier" (real
	// exact degree-elevation to cubic - curveDegreeElevateQuadratic's
	// own comment), and cubic "bspline" (real exact uniform-B-spline-
	// to-Bezier conversion - curveBsplineSegmentToBezierCubic's own
	// comment). Quadratic "bspline" is the one remaining combination
	// left unsupported - real pbrt-v4 scenes essentially never
	// combine the two, and closing it needs a second, quadratic-
	// specific B-spline conversion matrix for marginal real-world
	// value; a disclosed scope cut, not an oversight.
	if (degree != 2 && degree != 3) {
		warn("a curve uses degree " + std::to_string(degree) +
			 " - only degree 2 (quadratic) or 3 (cubic) is supported; skipped");
		return true;
	}
	if (basis != "bezier" && basis != "bspline") {
		warn("a curve uses basis '" + basis + "' - only \"bezier\"/"
			 "\"bspline\" are supported; skipped");
		return true;
	}
	if (basis == "bspline" && degree != 3) {
		warn("a curve uses basis \"bspline\" with degree " +
			 std::to_string(degree) + " - bspline is only supported at "
			 "degree 3; skipped");
		return true;
	}
	if (!pp) {
		warn("a curve is missing its \"point3 P\" control points; skipped");
		return true;
	}
	const std::size_t numPoints = pp->numbers.size() / 3;
	// pp->numbers is already numPoints*3 contiguous doubles in the
	// exact (x,y,z)-per-point layout curveDegreeElevateQuadratic/
	// curveBsplineSegmentToBezierCubic/the cubic branch below all
	// want - indexed directly (&pts(i) below) rather than copied
	// into an intermediate std::vector<array<double,3>>, since a
	// hair/fur scene can have very many curves and this runs once
	// per curve at scene-load time.
	const auto pt = [&](std::size_t i) { return &pp->numbers[i * 3]; };

	// Object-space (pre-CTM) cubic Bezier control points, one 4-point
	// segment at a time, from whichever degree/basis conversion
	// applies - transformed to world space afterward, in one shared
	// loop below, same as every OTHER shape branch in this file
	// transforms its own already-resolved geometry once.
	int nSegments = 0;
	std::vector<double> objCp;
	if (basis == "bspline") {
		// nSegments = numPoints - 3 (a sliding 4-point window, one
		// more segment per additional control point beyond the
		// first 4) - pbrt-v4's own real point-count convention for
		// a cubic B-spline curve.
		if (numPoints < 4) {
			warn("a curve's \"point3 P\" has " + std::to_string(numPoints) +
				 " control points; a cubic B-spline curve needs at least "
				 "4; skipped");
			return true;
		}
		nSegments = static_cast<int>(numPoints) - 3;
		objCp.resize(static_cast<std::size_t>(nSegments) * 4 * 3);
		for (int seg = 0; seg < nSegments; ++seg) {
			double seg4[12];
			curveBsplineSegmentToBezierCubic(pt(static_cast<std::size_t>(seg)), pt(static_cast<std::size_t>(seg) + 1),
											  pt(static_cast<std::size_t>(seg) + 2), pt(static_cast<std::size_t>(seg) + 3), seg4);
			for (int j = 0; j < 12; ++j) objCp[static_cast<std::size_t>(seg) * 12 + j] = seg4[j];
		}
	} else if (degree == 2) {
		// nSegments = (numPoints - 1) / 2 - pbrt-v4's own real
		// point-count convention for a quadratic Bezier curve
		// (2*n+1 points for n segments, matching the existing
		// cubic "3*n+1" formula one degree down).
		if (numPoints < 3 || (numPoints - 1) % 2 != 0) {
			warn("a curve's \"point3 P\" has " + std::to_string(numPoints) +
				 " control points; a quadratic Bezier curve needs 2*n+1 "
				 "control points (3, 5, 7, ...); skipped");
			return true;
		}
		nSegments = static_cast<int>((numPoints - 1) / 2);
		objCp.resize(static_cast<std::size_t>(nSegments) * 4 * 3);
		for (int seg = 0; seg < nSegments; ++seg) {
			double seg4[12];
			curveDegreeElevateQuadratic(pt(static_cast<std::size_t>(seg) * 2),
										 pt(static_cast<std::size_t>(seg) * 2 + 1),
										 pt(static_cast<std::size_t>(seg) * 2 + 2), seg4);
			for (int j = 0; j < 12; ++j) objCp[static_cast<std::size_t>(seg) * 12 + j] = seg4[j];
		}
	} else {
		// degree 3, "bezier" - the original path, unchanged.
		if (numPoints < 4 || (numPoints - 1) % 3 != 0) {
			warn("a curve's \"point3 P\" has " + std::to_string(numPoints) +
				 " control points; a cubic Bezier curve needs 3*n+1 control "
				 "points (4, 7, 10, ...); skipped");
			return true;
		}
		nSegments = static_cast<int>((numPoints - 1) / 3);
		objCp.resize(static_cast<std::size_t>(nSegments) * 4 * 3);
		for (int seg = 0; seg < nSegments; ++seg)
			for (int i = 0; i < 4; ++i)
				for (int k = 0; k < 3; ++k)
					objCp[(static_cast<std::size_t>(seg) * 4 + i) * 3 + k] =
						pt(static_cast<std::size_t>(seg) * 3 + static_cast<std::size_t>(i))[k];
	}

	Curve c;
	c.nSegments = nSegments;
	c.cp.resize(objCp.size());
	for (std::size_t i = 0; i < objCp.size() / 3; ++i) {
		double world[3];
		transformPoint(xform, objCp[i * 3], objCp[i * 3 + 1], objCp[i * 3 + 2], world);
		c.cp[i * 3] = world[0]; c.cp[i * 3 + 1] = world[1]; c.cp[i * 3 + 2] = world[2];
	}

	const double width = shape.params.getFloat("width", 1.0);
	c.width0 = shape.params.getFloat("width0", width);
	c.width1 = shape.params.getFloat("width1", width);
	c.curveType = shape.params.getString("type", "flat");
	if (c.curveType != "flat" && c.curveType != "cylinder" && c.curveType != "ribbon") {
		warn("a curve has unknown \"type\" '" + c.curveType +
			 "'; using \"flat\" instead");
		c.curveType = "flat";
	}

	if (c.curveType == "ribbon") {
		const pbrt_scene::Param *pn = shape.params.find("N");
		const std::size_t needed = static_cast<std::size_t>(c.nSegments) + 1;
		if (!pn || pn->numbers.size() != needed * 3) {
			warn("a ribbon curve needs " + std::to_string(needed) +
				 " \"normal N\" endpoint normals (one per segment "
				 "endpoint); skipped");
			return true;
		}
		c.n.resize(needed * 3);
		for (std::size_t i = 0; i < needed; ++i) {
			double worldN[3];
			transformNormal(xform, pn->numbers[i * 3], pn->numbers[i * 3 + 1],
							 pn->numbers[i * 3 + 2], worldN);
			normalizeOrDefault(worldN, 0.0, 1.0, 0.0);
			c.n[i * 3] = worldN[0]; c.n[i * 3 + 1] = worldN[1]; c.n[i * 3 + 2] = worldN[2];
		}
	}

	c.material = shape.materialIndex;
	c.areaLight = shape.areaLightIndex;

	// Real object motion blur (AnimatedCurve's own comment) - same
	// isTopLevelScene/real-inequality gate as mesh/bilinear-patch
	// motion blur, plus the same emissive exclusion (warned the
	// identical way) and a ribbon-type exclusion (its per-segment
	// shading normals have no per-ray-time transform of their own -
	// AnimatedCurve's own comment).
	const bool curveCouldAnimate =
		w.isTopLevelScene && xform.differsFrom(w.xformEnd);
	const bool curveAnimated =
		curveCouldAnimate && shape.areaLightIndex < 0 && c.curveType != "ribbon";
	if (curveCouldAnimate && shape.areaLightIndex >= 0) {
		warn("an emissive '" + shape.type + "' has an ActiveTransform "
			 "\"StartTime\"/\"EndTime\" pair, but this loader's curve motion "
			 "blur excludes emissive shapes (NEE sampling needs enumerable "
			 "world-space geometry); it will render static at its StartTime "
			 "pose instead");
	}
	if (curveCouldAnimate && shape.areaLightIndex < 0 && c.curveType == "ribbon") {
		warn("a ribbon curve has an ActiveTransform \"StartTime\"/\"EndTime\" "
			 "pair, but this loader's curve motion blur excludes ribbon-type "
			 "curves (their per-segment shading normals have no per-ray-time "
			 "transform yet); it will render static at its StartTime pose "
			 "instead");
	}
	if (curveAnimated) {
		c.gpuOnlyStaticFallback = true;
		AnimatedCurve ac;
		ac.cp = objCp;
		ac.nSegments = nSegments;
		ac.width0 = c.width0;
		ac.width1 = c.width1;
		ac.curveType = c.curveType;
		ac.material = c.material;
		fromMatrix4(xform, ac.xform);
		fromMatrix4(w.xformEnd, ac.xformEnd);
		out.animatedCurves.push_back(std::move(ac));
	}
	w.curves->push_back(std::move(c));
	return true;
	return false;   // not handled here: fall through to the generic 'not supported' warning
}

// Every Shape (and object instance) -> triangles, spheres, patches, curves, groups and instances.
inline void flattenShapes(const pbrt_scene::Scene &scene, FlatScene &out, const MeshResolver &meshes) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- decide what goes where before emitting anything ------------------
	//
	// Three destinations:
	//   * plain scene shapes            -> world space, as before
	//   * an object's non-emissive shapes -> its group, in OBJECT space
	//   * an object's EMISSIVE shapes   -> world space, once per instance
	//
	// That last one is the interesting decision. Instancing saves memory
	// because geometry is shared during traversal, but lights are not
	// traversed - they are enumerated into a flat list and sampled from a
	// distribution, so every copy of an emitter needs its own entry whatever
	// we do. Instancing them would save nothing and would force the MIS path
	// to recover which instance a ray struck. Baking them costs one entry per
	// copy, which for the handful of emitters an object has is nothing.
	//
	// pbrt declines this case outright ("Area lights not supported with object
	// instancing"). Baking is cheap enough that we do not have to.
	std::vector<flatten_detail::ShapeWork> work;

	for (const pbrt_scene::ShapeDecl &shape : scene.shapes)
		work.push_back({&shape, shape.xform, shape.xformEnd, &out.triangles, &out.spheres, &out.bilinearPatches,
						&out.disks, &out.cylinders, &out.curves,
						&out.cones, &out.paraboloids, /*isTopLevelScene=*/true});

	// Sized up front so the pointers taken below stay valid as work is added.
	out.groups.resize(scene.objects.size());
	for (std::size_t g = 0; g < scene.objects.size(); ++g) {
		out.groups[g].name = scene.objects[g].name;
		for (const pbrt_scene::ShapeDecl &shape : scene.objects[g].shapes) {
			if (shape.areaLightIndex >= 0) continue;    // emissive: baked below
			// bilinearPatches left null: an instanced (non-emissive)
			// bilinearmesh has no scene in this loader's own corpus and no
			// InstanceGroup storage for it - the bilinearmesh branch below
			// treats a null output the same way it treats any other
			// unsupported case, rather than silently dropping the shape
			// with no explanation.
			work.push_back({&shape, shape.xform, shape.xformEnd,
							&out.groups[g].triangles, &out.groups[g].spheres});
		}
	}

	for (const pbrt_scene::InstanceDecl &inst : scene.instances) {
		int group = -1;
		// Last match wins, matching pbrt's own name-rebinding semantics and
		// the parser's own warning when ObjectBegin redefines a name ("the
		// later definition replaces the earlier one" - pbrt_scene.h). A
		// redefined object still leaves BOTH definitions in scene.objects
		// (the parser doesn't erase the old one, only warns), so stopping at
		// the first match here silently placed the earlier, supposedly-
		// superseded definition instead - the exact opposite of what the
		// warning told the user would happen. No `break`: keep scanning so
		// the last (most recent) definition's index is what survives.
		for (std::size_t g = 0; g < scene.objects.size(); ++g)
			if (scene.objects[g].name == inst.name) group = static_cast<int>(g);
		if (group < 0) {
			warn("ObjectInstance names '" + inst.name +
				 "', which was never defined; the instance is skipped");
			continue;
		}

		Instance placed;
		placed.group = group;
		fromMatrix4(inst.xform, placed.xform);
		out.instances.push_back(placed);

		// Emissive shapes in the definition, baked into world space for this
		// placement so they can be sampled like any other light.
		for (const pbrt_scene::ShapeDecl &shape :
				 scene.objects[static_cast<std::size_t>(group)].shapes) {
			if (shape.areaLightIndex < 0) continue;
			work.push_back({&shape, flatten_detail::compose(inst.xform, shape.xform),
							flatten_detail::compose(inst.xform, shape.xformEnd),
							&out.triangles, &out.spheres, &out.bilinearPatches,
							&out.disks, &out.cylinders, &out.curves});
		}
	}

	// Which primitives each top-level scene shape produced (FlatScene::shapeRanges). The branches below end with `continue`, so a shape's range is closed
	// when the next one starts (and after the loop).
	const flatten_detail::ShapeWork *openShape = nullptr;
	std::size_t openTri = 0, openSphere = 0, openDisk = 0, openCylinder = 0;
	const auto closeRange = [&]() {
		if (!openShape) return;
		ShapeRange r;
		r.type = openShape->shape->type;
		r.group = openShape->shape->group;
		r.srcFile = openShape->shape->srcFile;
		r.srcBegin = openShape->shape->srcBegin;
		r.srcEnd = openShape->shape->srcEnd;
		for (int k = 0; k < 16; ++k) r.ctm[k] = openShape->shape->xform.m[k];
		r.triBegin = openTri; r.triEnd = out.triangles.size();
		r.sphereBegin = openSphere; r.sphereEnd = out.spheres.size();
		r.diskBegin = openDisk; r.diskEnd = out.disks.size();
		r.cylinderBegin = openCylinder; r.cylinderEnd = out.cylinders.size();
		if (r.triEnd > r.triBegin) r.material = out.triangles[r.triBegin].material;
		else if (r.sphereEnd > r.sphereBegin) r.material = out.spheres[r.sphereBegin].material;
		else if (r.diskEnd > r.diskBegin) r.material = out.disks[r.diskBegin].material;
		else if (r.cylinderEnd > r.cylinderBegin) r.material = out.cylinders[r.cylinderBegin].material;
		if (r.triEnd > r.triBegin || r.sphereEnd > r.sphereBegin || r.diskEnd > r.diskBegin || r.cylinderEnd > r.cylinderBegin) out.shapeRanges.push_back(r);
		openShape = nullptr;
	};
	for (const flatten_detail::ShapeWork &w : work) {
		closeRange();
		if (w.isTopLevelScene) {
			openShape = &w;
			openTri = out.triangles.size(); openSphere = out.spheres.size(); openDisk = out.disks.size(); openCylinder = out.cylinders.size();
		}
		const pbrt_scene::ShapeDecl &shape = *w.shape;
		const pbrt_scene::Matrix4 &xform = w.xform;
		// ReverseOrientation only flips a normal on a trianglemesh/plymesh/
		// loopsubdiv/bilinearmesh shape (see each branch's own comment for
		// its flip mechanism) - sphere/disk/cylinder derive their normal
		// analytically at intersection time in every backend, with no
		// per-shape "flip" input either backend's intersection code reads,
		// and curve has no such mechanism wired either. Warn rather than
		// silently ignore, matching this loop's established convention for
		// a parseable-but-unsupported combination.
		if (shape.reverseOrientation && shape.type != "trianglemesh" &&
				shape.type != "plymesh" && shape.type != "loopsubdiv" &&
				shape.type != "bilinearmesh") {
			warn("shape '" + shape.type + "' has ReverseOrientation set, but "
				 "only trianglemesh/plymesh/loopsubdiv/bilinearmesh honor it - "
				 "this shape's normal is unaffected");
		}
		if (shape.type == "sphere") {
			if (flattenSphereShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "disk" && w.disks) {
			if (flattenDiskShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "cylinder" && w.cylinders) {
			if (flattenCylinderShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "cone" && w.cones) {
			if (flattenConeShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "paraboloid" && w.paraboloids) {
			if (flattenParaboloidShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "bilinearmesh") {
			if (flattenBilinearMeshShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "trianglemesh" || shape.type == "plymesh"
				|| shape.type == "loopsubdiv") {
			if (flattenTriangleMeshShape(scene, out, meshes, w)) continue;
		}

		if (shape.type == "curve" && w.curves) {
			if (flattenCurveShape(scene, out, meshes, w)) continue;
		}

		// disk, cylinder, bilinearmesh, ... Recognised as geometry we cannot
		// build, which is worth saying: the scene will render with a hole in
		// it rather than looking subtly wrong.
		warn("shape '" + shape.type + "' is not supported; skipped");
	}
	closeRange();
}

// Camera-medium and area-light-power resolution: needs the shapes (areas) and media to be final.
inline void resolveAfterShapes(const pbrt_scene::Scene &scene, FlatScene &out) {
	const auto warn = [&out](const std::string &msg) {
		out.warnings.push_back({0, std::string(), msg});
	};

	// ---- camera medium resolution ------------------------------------------
	// pbrt-v4's own "camera medium" (Scene::cameraMediumIndex's own comment,
	// pbrt_scene.h) - scene.cameraMediumIndex is already a valid index into
	// scene.media (resolved at parse time, same as ShapeDecl::insideMedium),
	// and out.media mirrors scene.media 1:1 (see the "---- media ----"
	// block's own comment above), so this is a direct copy, no second
	// lookup needed - gated on two scope cuts specific to THIS feature
	// (homogeneous-only, and not combined with a real per-shape medium),
	// each warned rather than silently ignored or silently wrong.
	if (scene.cameraMediumIndex >= 0 &&
		static_cast<std::size_t>(scene.cameraMediumIndex) < out.media.size()) {
		if (out.media[static_cast<std::size_t>(scene.cameraMediumIndex)].type != "homogeneous") {
			warn("the camera's own medium (MediumInterface's \"outside\" name active "
				 "at the Camera directive) is a \"" +
				 out.media[static_cast<std::size_t>(scene.cameraMediumIndex)].type +
				 "\" medium, which is not supported there yet (homogeneous only); ignored");
		} else {
			// A real per-shape medium ANYWHERE in the scene means this
			// loader's own camera-medium implementation (ray_color()'s own
			// comment, camera.h) - which treats the camera medium as a
			// permanent, everywhere-present attenuator for the whole path,
			// with no true "exit" when a ray passes through another
			// medium's own boundary shape - would double-count or otherwise
			// disagree with that shape's own medium there. Warn and skip
			// rather than silently render an approximation nobody asked
			// for; a scene that genuinely only wants the ambient fog (the
			// common, motivating case for this feature) is unaffected.
			bool hasShapeMedium = false;
			for (const Sphere &s : out.spheres) if (s.medium >= 0) { hasShapeMedium = true; break; }
			for (const Disk &d : out.disks) if (!hasShapeMedium && d.medium >= 0) { hasShapeMedium = true; break; }
			for (const Cylinder &c : out.cylinders) if (!hasShapeMedium && c.medium >= 0) { hasShapeMedium = true; break; }
			if (hasShapeMedium) {
				warn("the scene declares both a camera medium (MediumInterface's "
					 "\"outside\" name active at the Camera directive) and at least "
					 "one real per-shape medium; combining the two is not supported "
					 "yet, so the camera medium is ignored (the per-shape media "
					 "still render normally)");
			} else {
				out.cameraMediumIndex = scene.cameraMediumIndex;
			}
		}
	}

	// ---- area light "power" resolution ------------------------------------
	// pbrt-v4's own formula (DiffuseAreaLight::Create): Phi = L * pi * area *
	// (twoSided ? 2 : 1) - inverted here into the `scale` multiplier that
	// achieves the requested Phi. Can only run now, not in the area-light
	// parsing loop above: AreaLightSource is declared BEFORE the Shape it
	// attaches to in pbrt syntax, so a light's attached area isn't known
	// until every shape referencing it (by `areaLight` index, possibly more
	// than one - a mesh light is often several triangles) has been built,
	// which just finished above.
	//
	// Triangle/unclipped-Sphere/BilinearPatch carry a plain world-space area
	// formula. Disk/Cylinder/Cone/Paraboloid are excluded: unlike Sphere,
	// they are never baked to world space (see Disk/Cylinder's own struct
	// comment) - their radius/height fields are object-space, and getting a
	// world-space area right would need the attached `xform`'s own scale
	// factored in, per axis, which their non-uniform-scale-tolerant
	// intersection path doesn't reduce to a single number the way Sphere's
	// "warn and use the largest axis" approximation does. A clipped Sphere
	// is excluded for the same reason (its radiusLocal/xform are object-
	// space too). Curve is excluded because no closed-form area() exists
	// anywhere in this codebase for it. Each exclusion warns rather than
	// silently mis-stating the light's total power.
	if (std::any_of(out.areaLights.begin(), out.areaLights.end(),
					 [](const Emission &e) { return e.hasPower; })) {
		std::vector<double> area(out.areaLights.size(), 0.0);
		std::vector<bool> excluded(out.areaLights.size(), false);
		auto triangleArea = [](const double a[3], const double b[3], const double c[3]) {
			double e1[3], e2[3];
			for (int k = 0; k < 3; ++k) { e1[k] = b[k] - a[k]; e2[k] = c[k] - a[k]; }
			const double cx = e1[1]*e2[2] - e1[2]*e2[1];
			const double cy = e1[2]*e2[0] - e1[0]*e2[2];
			const double cz = e1[0]*e2[1] - e1[1]*e2[0];
			return 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
		};
		for (const Triangle &t : out.triangles) {
			if (t.areaLight < 0) continue;
			area[t.areaLight] += triangleArea(&t.v[0], &t.v[3], &t.v[6]);
		}
		for (const Sphere &s : out.spheres) {
			if (s.areaLight < 0) continue;
			if (s.clipped) { excluded[s.areaLight] = true; continue; }
			area[s.areaLight] += 4.0 * 3.14159265358979323846 * s.radius * s.radius;
		}
		for (const Disk &d : out.disks) {
			if (d.areaLight >= 0) excluded[d.areaLight] = true;
		}
		for (const Cylinder &c : out.cylinders) {
			if (c.areaLight >= 0) excluded[c.areaLight] = true;
		}
		for (const BilinearPatch &bp : out.bilinearPatches) {
			if (bp.areaLight < 0) continue;
			// Split along the (p00,p10,p01,p11) diagonal into two triangles -
			// exact for a planar patch (the common case a flat panel light
			// actually is) and the same approximation real pbrt-v4's own
			// BilinearPatch::Area() falls back to for a non-rectangular one.
			area[bp.areaLight] += triangleArea(bp.p[0], bp.p[1], bp.p[2]) +
								   triangleArea(bp.p[1], bp.p[3], bp.p[2]);
		}
		for (const Curve &c : out.curves) {
			if (c.areaLight >= 0) excluded[c.areaLight] = true;
		}
		// Cone/Paraboloid - same object-space-radius-plus-unbaked-xform
		// exclusion reason as Disk/Cylinder above.
		for (const Cone &cn : out.cones) {
			if (cn.areaLight >= 0) excluded[cn.areaLight] = true;
		}
		for (const Paraboloid &pb : out.paraboloids) {
			if (pb.areaLight >= 0) excluded[pb.areaLight] = true;
		}

		for (std::size_t i = 0; i < out.areaLights.size(); ++i) {
			Emission &e = out.areaLights[i];
			if (!e.hasPower) continue;
			// A zero or negative "power" is treated as "not given", matching
			// pbrt-v4's own DiffuseAreaLight::Create (`if (phi_v > 0)`) -
			// leaves `scale` at whatever "scale"/L already resolved it to,
			// rather than a bare multiply zeroing or negating the light.
			if (e.power <= 0.0) continue;
			// "filename" (a spatially-varying emission image) wins over L
			// entirely for what actually gets rendered (Emission::filename's
			// own comment) - but the formula below is derived from L, which
			// has no relationship to the image's real average radiance. Warn
			// rather than silently computing a scale against a fictitious
			// flat colour, matching the goniometric/projection punctual-light
			// cases' own "power depends on the image, not supported" warning.
			if (!e.filename.empty()) {
				warn("area light \"power\" is not supported together with "
					 "\"filename\" (its real total output depends on the "
					 "image, not L); \"scale\"/L used as given instead");
				continue;
			}
			if (excluded[i]) {
				if (area[i] > 1e-12) {
					warn("area light \"power\" is not supported: attached to a "
						 "mix of a measurable shape and a disk/cylinder/cone/"
						 "paraboloid/clipped-sphere/curve shape sharing the same "
						 "AreaLightSource - the measurable shape's own area "
						 "cannot be used on its own without mis-stating the "
						 "light's total power; \"scale\"/L used as given instead");
				} else {
					warn("area light \"power\" is not supported when attached to a "
						 "disk/cylinder/cone/paraboloid/clipped-sphere/curve "
						 "shape; \"scale\"/L used as given instead");
				}
				continue;
			}
			if (area[i] <= 1e-12) {
				warn("area light \"power\" given but the attached shape has no "
					 "measurable area; \"scale\"/L used as given instead");
				continue;
			}
			const double kE = 3.14159265358979323846 * area[i] * (e.twoSided ? 2.0 : 1.0);
			e.scale *= (e.power / kE) / relativeLuminance(e.L);
		}
	}
}

} // namespace flatten_detail
