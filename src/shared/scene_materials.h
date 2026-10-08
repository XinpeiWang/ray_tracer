#pragma once
// scene_materials.h -- ready-made materials for the Scene Builder: a name, and the kind/colour/roughness/index of refraction that makes it look like gold, frosted
// glass, red plastic, wax... Choosing one in the inspector sets those fields on the object's material (a preset is not remembered afterwards: the object keeps
// ordinary material values that can be tuned further). Standard-library only, like the rest of the document.
//
// Metal colours are the reflectance at normal incidence (F0) of the real metals, linear RGB; glass and water use their real indices of refraction.

#include "scene_model.h"

#include <string>
#include <vector>

namespace scene_doc {

struct MaterialPreset {
	const char* id;     // stable, never shown; the name below is what a user sees (translated by the GUI)
	const char* name;
	const char* group;  // "Matte", "Plastic", "Metal", "Glass", "Translucent"
	Material material;
};

namespace presets_detail {
inline Material make(MaterialKind kind, Rgb color, double roughness = 0.0, double ior = 1.5, Rgb transmittance = {0.5, 0.5, 0.5}) {
	Material m;
	m.kind = kind;
	m.color = color;
	m.roughness = roughness;
	m.ior = ior;
	m.transmittance = transmittance;
	return m;
}
}  // namespace presets_detail

inline const std::vector<MaterialPreset>& materialPresets() {
	using namespace presets_detail;
	static const std::vector<MaterialPreset> all = {
		{"chalk", "Chalk", "Matte", make(MaterialKind::Diffuse, {0.88, 0.88, 0.85})},
		{"rubber", "Black rubber", "Matte", make(MaterialKind::Diffuse, {0.04, 0.04, 0.045})},
		{"clay", "Terracotta", "Matte", make(MaterialKind::Diffuse, {0.62, 0.30, 0.18})},
		{"concrete", "Concrete", "Matte", make(MaterialKind::Diffuse, {0.48, 0.48, 0.47})},
		{"red-plastic", "Red plastic", "Plastic", make(MaterialKind::CoatedDiffuse, {0.70, 0.04, 0.04}, 0.08, 1.5)},
		{"blue-plastic", "Blue plastic", "Plastic", make(MaterialKind::CoatedDiffuse, {0.05, 0.12, 0.65}, 0.08, 1.5)},
		{"ceramic", "White ceramic", "Plastic", make(MaterialKind::CoatedDiffuse, {0.90, 0.90, 0.88}, 0.02, 1.55)},
		{"car-paint", "Car paint", "Plastic", make(MaterialKind::CoatedDiffuse, {0.45, 0.02, 0.06}, 0.03, 1.6)},
		{"gold", "Gold", "Metal", make(MaterialKind::Conductor, {1.000, 0.766, 0.336}, 0.08)},
		{"copper", "Copper", "Metal", make(MaterialKind::Conductor, {0.955, 0.638, 0.538}, 0.10)},
		{"silver", "Silver", "Metal", make(MaterialKind::Conductor, {0.972, 0.960, 0.915}, 0.05)},
		{"aluminium", "Aluminium", "Metal", make(MaterialKind::Conductor, {0.913, 0.922, 0.924}, 0.12)},
		{"chrome", "Chrome", "Metal", make(MaterialKind::Conductor, {0.550, 0.556, 0.554}, 0.02)},
		{"brushed-steel", "Brushed steel", "Metal", make(MaterialKind::Conductor, {0.62, 0.62, 0.64}, 0.35)},
		{"glass", "Clear glass", "Glass", make(MaterialKind::Dielectric, {0.8, 0.8, 0.8}, 0.0, 1.5)},
		{"frosted-glass", "Frosted glass", "Glass", make(MaterialKind::Dielectric, {0.8, 0.8, 0.8}, 0.25, 1.5)},
		{"water", "Water", "Glass", make(MaterialKind::Dielectric, {0.8, 0.8, 0.8}, 0.0, 1.33)},
		{"diamond", "Diamond", "Glass", make(MaterialKind::Dielectric, {0.8, 0.8, 0.8}, 0.0, 2.42)},
		{"wax", "Wax", "Translucent", make(MaterialKind::DiffuseTransmission, {0.70, 0.62, 0.40}, 0.0, 1.5, {0.40, 0.34, 0.16})},
		{"leaf", "Leaf", "Translucent", make(MaterialKind::DiffuseTransmission, {0.10, 0.38, 0.07}, 0.0, 1.5, {0.12, 0.34, 0.05})},
		{"paper", "Paper", "Translucent", make(MaterialKind::DiffuseTransmission, {0.78, 0.78, 0.74}, 0.0, 1.5, {0.45, 0.45, 0.42})},
	};
	return all;
}

inline const MaterialPreset* findMaterialPreset(const std::string& id) {
	for (const MaterialPreset& p : materialPresets())
		if (id == p.id) return &p;
	return nullptr;
}

// Gives `m` the preset's kind and values. The picture and the checker pattern are dropped (they would replace the preset's colour); everything else about the
// material is the preset's. Returns false for an unknown id.
inline bool applyMaterialPreset(Material& m, const std::string& id) {
	const MaterialPreset* p = findMaterialPreset(id);
	if (!p) return false;
	m = p->material;
	m.imageFile.clear();
	m.checker = false;
	return true;
}

}  // namespace scene_doc
