/// @file scene_metadata_snapshot.h
/// @brief Plain-data struct shared, by #include, across the scene_metadata.dll
/// ABI boundary - the MSVC-built cpu_renderer/scene_metadata.dll side and the
/// MinGW-built Qt GUI client side both compile this exact header directly,
/// rather than each maintaining their own hand-typed copy of the struct.
///
/// This works because x64 Windows has one calling convention and one struct-
/// layout rule for standard-layout plain-data types regardless of compiler -
/// the same guarantee cpu_interface.h's every extern "C" function already
/// relies on for its individual int/double/const char* parameters (see
/// qt_gui/scene_metadata_client.cpp's own top-of-file ABI comment). A struct
/// made ONLY of those same primitive types (no STL, no virtual functions, no
/// exceptions crossing the boundary) is just as ABI-stable, so sharing this
/// header lets a field-order/type mismatch between the two sides become a
/// compile error instead of a silent one - unlike the hand-mirrored struct
/// this file replaced, where nothing but a comment enforced the agreement.
///
/// Keep this file exactly this narrow: plain data only, no other #includes,
/// no dependency on anything that isn't itself this simple - the moment it
/// needs to pull in an STL/Qt/project header, the two sides stop being able
/// to compile it identically and the whole point of sharing it is lost.
#ifndef SCENE_METADATA_SNAPSHOT_H
#define SCENE_METADATA_SNAPSHOT_H

#ifdef __cplusplus
extern "C" {
#endif

/// A single-call bundle of a scene's presentational/recommendation/camera
/// fields - see cpu_interface.h's cpu_scene_metadata_snapshot() and
/// qt_gui/scene_metadata_client.h's SceneMetadata for the two sides that
/// populate and consume this. Every const char* here has the same lifetime
/// as cpu_interface.h's individual by-id accessors' own return values
/// (points into the static scene registry, valid for the process lifetime,
/// never owned by the caller).
struct SceneMetadataSnapshot {
	const char* name;
	const char* category;
	const char* description;
	const char* performance;
	int recommended_spp;
	int requires_files;              // 1/0, C-bool
	int gpu_compatible;               // 1/0, C-bool
	double recommended_exposure;
	const char* recommended_integrator;    // "" if the scene doesn't declare one
	const char* recommended_sampler;       // ""
	const char* recommended_light_sampler; // ""
	// Same fields cpu_scene_recommended_camera() returns, unconditionally
	// populated here (unlike that function, which leaves its out-params
	// untouched on failure) - safe because this whole struct is only valid
	// when cpu_scene_metadata_snapshot() itself returned 1, at which point
	// every SceneDescriptor's camera (never optional - see CameraConfig)
	// is known-good.
	double cam_lookfrom_x, cam_lookfrom_y, cam_lookfrom_z;
	double cam_lookat_x, cam_lookat_y, cam_lookat_z;
};

#ifdef __cplusplus
}
#endif

#endif // SCENE_METADATA_SNAPSHOT_H
