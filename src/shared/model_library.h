// model_library.h - the Scene Builder's model library: the ready-made meshes in the models/ folder (scans and sculpts from public 3D test collections),
// with a display name each, and the rule that sets a new one down: scaled to a handy size, centred on the drop point, standing on the floor.
// Qt-free; docs/MODELS.md says where each model comes from and what that source's licence asks.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace model_library {

struct Entry {
	const char* stem;      // the file name without ".obj" (models/<stem>.obj, models/thumbnails/<stem>.png)
	const char* name;      // what the menu shows
	const char* source;    // where it comes from, in a few words
	int kiloTriangles;     // roughly, to warn about heavy ones
	bool bundled;          // shipped inside the macOS and Windows installers (small, and clear to redistribute); the others come with a source checkout
};

inline const std::vector<Entry>& catalog() {
	static const std::vector<Entry> all = {
		{"spot", "Spot (cow)", "Keenan Crane, CC0", 6, true},
		{"suzanne", "Suzanne (monkey)", "Blender", 1, true},
		{"teapot", "Utah teapot", "Martin Newell, public domain", 6, true},
		{"stanford-bunny", "Stanford bunny", "Stanford 3D Scanning Repository", 69, false},
		{"armadillo", "Armadillo", "Stanford 3D Scanning Repository", 100, false},
		{"happy-buddha", "Happy Buddha", "Stanford 3D Scanning Repository", 99, false},
		{"lucy", "Lucy (angel)", "Stanford 3D Scanning Repository", 100, false},
		{"xyzrgb_dragon", "Dragon", "Stanford 3D Scanning Repository (XYZ RGB)", 250, false},
		{"cow", "Cow", "public 3D test-model collection", 6, false},
		{"horse", "Horse", "public 3D test-model collection", 97, false},
		{"ogre", "Ogre", "public 3D test-model collection", 124, false},
		{"beast", "Beast", "public 3D test-model collection", 32, false},
		{"homer", "Homer", "public 3D test-model collection", 12, false},
		{"cheburashka", "Cheburashka", "public 3D test-model collection", 13, false},
		{"beetle", "Beetle (car, light)", "public 3D test-model collection", 2, false},
		{"beetle-alt", "Beetle (car, detailed)", "public 3D test-model collection", 39, false},
		{"nefertiti", "Nefertiti (bust)", "public 3D test-model collection", 100, false},
		{"max-planck", "Max Planck (bust)", "public 3D test-model collection", 100, false},
		{"bimba", "Bimba (bust)", "public 3D test-model collection", 225, false},
		{"igea", "Igea (head)", "public 3D test-model collection", 269, false},
		{"fandisk", "Fandisk (mechanical part)", "public 3D test-model collection", 13, false},
		{"rocker-arm", "Rocker arm", "public 3D test-model collection", 20, false},
	};
	return all;
}

inline const Entry* findEntry(const char* stem) {
	for (const Entry& e : catalog())
		if (std::string(e.stem) == stem) return &e;
	return nullptr;
}

// Where a library model goes: `scale` makes its largest side `targetSize` scene units; `offset` is the Scene Builder position that puts the model's
// bounding-box centre over the drop point (x and z) and its lowest point on the floor (y = 0). lo and hi are the file's own bounding box.
struct Placement {
	double scale = 1.0;
	double offset[3] = {0.0, 0.0, 0.0};
};

inline Placement placeOnFloor(const double lo[3], const double hi[3], double dropX, double dropZ, double targetSize = 1.6) {
	Placement p;
	const double largest = std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
	p.scale = largest > 1e-12 ? targetSize / largest : 1.0;
	p.offset[0] = dropX - 0.5 * (lo[0] + hi[0]) * p.scale;
	p.offset[1] = -lo[1] * p.scale;
	p.offset[2] = dropZ - 0.5 * (lo[2] + hi[2]) * p.scale;
	return p;
}

}  // namespace model_library
