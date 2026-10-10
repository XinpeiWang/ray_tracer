#pragma once
// scene_blocks.h -- the Scene Builder's blocky objects: coloured blocks (grass, TNT, lava...), blocky creatures (a green monster, a pig, a sheep, an explorer, a skeleton)
// and things (a tree, a cottage, a torch, a chest, a bed, a rainbow row of wool, a glowing portal), in the look of block-building games. Not affiliated with any such game.
//
// Each one is drawn as voxels on a small grid (a letter per colour), then merged into as few boxes as possible (a greedy merge: every box holds one colour and no two
// overlap), so a creature is a few dozen ordinary Box objects. Adding one adds those boxes, as props do (scene_props.h): from then on they are plain objects that can be
// moved, recoloured or deleted one by one. One block of the world is 0.5 scene units, so a block is half a metre-ish table-height unit: a creature stands about 0.8 tall and a
// tree 3.5. The origin is the middle of the footprint on the floor.
//
// std-only, like scene_props.h.

#include "scene_props.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace scene_doc {

enum class BlockyKind {
	// blocks
	GrassBlock, DirtBlock, StoneBlock, PlanksBlock, LogBlock, SandBlock, GlassBlock, WaterBlock, TntBlock, GoldBlock, DiamondOre, GlowBlock, LavaBlock, Pumpkin, LanternPumpkin,
	// creatures
	GreenMonster, Pig, Sheep, Explorer, Skeleton,
	// things
	OakTree, Cottage, Torch, Chest, CraftingTable, Furnace, Bed, Poppy, Dandelion, Fence, RainbowWool, PurplePortal
};

enum class BlockyGroup { Blocks, Creatures, Things };

inline const std::vector<BlockyKind>& allBlockyKinds() {
	static const std::vector<BlockyKind> all = {
		BlockyKind::GrassBlock, BlockyKind::DirtBlock, BlockyKind::StoneBlock, BlockyKind::PlanksBlock, BlockyKind::LogBlock, BlockyKind::SandBlock, BlockyKind::GlassBlock,
		BlockyKind::WaterBlock, BlockyKind::TntBlock, BlockyKind::GoldBlock, BlockyKind::DiamondOre, BlockyKind::GlowBlock, BlockyKind::LavaBlock, BlockyKind::Pumpkin,
		BlockyKind::LanternPumpkin, BlockyKind::GreenMonster, BlockyKind::Pig, BlockyKind::Sheep, BlockyKind::Explorer, BlockyKind::Skeleton, BlockyKind::OakTree,
		BlockyKind::Cottage, BlockyKind::Torch, BlockyKind::Chest, BlockyKind::CraftingTable, BlockyKind::Furnace, BlockyKind::Bed, BlockyKind::Poppy, BlockyKind::Dandelion,
		BlockyKind::Fence, BlockyKind::RainbowWool, BlockyKind::PurplePortal};
	return all;
}

inline BlockyGroup blockyGroup(BlockyKind k) {
	if (k <= BlockyKind::LanternPumpkin) return BlockyGroup::Blocks;
	if (k <= BlockyKind::Skeleton) return BlockyGroup::Creatures;
	return BlockyGroup::Things;
}

inline const char* toString(BlockyKind k) {
	switch (k) {
		case BlockyKind::GrassBlock: return "grass block";
		case BlockyKind::DirtBlock: return "dirt block";
		case BlockyKind::StoneBlock: return "stone block";
		case BlockyKind::PlanksBlock: return "planks block";
		case BlockyKind::LogBlock: return "log block";
		case BlockyKind::SandBlock: return "sand block";
		case BlockyKind::GlassBlock: return "glass block";
		case BlockyKind::WaterBlock: return "water block";
		case BlockyKind::TntBlock: return "TNT block";
		case BlockyKind::GoldBlock: return "gold block";
		case BlockyKind::DiamondOre: return "diamond ore";
		case BlockyKind::GlowBlock: return "glowing block";
		case BlockyKind::LavaBlock: return "lava block";
		case BlockyKind::Pumpkin: return "pumpkin";
		case BlockyKind::LanternPumpkin: return "lantern pumpkin";
		case BlockyKind::GreenMonster: return "green monster";
		case BlockyKind::Pig: return "pig";
		case BlockyKind::Sheep: return "sheep";
		case BlockyKind::Explorer: return "explorer";
		case BlockyKind::Skeleton: return "skeleton";
		case BlockyKind::OakTree: return "oak tree";
		case BlockyKind::Cottage: return "cottage";
		case BlockyKind::Torch: return "torch";
		case BlockyKind::Chest: return "chest";
		case BlockyKind::CraftingTable: return "crafting table";
		case BlockyKind::Furnace: return "furnace";
		case BlockyKind::Bed: return "bed";
		case BlockyKind::Poppy: return "poppy";
		case BlockyKind::Dandelion: return "dandelion";
		case BlockyKind::Fence: return "fence";
		case BlockyKind::RainbowWool: return "rainbow wool";
		case BlockyKind::PurplePortal: return "purple portal";
	}
	return "block";
}

namespace blocks_detail {

// A grid of voxels, a letter per colour (0 = empty). x to the right, y up, z toward the front (the face the creatures show).
struct Voxels {
	int nx, ny, nz;
	std::vector<char> cell;
	Voxels(int x, int y, int z) : nx(x), ny(y), nz(z), cell(static_cast<size_t>(x) * y * z, 0) {}
	char& at(int x, int y, int z) { return cell[(static_cast<size_t>(y) * nz + z) * nx + x]; }
	char at(int x, int y, int z) const { return cell[(static_cast<size_t>(y) * nz + z) * nx + x]; }
	bool inside(int x, int y, int z) const { return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz; }
	void fill(int x0, int y0, int z0, int w, int h, int d, char c) {   // later fills paint over earlier ones; anything outside the grid is clipped
		for (int y = y0; y < y0 + h; ++y)
			for (int z = z0; z < z0 + d; ++z)
				for (int x = x0; x < x0 + w; ++x)
					if (inside(x, y, z)) at(x, y, z) = c;
	}
	size_t filled() const { return static_cast<size_t>(std::count_if(cell.begin(), cell.end(), [](char c) { return c != 0; })); }
};

enum class Look { Matte, Glass, Water, Gold, Glow };
struct Paint {
	char key;
	const char* name;
	Rgb color;
	Look look;
	double strength;   // Glow: the emission strength
};

inline const Paint* findPaint(const std::vector<Paint>& palette, char key) {
	for (const Paint& p : palette)
		if (p.key == key) return &p;
	return nullptr;
}

// Greedy merge: scans the grid in order and grows a box from each unclaimed voxel along x, then z, then y while every voxel in the next slab is the same colour and
// unclaimed. The boxes cover exactly the filled voxels, once each.
inline std::vector<Object> boxesFromVoxels(const std::string& model, const Voxels& v, const std::vector<Paint>& palette, double unit) {
	std::vector<Object> out;
	std::vector<char> done(v.cell.size(), 0);
	auto idx = [&](int x, int y, int z) { return (static_cast<size_t>(y) * v.nz + z) * v.nx + x; };
	std::vector<std::pair<char, int>> counters;
	for (int y = 0; y < v.ny; ++y)
		for (int z = 0; z < v.nz; ++z)
			for (int x = 0; x < v.nx; ++x) {
				const char c = v.at(x, y, z);
				if (c == 0 || done[idx(x, y, z)]) continue;
				int w = 1, d = 1, h = 1;
				while (x + w < v.nx && v.at(x + w, y, z) == c && !done[idx(x + w, y, z)]) ++w;
				auto rowOk = [&](int zz, int yy) {
					for (int xx = x; xx < x + w; ++xx)
						if (v.at(xx, yy, zz) != c || done[idx(xx, yy, zz)]) return false;
					return true;
				};
				while (z + d < v.nz && rowOk(z + d, y)) ++d;
				auto slabOk = [&](int yy) {
					for (int zz = z; zz < z + d; ++zz)
						if (!rowOk(zz, yy)) return false;
					return true;
				};
				while (y + h < v.ny && slabOk(y + h)) ++h;
				for (int yy = y; yy < y + h; ++yy)
					for (int zz = z; zz < z + d; ++zz)
						for (int xx = x; xx < x + w; ++xx) done[idx(xx, yy, zz)] = 1;
				const Paint* p = findPaint(palette, c);
				Object o = makeObject(ShapeKind::Box, model);
				int* count = nullptr;
				for (auto& pr : counters)
					if (pr.first == c) count = &pr.second;
				if (!count) { counters.push_back({c, 0}); count = &counters.back().second; }
				++*count;
				o.name = model + " " + (p ? p->name : "part") + (*count > 1 ? " (" + std::to_string(*count) + ")" : std::string());
				o.size = {w * unit, h * unit, d * unit};
				o.position = {(x + 0.5 * w - 0.5 * v.nx) * unit, (y + 0.5 * h) * unit, (z + 0.5 * d - 0.5 * v.nz) * unit};
				if (p) {
					o.material.color = p->color;
					switch (p->look) {
						case Look::Matte: break;
						case Look::Glass: o.material.kind = MaterialKind::Dielectric; o.material.ior = 1.5; break;
						case Look::Water:
							o.material.kind = MaterialKind::DiffuseTransmission;
							o.material.transmittance = {p->color.r * 0.8 + 0.1, p->color.g * 0.8 + 0.1, p->color.b * 0.8 + 0.1};
							break;
						case Look::Gold: o.material.kind = MaterialKind::Conductor; o.material.roughness = 0.2; break;
						case Look::Glow: o.emissive = true; o.emission = p->color; o.emissionStrength = p->strength; break;
					}
				}
				out.push_back(o);
			}
	return out;
}

// A cube block of 16 voxels (one block = 0.5 units, so a voxel is 1/32 unit).
constexpr double kPx = 0.5 / 16.0;
constexpr double kBlock = 0.5;

inline std::vector<Object> make(const std::string& name, const Voxels& v, const std::vector<Paint>& pal, double unit) { return boxesFromVoxels(name, v, pal, unit); }

}  // namespace blocks_detail

inline std::vector<Object> makeBlocky(BlockyKind kind) {
	using namespace blocks_detail;
	using L = Look;
	switch (kind) {
		case BlockyKind::GrassBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'd');
			v.fill(0, 14, 0, 16, 2, 16, 'g');
			for (int x = 0; x < 16; ++x)
				for (int z = 0; z < 16; ++z)
					if ((x == 0 || x == 15 || z == 0 || z == 15) && (x * 7 + z * 3) % 5 < 2) v.at(x, 13, z) = 'g';
			return make("Grass block", v, {{'d', "dirt", {0.42, 0.28, 0.16}, L::Matte, 0}, {'g', "grass", {0.28, 0.55, 0.14}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::DirtBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'd');
			for (int i = 0; i < 6; ++i) v.fill((i * 5) % 12 + 1, 15, (i * 7) % 12 + 1, 2, 1, 2, 's');   // a few darker flecks on the top
			return make("Dirt block", v, {{'d', "dirt", {0.42, 0.28, 0.16}, L::Matte, 0}, {'s', "fleck", {0.3, 0.19, 0.1}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::StoneBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 's');
			for (int i = 0; i < 6; ++i) v.fill((i * 5) % 12 + 1, 15, (i * 7) % 12 + 1, 3, 1, 2, 'd');
			return make("Stone block", v, {{'s', "stone", {0.5, 0.5, 0.5}, L::Matte, 0}, {'d', "fleck", {0.38, 0.38, 0.38}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::PlanksBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'p');
			for (int y : {3, 7, 11, 15})
				for (int x = 0; x < 16; ++x)
					for (int z = 0; z < 16; ++z)
						if (x == 0 || x == 15 || z == 0 || z == 15) v.at(x, y, z) = 'l';
			return make("Planks block", v, {{'p', "planks", {0.66, 0.5, 0.3}, L::Matte, 0}, {'l', "seam", {0.42, 0.3, 0.16}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::LogBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'b');
			for (int y : {0, 15}) {
				v.fill(1, y, 1, 14, 1, 14, 'w');
				v.fill(5, y, 5, 6, 1, 6, 'r');
			}
			return make("Log block", v, {{'b', "bark", {0.32, 0.22, 0.11}, L::Matte, 0}, {'w', "wood", {0.66, 0.52, 0.32}, L::Matte, 0}, {'r', "ring", {0.5, 0.38, 0.2}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::SandBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 's');
			return make("Sand block", v, {{'s', "sand", {0.82, 0.76, 0.52}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::GlassBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'g');
			return make("Glass block", v, {{'g', "glass", {1, 1, 1}, L::Glass, 0}}, kPx);
		}
		case BlockyKind::WaterBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'w');
			return make("Water block", v, {{'w', "water", {0.2, 0.42, 0.9}, L::Water, 0}}, kPx);
		}
		case BlockyKind::TntBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'r');
			for (int x = 0; x < 16; ++x)
				for (int z = 0; z < 16; ++z)
					if (x == 0 || x == 15 || z == 0 || z == 15) for (int y = 5; y < 11; ++y) v.at(x, y, z) = 'w';
			v.fill(6, 15, 6, 4, 1, 4, 'k');
			return make("TNT block", v, {{'r', "red", {0.78, 0.1, 0.07}, L::Matte, 0}, {'w', "band", {0.88, 0.88, 0.85}, L::Matte, 0}, {'k', "fuse", {0.12, 0.12, 0.12}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::GoldBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'g');
			return make("Gold block", v, {{'g', "gold", {1.0, 0.78, 0.3}, L::Gold, 0}}, kPx);
		}
		case BlockyKind::DiamondOre: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 's');
			const int spots[6][2] = {{2, 9}, {9, 11}, {5, 3}, {11, 4}, {7, 7}, {2, 2}};
			for (const auto& s : spots) {
				v.fill(s[0], s[1], 15, 3, 3, 1, 'd');   // front face
				v.fill(15, s[1], s[0], 1, 3, 3, 'd');   // right face
			}
			return make("Diamond ore", v, {{'s', "stone", {0.5, 0.5, 0.5}, L::Matte, 0}, {'d', "diamond", {0.3, 0.88, 0.84}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::GlowBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'g');
			return make("Glowing block", v, {{'g', "glow", {1.0, 0.8, 0.42}, L::Glow, 6.0}}, kPx);
		}
		case BlockyKind::LavaBlock: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'o');
			for (int i = 0; i < 5; ++i) v.fill((i * 5) % 11 + 1, 15, (i * 7) % 11 + 1, 4, 1, 3, 'y');
			return make("Lava block", v, {{'o', "lava", {1.0, 0.34, 0.04}, L::Glow, 4.0}, {'y', "bright", {1.0, 0.7, 0.15}, L::Glow, 6.0}}, kPx);
		}
		case BlockyKind::Pumpkin:
		case BlockyKind::LanternPumpkin: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'o');
			for (int x : {3, 7, 11})
				for (int y = 0; y < 16; ++y) v.at(x, y, 15) = 'r';   // the ribs on the front
			v.fill(7, 15, 7, 2, 1, 2, 'k');   // the stem on top
			if (kind == BlockyKind::LanternPumpkin) {
				v.fill(3, 8, 15, 3, 3, 1, 'f');
				v.fill(10, 8, 15, 3, 3, 1, 'f');
				v.fill(3, 3, 15, 10, 2, 1, 'f');
				v.fill(5, 5, 15, 6, 1, 1, 'f');
			}
			return make(kind == BlockyKind::Pumpkin ? "Pumpkin" : "Lantern pumpkin", v,
			            {{'o', "orange", {0.86, 0.46, 0.08}, L::Matte, 0}, {'r', "rib", {0.7, 0.34, 0.05}, L::Matte, 0}, {'k', "stem", {0.25, 0.4, 0.12}, L::Matte, 0},
			             {'f', "light", {1.0, 0.78, 0.25}, L::Glow, 8.0}}, kPx);
		}
		case BlockyKind::GreenMonster: {
			Voxels v(8, 26, 8);
			for (int x : {0, 5})
				for (int z : {0, 5}) v.fill(x, 0, z, 3, 6, 3, 'g');   // four legs
			v.fill(0, 6, 2, 8, 12, 4, 'g');                       // body
			v.fill(0, 18, 0, 8, 8, 8, 'g');                       // head
			v.fill(1, 22, 7, 2, 2, 1, 'k'); v.fill(5, 22, 7, 2, 2, 1, 'k');   // eyes
			v.fill(3, 20, 7, 2, 2, 1, 'k'); v.fill(2, 19, 7, 4, 1, 1, 'k');   // nose and mouth
			v.fill(2, 18, 7, 1, 1, 1, 'k'); v.fill(5, 18, 7, 1, 1, 1, 'k');
			for (int i = 0; i < 7; ++i) v.fill((i * 3) % 6, 6 + (i * 5) % 10, 6, 2, 2, 1, 'd');   // darker patches on the front of the body
			return make("Green monster", v, {{'g', "green", {0.36, 0.66, 0.3}, L::Matte, 0}, {'k', "face", {0.04, 0.05, 0.04}, L::Matte, 0}, {'d', "patch", {0.24, 0.5, 0.2}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Pig: {
			Voxels v(10, 14, 25);
			for (int x : {0, 7})
				for (int z : {1, 12}) v.fill(x, 0, z, 3, 6, 3, 'p');   // legs
			v.fill(0, 6, 0, 10, 8, 16, 'p');                       // body
			v.fill(1, 6, 16, 8, 8, 8, 'p');                        // head
			v.fill(3, 7, 24, 4, 3, 1, 's');                        // snout
			v.fill(3, 8, 24, 1, 1, 1, 'k'); v.fill(6, 8, 24, 1, 1, 1, 'k');
			v.fill(2, 11, 23, 1, 1, 1, 'k'); v.fill(7, 11, 23, 1, 1, 1, 'k');   // eyes
			return make("Pig", v, {{'p', "pink", {0.96, 0.62, 0.64}, L::Matte, 0}, {'s', "snout", {0.88, 0.48, 0.52}, L::Matte, 0}, {'k', "dark", {0.08, 0.05, 0.05}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Sheep: {
			Voxels v(10, 15, 22);
			for (int x : {0, 7})
				for (int z : {1, 10}) v.fill(x, 0, z, 3, 6, 3, 'l');   // legs
			v.fill(0, 6, 0, 10, 9, 14, 'w');                       // woolly body
			v.fill(2, 8, 14, 6, 6, 8, 's');                        // head
			v.fill(2, 13, 14, 6, 2, 6, 'w');                       // wool on top of the head
			v.fill(3, 11, 21, 1, 1, 1, 'k'); v.fill(6, 11, 21, 1, 1, 1, 'k');   // eyes
			v.fill(4, 9, 21, 2, 1, 1, 'n');                        // nose
			return make("Sheep", v, {{'w', "wool", {0.92, 0.92, 0.9}, L::Matte, 0}, {'l', "leg", {0.78, 0.68, 0.58}, L::Matte, 0}, {'s', "head", {0.76, 0.64, 0.54}, L::Matte, 0},
			                         {'k', "eye", {0.05, 0.05, 0.05}, L::Matte, 0}, {'n', "nose", {0.6, 0.45, 0.4}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Explorer: {
			Voxels v(16, 32, 8);
			v.fill(4, 6, 2, 4, 6, 4, 'p'); v.fill(8, 6, 2, 4, 6, 4, 'p');     // legs (trousers)
			v.fill(4, 0, 2, 4, 6, 4, 'h'); v.fill(8, 0, 2, 4, 6, 4, 'h');     // legs (shoes)
			v.fill(4, 12, 2, 8, 12, 4, 'c');                                    // shirt
			v.fill(0, 12, 2, 4, 12, 4, 'k'); v.fill(12, 12, 2, 4, 12, 4, 'k');   // arms (skin)
			v.fill(0, 20, 2, 4, 4, 4, 'c'); v.fill(12, 20, 2, 4, 4, 4, 'c');   // sleeves
			v.fill(4, 24, 0, 8, 8, 8, 'k');                                     // head
			v.fill(4, 30, 0, 8, 2, 8, 'r');                                     // hair on top
			v.fill(4, 28, 0, 8, 2, 1, 'r');
			v.fill(5, 27, 7, 1, 1, 1, 'w'); v.fill(6, 27, 7, 1, 1, 1, 'e');     // eyes
			v.fill(10, 27, 7, 1, 1, 1, 'e'); v.fill(9, 27, 7, 1, 1, 1, 'w');
			v.fill(6, 25, 7, 4, 1, 1, 'm');                                     // mouth
			return make("Explorer", v, {{'p', "trousers", {0.24, 0.26, 0.62}, L::Matte, 0}, {'h', "shoes", {0.42, 0.42, 0.44}, L::Matte, 0}, {'c', "shirt", {0.2, 0.68, 0.72}, L::Matte, 0},
			                            {'k', "skin", {0.78, 0.58, 0.44}, L::Matte, 0}, {'r', "hair", {0.28, 0.17, 0.09}, L::Matte, 0}, {'w', "eye white", {0.92, 0.92, 0.92}, L::Matte, 0},
			                            {'e', "eye", {0.25, 0.28, 0.65}, L::Matte, 0}, {'m', "mouth", {0.5, 0.28, 0.2}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Skeleton: {
			Voxels v(14, 32, 8);
			v.fill(4, 0, 3, 2, 12, 2, 'b'); v.fill(8, 0, 3, 2, 12, 2, 'b');      // legs
			v.fill(3, 12, 2, 8, 12, 4, 'b');                                      // ribs
			for (int y : {14, 17, 20}) v.fill(3, y, 5, 8, 1, 1, 'd');             // gaps between the ribs
			v.fill(1, 12, 3, 2, 12, 2, 'b'); v.fill(11, 12, 3, 2, 12, 2, 'b');    // arms
			v.fill(3, 24, 0, 8, 8, 8, 'b');                                       // skull
			v.fill(4, 27, 7, 2, 2, 1, 'd'); v.fill(8, 27, 7, 2, 2, 1, 'd');       // eyes
			v.fill(5, 25, 7, 4, 1, 1, 'd');                                       // mouth
			return make("Skeleton", v, {{'b', "bone", {0.84, 0.84, 0.8}, L::Matte, 0}, {'d', "dark", {0.12, 0.12, 0.12}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::OakTree: {
			Voxels v(5, 7, 5);
			v.fill(2, 0, 2, 1, 5, 1, 'b');
			v.fill(0, 3, 0, 5, 2, 5, 'l');
			v.fill(2, 3, 2, 1, 2, 1, 'b');
			v.fill(1, 5, 1, 3, 1, 3, 'l');
			v.fill(2, 6, 1, 1, 1, 3, 'l'); v.fill(1, 6, 2, 3, 1, 1, 'l');
			return make("Oak tree", v, {{'b', "trunk", {0.34, 0.23, 0.12}, L::Matte, 0}, {'l', "leaves", {0.16, 0.46, 0.13}, L::Matte, 0}}, kBlock);
		}
		case BlockyKind::Cottage: {
			Voxels v(9, 8, 9);
			v.fill(1, 0, 1, 7, 1, 7, 's');                                  // floor
			for (int x = 1; x <= 7; ++x)
				for (int z = 1; z <= 7; ++z)
					if (x == 1 || x == 7 || z == 1 || z == 7) v.fill(x, 1, z, 1, 3, 1, 'p');   // walls
			for (int x : {1, 7})
				for (int z : {1, 7}) v.fill(x, 0, z, 1, 4, 1, 'l');         // corner posts
			v.fill(4, 1, 7, 1, 2, 1, 'd');                                  // door
			for (const auto& w : std::vector<std::pair<int, int>>{{2, 7}, {6, 7}, {1, 4}, {7, 4}, {4, 1}}) v.fill(w.first, 2, w.second, 1, 1, 1, 'g');   // windows
			v.fill(0, 4, 0, 9, 1, 9, 'r');                                  // the roof, stepping in
			v.fill(0, 5, 1, 9, 1, 7, 'r'); v.fill(0, 6, 2, 9, 1, 5, 'r'); v.fill(0, 7, 3, 9, 1, 3, 'r');
			return make("Cottage", v, {{'s', "floor", {0.5, 0.5, 0.5}, L::Matte, 0}, {'p', "wall", {0.66, 0.5, 0.3}, L::Matte, 0}, {'l', "post", {0.32, 0.22, 0.11}, L::Matte, 0},
			                           {'d', "door", {0.38, 0.25, 0.12}, L::Matte, 0}, {'g', "window", {1, 1, 1}, L::Glass, 0}, {'r', "roof", {0.6, 0.22, 0.15}, L::Matte, 0}}, kBlock);
		}
		case BlockyKind::Torch: {
			Voxels v(2, 10, 2);
			v.fill(0, 0, 0, 2, 9, 2, 's');
			v.fill(0, 9, 0, 2, 1, 2, 'f');
			return make("Torch", v, {{'s', "stick", {0.4, 0.27, 0.13}, L::Matte, 0}, {'f', "flame", {1.0, 0.72, 0.22}, L::Glow, 14.0}}, kPx);
		}
		case BlockyKind::Chest: {
			Voxels v(14, 14, 14);
			v.fill(0, 0, 0, 14, 10, 14, 'b');
			v.fill(0, 10, 0, 14, 4, 14, 'l');
			for (int x = 0; x < 14; ++x)
				for (int z = 0; z < 14; ++z)
					if (x == 0 || x == 13 || z == 0 || z == 13) v.at(x, 9, z) = 'd';
			v.fill(6, 8, 13, 2, 4, 1, 'm');
			return make("Chest", v, {{'b', "body", {0.55, 0.37, 0.17}, L::Matte, 0}, {'l', "lid", {0.62, 0.43, 0.2}, L::Matte, 0}, {'d', "band", {0.3, 0.19, 0.09}, L::Matte, 0},
			                         {'m', "latch", {0.72, 0.72, 0.74}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::CraftingTable: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 'p');
			v.fill(0, 0, 0, 16, 3, 16, 'd');
			v.fill(0, 15, 0, 16, 1, 16, 't');
			v.fill(2, 15, 2, 4, 1, 4, 'p'); v.fill(10, 15, 10, 4, 1, 4, 'p'); v.fill(10, 15, 2, 4, 1, 4, 'p'); v.fill(2, 15, 10, 4, 1, 4, 'p');
			return make("Crafting table", v, {{'p', "planks", {0.66, 0.5, 0.3}, L::Matte, 0}, {'d', "base", {0.34, 0.22, 0.1}, L::Matte, 0}, {'t', "top", {0.42, 0.28, 0.14}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Furnace: {
			Voxels v(16, 16, 16);
			v.fill(0, 0, 0, 16, 16, 16, 's');
			v.fill(0, 14, 0, 16, 2, 16, 'd');
			v.fill(4, 3, 15, 8, 6, 1, 'k');
			v.fill(5, 3, 15, 6, 2, 1, 'f');
			return make("Furnace", v, {{'s', "stone", {0.52, 0.52, 0.52}, L::Matte, 0}, {'d', "top", {0.4, 0.4, 0.4}, L::Matte, 0}, {'k', "opening", {0.08, 0.08, 0.08}, L::Matte, 0},
			                           {'f', "fire", {1.0, 0.5, 0.1}, L::Glow, 7.0}}, kPx);
		}
		case BlockyKind::Bed: {
			Voxels v(16, 9, 32);
			for (int x : {0, 14})
				for (int z : {0, 30}) v.fill(x, 0, z, 2, 2, 2, 'p');
			v.fill(0, 2, 0, 16, 2, 32, 'p');
			v.fill(0, 4, 0, 16, 3, 32, 'w');
			v.fill(0, 4, 12, 16, 4, 20, 'r');
			v.fill(2, 7, 1, 12, 2, 8, 'w');
			return make("Bed", v, {{'p', "frame", {0.62, 0.46, 0.26}, L::Matte, 0}, {'w', "sheet", {0.9, 0.9, 0.9}, L::Matte, 0}, {'r', "blanket", {0.74, 0.1, 0.1}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Poppy:
		case BlockyKind::Dandelion: {
			Voxels v(6, 12, 6);
			v.fill(2, 0, 2, 2, 8, 2, 's');
			v.fill(0, 8, 0, 6, 4, 6, 'f');
			v.fill(2, 11, 2, 2, 1, 2, 'c');
			const bool poppy = kind == BlockyKind::Poppy;
			return make(poppy ? "Poppy" : "Dandelion", v,
			            {{'s', "stem", {0.2, 0.55, 0.15}, L::Matte, 0}, {'f', "petals", poppy ? Rgb{0.86, 0.1, 0.1} : Rgb{0.96, 0.82, 0.12}, L::Matte, 0},
			             {'c', "centre", poppy ? Rgb{0.15, 0.1, 0.06} : Rgb{0.9, 0.55, 0.08}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::Fence: {
			Voxels v(16, 16, 4);
			v.fill(6, 0, 0, 4, 16, 4, 'p');
			v.fill(0, 5, 1, 16, 3, 2, 'p'); v.fill(0, 11, 1, 16, 3, 2, 'p');
			return make("Fence", v, {{'p', "wood", {0.62, 0.46, 0.26}, L::Matte, 0}}, kPx);
		}
		case BlockyKind::RainbowWool: {
			Voxels v(8, 1, 1);
			const char keys[8] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
			for (int i = 0; i < 8; ++i) v.at(i, 0, 0) = keys[i];
			return make("Wool", v, {{'a', "red", {0.84, 0.14, 0.14}, L::Matte, 0}, {'b', "orange", {0.92, 0.5, 0.1}, L::Matte, 0}, {'c', "yellow", {0.95, 0.88, 0.2}, L::Matte, 0},
			                        {'d', "lime", {0.5, 0.8, 0.15}, L::Matte, 0}, {'e', "cyan", {0.15, 0.7, 0.8}, L::Matte, 0}, {'f', "blue", {0.2, 0.3, 0.82}, L::Matte, 0},
			                        {'g', "purple", {0.56, 0.2, 0.7}, L::Matte, 0}, {'h', "pink", {0.96, 0.56, 0.72}, L::Matte, 0}}, kBlock);
		}
		case BlockyKind::PurplePortal: {
			Voxels v(4, 5, 1);
			v.fill(0, 0, 0, 4, 5, 1, 'o');
			v.fill(1, 1, 0, 2, 3, 1, 'p');
			return make("Portal", v, {{'o', "frame", {0.09, 0.06, 0.16}, L::Matte, 0}, {'p', "glow", {0.52, 0.12, 0.85}, L::Glow, 4.0}}, kBlock);
		}
	}
	return {};
}

}  // namespace scene_doc
