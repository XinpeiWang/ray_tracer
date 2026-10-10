#pragma once
// scene_array.h -- making many objects from one for the Scene Builder: a numbered copy, an array of copies (a grid or a ring) and a random scatter.
//
// Blender's Array modifier and particle scatter, in the Builder's own terms: every function takes the original and gives back NEW ordinary objects, which the
// caller adds to the document as one undo step; the original is never changed. A copy is of a UNIT: the object you selected (the anchor) and any companions that go
// with it, such as the trunk and crown of a tree or the parts of a table. Every copy keeps the parts' places relative to the anchor; a ring turns the whole unit, a
// scatter scales and turns it as one. The copies are plain objects (move, recolour or delete them one by one), and the numbers are deterministic: the same seed gives
// the same scatter on every platform, because the random numbers come from a small generator written out here instead of the library's distributions (which differ
// between standard libraries).
//
// std-only, like scene_document.h.

#include "scene_document.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace scene_doc {

// The most copies one array or scatter makes, and the most objects it adds in all (a copy of a three-part tree is three objects). Each object is a few hundred
// bytes of JSON in every undo step and a shape in every view.
constexpr int kMaxArrayCopies = 500;
constexpr int kMaxArrayObjects = 2000;

namespace array_detail {

constexpr double kPi = 3.14159265358979323846;

inline bool nameTaken(const std::string& name, const std::vector<Object>& objects) {
	for (const Object& o : objects)
		if (o.name == name) return true;
	return false;
}

// "Chair 3" -> "Chair", "Chair copy" -> "Chair": what a numbered copy is numbered from.
inline std::string baseNameOf(std::string name) {
	const std::string copy = " copy";
	if (name.size() > copy.size() && name.compare(name.size() - copy.size(), copy.size(), copy) == 0) name.erase(name.size() - copy.size());
	std::size_t end = name.size();
	while (end > 0 && std::isdigit(static_cast<unsigned char>(name[end - 1]))) --end;
	if (end < name.size() && end > 0 && name[end - 1] == ' ') name.erase(end - 1);
	return name;
}

// SplitMix64: a few lines, the same numbers everywhere.
class Rng {
public:
	explicit Rng(std::uint32_t seed) : state_(static_cast<std::uint64_t>(seed) * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull) {}
	std::uint64_t next() {
		state_ += 0x9E3779B97F4A7C15ull;
		std::uint64_t z = state_;
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }   // [0, 1)
	double range(double lo, double hi) { return lo + (hi - lo) * uniform(); }
private:
	std::uint64_t state_;
};

// The names in use, kept as a set so an array of hundreds of copies does not rescan the document for every name: the same answers as uniqueObjectName() below.
class NameBook {
public:
	explicit NameBook(const std::vector<Object>& existing) {
		for (const Object& o : existing) used_.insert(o.name);
	}
	std::string unique(const std::string& wanted) {
		const std::string base = baseNameOf(wanted);
		int& next = next_[base];   // names are only ever added, so the first free number never goes back down
		if (next < 2) next = 2;
		for (;; ++next) {
			std::string candidate = base + " " + std::to_string(next);
			if (used_.insert(candidate).second) { ++next; return candidate; }
		}
	}
private:
	std::unordered_set<std::string> used_;
	std::unordered_map<std::string, int> next_;
};

}  // namespace array_detail

// A name nobody uses yet, numbered from the source's: "Chair" gives "Chair 2" (then "Chair 3"...), and "Chair 2" gives the next free number rather than "Chair 2 copy".
// `pending` are objects about to be added together with this one.
inline std::string uniqueObjectName(const std::string& wanted, const std::vector<Object>& existing, const std::vector<Object>& pending = {}) {
	const std::string base = array_detail::baseNameOf(wanted);
	for (int n = 2;; ++n) {
		const std::string candidate = base + " " + std::to_string(n);
		if (!array_detail::nameTaken(candidate, existing) && !array_detail::nameTaken(candidate, pending)) return candidate;
	}
}

// Multiplies every length of the object by `s` (a mesh's scale included). Its position is not touched.
inline void scaleObject(Object& o, double s) {
	o.radius *= s;
	o.radius2 *= s;
	o.height *= s;
	o.size.x *= s;
	o.size.y *= s;
	o.size.z *= s;
	o.meshScale *= s;
}

// The radius of the object's footprint on the floor (what stops two trees standing in each other): sideways reach only, whatever the height.
inline double footprintRadius(const Object& o) {
	switch (o.shape) {
		case ShapeKind::Box:
		case ShapeKind::Wedge:
		case ShapeKind::Stairs:
		case ShapeKind::Quad:
		case ShapeKind::Pyramid: return 0.5 * std::sqrt(o.size.x * o.size.x + o.size.z * o.size.z);
		case ShapeKind::Mesh: return std::fabs(o.meshScale);
		case ShapeKind::Torus: return std::fabs(o.radius) + std::fabs(o.radius2);
		default: return std::fabs(o.radius);   // sphere, disk, cylinder, cone, capsule, dome, tube
	}
}

// A copy for the Duplicate button: numbered, and moved sideways by about the object's own width so it does not land inside it (snapped to the 0.25 grid).
inline Object duplicateOf(const Object& src, const std::vector<Object>& existing) {
	Object o = src;
	o.name = uniqueObjectName(src.name, existing);
	const double width = std::min(std::max(2.0 * footprintRadius(src) * 1.1, 0.5), 4.0);
	o.position.x += std::round(width * 4.0) / 4.0;
	return o;
}

// ---- units: the selected object and the parts that go with it -------------------------------------------------------------------------

// How many copies an array may make of a unit of `parts` objects, within the limits.
inline int unitLimit(std::size_t parts) {
	return std::max(1, std::min(kMaxArrayCopies, kMaxArrayObjects / static_cast<int>(std::max<std::size_t>(parts, 1))));
}

namespace array_detail {

struct Placement {
	Float3 anchor{};            // where the unit's anchor goes
	double turn = 0.0;          // degrees about the vertical, round the anchor
	double scale = 1.0;
	bool onGround = false;      // scale heights about `groundY` instead of moving the unit by the anchor's shift
	double groundY = 0.0;
	double tiltX = 0.0, tiltZ = 0.0;   // degrees added to every part's rotation about X and Z
};

// One copy of the unit at `at`, its parts named apart from everything `names` knows; appended to `out`.
inline void placeUnit(const std::vector<Object>& unit, const Placement& at, NameBook& names, std::vector<Object>& out) {
	const Float3 a = unit[0].position;
	const double t = at.turn * kPi / 180.0, c = std::cos(t), s = std::sin(t);
	for (const Object& part : unit) {
		Object o = part;
		const double dx = part.position.x - a.x, dz = part.position.z - a.z;
		scaleObject(o, at.scale);
		o.position.x = at.anchor.x + (dx * c + dz * s) * at.scale;   // a turn by t about Y: x' = x cos t + z sin t, z' = -x sin t + z cos t
		o.position.z = at.anchor.z + (-dx * s + dz * c) * at.scale;
		o.position.y = at.onGround ? at.groundY + (part.position.y - at.groundY) * at.scale : part.position.y + (at.anchor.y - a.y);
		o.rotation.y = part.rotation.y + at.turn;
		o.rotation.x = part.rotation.x + at.tiltX;
		o.rotation.z = part.rotation.z + at.tiltZ;
		o.name = names.unique(part.name);
		out.push_back(o);
	}
}

// The radius round the anchor on the floor that the whole unit covers.
inline double unitFootprint(const std::vector<Object>& unit) {
	double r = 0.0;
	for (const Object& part : unit)
		r = std::max(r, std::hypot(part.position.x - unit[0].position.x, part.position.z - unit[0].position.z) + footprintRadius(part));
	return r;
}

}  // namespace array_detail

// Which of the document's other objects most likely belong with `source` (the parts of the same prop): the same first word in the name ("Tree trunk" and "Tree
// crown 2" share "Tree") and close by. The Array window ticks these first; the user can change the choice. Returns indices into `all`, `sourceIndex` excluded.
inline std::vector<int> likelyCompanions(const Object& source, const std::vector<Object>& all, int sourceIndex) {
	const auto firstWord = [](const std::string& name) {
		std::string w = name.substr(0, name.find(' '));
		for (char& c : w) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
		return w;
	};
	std::vector<int> out;
	const std::string word = firstWord(source.name);
	const double reach = 3.0 + boundingRadius(source);
	for (int i = 0; i < static_cast<int>(all.size()); ++i) {
		if (i == sourceIndex || firstWord(all[i].name) != word) continue;
		const Object& o = all[i];
		const double d = std::sqrt((o.position.x - source.position.x) * (o.position.x - source.position.x) + (o.position.y - source.position.y) * (o.position.y - source.position.y) +
		                           (o.position.z - source.position.z) * (o.position.z - source.position.z));
		if (d <= reach) out.push_back(i);
	}
	return out;
}

// ---- a grid (a line, a rectangle or a block) of copies --------------------------------------------------------------------------------

struct GridParams {
	int countX = 3, countY = 1, countZ = 1;   // copies along each axis, the original included
	Float3 spacing{2.0, 2.0, 2.0};            // the distance between neighbours along X, Y and Z
};

// How many new copies a grid asks for (countX * countY * countZ - 1), before the limit.
inline long long gridRequested(const GridParams& p) {
	return static_cast<long long>(std::max(p.countX, 1)) * std::max(p.countY, 1) * std::max(p.countZ, 1) - 1;
}

inline std::vector<Object> makeGrid(const std::vector<Object>& unit, const GridParams& p, const std::vector<Object>& existing) {
	std::vector<Object> out;
	if (unit.empty()) return out;
	const int nx = std::max(p.countX, 1), ny = std::max(p.countY, 1), nz = std::max(p.countZ, 1), limit = unitLimit(unit.size());
	int made = 0;
	array_detail::NameBook names(existing);
	for (int k = 0; k < nz; ++k)
		for (int j = 0; j < ny; ++j)
			for (int i = 0; i < nx; ++i) {
				if (i == 0 && j == 0 && k == 0) continue;   // the original
				if (made >= limit) return out;
				array_detail::Placement at;
				at.anchor = {unit[0].position.x + i * p.spacing.x, unit[0].position.y + j * p.spacing.y, unit[0].position.z + k * p.spacing.z};
				array_detail::placeUnit(unit, at, names, out);
				++made;
			}
	return out;
}
inline std::vector<Object> makeGrid(const Object& src, const GridParams& p, const std::vector<Object>& existing) { return makeGrid(std::vector<Object>{src}, p, existing); }

// ---- a ring of copies -------------------------------------------------------------------------------------------------------------------

struct RingParams {
	int count = 8;                  // copies in the whole ring, the original included (so count - 1 are new)
	double radius = 3.0;            // distance from the centre on the floor plane
	double centreX = 0.0, centreZ = 0.0;
	double startAngle = 0.0;        // degrees added to every copy's angle (0 = the first copy is one step on from the original)
	bool turnWithRing = true;       // turn each copy as far as it goes round, so it faces the centre the way the original does
};

// The angle of a point seen from the ring's centre, in degrees: 0 along +X, 90 along +Z. The ring's copies are spaced from the original's own angle.
inline double angleAround(double centreX, double centreZ, double x, double z) {
	return std::atan2(z - centreZ, x - centreX) * 180.0 / array_detail::kPi;
}

inline std::vector<Object> makeRing(const std::vector<Object>& unit, const RingParams& p, const std::vector<Object>& existing) {
	std::vector<Object> out;
	if (unit.empty()) return out;
	const int n = std::max(p.count, 1), limit = unitLimit(unit.size());
	const double a0 = angleAround(p.centreX, p.centreZ, unit[0].position.x, unit[0].position.z);
	array_detail::NameBook names(existing);
	for (int k = 1; k < n && k <= limit; ++k) {
		const double step = p.startAngle + 360.0 * k / n;   // degrees on from the original, towards +Z from +X
		const double a = (a0 + step) * array_detail::kPi / 180.0;
		array_detail::Placement at;
		at.anchor = {p.centreX + p.radius * std::cos(a), unit[0].position.y, p.centreZ + p.radius * std::sin(a)};
		at.turn = p.turnWithRing ? -step : 0.0;   // a turn about +Y by phi moves a point's angle by -phi
		array_detail::placeUnit(unit, at, names, out);
	}
	return out;
}
inline std::vector<Object> makeRing(const Object& src, const RingParams& p, const std::vector<Object>& existing) { return makeRing(std::vector<Object>{src}, p, existing); }

// ---- a random scatter -------------------------------------------------------------------------------------------------------------------

struct ScatterParams {
	int count = 20;
	bool disk = true;                       // a disk (width is its diameter) or a rectangle (width by depth)
	double centreX = 0.0, centreZ = 0.0;
	double width = 10.0, depth = 10.0;
	double groundY = 0.0;                   // the floor the copies stand on: a scaled copy keeps its own distance above it
	double scaleMin = 0.8, scaleMax = 1.2;  // each copy is scaled by a random factor in this range
	bool randomSpin = true;                 // a random turn about the vertical, the whole unit together (best for upright objects)
	double tiltDegrees = 0.0;               // a random lean of up to this many degrees about X and Z
	bool keepApart = true;                  // no two footprints (and none and the original) overlap
	double extraGap = 0.0;                  // and this much clear floor between them
	std::uint32_t seed = 1;
};

struct ScatterResult {
	std::vector<Object> objects;
	int requested = 0;   // how many copies were asked for (within the limit); fewer are placed when keepApart leaves no room
	int placed = 0;      // how many copies were made (objects.size() divided by the unit's size)
};

inline ScatterResult scatter(const std::vector<Object>& unit, const ScatterParams& p, const std::vector<Object>& existing) {
	ScatterResult result;
	if (unit.empty()) return result;
	result.requested = std::min(std::max(p.count, 0), unitLimit(unit.size()));
	array_detail::Rng rng(p.seed);
	const double scaleLo = std::min(p.scaleMin, p.scaleMax), scaleHi = std::max(p.scaleMin, p.scaleMax);
	const double baseFootprint = array_detail::unitFootprint(unit);
	struct Placed { double x, z, r; };
	std::vector<Placed> placed;
	placed.push_back({unit[0].position.x, unit[0].position.z, baseFootprint});   // the original is in the way too
	array_detail::NameBook names(existing);
	constexpr int kAttempts = 40;
	for (int i = 0; i < result.requested; ++i) {
		const double s = rng.range(scaleLo, scaleHi);
		const double r = baseFootprint * s;
		bool found = false;
		double x = 0.0, z = 0.0;
		for (int attempt = 0; attempt < kAttempts && !found; ++attempt) {
			if (p.disk) {
				const double radius = 0.5 * p.width * std::sqrt(rng.uniform());
				const double angle = rng.range(0.0, 2.0 * array_detail::kPi);
				x = p.centreX + radius * std::cos(angle);
				z = p.centreZ + radius * std::sin(angle);
			} else {
				x = p.centreX + (rng.uniform() - 0.5) * p.width;
				z = p.centreZ + (rng.uniform() - 0.5) * p.depth;
			}
			found = true;
			if (p.keepApart) {
				for (const Placed& q : placed) {
					const double need = q.r + r + p.extraGap;
					if ((x - q.x) * (x - q.x) + (z - q.z) * (z - q.z) < need * need) { found = false; break; }
				}
			}
		}
		// The random numbers below are drawn whether or not a spot was found, so a copy's turn and lean do not depend on how crowded the earlier ones were.
		const double spin = rng.range(0.0, 360.0), tiltX = rng.range(-1.0, 1.0), tiltZ = rng.range(-1.0, 1.0);
		if (!found) continue;
		array_detail::Placement at;
		at.anchor = {x, 0.0, z};
		at.scale = s;
		at.onGround = true;
		at.groundY = p.groundY;
		at.turn = p.randomSpin ? spin : 0.0;
		if (p.tiltDegrees > 0.0) {
			at.tiltX = tiltX * p.tiltDegrees;
			at.tiltZ = tiltZ * p.tiltDegrees;
		}
		array_detail::placeUnit(unit, at, names, result.objects);
		placed.push_back({x, z, r});
		++result.placed;
	}
	return result;
}
inline ScatterResult scatter(const Object& src, const ScatterParams& p, const std::vector<Object>& existing) { return scatter(std::vector<Object>{src}, p, existing); }

}  // namespace scene_doc
