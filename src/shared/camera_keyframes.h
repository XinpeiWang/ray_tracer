#pragma once
// camera_keyframes.h -- a camera path made of places the camera must pass through (the Scene Builder's saved views), for the renderer's --video mode
// (--camera-keyframes FILE) and the Scene Builder's flythrough. std-only, so the launcher, the GUI and the unit tests share it.
//
// The file is plain text, one thing per line, '#' starts a comment:
//
//     ease 1                           # 1 (the default): start and end gently; 0: move at the path's own pace from the first frame to the last
//     key  px py pz   tx ty tz         # a camera place: where it is, and what it looks at (two or more, in the order they are visited)
//     key  px py pz   tx ty tz  fov    # ... and its vertical field of view in degrees
//     key  px py pz   tx ty tz  fov lensRadius focusDistance    # ... and its lens: the radius of the aperture (0: sharp everywhere) and the distance in focus
//
// The first frame is exactly the first key and the last frame exactly the last. In between the camera goes through every key on a smooth curve (a Catmull-Rom
// spline through the places, and another through the targets), spending time on each stretch in proportion to how far it is (the distance the camera and its
// target travel), so the speed is about the same all the way instead of racing through a short stretch and crawling through a long one. A key twice in a row is a
// pause-free duplicate (the stretch between them has no length and takes no time).
//
// The field of view and the lens are on the same curve when EVERY key gives them (a view that does not is the scene's own, so a path where only some keys say is
// not mixed: those values are then left to the scene, and at() reports them as not given: fov 0, lensRadius -1, focus 0).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace camera_keyframes {

struct Key {
	double pos[3] = {0, 0, 0};      // where the camera is
	double target[3] = {0, 0, 1};   // what it looks at
	double fov = 0;                 // vertical field of view in degrees; 0: not given
	double lensRadius = -1;         // aperture radius; < 0: not given (0 is a real value: no depth of field)
	double focus = 0;               // distance in focus; 0: not given
	bool hasLens() const { return lensRadius >= 0 && focus > 0; }
};

struct Path {
	std::vector<Key> keys;
	bool ease = true;
};

struct ParseResult {
	bool ok = false;
	std::string error;   // when !ok: what is wrong, with the line number
	bool unreadable = false;   // load(): the file could not be opened (as opposed to its contents being wrong)
	Path path;
};

namespace detail {

inline bool finiteNumber(double v) { return std::isfinite(v); }

inline double distance(const double a[3], const double b[3]) { return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])); }

// A Catmull-Rom curve between p1 and p2 (u in [0, 1]) with p0 before and p3 after; at the ends a missing neighbour is the end point reflected, which makes the end
// tangent the chord.
inline double catmullRom(double p0, double p1, double p2, double p3, double u) {
	const double u2 = u * u, u3 = u2 * u;
	return 0.5 * ((2.0 * p1) + (-p0 + p2) * u + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2 + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3);
}

// The key a neighbour of `from` would be, past the end of the path on the side away from `towards`: `from` reflected through its other neighbour, in every field (which
// makes the end tangent the chord).
inline Key reflected(const Key& from, const Key& towards) {
	Key r;
	for (int i = 0; i < 3; ++i) {
		r.pos[i] = 2 * from.pos[i] - towards.pos[i];
		r.target[i] = 2 * from.target[i] - towards.target[i];
	}
	r.fov = 2 * from.fov - towards.fov;
	r.lensRadius = 2 * from.lensRadius - towards.lensRadius;
	r.focus = 2 * from.focus - towards.focus;
	return r;
}

}  // namespace detail

inline ParseResult parse(const std::string& text) {
	ParseResult r;
	std::istringstream in(text);
	in.imbue(std::locale::classic());
	std::string line;
	int lineNumber = 0;
	while (std::getline(in, line)) {
		++lineNumber;
		const std::size_t hash = line.find('#');
		if (hash != std::string::npos) line.erase(hash);
		std::istringstream words(line);
		words.imbue(std::locale::classic());
		std::string word;
		if (!(words >> word)) continue;   // blank or only a comment
		const auto fail = [&](const std::string& why) {
			r.ok = false;
			r.error = "line " + std::to_string(lineNumber) + ": " + why;
			r.path.keys.clear();
			return r;
		};
		if (word == "ease") {
			int v = 1;
			if (!(words >> v) || (v != 0 && v != 1)) return fail("'ease' is 0 or 1");
			r.path.ease = v == 1;
		} else if (word == "key") {
			Key k;
			std::vector<double> v;
			double x;
			while (words >> x) {
				if (!detail::finiteNumber(x)) return fail("'key' has a number that is not finite");
				v.push_back(x);
			}
			if (!words.eof()) return fail("'key' has something that is not a number");
			if (v.size() != 6 && v.size() != 7 && v.size() != 9) return fail("'key' needs 6 numbers (px py pz tx ty tz), or 7 (and fov), or 9 (and fov lensRadius focusDistance)");
			for (int i = 0; i < 3; ++i) {
				k.pos[i] = v[i];
				k.target[i] = v[3 + i];
			}
			if (v.size() >= 7) {
				if (!(v[6] > 0.0 && v[6] < 180.0)) return fail("the field of view is between 0 and 180 degrees");
				k.fov = v[6];
			}
			if (v.size() == 9) {
				if (!(v[7] >= 0.0)) return fail("the lens radius is 0 or more");
				if (!(v[8] > 0.0)) return fail("the focus distance is more than 0");
				k.lensRadius = v[7];
				k.focus = v[8];
			}
			if (detail::distance(k.pos, k.target) < 1e-9) return fail("a camera cannot look at its own place");
			r.path.keys.push_back(k);
		} else {
			return fail("unknown word '" + word + "' (expected 'key' or 'ease')");
		}
	}
	if (r.path.keys.size() < 2) {
		r.error = "a path needs at least two 'key' lines (it has " + std::to_string(r.path.keys.size()) + ")";
		r.path.keys.clear();
		return r;
	}
	r.ok = true;
	return r;
}

// parse() of a file's contents; the error names the file.
inline ParseResult load(const std::string& file) {
	std::ifstream in(file, std::ios::binary);
	if (!in) {
		ParseResult r;
		r.error = "cannot read the camera keyframes file " + file;
		r.unreadable = true;
		return r;
	}
	std::ostringstream text;
	text << in.rdbuf();
	ParseResult r = parse(text.str());
	if (!r.ok) r.error = file + ": " + r.error;
	return r;
}

inline std::string toText(const Path& p) {
	std::ostringstream out;
	out.imbue(std::locale::classic());
	out.precision(17);
	out << "# A camera path: the camera goes through each key in turn (see src/shared/camera_keyframes.h).\n";
	out << "ease " << (p.ease ? 1 : 0) << "\n";
	for (const Key& k : p.keys) {
		out << "key " << k.pos[0] << " " << k.pos[1] << " " << k.pos[2] << "  " << k.target[0] << " " << k.target[1] << " " << k.target[2];
		if (k.fov > 0) out << "  " << k.fov;
		if (k.fov > 0 && k.hasLens()) out << " " << k.lensRadius << " " << k.focus;
		out << "\n";
	}
	return out.str();
}

// How far along the path (0 first frame, 1 last) a frame is: frame 0 of `total` is the start and frame total-1 the end; one frame is the start.
inline double progressOf(int frame, int total) { return total <= 1 ? 0.0 : std::min(1.0, std::max(0.0, static_cast<double>(frame) / static_cast<double>(total - 1))); }

// The camera at progress t in [0, 1] (clamped). A path with fewer than two keys is its first key (or a camera at the origin looking down +Z if empty).
inline Key at(const Path& path, double t) {
	const std::vector<Key>& k = path.keys;
	if (k.empty()) return Key{};
	if (k.size() == 1) return k[0];
	t = std::min(1.0, std::max(0.0, t));
	if (path.ease) t = t * t * (3.0 - 2.0 * t);   // smoothstep: the ends stay where they are
	const std::size_t segments = k.size() - 1;
	std::vector<double> length(segments);
	double total = 0.0;
	for (std::size_t i = 0; i < segments; ++i) {
		length[i] = detail::distance(k[i].pos, k[i + 1].pos) + detail::distance(k[i].target, k[i + 1].target);
		total += length[i];
	}
	if (total <= 1e-12) return k.front();
	// The segment this progress falls in, and how far into it.
	double along = t * total;
	std::size_t seg = 0;
	while (seg + 1 < segments && along > length[seg]) {
		along -= length[seg];
		++seg;
	}
	// (A zero-length stretch is skipped: along <= 0 there leaves seg where the loop stopped, which has length unless it is the last one.)
	const double u = length[seg] > 1e-12 ? std::min(1.0, std::max(0.0, along / length[seg])) : 1.0;
	const Key& p1 = k[seg];
	const Key& p2 = k[seg + 1];
	const Key p0 = seg > 0 ? k[seg - 1] : detail::reflected(p1, p2);
	const Key p3 = seg + 2 < k.size() ? k[seg + 2] : detail::reflected(p2, p1);
	Key out;
	for (int a = 0; a < 3; ++a) {
		out.pos[a] = detail::catmullRom(p0.pos[a], p1.pos[a], p2.pos[a], p3.pos[a], u);
		out.target[a] = detail::catmullRom(p0.target[a], p1.target[a], p2.target[a], p3.target[a], u);
	}
	// The field of view and lens ride the same curve, if every key gives them (the curve can overshoot between keys, so they are kept to what a camera can do).
	const auto all = [&k](auto given) { return std::all_of(k.begin(), k.end(), given); };
	if (all([](const Key& key) { return key.fov > 0; })) {
		out.fov = std::min(170.0, std::max(1.0, detail::catmullRom(p0.fov, p1.fov, p2.fov, p3.fov, u)));
		if (all([](const Key& key) { return key.hasLens(); })) {
			out.lensRadius = std::max(0.0, detail::catmullRom(p0.lensRadius, p1.lensRadius, p2.lensRadius, p3.lensRadius, u));
			out.focus = std::max(1e-3, detail::catmullRom(p0.focus, p1.focus, p2.focus, p3.focus, u));
		}
	}
	return out;
}

}  // namespace camera_keyframes
