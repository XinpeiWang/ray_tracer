#pragma once
// scene_selection.h -- acting on SEVERAL items of a Scene Builder document at once: a set of objects and lights that is moved, copied, deleted, recoloured or grouped
// together. Like Blender's selection, with groups as the thing that makes a table (a top and four legs) one object to click.
//
// A group is just a name on each member (Object::group); objects with the same non-empty name are a group, a group has at least two members (a name left on one
// object is removed), and a group lives only in the Scene Builder's own copy of the document: the .pbrt text has no groups, so Live Preview and the renderers see the
// members as the separate objects they are.
//
// Every function here changes the Document it is given and does nothing else (no selection, no undo): the GUI wraps one call in one undo step. std-only, like
// scene_document.h, so the unit tests and the GUI use the same code.

#include "scene_array.h"
#include "scene_document.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace scene_doc {

// The items picked: indices into Document::objects and Document::lights, ascending and without repeats.
struct ItemSet {
	std::vector<int> objects;
	std::vector<int> lights;
	bool empty() const { return objects.empty() && lights.empty(); }
	std::size_t size() const { return objects.size() + lights.size(); }
};

namespace selection_detail {

inline void normalise(std::vector<int>& v, int count) {
	std::sort(v.begin(), v.end());
	v.erase(std::unique(v.begin(), v.end()), v.end());
	v.erase(std::remove_if(v.begin(), v.end(), [count](int i) { return i < 0 || i >= count; }), v.end());
}

inline bool nameTaken(const std::string& name, const std::vector<Light>& lights) {
	for (const Light& l : lights)
		if (l.name == name) return true;
	return false;
}

inline std::set<std::string> groupNamesIn(const Document& d) {
	std::set<std::string> used;
	for (const Object& o : d.objects)
		if (!o.group.empty()) used.insert(o.group);
	return used;
}

// `wanted` itself if nobody uses it, else "Base 2", "Base 3"...; the name is added to `taken`.
inline std::string pickGroupName(std::set<std::string>& taken, const std::string& wanted) {
	const std::string base = array_detail::baseNameOf(wanted.empty() ? std::string("Group") : wanted);
	std::string name = wanted.empty() ? base : wanted;
	for (int n = 2; taken.count(name); ++n) name = base + " " + std::to_string(n);
	taken.insert(name);
	return name;
}

}  // namespace selection_detail

// Cleans an index list: ascending, unique, and only indices the document has.
inline ItemSet makeItemSet(const Document& d, std::vector<int> objects, std::vector<int> lights = {}) {
	selection_detail::normalise(objects, static_cast<int>(d.objects.size()));
	selection_detail::normalise(lights, static_cast<int>(d.lights.size()));
	return ItemSet{std::move(objects), std::move(lights)};
}

// The set with every member of any group it touches added: clicking one leg picks the whole table.
inline ItemSet withGroupMates(const Document& d, ItemSet s) {
	std::set<std::string> groups;
	for (int i : s.objects)
		if (i >= 0 && i < static_cast<int>(d.objects.size()) && !d.objects[i].group.empty()) groups.insert(d.objects[i].group);
	if (groups.empty()) return s;
	for (int i = 0; i < static_cast<int>(d.objects.size()); ++i)
		if (!d.objects[i].group.empty() && groups.count(d.objects[i].group)) s.objects.push_back(i);
	selection_detail::normalise(s.objects, static_cast<int>(d.objects.size()));
	return s;
}

// A group name no object uses yet, made from `wanted` ("Table" gives "Table", then "Table 2"...).
inline std::string uniqueGroupName(const Document& d, const std::string& wanted = "Group") {
	std::set<std::string> taken = selection_detail::groupNamesIn(d);
	return selection_detail::pickGroupName(taken, wanted);
}

// A name left on a single object is not a group: removes it.
inline void tidyGroups(Document& d) {
	std::set<std::string> seen, repeated;
	for (const Object& o : d.objects) {
		if (o.group.empty()) continue;
		if (!seen.insert(o.group).second) repeated.insert(o.group);
	}
	for (Object& o : d.objects)
		if (!o.group.empty() && !repeated.count(o.group)) o.group.clear();
}

// Puts the objects in one new group (named from `wanted`) and returns its name; "" and nothing changed if fewer than two objects were given. Objects leave the groups
// they were in; a group that is left with one member is dissolved.
inline std::string groupObjects(Document& d, const std::vector<int>& objects, const std::string& wanted = "Group") {
	std::vector<int> idx = objects;
	selection_detail::normalise(idx, static_cast<int>(d.objects.size()));
	if (idx.size() < 2) return std::string();
	const std::string name = uniqueGroupName(d, wanted);
	for (int i : idx) d.objects[i].group = name;
	tidyGroups(d);
	return name;
}

// Dissolves every group the objects belong to (all of each group's members, not just the ones given); returns how many objects were freed.
inline int ungroupObjects(Document& d, const std::vector<int>& objects) {
	std::set<std::string> groups;
	for (int i : objects)
		if (i >= 0 && i < static_cast<int>(d.objects.size()) && !d.objects[i].group.empty()) groups.insert(d.objects[i].group);
	int freed = 0;
	for (Object& o : d.objects)
		if (!o.group.empty() && groups.count(o.group)) { o.group.clear(); ++freed; }
	return freed;
}

// Moves the items by `delta`; a light's target moves with its position, so a spot keeps pointing the same way.
inline void translateItems(Document& d, const ItemSet& s, const Float3& delta) {
	for (int i : s.objects) {
		if (i < 0 || i >= static_cast<int>(d.objects.size())) continue;
		d.objects[i].position.x += delta.x;
		d.objects[i].position.y += delta.y;
		d.objects[i].position.z += delta.z;
	}
	for (int i : s.lights) {
		if (i < 0 || i >= static_cast<int>(d.lights.size())) continue;
		Light& l = d.lights[i];
		l.position.x += delta.x;
		l.position.y += delta.y;
		l.position.z += delta.z;
		if (l.kind == LightKind::Spot || l.kind == LightKind::Distant) {
			l.target.x += delta.x;
			l.target.y += delta.y;
			l.target.z += delta.z;
		}
	}
}

// Copies of the items, added at the end of the document and returned as the set of the new indices. The copies keep their places relative to each other and are
// moved sideways together by about the width of what was copied (snapped to the 0.25 grid, like duplicateOf for one object), so they do not land inside the
// originals. Names are numbered ("Leg" gives "Leg 2"); every group among the originals gets a group of its own for the copies, so a copied table is a table.
inline ItemSet duplicateItems(Document& d, const ItemSet& wanted) {
	const ItemSet s = makeItemSet(d, wanted.objects, wanted.lights);
	ItemSet added;
	if (s.empty()) return added;
	double lo = 1e300, hi = -1e300;
	for (int i : s.objects) {
		const Object& o = d.objects[i];
		const double r = footprintRadius(o);
		lo = std::min(lo, o.position.x - r);
		hi = std::max(hi, o.position.x + r);
	}
	double dx = 0.5;   // a light on its own, as before
	if (!s.objects.empty()) dx = std::round(std::min(std::max((hi - lo) * 1.1, 0.5), 4.0) * 4.0) / 4.0;
	std::map<std::string, std::string> newGroup;   // each group among the originals -> the group its copies form
	std::set<std::string> takenGroups = selection_detail::groupNamesIn(d);
	array_detail::NameBook names(d.objects);
	const std::vector<Object> originals = d.objects;
	for (int i : s.objects) {
		Object o = originals[i];
		o.name = names.unique(o.name);
		o.position.x += dx;
		if (!o.group.empty()) {
			auto it = newGroup.find(o.group);
			if (it == newGroup.end()) it = newGroup.emplace(o.group, selection_detail::pickGroupName(takenGroups, o.group)).first;
			o.group = it->second;
		}
		d.objects.push_back(o);
		added.objects.push_back(static_cast<int>(d.objects.size()) - 1);
	}
	for (int i : s.lights) {
		Light l = d.lights[i];
		std::string base = array_detail::baseNameOf(l.name);
		std::string name;
		for (int n = 2;; ++n) {
			name = base + " " + std::to_string(n);
			if (!selection_detail::nameTaken(name, d.lights)) break;
		}
		l.name = name;
		l.position.x += dx;
		if (l.kind == LightKind::Spot || l.kind == LightKind::Distant) l.target.x += dx;
		d.lights.push_back(l);
		added.lights.push_back(static_cast<int>(d.lights.size()) - 1);
	}
	return added;
}

// Removes the items. Groups left with one member are dissolved.
inline void eraseItems(Document& d, const ItemSet& wanted) {
	const ItemSet s = makeItemSet(d, wanted.objects, wanted.lights);
	for (auto it = s.objects.rbegin(); it != s.objects.rend(); ++it) d.objects.erase(d.objects.begin() + *it);
	for (auto it = s.lights.rbegin(); it != s.lights.rend(); ++it) d.lights.erase(d.lights.begin() + *it);
	tidyGroups(d);
}

// Gives every object in `to` the material (and emission) of object `from`; nothing else about them changes. False for a bad index.
inline bool copyLook(Document& d, int from, const std::vector<int>& to) {
	if (from < 0 || from >= static_cast<int>(d.objects.size())) return false;
	const Object src = d.objects[from];
	for (int i : to) {
		if (i < 0 || i >= static_cast<int>(d.objects.size()) || i == from) continue;
		Object& o = d.objects[i];
		o.material = src.material;
		o.emissive = src.emissive;
		o.emission = src.emission;
		o.emissionStrength = src.emissionStrength;
		o.twoSided = src.twoSided;
	}
	return true;
}

// The copies an array made (blocks of unit.size() objects, one block per copy of the unit): when every part of the unit was in the same group, each copy is a group of
// its own, named from that group. Returns how many groups were made; nothing changes for a unit that was not one group.
inline int groupCopies(const Document& d, std::vector<Object>& copies, const std::vector<Object>& unit) {
	if (unit.size() < 2 || unit[0].group.empty()) return 0;
	for (const Object& part : unit)
		if (part.group != unit[0].group) return 0;
	std::set<std::string> taken = selection_detail::groupNamesIn(d);
	for (const Object& c : copies)
		if (!c.group.empty()) taken.insert(c.group);
	int made = 0;
	for (std::size_t from = 0; from + unit.size() <= copies.size(); from += unit.size()) {
		const std::string name = selection_detail::pickGroupName(taken, unit[0].group);
		for (std::size_t k = 0; k < unit.size(); ++k) copies[from + k].group = name;
		++made;
	}
	return made;
}

// The objects of the set as a unit for an array (scene_array.h): `primary` first (the anchor), then the others in document order.
inline std::vector<Object> unitOfObjects(const Document& d, int primary, const std::vector<int>& objects) {
	std::vector<Object> unit;
	if (primary < 0 || primary >= static_cast<int>(d.objects.size())) return unit;
	unit.push_back(d.objects[primary]);
	for (int i : objects)
		if (i != primary && i >= 0 && i < static_cast<int>(d.objects.size())) unit.push_back(d.objects[i]);
	return unit;
}

}  // namespace scene_doc
