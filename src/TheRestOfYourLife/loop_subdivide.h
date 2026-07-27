#ifndef LOOP_SUBDIVIDE_H
#define LOOP_SUBDIVIDE_H
//==============================================================================
// loop_subdivide.h -- Loop subdivision surfaces (pbrt-v4 §6.5)
//
// pbrt-v4 reference: src/pbrt/util/loopsubdiv.cpp
//
// Algorithm (Charles Loop 1987):
//   Each subdivision level replaces every triangle with 4 children:
//     - 3 corner triangles (one per original vertex)
//     - 1 central triangle (new edge midpoints)
//
//   Even vertex rule (interior, valence k):
//     p_new = (1 - k*beta) * p_old  +  beta * sum(one-ring neighbors)
//     beta = 3/(8k) for irregular (k != 6), 1/16 for regular (k == 6)
//
//   Odd vertex rule (new edge midpoints, interior edge):
//     p_new = 3/8*(v0+v1)  +  1/8*(opp0+opp1)
//
//   Boundary rules:
//     Even: p_new = 3/4*p_old + 1/8*(prev_boundary + next_boundary)
//     Odd:  p_new = 1/2*(v0 + v1)
//
//   Limit surface push (applied once after all levels):
//     Interior: weightOneRing(v, loopGamma(valence))
//     Boundary: weightBoundary(v, 1/5)
//
//   Limit normals from tangents (cross product of S and T vectors).
//
// Input/output: triangle_mesh_data (same struct as triangle.h).
// UVs are not interpolated by Loop subdivision (discarded in output).
//==============================================================================

#include "triangle.h"  // triangle_mesh_data, point3, vec3

#include <vector>
#include <map>
#include <cmath>
#include <cassert>

// ---------------------------------------------------------------------------
// Internal subdivision mesh structures
// ---------------------------------------------------------------------------
namespace loopsubdiv_detail {

struct SDFace;

struct SDVertex {
	point3 p;
	SDFace* startFace = nullptr;
	SDVertex* child   = nullptr;
	bool regular      = false;
	bool boundary     = false;

	explicit SDVertex(const point3& p_ = point3(0,0,0)) : p(p_) {}

	int valence() const;
	void one_ring(std::vector<point3>& ring) const;
};

struct SDFace {
	SDVertex* v[3]        = {nullptr,nullptr,nullptr};
	SDFace*   f[3]        = {nullptr,nullptr,nullptr};
	SDFace*   children[4] = {nullptr,nullptr,nullptr,nullptr};

	int vnum(const SDVertex* vert) const {
		for (int i = 0; i < 3; ++i)
			if (v[i] == vert) return i;
		return -1;
	}
	SDFace* next_face(SDVertex* vert) { return f[vnum(vert)]; }
	SDFace* prev_face(SDVertex* vert) { return f[(vnum(vert)+2)%3]; }
	SDVertex* next_vert(SDVertex* vert) { return v[(vnum(vert)+1)%3]; }
	SDVertex* prev_vert(SDVertex* vert) { return v[(vnum(vert)+2)%3]; }
	SDVertex* other_vert(SDVertex* v0, SDVertex* v1) {
		for (int i = 0; i < 3; ++i)
			if (v[i] != v0 && v[i] != v1) return v[i];
		return nullptr;
	}
};

struct SDEdge {
	SDVertex* v[2];
	SDFace*   f[2]    = {nullptr,nullptr};
	int       f_idx[2] = {0,0};

	SDEdge(SDVertex* v0 = nullptr, SDVertex* v1 = nullptr) {
		v[0] = std::min(v0, v1);
		v[1] = std::max(v0, v1);
	}
	bool operator<(const SDEdge& e) const {
		if (v[0] != e.v[0]) return v[0] < e.v[0];
		return v[1] < e.v[1];
	}
};

inline int SDVertex::valence() const {
	SDFace* f = startFace;
	if (!boundary) {
		int n = 1;
		while ((f = f->next_face(const_cast<SDVertex*>(this))) != startFace)
			++n;
		return n;
	} else {
		int n = 1;
		while ((f = f->next_face(const_cast<SDVertex*>(this))) != nullptr)
			++n;
		f = startFace;
		while ((f = f->prev_face(const_cast<SDVertex*>(this))) != nullptr)
			++n;
		return n + 1;
	}
}

inline void SDVertex::one_ring(std::vector<point3>& ring) const {
	ring.clear();
	if (!boundary) {
		SDFace* face = startFace;
		do {
			ring.push_back(face->next_vert(const_cast<SDVertex*>(this))->p);
			face = face->next_face(const_cast<SDVertex*>(this));
		} while (face != startFace);
	} else {
		// Walk to start of boundary
		SDFace* face = startFace;
		SDFace* f2;
		while ((f2 = face->next_face(const_cast<SDVertex*>(this))) != nullptr)
			face = f2;
		ring.push_back(face->next_vert(const_cast<SDVertex*>(this))->p);
		do {
			ring.push_back(face->prev_vert(const_cast<SDVertex*>(this))->p);
			face = face->prev_face(const_cast<SDVertex*>(this));
		} while (face != nullptr);
	}
}

// Loop beta for irregular interior vertex of valence k
inline double loop_beta(int k) {
	const double Pi = 3.14159265358979323846;
	double t = 3.0 + 2.0 * std::cos(2.0 * Pi / k);
	return (1.0 / k) * (5.0/8.0 - t*t/64.0);
}

// loopGamma for limit-surface push of interior vertex (valence k)
inline double loop_gamma(int k) {
	return 1.0 / (k + 3.0 / (8.0 * loop_beta(k)));
}

inline point3 weight_one_ring(SDVertex* vert, double beta) {
	std::vector<point3> ring;
	vert->one_ring(ring);
	int k = (int)ring.size();
	point3 p = (1.0 - k * beta) * vert->p;
	for (const auto& r : ring)
		p = p + beta * r;
	return p;
}

inline point3 weight_boundary(SDVertex* vert, double beta) {
	std::vector<point3> ring;
	vert->one_ring(ring);
	// Boundary: weight first and last neighbor (the two boundary neighbors)
	return (1.0 - 2.0*beta) * vert->p + beta * ring.front() + beta * ring.back();
}

} // namespace loopsubdiv_detail


// ---------------------------------------------------------------------------
// loop_subdivide -- main entry point
//
// Parameters:
//   mesh_in  : input triangle_mesh_data (positions + indices required;
//               normals and UVs are ignored — smooth normals are recomputed)
//   nLevels  : number of subdivision levels (0 returns a copy of input)
//
// Returns:
//   New triangle_mesh_data with smooth positions and limit-surface normals.
//   Index count = 3 * nTris_in * 4^nLevels.
// ---------------------------------------------------------------------------
inline std::shared_ptr<triangle_mesh_data> loop_subdivide(
	const triangle_mesh_data& mesh_in,
	int nLevels)
{
	using namespace loopsubdiv_detail;

	const int nVerts = (int)mesh_in.positions.size();
	const int nTris  = mesh_in.num_triangles();

	if (nVerts == 0 || nTris == 0 || nLevels < 0) {
		// Return copy
		auto out = std::make_shared<triangle_mesh_data>();
		*out = mesh_in;
		return out;
	}

	// ------------------------------------------------------------------
	// 1. Allocate SDVertex and SDFace objects using std::vector (stable ptr)
	// ------------------------------------------------------------------
	std::vector<SDVertex> vbuf(nVerts);
	std::vector<SDFace>   fbuf(nTris);

	for (int i = 0; i < nVerts; ++i)
		vbuf[i].p = mesh_in.positions[i];

	// Build face/vertex adjacency
	std::map<SDEdge, SDEdge> edge_map;  // maps canonical edge -> edge with faces

	for (int i = 0; i < nTris; ++i) {
		SDFace* face = &fbuf[i];
		for (int j = 0; j < 3; ++j) {
			int vi = mesh_in.indices[3*i + j];
			face->v[j] = &vbuf[vi];
			face->v[j]->startFace = face;
		}
		// Register edges
		for (int j = 0; j < 3; ++j) {
			SDVertex* v0 = face->v[j];
			SDVertex* v1 = face->v[(j+1)%3];
			SDEdge e(v0, v1);
			auto it = edge_map.find(e);
			if (it == edge_map.end()) {
				SDEdge ne(v0, v1);
				ne.f[0] = face;
				ne.f_idx[0] = j;
				edge_map[ne] = ne;
			} else {
				it->second.f[1] = face;
				it->second.f_idx[1] = j;
			}
		}
	}

	// Set face neighbor pointers from edge map
	for (auto& [key, e] : edge_map) {
		if (e.f[0]) e.f[0]->f[e.f_idx[0]] = e.f[1];  // may be nullptr for boundary
		if (e.f[1]) e.f[1]->f[e.f_idx[1]] = e.f[0];
	}

	// Determine boundary / regular
	for (int i = 0; i < nVerts; ++i) {
		SDVertex* vtx = &vbuf[i];
		SDFace* f = vtx->startFace;
		// Walk forward until we complete the ring or hit nullptr
		do { f = f->next_face(vtx); }
		while (f != nullptr && f != vtx->startFace);
		vtx->boundary = (f == nullptr);
		int val = vtx->valence();
		vtx->regular = vtx->boundary ? (val == 4) : (val == 6);
	}

	// ------------------------------------------------------------------
	// 2. Subdivide nLevels times
	//    Use pools so pointers remain stable
	// ------------------------------------------------------------------
	std::vector<std::vector<SDVertex>> vertex_pools;
	std::vector<std::vector<SDFace>>   face_pools;

	// Working sets (pointers into buffers)
	std::vector<SDVertex*> cur_verts(nVerts);
	std::vector<SDFace*>   cur_faces(nTris);
	for (int i = 0; i < nVerts; ++i) cur_verts[i] = &vbuf[i];
	for (int i = 0; i < nTris;  ++i) cur_faces[i] = &fbuf[i];

	for (int level = 0; level < nLevels; ++level) {
		int nV = (int)cur_verts.size();
		int nF = (int)cur_faces.size();

		// Allocate child storage
		vertex_pools.emplace_back(nV);   // even children
		face_pools.emplace_back(4 * nF);

		std::vector<SDVertex>& new_vbuf = vertex_pools.back();
		std::vector<SDFace>&   new_fbuf = face_pools.back();

		std::vector<SDVertex*> new_verts;
		std::vector<SDFace*>   new_faces;

		// Link even vertices to their children
		for (int i = 0; i < nV; ++i) {
			cur_verts[i]->child = &new_vbuf[i];
			new_vbuf[i].regular  = cur_verts[i]->regular;
			new_vbuf[i].boundary = cur_verts[i]->boundary;
			new_verts.push_back(&new_vbuf[i]);
		}
		// Reserve face children
		for (int i = 0; i < nF; ++i) {
			for (int k = 0; k < 4; ++k) {
				cur_faces[i]->children[k] = &new_fbuf[4*i + k];
				new_faces.push_back(&new_fbuf[4*i + k]);
			}
		}

		// -- Even vertex positions --
		for (SDVertex* v : cur_verts) {
			if (!v->boundary) {
				v->child->p = v->regular
					? weight_one_ring(v, 1.0/16.0)
					: weight_one_ring(v, loop_beta(v->valence()));
			} else {
				v->child->p = weight_boundary(v, 1.0/8.0);
			}
		}

		// -- Odd edge vertices --
		std::map<SDEdge, SDVertex*> edge_verts;
		// We need a second pool for odd verts (appended to new_vbuf would invalidate ptrs)
		std::vector<SDVertex> odd_buf;
		odd_buf.reserve(3 * nF);  // at most 3 per face, but shared

		for (SDFace* face : cur_faces) {
			for (int k = 0; k < 3; ++k) {
				SDVertex* v0 = face->v[k];
				SDVertex* v1 = face->v[(k+1)%3];
				SDEdge e(v0, v1);
				if (edge_verts.count(e) == 0) {
					odd_buf.push_back(SDVertex());
					SDVertex* ov = &odd_buf.back();
					ov->regular  = true;
					ov->boundary = (face->f[k] == nullptr);
					ov->startFace = face->children[3];

					if (ov->boundary) {
						ov->p = 0.5 * v0->p + 0.5 * v1->p;
					} else {
						ov->p = (3.0/8.0) * v0->p + (3.0/8.0) * v1->p
							  + (1.0/8.0) * face->other_vert(v0,v1)->p
							  + (1.0/8.0) * face->f[k]->other_vert(v0,v1)->p;
					}
					edge_verts[e] = ov;
					new_verts.push_back(ov);
				}
			}
		}

		// -- Even vertex startFace update --
		for (SDVertex* v : cur_verts) {
			int vn = v->startFace->vnum(v);
			v->child->startFace = v->startFace->children[vn];
		}

		// -- Face neighbor pointers (verbatim pbrt-v4 topology update) --
		for (SDFace* face : cur_faces) {
			for (int j = 0; j < 3; ++j) {
				// Siblings: central tri <-> corner tri
				face->children[3]->f[j]        = face->children[(j+1)%3];
				face->children[j]->f[(j+1)%3]  = face->children[3];

				// Outer neighbors across original edge j
				SDFace* f2 = face->f[j];
				face->children[j]->f[j] = (f2 != nullptr)
					? f2->children[f2->vnum(face->v[j])]
					: nullptr;

				// Outer neighbors across original edge PREV(j)
				f2 = face->f[(j+2)%3];
				face->children[j]->f[(j+2)%3] = (f2 != nullptr)
					? f2->children[f2->vnum(face->v[j])]
					: nullptr;
			}
		}

		// -- Face vertex pointers --
		for (SDFace* face : cur_faces) {
			for (int j = 0; j < 3; ++j) {
				face->children[j]->v[j] = face->v[j]->child;
				SDVertex* ov = edge_verts[SDEdge(face->v[j], face->v[(j+1)%3])];
				face->children[j]->v[(j+1)%3] = ov;
				face->children[(j+1)%3]->v[j] = ov;
				face->children[3]->v[j] = ov;
			}
		}

		// Prepare for next level — need to move odd_buf into a stable pool
		// We store odd_buf per-level in a side storage
		// (already stored in odd_buf; new_verts points into it — we must keep it alive)
		// Store it in vertex_pools as a separate entry:
		vertex_pools.push_back(std::move(odd_buf));

		cur_verts = new_verts;
		cur_faces = new_faces;
	}

	// ------------------------------------------------------------------
	// 3. Push to limit surface
	// ------------------------------------------------------------------
	std::vector<point3> p_limit(cur_verts.size());
	for (size_t i = 0; i < cur_verts.size(); ++i) {
		SDVertex* v = cur_verts[i];
		p_limit[i] = v->boundary
			? weight_boundary(v, 1.0/5.0)
			: weight_one_ring(v, loop_gamma(v->valence()));
	}
	for (size_t i = 0; i < cur_verts.size(); ++i)
		cur_verts[i]->p = p_limit[i];

	// ------------------------------------------------------------------
	// 4. Compute limit normals (tangent cross product, pbrt-v4 §6.5)
	// ------------------------------------------------------------------
	const double Pi = 3.14159265358979323846;
	std::vector<vec3> normals(cur_verts.size());
	for (size_t i = 0; i < cur_verts.size(); ++i) {
		SDVertex* v = cur_verts[i];
		std::vector<point3> ring;
		v->one_ring(ring);
		int k = (int)ring.size();

		vec3 S(0,0,0), T(0,0,0);
		if (!v->boundary) {
			for (int j = 0; j < k; ++j) {
				double c = std::cos(2.0*Pi*j/k);
				double s = std::sin(2.0*Pi*j/k);
				S = S + c * (ring[j] - point3(0,0,0));
				T = T + s * (ring[j] - point3(0,0,0));
			}
		} else {
			S = ring.back() - ring.front();
			if (k == 2)
				T = vec3(ring[0]) + vec3(ring[1]) - 2.0*vec3(v->p);
			else if (k == 3)
				T = vec3(ring[1]) - vec3(v->p);
			else if (k == 4)
				T = vec3(-ring[0]) + 2.0*vec3(ring[1]) + 2.0*vec3(ring[2]) + (-1.0)*vec3(ring[3]) - 2.0*vec3(v->p);
			else {
				double theta = Pi / (k - 1);
				T = std::sin(theta) * (vec3(ring[0]) + vec3(ring[k-1]));
				for (int j = 1; j < k-1; ++j) {
					double wt = (2.0*std::cos(theta) - 2.0) * std::sin(j*theta);
					T = T + wt * vec3(ring[j]);
				}
				T = -T;
			}
		}
		vec3 n = cross(S, T);
		double nl = n.length();
		normals[i] = (nl > 1e-14) ? n / nl : vec3(0,1,0);
	}

	// ------------------------------------------------------------------
	// 5. Build output triangle_mesh_data
	// ------------------------------------------------------------------
	auto out = std::make_shared<triangle_mesh_data>();
	out->positions.resize(cur_verts.size());
	out->normals.resize(cur_verts.size());

	std::map<SDVertex*, int> vert_idx;
	for (size_t i = 0; i < cur_verts.size(); ++i) {
		out->positions[i] = cur_verts[i]->p;
		out->normals[i]   = normals[i];
		vert_idx[cur_verts[i]] = (int)i;
	}

	out->indices.resize(3 * cur_faces.size());
	for (size_t i = 0; i < cur_faces.size(); ++i) {
		for (int j = 0; j < 3; ++j)
			out->indices[3*i + j] = vert_idx[cur_faces[i]->v[j]];
	}

	return out;
}


#endif
