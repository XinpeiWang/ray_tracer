#pragma once
// ---------------------------------------------------------------------------
// loop_subdivide.h -- Loop subdivision surface refinement
//
// Ports pbrt-v4 util/loopsubdiv.cpp (Apache-2.0) as a self-contained
// header-only C++ implementation.
//
// Public API:
//   template<typename T>
//   struct LoopSubdivResult {
//       std::vector<std::array<T,3>> positions;  // limit-surface positions
//       std::vector<std::array<T,3>> normals;    // limit-surface normals
//       std::vector<int>             indices;    // triangle index triples
//   };
//
//   template<typename T>
//   LoopSubdivResult<T> loop_subdivide(
//       const std::vector<std::array<T,3>>& positions,
//       const std::vector<int>&             indices,
//       int                                 n_levels);
//
// Adaptations from pbrt-v4:
//   - No pbrt Allocator / TriangleMesh / Transform dependencies
//   - Template parameter T (float or double)
//   - Output returned by value as LoopSubdivResult<T>
//   - Internal mesh structures in anonymous namespace
//
// Reference: pbrt-v4 src/pbrt/util/loopsubdiv.cpp (Apache-2.0)
//            Charles Loop, "Smooth Subdivision Surfaces Based on Triangles", 1987
// ---------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <vector>

// ---------------------------------------------------------------------------
// LoopSubdivResult
// ---------------------------------------------------------------------------
template<typename T>
struct LoopSubdivResult {
	std::vector<std::array<T,3>> positions;  // one per vertex
	std::vector<std::array<T,3>> normals;    // one per vertex (limit surface)
	std::vector<int>             indices;    // 3 per triangle
};

// ---------------------------------------------------------------------------
// Implementation details in anonymous namespace
// ---------------------------------------------------------------------------
namespace {

// pbrt-v4: NEXT / PREV macros for face edge cycling
inline int sd_next(int i) { return (i + 1) % 3; }
inline int sd_prev(int i) { return (i + 2) % 3; }

struct SDFace;

// ---------------------------------------------------------------------------
// SDVertex
// ---------------------------------------------------------------------------
template<typename T>
struct SDVertex {
	std::array<T,3> p = {T(0), T(0), T(0)};
	SDFace*         startFace = nullptr;
	SDVertex*       child     = nullptr;
	bool            regular   = false;
	bool            boundary  = false;

	SDVertex() = default;
	explicit SDVertex(const std::array<T,3>& pos) : p(pos) {}

	int  valence() const;
	void oneRing(std::array<T,3>* ring) const;
};

// ---------------------------------------------------------------------------
// SDFace
// ---------------------------------------------------------------------------
template<typename T>
struct SDFaceT {
	SDVertex<T>*  v[3]        = {};
	SDFaceT<T>*   f[3]        = {};
	SDFaceT<T>*   children[4] = {};

	// Index of vertex vert within this face (0,1,2)
	int vnum(SDVertex<T>* vert) const {
		for (int i = 0; i < 3; ++i)
			if (v[i] == vert) return i;
		assert(false && "SDFace::vnum: vertex not in face");
		return -1;
	}

	SDFaceT<T>*  nextFace(SDVertex<T>* vert) { return f[vnum(vert)]; }
	SDFaceT<T>*  prevFace(SDVertex<T>* vert) { return f[sd_prev(vnum(vert))]; }
	SDVertex<T>* nextVert(SDVertex<T>* vert) { return v[sd_next(vnum(vert))]; }
	SDVertex<T>* prevVert(SDVertex<T>* vert) { return v[sd_prev(vnum(vert))]; }

	SDVertex<T>* otherVert(SDVertex<T>* v0, SDVertex<T>* v1) {
		for (int i = 0; i < 3; ++i)
			if (v[i] != v0 && v[i] != v1) return v[i];
		assert(false && "SDFace::otherVert: no other vertex");
		return nullptr;
	}
};

// Alias SDFace to SDFaceT<T> -- SDVertex::startFace must be untyped for the
// template, so we use void* internally and cast. Instead, we just duplicate
// the face type per T instantiation via the SDFaceT template.
// SDVertex::startFace is stored as SDFaceT<T>* via reinterpret.
// Simpler: store as void* and cast in valence/oneRing.
// Instead: make SDVertex hold SDFaceT<T>* directly via template.

// Re-define SDVertex to be fully templated with SDFaceT:
// (The forward declaration above is just for the SDFace reference.)

// ---------------------------------------------------------------------------
// SDEdge
// ---------------------------------------------------------------------------
template<typename T>
struct SDEdge {
	SDVertex<T>* v[2] = {};
	SDFaceT<T>*  f[2] = {};
	int          f0edgeNum = -1;

	SDEdge() = default;
	SDEdge(SDVertex<T>* v0, SDVertex<T>* v1) {
		v[0] = std::min(v0, v1);
		v[1] = std::max(v0, v1);
		f[0] = f[1] = nullptr;
		f0edgeNum = -1;
	}

	bool operator<(const SDEdge<T>& e2) const {
		if (v[0] == e2.v[0]) return v[1] < e2.v[1];
		return v[0] < e2.v[0];
	}
};

// ---------------------------------------------------------------------------
// valence() -- count one-ring neighbors
// ---------------------------------------------------------------------------
template<typename T>
int SDVertexValence(SDVertex<T>* vert) {
	SDFaceT<T>* f = reinterpret_cast<SDFaceT<T>*>(vert->startFace);
	if (!vert->boundary) {
		int nf = 1;
		while ((f = f->nextFace(vert)) != reinterpret_cast<SDFaceT<T>*>(vert->startFace))
			++nf;
		return nf;
	} else {
		int nf = 1;
		while ((f = f->nextFace(vert)) != nullptr) ++nf;
		f = reinterpret_cast<SDFaceT<T>*>(vert->startFace);
		while ((f = f->prevFace(vert)) != nullptr) ++nf;
		return nf + 1;
	}
}

// ---------------------------------------------------------------------------
// oneRing() -- collect one-ring positions into ring[]
// ---------------------------------------------------------------------------
template<typename T>
void SDVertexOneRing(SDVertex<T>* vert, std::array<T,3>* ring) {
	SDFaceT<T>* face = reinterpret_cast<SDFaceT<T>*>(vert->startFace);
	if (!vert->boundary) {
		SDFaceT<T>* start = face;
		do {
			*ring++ = face->nextVert(vert)->p;
			face = face->nextFace(vert);
		} while (face != start);
	} else {
		SDFaceT<T>* f2;
		while ((f2 = face->nextFace(vert)) != nullptr) face = f2;
		*ring++ = face->nextVert(vert)->p;
		do {
			*ring++ = face->prevVert(vert)->p;
			face = face->prevFace(vert);
		} while (face != nullptr);
	}
}

// ---------------------------------------------------------------------------
// Loop subdivision weights
// pbrt-v4: beta(), loopGamma(), weightOneRing(), weightBoundary()
// ---------------------------------------------------------------------------
template<typename T>
inline T sd_beta(int valence) {
	if (valence == 3) return T(3) / T(16);
	return T(3) / (T(8) * T(valence));
}

template<typename T>
inline T sd_loop_gamma(int valence) {
	return T(1) / (T(valence) + T(3) / (T(8) * sd_beta<T>(valence)));
}

template<typename T>
std::array<T,3> sd_weight_one_ring(SDVertex<T>* vert, T beta_val) {
	int valence = SDVertexValence(vert);
	std::vector<std::array<T,3>> pRing(valence);
	SDVertexOneRing(vert, pRing.data());
	// (1 - valence*beta)*p + beta * sum(ring)
	T w = T(1) - T(valence) * beta_val;
	std::array<T,3> p = {w * vert->p[0], w * vert->p[1], w * vert->p[2]};
	for (int i = 0; i < valence; ++i) {
		p[0] += beta_val * pRing[i][0];
		p[1] += beta_val * pRing[i][1];
		p[2] += beta_val * pRing[i][2];
	}
	return p;
}

template<typename T>
std::array<T,3> sd_weight_boundary(SDVertex<T>* vert, T beta_val) {
	int valence = SDVertexValence(vert);
	std::vector<std::array<T,3>> pRing(valence);
	SDVertexOneRing(vert, pRing.data());
	// (1 - 2*beta)*p + beta*(ring[0] + ring[last])
	T w = T(1) - T(2) * beta_val;
	std::array<T,3> p = {w * vert->p[0], w * vert->p[1], w * vert->p[2]};
	p[0] += beta_val * pRing[0][0] + beta_val * pRing[valence - 1][0];
	p[1] += beta_val * pRing[0][1] + beta_val * pRing[valence - 1][1];
	p[2] += beta_val * pRing[0][2] + beta_val * pRing[valence - 1][2];
	return p;
}

// Simple cross product
template<typename T>
std::array<T,3> sd_cross(const std::array<T,3>& a, const std::array<T,3>& b) {
	return {a[1]*b[2] - a[2]*b[1],
			a[2]*b[0] - a[0]*b[2],
			a[0]*b[1] - a[1]*b[0]};
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// loop_subdivide<T>
//
// pbrt-v4: LoopSubdivide() (util/loopsubdiv.cpp)
//
// positions : flat array of vertex positions, one per vertex
// indices   : flat array of vertex index triples (one per triangle)
// n_levels  : number of subdivision levels (0 = return input mesh)
// ---------------------------------------------------------------------------
template<typename T>
LoopSubdivResult<T> loop_subdivide(
	const std::vector<std::array<T,3>>& positions,
	const std::vector<int>&             indices,
	int                                 n_levels)
{
	using Vertex = SDVertex<T>;
	using Face   = SDFaceT<T>;
	using Edge   = SDEdge<T>;

	const int nFaces0 = static_cast<int>(indices.size() / 3);

	// Allocate initial vertices and faces as contiguous arrays
	std::unique_ptr<Vertex[]> verts_mem(new Vertex[positions.size()]);
	std::unique_ptr<Face[]>   faces_mem(new Face[nFaces0]);

	std::vector<Vertex*> v;
	std::vector<Face*>   f;
	v.reserve(positions.size());
	f.reserve(nFaces0);

	for (int i = 0; i < (int)positions.size(); ++i) {
		verts_mem[i] = Vertex(positions[i]);
		v.push_back(&verts_mem[i]);
	}
	for (int i = 0; i < nFaces0; ++i)
		f.push_back(&faces_mem[i]);

	// Set face->vertex pointers
	const int* vp = indices.data();
	for (int i = 0; i < nFaces0; ++i, vp += 3) {
		for (int j = 0; j < 3; ++j) {
			Vertex* vv = v[vp[j]];
			f[i]->v[j] = vv;
			vv->startFace = reinterpret_cast<SDFace*>(f[i]);
		}
	}

	// Set face neighbor pointers using edge set
	std::set<Edge> edges;
	for (int i = 0; i < nFaces0; ++i) {
		for (int edgeNum = 0; edgeNum < 3; ++edgeNum) {
			int v0 = edgeNum, v1 = sd_next(edgeNum);
			Edge e(f[i]->v[v0], f[i]->v[v1]);
			auto it = edges.find(e);
			if (it == edges.end()) {
				e.f[0] = f[i];
				e.f0edgeNum = edgeNum;
				edges.insert(e);
			} else {
				Edge found = *it;
				found.f[0]->f[found.f0edgeNum] = f[i];
				f[i]->f[edgeNum] = found.f[0];
				edges.erase(it);
			}
		}
	}

	// Determine boundary and regularity for each vertex
	for (int i = 0; i < (int)positions.size(); ++i) {
		Vertex* vv = v[i];
		Face* face = reinterpret_cast<Face*>(vv->startFace);
		do { face = face->nextFace(vv); } while (face != nullptr && face != reinterpret_cast<Face*>(vv->startFace));
		vv->boundary = (face == nullptr);
		int val = SDVertexValence(vv);
		vv->regular = (!vv->boundary && val == 6) || (vv->boundary && val == 4);
	}

	// Allocate pool for all refinement levels
	// Each level: #v grows by #edges-new; #f grows 4x
	// We pool everything in a single vector of unique_ptrs per level
	std::vector<std::unique_ptr<Vertex[]>> vert_pools;
	std::vector<std::unique_ptr<Face[]>>   face_pools;

	// ---------------------------------------------------------------------------
	// Refinement loop  (pbrt-v4: "Refine _LoopSubdiv_ into triangles")
	// ---------------------------------------------------------------------------
	for (int level = 0; level < n_levels; ++level) {
		std::vector<Vertex*> newVertices;
		std::vector<Face*>   newFaces;

		// Allocate child vertices (one per current vertex + one per edge)
		// We don't know edge count yet; allocate lazily via a vector
		// Use a map from edge -> new odd vertex
		// Allocate child-vertex memory as one block per face (3 edges * nFaces worst case)
		size_t maxNewVerts = v.size() + f.size() * 3; // over-estimate
		auto   child_verts = std::make_unique<Vertex[]>(maxNewVerts);
		auto   child_faces = std::make_unique<Face[]>(f.size() * 4);
		vert_pools.push_back(std::move(child_verts));
		face_pools.push_back(std::move(child_faces));
		Vertex* cv_pool = vert_pools.back().get();
		Face*   cf_pool = face_pools.back().get();
		int     cv_used = 0;

		// Allocate one child vertex per current vertex (even vertices)
		for (Vertex* vert : v) {
			vert->child = &cv_pool[cv_used++];
			*vert->child = Vertex();
			vert->child->regular = vert->regular;
			vert->child->boundary = vert->boundary;
			newVertices.push_back(vert->child);
		}

		// Allocate 4 child faces per current face
		for (int i = 0; i < (int)f.size(); ++i) {
			for (int k = 0; k < 4; ++k) {
				f[i]->children[k] = &cf_pool[i * 4 + k];
				*f[i]->children[k] = Face();
				newFaces.push_back(f[i]->children[k]);
			}
		}

		// Update even vertex positions  (pbrt-v4: "Update vertex positions")
		for (Vertex* vert : v) {
			if (!vert->boundary) {
				if (vert->regular)
					vert->child->p = sd_weight_one_ring(vert, T(1) / T(16));
				else
					vert->child->p = sd_weight_one_ring(vert, sd_beta<T>(SDVertexValence(vert)));
			} else {
				vert->child->p = sd_weight_boundary(vert, T(1) / T(8));
			}
		}

		// Create odd (edge) vertices and update new face vertex pointers
		std::map<Edge, Vertex*> edgeVerts;
		for (Face* face : f) {
			for (int k = 0; k < 3; ++k) {
				Edge edge(face->v[k], face->v[sd_next(k)]);
				Vertex* vert = edgeVerts[edge];
				if (!vert) {
					// New odd vertex
					vert = &cv_pool[cv_used++];
					*vert = Vertex();
					vert->regular = true;
					vert->boundary = (face->f[k] == nullptr);
					vert->startFace = reinterpret_cast<SDFace*>(face->children[3]);
					// Position: edge rule
					if (vert->boundary) {
						vert->p[0] = T(0.5) * (face->v[k]->p[0] + face->v[sd_next(k)]->p[0]);
						vert->p[1] = T(0.5) * (face->v[k]->p[1] + face->v[sd_next(k)]->p[1]);
						vert->p[2] = T(0.5) * (face->v[k]->p[2] + face->v[sd_next(k)]->p[2]);
					} else {
						// Interior edge: 3/8 * (v0+v1) + 1/8 * (opp0+opp1)
						auto& a = face->v[k]->p;
						auto& b = face->v[sd_next(k)]->p;
						auto& c = face->otherVert(face->v[k], face->v[sd_next(k)])->p;
						auto& d = face->f[k]->otherVert(face->v[k], face->v[sd_next(k)])->p;
						for (int ax = 0; ax < 3; ++ax)
							vert->p[ax] = T(3)/T(8)*(a[ax]+b[ax]) + T(1)/T(8)*(c[ax]+d[ax]);
					}
					edgeVerts[edge] = vert;
					newVertices.push_back(vert);
				}
			}
		}

		// Update even vertex face pointers
		for (Vertex* vert : v) {
			int vertNum = reinterpret_cast<Face*>(vert->startFace)->vnum(vert);
			vert->child->startFace = reinterpret_cast<SDFace*>(
				reinterpret_cast<Face*>(vert->startFace)->children[vertNum]);
		}

		// Update face neighbor pointers (pbrt-v4: "Update face neighbor pointers")
		for (Face* face : f) {
			for (int j = 0; j < 3; ++j) {
				// Center child's j-th neighbor is child j+1
				face->children[3]->f[j] = face->children[sd_next(j)];
				// Child j's sd_next(j) neighbor is center child
				face->children[j]->f[sd_next(j)] = face->children[3];
				// Child j's j-th neighbor is the corresponding child of the neighboring face
				Face* f2 = face->f[j];
				face->children[j]->f[j] = (f2 != nullptr)
					? f2->children[f2->vnum(face->v[j])] : nullptr;
				f2 = face->f[sd_prev(j)];
				face->children[j]->f[sd_prev(j)] = (f2 != nullptr)
					? f2->children[f2->vnum(face->v[j])] : nullptr;
			}
		}

		// Update face vertex pointers (pbrt-v4: "Update face vertex pointers")
		for (Face* face : f) {
			for (int j = 0; j < 3; ++j) {
				face->children[j]->v[j] = face->v[j]->child;
				Vertex* vert = edgeVerts[Edge(face->v[j], face->v[sd_next(j)])];
				face->children[j]->v[sd_next(j)] = vert;
				face->children[sd_next(j)]->v[j] = vert;
				face->children[3]->v[j] = vert;
			}
		}

		f = newFaces;
		v = newVertices;
	}

	// ---------------------------------------------------------------------------
	// Push vertices to limit surface  (pbrt-v4: "Push vertices to limit surface")
	// ---------------------------------------------------------------------------
	std::vector<std::array<T,3>> pLimit(v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		if (v[i]->boundary)
			pLimit[i] = sd_weight_boundary(v[i], T(1) / T(5));
		else
			pLimit[i] = sd_weight_one_ring(v[i], sd_loop_gamma<T>(SDVertexValence(v[i])));
	}
	for (size_t i = 0; i < v.size(); ++i)
		v[i]->p = pLimit[i];

	// ---------------------------------------------------------------------------
	// Compute limit-surface normals  (pbrt-v4: "Compute vertex tangents on limit surface")
	// ---------------------------------------------------------------------------
	constexpr T Pi = T(3.14159265358979323846);
	std::vector<std::array<T,3>> Ns;
	Ns.reserve(v.size());
	for (Vertex* vert : v) {
		int valence = SDVertexValence(vert);
		std::vector<std::array<T,3>> pRing(valence);
		SDVertexOneRing(vert, pRing.data());

		std::array<T,3> S = {T(0), T(0), T(0)};
		std::array<T,3> Tv = {T(0), T(0), T(0)};

		if (!vert->boundary) {
			for (int j = 0; j < valence; ++j) {
				T cosA = std::cos(T(2) * Pi * T(j) / T(valence));
				T sinA = std::sin(T(2) * Pi * T(j) / T(valence));
				for (int ax = 0; ax < 3; ++ax) {
					S[ax]  += cosA * pRing[j][ax];
					Tv[ax] += sinA * pRing[j][ax];
				}
			}
		} else {
			// Boundary tangent
			for (int ax = 0; ax < 3; ++ax)
				S[ax] = pRing[valence - 1][ax] - pRing[0][ax];
			if (valence == 2) {
				for (int ax = 0; ax < 3; ++ax)
					Tv[ax] = pRing[0][ax] + pRing[1][ax] - T(2) * vert->p[ax];
			} else if (valence == 3) {
				for (int ax = 0; ax < 3; ++ax)
					Tv[ax] = pRing[1][ax] - vert->p[ax];
			} else if (valence == 4) {
				for (int ax = 0; ax < 3; ++ax)
					Tv[ax] = -pRing[0][ax] + T(2)*pRing[1][ax] + T(2)*pRing[2][ax]
							 - pRing[3][ax] - T(2)*vert->p[ax];
			} else {
				T theta = Pi / T(valence - 1);
				for (int ax = 0; ax < 3; ++ax)
					Tv[ax] = std::sin(theta) * (pRing[0][ax] + pRing[valence - 1][ax]);
				for (int k = 1; k < valence - 1; ++k) {
					T wt = (T(2) * std::cos(theta) - T(2)) * std::sin(T(k) * theta);
					for (int ax = 0; ax < 3; ++ax)
						Tv[ax] += wt * pRing[k][ax];
				}
				for (int ax = 0; ax < 3; ++ax)
					Tv[ax] = -Tv[ax];
			}
		}
		Ns.push_back(sd_cross(S, Tv));
	}

	// ---------------------------------------------------------------------------
	// Build output index buffer  (pbrt-v4: "Create triangle mesh from subdivision mesh")
	// ---------------------------------------------------------------------------
	std::map<Vertex*, int> usedVerts;
	for (int i = 0; i < (int)v.size(); ++i)
		usedVerts[v[i]] = i;

	std::vector<int> outIndices;
	outIndices.reserve(f.size() * 3);
	for (Face* face : f) {
		for (int j = 0; j < 3; ++j)
			outIndices.push_back(usedVerts[face->v[j]]);
	}

	LoopSubdivResult<T> result;
	result.positions = pLimit;
	result.normals   = Ns;
	result.indices   = outIndices;
	return result;
}
