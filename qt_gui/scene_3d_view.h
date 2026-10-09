#pragma once
// scene_3d_view.h - the Scene Builder's 3D view: the scene seen from an orbiting camera, where things are picked and moved, turned and resized. It is
// drawn with QPainter (flat-shaded faces sorted back to front), so it needs no OpenGL and no extra libraries; its geometry is
// src/shared/scene_view_math.h, the tools' decisions are src/shared/scene_gizmo.h and a mesh file's bounds and vertex sample come from
// src/shared/mesh_preview.h (read on a worker thread, never while painting).
//
//   drag the background  orbit        right- or middle-drag  pan        wheel  zoom
//   click an item        select it    W / E / R  (or the buttons)  the Move / Rotate / Scale tool
//   Move:   drag an item to move it on the floor (Shift: up and down), or drag a coloured arrow to move along that axis only (X red, Y green, Z blue)
//   Rotate: drag a coloured ring to turn the object about that world axis
//   Scale:  drag a square handle to stretch the object along that one of its own axes (a sphere, disk or mesh scales all round)
//
// Moves use the same signals as the 2D layout view, so the Scene Builder treats them the same whichever view they came from; turns and stretches
// arrive as a whole edited object.

#include <QElapsedTimer>
#include <QPointF>
#include <QWidget>

#include <map>
#include <string>
#include <vector>

#include "../src/shared/mesh_preview.h"
#include "../src/shared/scene_document.h"
#include "../src/shared/scene_gizmo.h"
#include "../src/shared/scene_view_math.h"
#include "scene_layout_view.h"

class Scene3DView : public QWidget {
	Q_OBJECT
public:
	enum class GizmoMode { Move, Rotate, Scale };

	explicit Scene3DView(QWidget *parent = nullptr);
	~Scene3DView() override;  // out of line: Face is only defined in the .cpp

	void setDocument(const scene_doc::Document *doc) { m_doc = doc; update(); }
	void setSelection(const BuilderSelection &s) { m_sel = s; update(); }
	void setSnap(bool on) { m_snap = on; }
	void setGizmoMode(GizmoMode m);
	GizmoMode gizmoMode() const { return m_gizmo; }
	// Where a new object is dropped: the floor (y = 0) point under the middle of the view when that is near what the camera looks at, else the floor under
	// the point it looks at (a level camera would otherwise drop it near the horizon, far from anything visible).
	scene_doc::Float3 centerInWorld() const;
	void frameAll();
	QPointF itemScreenPos(const BuilderSelection &s) const;        // where an item is drawn, in this widget's pixels (null if behind the camera)
	// Points on the selected object's tool, for tests: along a move arrow, on a rotate ring (angle from its reference direction), at a scale handle.
	QPointF axisArrowPoint(int axis, double fraction) const;
	QPointF ringPoint(int axis, double deg) const;
	QPointF scaleHandlePoint(int axis, double fraction) const;
	// For tests: set the camera's angles, and whether a mesh file's preview has been read yet.
	void orbitForTest(double yawDeg, double pitchDeg) { m_cam.yawDeg = yawDeg; m_cam.pitchDeg = pitchDeg; m_userView = true; update(); }
	bool meshPreviewReady(const std::string &path) const { return meshPreview(path) != nullptr; }

	QSize sizeHint() const override { return QSize(520, 380); }

signals:
	void selectionRequested(const BuilderSelection &s);
	void dragBegan();
	void positionDragged(const BuilderSelection &s, int which, const scene_doc::Float3 &world);
	// A turn or a stretch: the object as it now is (the owner takes its rotation, radius, height, size and meshScale).
	void objectEdited(const BuilderSelection &s, const scene_doc::Object &updated);
	void gizmoModeChanged(int mode);

protected:
	void paintEvent(QPaintEvent *) override;
	void resizeEvent(QResizeEvent *e) override;
	void mousePressEvent(QMouseEvent *e) override;
	void mouseMoveEvent(QMouseEvent *e) override;
	void mouseReleaseEvent(QMouseEvent *e) override;
	void wheelEvent(QWheelEvent *e) override;
	void keyPressEvent(QKeyEvent *e) override;

private:
	struct Face;
	struct Ctx;
	struct Hit {
		enum class Kind { None, Item, Axis, Ring, ScaleHandle, FreeHandle } kind = Kind::None;
		BuilderSelection sel;
		int which = 0;   // 0: the item's position, 1: its target
		int axis = 0;    // for an arrow, ring or handle
	};
	enum class Mode { None, Orbit, Pan, Ground, Vertical, Free, Axis, Rotate, Scale };

	scene_view::View view() const;
	// The shapes as world-space polygons. Tessellating them is the expensive part of a repaint, so they are kept until the document changes (found by
	// comparing a signature of everything that shapes them), and only projected again for each frame.
	std::vector<Face> &faces() const;
	void projectFaces(const scene_view::View &v) const;
	std::uint64_t geometrySignature() const;
	const scene_doc::Float3 *handle(const BuilderSelection &s, int which) const;
	const scene_doc::Object *selectedObject() const;
	GizmoMode effectiveGizmo() const;   // Rotate and Scale only apply to objects; for a light or the camera the Move tool shows
	double gizmoLength(const scene_view::View &v, const scene_view::V3 &at) const;
	scene_view::V3 localAxis(const scene_doc::Object &o, int axis) const;   // the object's own axis, in the world
	// A mesh file's bounds and vertex sample, or null while it is still being read (on a worker thread) or if it cannot be.
	const mesh_preview::MeshPreview *meshPreview(const std::string &path) const;
	void startMeshLoad(const std::string &path, qint64 modified, qint64 size) const;
	void meshLoaded(const std::string &path, mesh_preview::MeshPreview preview, qint64 modified, qint64 size);
	Hit hitTest(const QPointF &px) const;
	void applyDrag(const QPointF &px, Qt::KeyboardModifiers mods);

	// paintEvent's parts
	void drawGrid(Ctx &c) const;
	void drawFaces(Ctx &c) const;
	void drawMeshPoints(Ctx &c, int objectIndex) const;
	void drawNames(Ctx &c) const;
	void drawLights(Ctx &c) const;
	void drawCamera(Ctx &c) const;
	void drawTool(Ctx &c) const;

	const scene_doc::Document *m_doc = nullptr;
	BuilderSelection m_sel;
	scene_view::OrbitCamera m_cam;
	bool m_snap = true;
	GizmoMode m_gizmo = GizmoMode::Move;
	bool m_userView = false;   // the user has orbited, panned or zoomed, so a resize keeps the view instead of re-framing the scene

	Mode m_mode = Mode::None;
	Hit m_drag;
	QPointF m_last;
	scene_view::V3 m_dragStart;     // the item's position when the drag began
	scene_view::V3 m_grabOffset;    // from where the mouse met the floor to the item (ground drags)
	double m_axisT0 = 0.0;          // where along the axis the mouse was when the drag began
	scene_doc::Object m_dragObject; // the object as it was when a turn or stretch began
	double m_angle0 = 0.0;          // the mouse's angle round the ring when a turn began

	struct CachedMesh {
		mesh_preview::MeshPreview preview;
		qint64 modified = 0, size = 0;
		qint64 checkedAt = 0;   // when the file's date was last looked at (ms on m_clock), so a repaint does not stat it every time
		bool pending = false;   // being read now
	};
	mutable std::map<std::string, CachedMesh> m_meshes;
	mutable int m_meshVersion = 0;   // counts finished mesh reads, so the cached faces are rebuilt when one lands
	QElapsedTimer m_clock;

	mutable std::vector<Face> m_faces;
	mutable std::uint64_t m_facesSignature = 0;
	mutable bool m_facesValid = false;
};
