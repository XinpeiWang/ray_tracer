#pragma once
// scene_3d_view.h - the Scene Builder's 3D view: the scene seen from an orbiting camera, where things are picked and moved. It is drawn with
// QPainter (flat-shaded faces sorted back to front), so it needs no OpenGL and no extra libraries; its geometry is src/shared/scene_view_math.h.
//
//   drag the background  orbit        right- or middle-drag  pan        wheel  zoom
//   click an item        select it    drag an item           move it on the floor (Shift: up and down)
//   drag a coloured arrow on the selected item: move it along that axis only (X red, Y green, Z blue)
//
// It uses the same signals as the 2D layout view, so the Scene Builder treats a move the same way whichever view it came from.

#include <QPointF>
#include <QWidget>

#include "../src/shared/scene_document.h"
#include "../src/shared/scene_view_math.h"
#include "scene_layout_view.h"

class Scene3DView : public QWidget {
	Q_OBJECT
public:
	explicit Scene3DView(QWidget *parent = nullptr);

	void setDocument(const scene_doc::Document *doc) { m_doc = doc; update(); }
	void setSelection(const BuilderSelection &s) { m_sel = s; update(); }
	void setSnap(bool on) { m_snap = on; }
	// Where a new object is dropped: the floor (y = 0) point under the middle of the view, else the point the camera looks at.
	scene_doc::Float3 centerInWorld() const;
	void frameAll();
	QPointF itemScreenPos(const BuilderSelection &s) const;        // where an item is drawn, in this widget's pixels (null if behind the camera)
	QPointF axisArrowPoint(int axis, double fraction) const;       // a point along the selected item's arrow (0 = X, 1 = Y, 2 = Z); for tests

	QSize sizeHint() const override { return QSize(520, 380); }

signals:
	void selectionRequested(const BuilderSelection &s);
	void dragBegan();
	void positionDragged(const BuilderSelection &s, int which, const scene_doc::Float3 &world);

protected:
	void paintEvent(QPaintEvent *) override;
	void resizeEvent(QResizeEvent *e) override;
	void mousePressEvent(QMouseEvent *e) override;
	void mouseMoveEvent(QMouseEvent *e) override;
	void mouseReleaseEvent(QMouseEvent *e) override;
	void wheelEvent(QWheelEvent *e) override;

private:
	struct Face;
	struct Hit {
		enum class Kind { None, Item, Axis } kind = Kind::None;
		BuilderSelection sel;
		int which = 0;   // 0: the item's position, 1: its target
		int axis = 0;    // for Kind::Axis
	};
	enum class Mode { None, Orbit, Pan, Ground, Vertical, Axis };

	scene_view::View view() const;
	QList<Face> buildFaces(const scene_view::View &v) const;
	void projectFaces(QList<Face> &faces, const scene_view::View &v) const;
	const scene_doc::Float3 *handle(const BuilderSelection &s, int which) const;
	double gizmoLength(const scene_view::View &v, const scene_view::V3 &at) const;
	Hit hitTest(const QPointF &px) const;
	void applyDrag(const QPointF &px, Qt::KeyboardModifiers mods);

	const scene_doc::Document *m_doc = nullptr;
	BuilderSelection m_sel;
	scene_view::OrbitCamera m_cam;
	bool m_snap = true;
	bool m_userView = false;   // the user has orbited, panned or zoomed, so a resize keeps the view instead of re-framing the scene

	Mode m_mode = Mode::None;
	Hit m_drag;
	QPointF m_last;
	scene_view::V3 m_dragStart;     // the item's position when the drag began
	scene_view::V3 m_grabOffset;    // from where the mouse met the floor to the item (ground drags)
	double m_axisT0 = 0.0;          // where along the axis the mouse was when the drag began
};
