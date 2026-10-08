// RT_GUI_SELFTEST=builder3d RT_GUI_SELFTEST_OUT=<prefix>: the Scene Builder's 3D view, driven with real mouse events: dragging an object on the floor
// (snapped to the 0.25 grid, height unchanged, one undo step), dragging each coloured arrow (only that coordinate changes), and a screenshot with an
// object selected (<prefix>_builder3d.png). Exit 0 if every step held.
#include "mainwindow.h"

#include "scene_builder_widget.h"

#include <QApplication>
#include <QCoreApplication>
#include <QEventLoop>
#include <QThread>
#include <QDir>
#include <QFile>
#include <QMouseEvent>
#include <QSplitter>
#include <QSplitterHandle>
#include <QTimer>

#include <cmath>

#include "../src/shared/scene_document.h"
#include "../src/shared/scene_view_math.h"

void MainWindow::runBuilder3dSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	if (m_sceneBuilder) m_tabWidget->setCurrentWidget(m_sceneBuilder);
	resize(1500, 950);
	SceneBuilderWidget *sb = m_sceneBuilder;
	if (!sb) { log("FAIL: no Scene Builder"); QApplication::exit(1); return; }
	sb->show3dView(true);
	QTimer::singleShot(500, this, [this, sb, log, shot]() {
		bool ok = true;
		auto check = [&ok, log](bool cond, const QString &what) {
			log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
			ok = ok && cond;
		};
		// The divider between the view and the preview can be dragged (it once could not: both panes were at their minimum height).
		{
			QSplitter *vertical = nullptr;
			for (QSplitter *sp : sb->findChildren<QSplitter *>())
				if (sp->orientation() == Qt::Vertical && sp->count() == 2) vertical = sp;
			check(vertical != nullptr, "found the splitter between the view and the preview");
			if (vertical) {
				const int before = vertical->sizes()[0];
				QSplitterHandle *h = vertical->handle(1);
				const QPointF mid(h->width() / 2.0, h->height() / 2.0);
				auto send = [h](QEvent::Type t, const QPointF &pos, Qt::MouseButton b, Qt::MouseButtons bs) {
					QMouseEvent e(t, pos, h->mapToGlobal(pos), b, bs, Qt::NoModifier);
					QApplication::sendEvent(h, &e);
				};
				send(QEvent::MouseButtonPress, mid, Qt::LeftButton, Qt::LeftButton);
				for (int step = 1; step <= 4; ++step) send(QEvent::MouseMove, mid + QPointF(0, -60.0 * step / 4.0), Qt::NoButton, Qt::LeftButton);
				send(QEvent::MouseButtonRelease, mid + QPointF(0, -60.0), Qt::LeftButton, Qt::NoButton);
				const int after = vertical->sizes()[0];
				check(after < before - 20, QString("dragging the divider up made the view shorter (%1 -> %2)").arg(before).arg(after));
			}
		}
		const scene_doc::Float3 ball0 = sb->document().objects[1].position;
		check(sb->dragObject3dForTest(1, QPointF(70, -25)), "dragging the glass ball in the 3D view moves it");
		const scene_doc::Float3 ball1 = sb->document().objects[1].position;
		check(ball1.y == ball0.y, "it stayed at its height (a drag on the floor plane)");
		check(std::fabs(ball1.x * 4 - std::round(ball1.x * 4)) < 1e-9 && std::fabs(ball1.z * 4 - std::round(ball1.z * 4)) < 1e-9, "its new position is on the grid");
		check(ball1.x != ball0.x || ball1.z != ball0.z, "it moved on the floor");
		check(sb->undo() && sb->document().objects[1].position.x == ball0.x && sb->document().objects[1].position.z == ball0.z, "one undo puts it back");
		{
			// Shift-drag on the background pans (the trackpad's way, with no right button): the scene follows the mouse instead of turning.
			const QPointF moved = sb->shiftPanBackground3dForTest(1, QPointF(60, 30));
			log(QString("Shift-drag moved the ball on screen by %1, %2").arg(moved.x()).arg(moved.y()));
			check(moved.x() > 35 && moved.x() < 90 && moved.y() > 15 && moved.y() < 45, "a Shift-drag on the background pans the 3D view (the scene follows the mouse)");
		}
		const char *names[3] = {"X", "Y", "Z"};
		for (int axis = 0; axis < 3; ++axis) {
			const scene_doc::Float3 before = sb->document().objects[2].position;
			check(sb->dragAxis3dForTest(2, axis, 60), QString("dragging the %1 arrow changes only %1").arg(names[axis]));
			check(sb->undo(), QString("and one undo reverts the %1 drag").arg(names[axis]));
			const scene_doc::Float3 after = sb->document().objects[2].position;
			check(after.x == before.x && after.y == before.y && after.z == before.z, QString("back where it was after %1").arg(names[axis]));
		}
		// Move: a press on the middle of the SELECTED object (where all three arrows start) is a free move on the floor, not an axis drag.
		{
			sb->selectObject(2);
			const scene_doc::Float3 b0 = sb->document().objects[2].position;
			check(sb->dragObject3dForTest(2, QPointF(70, 45)), "dragging the middle of the selected ball moves it");
			const scene_doc::Float3 b1 = sb->document().objects[2].position;
			check(b1.x != b0.x && b1.z != b0.z && b1.y == b0.y, "on the floor in both X and Z, not locked to one axis by an arrow");
			check(sb->undo(), "undo");
		}
		// The tool buttons follow what can be turned: a light or the camera has only Move, so Rotate and Scale are greyed and Move shows pressed.
		{
			sb->selectObject(3);
			check(sb->gizmoButtonEnabled(1) && sb->gizmoButtonEnabled(2), "Rotate and Scale are available for an object");
			sb->selectCameraForTest();
			check(!sb->gizmoButtonEnabled(1) && !sb->gizmoButtonEnabled(2) && sb->gizmoButtonEnabled(0) && sb->gizmoButtonChecked() == 0, "...but not for the camera, which shows Move");
			sb->selectObject(3);
			check(sb->gizmoButtonEnabled(1), "and are back for an object");
		}
		// A new object is dropped near what the camera looks at even when the camera is almost level with the floor (it used to land near the horizon).
		{
			sb->orbit3dForTest(30.0, 1.0);
			const scene_doc::Float3 drop = sb->dropPointForTest();
			check(std::fabs(drop.x) < 30 && std::fabs(drop.z) < 30 && drop.y == 0.0, QString("the drop point stays near the scene with a level camera (%1, %2)").arg(drop.x).arg(drop.z));
			sb->orbit3dForTest(30.0, 25.0);
		}
		// Rotate: the red box (object 3) turned 30 degrees about each world axis by dragging its ring, from whatever angles it already has.
		for (int axis = 0; axis < 3; ++axis) {
			const scene_doc::Float3 r0 = sb->document().objects[3].rotation;
			check(sb->dragRotate3dForTest(3, axis, 30.0), QString("dragging the %1 ring turns the box").arg(names[axis]));
			const scene_doc::Float3 r1 = sb->document().objects[3].rotation;
			const scene_view::V3 want = scene_view::turnAboutWorldAxis({r0.x, r0.y, r0.z}, scene_view::V3{axis == 0 ? 1.0 : 0.0, axis == 1 ? 1.0 : 0.0, axis == 2 ? 1.0 : 0.0}, 30.0);
			check(std::fabs(r1.x - want.x) < 1e-6 && std::fabs(r1.y - want.y) < 1e-6 && std::fabs(r1.z - want.z) < 1e-6,
			      QString("it is now exactly a 30 degree turn about world %1 (snapped to 5 degrees): got %2 %3 %4, wanted %5 %6 %7").arg(names[axis]).arg(r1.x).arg(r1.y).arg(r1.z).arg(want.x).arg(want.y).arg(want.z));
			check(sb->undo() && sb->document().objects[3].rotation.x == r0.x && sb->document().objects[3].rotation.y == r0.y && sb->document().objects[3].rotation.z == r0.z,
			      QString("one undo reverts the %1 turn").arg(names[axis]));
		}
		// Scale: the red box stretched along its own X, and the gold ball all round.
		{
			const scene_doc::Object box0 = sb->document().objects[3];
			check(sb->dragScale3dForTest(3, 0, 1.5), "dragging the box's X handle stretches it");
			const scene_doc::Object box1 = sb->document().objects[3];
			check(std::fabs(box1.size.x / box0.size.x - 1.5) < 0.1 && box1.size.y == box0.size.y && box1.size.z == box0.size.z, "only its X size grew, by about half");
			check(sb->undo() && sb->document().objects[3].size.x == box0.size.x, "one undo reverts the stretch");
			const double radius0 = sb->document().objects[2].radius;
			check(sb->dragScale3dForTest(2, 1, 0.5) && sb->document().objects[2].radius < radius0 * 0.6 && sb->document().objects[2].radius > radius0 * 0.4, "a sphere's handle scales its radius all round");
			check(sb->undo(), "undo");
			// A ready-made shape: a torus scales its ring and its tube together, a pyramid's height handle changes only its height.
			sb->addObject(scene_doc::ShapeKind::Torus);
			const int torus = static_cast<int>(sb->document().objects.size()) - 1;
			const scene_doc::Object t0 = sb->document().objects[torus];
			const bool torusDragged = sb->dragScale3dForTest(torus, 0, 1.5);
			check(torusDragged && sb->document().objects[torus].radius > t0.radius * 1.2 &&
			          std::fabs(sb->document().objects[torus].radius2 / t0.radius2 - sb->document().objects[torus].radius / t0.radius) < 1e-9,
			      "a torus' handle scales its ring and its tube together");
			sb->addObject(scene_doc::ShapeKind::Pyramid);
			const int pyr = static_cast<int>(sb->document().objects.size()) - 1;
			const scene_doc::Object p0 = sb->document().objects[pyr];
			check(sb->dragScale3dForTest(pyr, 1, 1.5) && sb->document().objects[pyr].height > p0.height * 1.2 && sb->document().objects[pyr].size.x == p0.size.x,
			      "a pyramid's height handle changes only its height");
			check(sb->undo() && sb->undo() && sb->undo() && sb->undo(), "undo the scales and the two shapes");
			check(static_cast<int>(sb->document().objects.size()) == torus, "back to the starter objects");
		}
		// A mesh file is drawn at its real size: open a scene whose mesh is a 2 x 4 x 6 box, to be looked at in the screenshot.
		{
			const QString objPath = QDir::tempPath() + "/builder3d_selftest_mesh.obj";
			{
				QFile f(objPath);
				if (f.open(QIODevice::WriteOnly)) {
					// a box 2 x 4 x 6, as an .obj
					const char *obj[] = {"v 0 0 0", "v 2 0 0", "v 2 4 0", "v 0 4 0", "v 0 0 6", "v 2 0 6", "v 2 4 6", "v 0 4 6", "f 1 2 3", "f 1 3 4", "f 5 6 7", "f 5 7 8"};
					for (const char *line : obj) f.write(QByteArray(line) + QByteArray(1, char(10)));
				}
			}
			scene_doc::Document d = scene_doc::makeStarterScene();
			scene_doc::Object mesh = scene_doc::makeObject(scene_doc::ShapeKind::Mesh, "Scan");
			mesh.meshFile = objPath.toStdString();
			mesh.meshScale = 0.5;
			mesh.position = {2.0, 0.0, 1.0};
			d.objects.push_back(mesh);
			const QString scenePath = QDir::tempPath() + "/builder3d_selftest_scene.pbrt";
			QFile sf(scenePath);
			check(sf.open(QIODevice::WriteOnly), "wrote a scene with a mesh object");
			sf.write(QByteArray::fromStdString(scene_doc::toPbrt(d)));
			sf.close();
			QString err;
			check(sb->openFile(scenePath, &err), "opened it " + err);
			sb->selectObject(static_cast<int>(sb->document().objects.size()) - 1);
			QFile::remove(scenePath);
		}
		// The mesh is read on a worker thread: it is ready a moment after the scene opens (and nothing waited for it).
		{
			const QString objPath = QDir::tempPath() + "/builder3d_selftest_mesh.obj";
			bool ready = false;
			for (int i = 0; i < 100 && !ready; ++i) {
				QCoreApplication::processEvents(QEventLoop::AllEvents, 50);
				ready = sb->meshReadyForTest(objPath);
				if (!ready) QThread::msleep(20);
			}
			check(ready, "the mesh preview was read in the background");
		}
		QTimer::singleShot(400, this, [this, shot, log, ok, sb]() {
			shot("builder3d");
			// RT_GUI_SELFTEST_OPEN=<scene.pbrt>: one more picture, of that builder scene in this view (to look at how a shape or scene draws).
			const QString open = qEnvironmentVariable("RT_GUI_SELFTEST_OPEN");
			if (!open.isEmpty()) {
				QString err;
				log(sb->openFile(open, &err) ? "opened " + open : "FAIL: could not open " + open + ": " + err);
				QTimer::singleShot(500, this, [this, shot, log, ok]() {
					shot("builder3d_open");
					log(ok ? "RESULT: OK" : "RESULT: FAIL");
					QApplication::exit(ok ? 0 : 1);
				});
				return;
			}
			log(ok ? "RESULT: OK" : "RESULT: FAIL");
			QApplication::exit(ok ? 0 : 1);
		});
	});
}
