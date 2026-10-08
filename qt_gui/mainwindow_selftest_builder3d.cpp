// RT_GUI_SELFTEST=builder3d RT_GUI_SELFTEST_OUT=<prefix>: the Scene Builder's 3D view, driven with real mouse events: dragging an object on the floor
// (snapped to the 0.25 grid, height unchanged, one undo step), dragging each coloured arrow (only that coordinate changes), and a screenshot with an
// object selected (<prefix>_builder3d.png). Exit 0 if every step held.
#include "mainwindow.h"

#include "scene_builder_widget.h"

#include <QApplication>
#include <QTimer>

#include <cmath>

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
		const scene_doc::Float3 ball0 = sb->document().objects[1].position;
		check(sb->dragObject3dForTest(1, QPointF(70, -25)), "dragging the glass ball in the 3D view moves it");
		const scene_doc::Float3 ball1 = sb->document().objects[1].position;
		check(ball1.y == ball0.y, "it stayed at its height (a drag on the floor plane)");
		check(std::fabs(ball1.x * 4 - std::round(ball1.x * 4)) < 1e-9 && std::fabs(ball1.z * 4 - std::round(ball1.z * 4)) < 1e-9, "its new position is on the grid");
		check(ball1.x != ball0.x || ball1.z != ball0.z, "it moved on the floor");
		check(sb->undo() && sb->document().objects[1].position.x == ball0.x && sb->document().objects[1].position.z == ball0.z, "one undo puts it back");
		const char *names[3] = {"X", "Y", "Z"};
		for (int axis = 0; axis < 3; ++axis) {
			const scene_doc::Float3 before = sb->document().objects[2].position;
			check(sb->dragAxis3dForTest(2, axis, 60), QString("dragging the %1 arrow changes only %1").arg(names[axis]));
			check(sb->undo(), QString("and one undo reverts the %1 drag").arg(names[axis]));
			const scene_doc::Float3 after = sb->document().objects[2].position;
			check(after.x == before.x && after.y == before.y && after.z == before.z, QString("back where it was after %1").arg(names[axis]));
		}
		sb->selectObject(3);
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
