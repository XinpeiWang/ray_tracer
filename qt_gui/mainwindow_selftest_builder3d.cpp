// RT_GUI_SELFTEST=builder3d RT_GUI_SELFTEST_OUT=<prefix>: the Scene Builder's 3D view, driven with real mouse events: dragging an object on the floor
// (snapped to the 0.25 grid, height unchanged, one undo step), dragging each coloured arrow (only that coordinate changes), and a screenshot with an
// object selected (<prefix>_builder3d.png). Exit 0 if every step held.
#include "mainwindow.h"

#include "scene_builder_widget.h"
#include "scene_3d_view.h"
#include "app_log.h"
#include "window_geometry.h"
#include "atomic_file.h"
#include "crash_recovery.h"
#include "render_queue_model.h"
#include "denoiser_installer.h"
#include "../src/shared/oidn_runtime.h"
#include <random>
#include "scene_metadata_client.h"
#include "scene_builder_widget.h"

#include <QApplication>
#include <QCheckBox>
#include <QKeyEvent>
#include <QListWidget>
#include <QRegularExpression>
#include <QTextEdit>
#include <QComboBox>
#include <QScreen>
#include <QGuiApplication>
#include <QScrollArea>
#include <QScrollBar>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWheelEvent>
#include <QFileInfo>
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

// Part of the "builder" mode: every ready-made shape and prop can be added, leaves the scene without problems, and is one undo step each.
void MainWindow::selfTestShapes(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	// Every ready-made shape can be added (and drawn in the layout and 3D views), leaves the scene without problems, and is one undo step each.
	{
		const size_t before = sb->document().objects.size();
		size_t added = 0;
		for (scene_doc::ShapeKind k : scene_doc::allShapeKinds()) {
			if (k == scene_doc::ShapeKind::Mesh) continue;   // needs a file
			sb->addObject(k);
			++added;
		}
		check(sb->document().objects.size() == before + added, QString("added all %1 ready-made shapes").arg(added));
		check(sb->problemsText().isEmpty(), "the ready-made shapes give no problems or notes");
		// And every prop (several objects in one step, a second copy named apart).
		const size_t withShapes = sb->document().objects.size();
		for (scene_doc::PropKind k : scene_doc::allPropKinds()) sb->addProp(k);
		size_t propObjects = 0;
		for (scene_doc::PropKind k : scene_doc::allPropKinds()) propObjects += scene_doc::makeProp(k).size();
		check(sb->document().objects.size() == withShapes + propObjects, QString("added all %1 props (%2 objects)").arg(scene_doc::allPropKinds().size()).arg(propObjects));
		sb->addProp(scene_doc::PropKind::Table);
		check(sb->document().objects.back().name.find("Table leg") != std::string::npos && sb->document().objects.back().name.back() == '2', "a second table is named apart (\"... 2\")");
		check(sb->problemsText().isEmpty(), "the props give no problems or notes");
		bool propsUndone = true;
		for (size_t n = 0; n <= scene_doc::allPropKinds().size(); ++n) propsUndone = sb->undo() && propsUndone;
		check(propsUndone && sb->document().objects.size() == withShapes, "each prop is one undo step");
		bool allUndone = true;
		for (size_t n = 0; n < added; ++n) allUndone = sb->undo() && allUndone;
		check(allUndone && sb->document().objects.size() == before, "undo removes them one by one");
		for (size_t n = 0; n < added; ++n) sb->redo();
		for (size_t n = 0; n < added; ++n) sb->undo();
	}
	// The Preset list in a material's properties: choosing "Gold" in the real combo box makes the selected object a gold conductor, as one undo step.
	{
		sb->selectObject(1);
		QApplication::processEvents();
		QComboBox *presets = nullptr;
		int goldIndex = -1;
		for (QComboBox *c : sb->findChildren<QComboBox *>()) {
			const int idx = c->findData(QStringLiteral("gold"));
			if (idx >= 0) { presets = c; goldIndex = idx; }
		}
		check(presets != nullptr, "the material properties have a Preset list");
		if (presets) {
			const scene_doc::MaterialKind kindBefore = sb->document().objects[1].material.kind;
			presets->setCurrentIndex(goldIndex);
			emit presets->activated(goldIndex);
			QApplication::processEvents();
			const scene_doc::Material &m = sb->document().objects[1].material;
			check(m.kind == scene_doc::MaterialKind::Conductor && m.roughness < 0.2 && m.color.r > m.color.b, "choosing Gold makes the object a gold metal");
			check(sb->problemsText().isEmpty(), "a preset material gives no problems or notes");
			check(sb->undo() && sb->document().objects[1].material.kind == kindBefore, "one undo takes the preset back");
			sb->redo();
			sb->undo();
		}
	}
}

// Part of the "builder" mode: the log file has what was just done.
void MainWindow::selfTestLog(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	{
		// The log file (app_log.h) has what was just done: the session header, the tab change, the edits (as readable differences), the save and the open.
		for (QCheckBox *box : sb->findChildren<QCheckBox *>())
			if (box->text() == SceneBuilderWidget::tr("Snap to grid")) { box->click(); box->click(); }   // a real click on a real control, off and on again
		// What the user does in the editor besides editing: picking things, deleting from the middle of the list, the 3D tool keys, scenes with problems,
		// a mesh the 3D view cannot read, and the last edit before the widget goes away.
		sb->addObject(scene_doc::ShapeKind::Box);
		sb->addObject(scene_doc::ShapeKind::Sphere);
		sb->addObject(scene_doc::ShapeKind::Box);
		if (QListWidget *list = sb->findChild<QListWidget *>()) {
			list->setCurrentRow(0);
			list->setCurrentRow(2);   // a click on the second item of the list (row 0 is "Camera and image")
		}
		const int middle = std::max(0, static_cast<int>(sb->document().objects.size()) - 2);
		const QString middleName = QString::fromStdString(sb->document().objects[middle].name);
		sb->selectObject(middle);
		sb->deleteSelected();
		sb->undo();   // (undo writes the pending edit to the log first)
		sb->show3dView(true);
		if (Scene3DView *view = sb->findChild<Scene3DView *>()) {
			QKeyEvent e(QEvent::KeyPress, Qt::Key_E, Qt::NoModifier);
			QCoreApplication::sendEvent(view, &e);
			QKeyEvent r(QEvent::KeyPress, Qt::Key_W, Qt::NoModifier);
			QCoreApplication::sendEvent(view, &r);
		}
		{
			const QString bad = QDir::temp().absoluteFilePath("rt_selftest_unreadable.ply");
			QFile(bad).open(QIODevice::WriteOnly);   // an empty file: not a mesh
			scene_doc::Document d = scene_doc::makeStarterScene();
			scene_doc::Object o = scene_doc::makeObject(scene_doc::ShapeKind::Mesh, "Unreadable mesh");
			o.meshFile = bad.toStdString();
			o.material.imageFile = "C:/rt_no_such_folder/missing_picture.png";   // a picture that is not there: the scene gets a note
			d.objects.push_back(o);
			const QString scenePath = QDir::temp().absoluteFilePath("rt_selftest_unreadable.pbrt");
			QFile f(scenePath);
			if (f.open(QIODevice::WriteOnly)) {
				const std::string t = scene_doc::toPbrt(d);
				f.write(t.data(), static_cast<qint64>(t.size()));
			}
			f.close();
			sb->openFile(scenePath);
			QEventLoop wait;
			QTimer::singleShot(900, &wait, &QEventLoop::quit);
			wait.exec();
			QFile::remove(scenePath);
			QFile::remove(bad);
		}
		{
			auto *temp = new SceneBuilderWidget();
			temp->setSceneName("quit-test-title");
			delete temp;   // the edit was still waiting for its pause when the widget went away
		}
		const QString text = AppLog::tail(3000).join('\n');
		check(text.contains("builder: select: object '") && text.contains("(list)"), "the log records picking an item in the list");
		check(text.contains("-1 object (" + middleName + ")"), "the log names the object that was deleted from the middle of the list: " + middleName);
		check(text.contains("3D view: tool Rotate (E key)") && text.contains("3D view: tool Move (W key)"), "the log records the 3D tool keys");
		check(text.contains("builder: scene problem - ") && text.contains("missing_picture.png"), "the log records a problem (a missing picture) the scene has");
		check(text.contains("3D view: cannot show the mesh ") && text.contains("rt_selftest_unreadable.ply"), "the log says which mesh the 3D view cannot show");
		check(text.contains("title '") && text.contains("-> 'quit-test-title'"), "the last edit is in the log although the widget was destroyed straight after it");
		check(!AppLog::filePath().isEmpty() && QFileInfo::exists(AppLog::filePath()), "the log file exists: " + AppLog::filePath());
		check(text.contains("[session]") || text.contains("session: started"), "the log has the session header");
		check(text.contains("startup: ") && text.contains("startup: window up "), "the log has the start-up timings and when the window was up");
		check(text.contains("ui: tab \"") , "the log records the tab change");
		check(text.contains("ui: Scene Builder > click \"Snap to grid\" -> off") && text.contains("click \"Snap to grid\" -> on"), "the log records a click on a checkbox, with its tab and new state");
		check(text.contains("builder: edit: ") && text.contains("+1 object (") , "the log records the added object as a readable edit");
		check(text.contains("builder: undo: ") && text.contains("builder: redo: "), "the log records undo and redo");
		check(text.contains("builder: save ") && text.contains("builder: open "), "the log records the save and the open");
		check(!text.contains("\n\n") && !AppLog::previousSessionEndedUnexpectedly(), "one entry per line, and the previous session ended cleanly");
	}
}

// Part of the "builder" mode: a spin box, combo box or slider that does not have focus lets the wheel scroll the page behind it instead of changing its value;
// once it has focus the wheel changes it.
void MainWindow::selfTestWheelGuard(const std::function<void(bool, const QString &)> &check) {
	QScrollArea area;
	area.resize(300, 200);
	auto *page = new QWidget;
	auto *layout = new QVBoxLayout(page);
	auto *spin = new QSpinBox(page);
	spin->setRange(0, 100);
	spin->setValue(50);
	auto *combo = new QComboBox(page);
	combo->addItems({"a", "b", "c"});
	auto *slider = new QSlider(Qt::Horizontal, page);
	slider->setRange(0, 100);
	slider->setValue(50);
	layout->addWidget(spin);
	layout->addWidget(combo);
	layout->addWidget(slider);
	layout->addSpacing(800);   // the page is much taller than the area, so it scrolls
	area.setWidget(page);
	area.setWidgetResizable(true);
	area.show();
	QApplication::processEvents();
	// A window that has just opened gives its first control the focus; the point here is a control that does not have it.
	if (QWidget *f = QApplication::focusWidget()) f->clearFocus();
	area.setFocus();
	QApplication::processEvents();
	auto wheel = [](QWidget *w, int delta) {
		const QPoint local = w->rect().center();
		QWheelEvent e(QPointF(local), QPointF(w->mapToGlobal(local)), QPoint(), QPoint(0, delta), Qt::NoButton, Qt::NoModifier, Qt::NoScrollPhase, false);
		QApplication::sendEvent(w, &e);
	};
	QScrollBar *bar = area.verticalScrollBar();
	check(bar->maximum() > 100, "the test page scrolls");
	const int before = bar->value();
	wheel(spin, -120);
	wheel(combo, -120);
	wheel(slider, -120);
	check(spin->value() == 50 && combo->currentIndex() == 0 && slider->value() == 50, "the wheel over an unfocused spin box, combo box and slider changes none of them");
	check(bar->value() > before, QString("... and scrolls the page instead (%1 -> %2 of %3)").arg(before).arg(bar->value()).arg(bar->maximum()));
	spin->setFocus();
	QApplication::processEvents();
	wheel(spin, 120);
	check(spin->hasFocus() ? spin->value() == 51 : true, "once the spin box has focus the wheel changes it");
}

// Part of the "builder" mode: the window's geometry is saved and comes back, but a saved place that is on no screen is refused.
void MainWindow::selfTestWindowGeometry(const std::function<void(bool, const QString &)> &check) {
	const QRect original = geometry();
	resize(700, 560);   // (the offscreen test screen is small)
	QApplication::processEvents();
	const QSize saved = size();
	window_geometry::save(this);
	resize(780, 600);
	check(window_geometry::restore(this) && size() == saved, QString("the saved window size comes back (%1 x %2)").arg(saved.width()).arg(saved.height()));
	move(40000, 40000);   // a place no screen has
	QApplication::processEvents();
	window_geometry::save(this);
	setGeometry(original);
	// Qt itself pulls such a window back onto a screen; either way it must never come back unreachable.
	const bool restored = window_geometry::restore(this);
	bool onScreen = false;
	for (const QScreen *s : QGuiApplication::screens()) onScreen = onScreen || s->availableGeometry().intersects(frameGeometry());
	check(!restored || onScreen, "a window saved on no connected screen does not come back off-screen");
	window_geometry::forget();
	check(!window_geometry::restore(this), "after Reset Window Layout nothing is restored");
	// The render form remembers the scene and the image size: change both, save, change them again, restore.
	if (m_sceneCombo && m_sceneCombo->count() > 1 && m_widthSpinBox && m_heightSpinBox) {
		const int sceneIndex = m_sceneCombo->count() - 1;
		const QString sceneId = m_sceneCombo->itemData(sceneIndex).toString();
		m_sceneCombo->setCurrentIndex(sceneIndex);
		m_widthSpinBox->setValue(640);
		m_heightSpinBox->setValue(360);
		saveRenderForm();
		m_sceneCombo->setCurrentIndex(0);
		m_widthSpinBox->setValue(800);
		m_heightSpinBox->setValue(800);
		restoreRenderForm();
		check(m_sceneCombo->currentData().toString() == sceneId && m_widthSpinBox->value() == 640 && m_heightSpinBox->value() == 360, "the render form brings back the saved scene and image size");
		m_sceneCombo->setCurrentIndex(0);
		m_widthSpinBox->setValue(800);
		m_heightSpinBox->setValue(800);
	}
	// Atomic writes: the new text replaces the old in one step and no temporary file is left beside it.
	{
		const QString dir = QDir::tempPath() + "/rt_atomic_selftest";
		QDir().mkpath(dir);
		const QString path = dir + "/file.txt";
		QFile::remove(path);
		check(writeFileAtomically(path, "first") && writeFileAtomically(path, "second\n"), "an atomic write succeeds twice over the same file");
		QFile f(path);
		check(f.open(QIODevice::ReadOnly) && f.readAll() == "second\n" && QDir(dir).entryList(QDir::Files).size() == 1, "it holds the new text and left no temporary file");
		f.close();
		QDir(dir).removeRecursively();
		check(!writeFileAtomically(dir + "/no/such/folder/file.txt", "x"), "writing into a missing folder fails cleanly");
	}
	setGeometry(original);
	QApplication::processEvents();
}

// Part of the "builder" mode: "Start fresh" after a crash forgets the saved window layout and sets the unsaved scene aside (renamed, not deleted).
void MainWindow::selfTestCrashRecovery(const std::function<void(bool, const QString &)> &check) {
	const QString dir = QDir::tempPath() + "/rt_recovery_selftest";
	QDir(dir).removeRecursively();
	QDir().mkpath(dir);
	qputenv("RAY_TRACER_STATE_DIR", dir.toUtf8());   // the unsaved scene lives here for this test, not in the real per-user folder
	check(!SceneBuilderWidget::hasAutosave() && crash_recovery::startFresh().isEmpty(), "with no unsaved scene, a fresh start sets nothing aside");
	check(writeFileAtomically(dir + "/scene_builder_autosave.pbrt", "# unsaved") && SceneBuilderWidget::hasAutosave(), "an unsaved scene is waiting");
	window_geometry::save(this);
	const QString aside = crash_recovery::startFresh();
	check(!aside.isEmpty() && QFileInfo::exists(aside) && !SceneBuilderWidget::hasAutosave(), "a fresh start renames the unsaved scene (it is not deleted): " + aside);
	QFile f(aside);
	check(f.open(QIODevice::ReadOnly) && f.readAll() == "# unsaved", "and the renamed file still has its content");
	f.close();
	check(!window_geometry::restore(this), "and forgets the saved window layout");
	qunsetenv("RAY_TRACER_STATE_DIR");
	QDir(dir).removeRecursively();
}

// Part of the "builder" mode: the Progress tab's render queue is a table of waiting, running and finished jobs; each button is available only where it means
// something, and finished jobs stay as a record and can be run again. (The model is driven directly: no render is started.)
void MainWindow::selfTestRenderQueue(const std::function<void(bool, const QString &)> &check) {
	using job_queue::State;
	RenderJob a, b, c;
	a.displayTitle = "Scene A";
	b.displayTitle = "Scene B";
	c.displayTitle = "Scene C";
	for (RenderJob *j : {&a, &b, &c}) { j->width = 320; j->height = 240; j->samples = 16; }
	const int ida = m_queueModel->add(a), idb = m_queueModel->add(b), idc = m_queueModel->add(c);
	check(m_queueView->model()->rowCount() == 3 && m_queueModel->waitingCount() == 3, "three queued jobs are three rows of the table");
	check(m_queueModel->data(m_queueModel->index(0, RenderQueueModel::SceneColumn)).toString() == "Scene A" &&
	          m_queueModel->data(m_queueModel->index(0, RenderQueueModel::StatusColumn)).toString() == RenderQueueModel::tr("Waiting"),
	      "a row shows the scene and its state");
	m_queueView->selectRow(0);
	QApplication::processEvents();
	check(m_queueUpButton->isEnabled() && m_queueDownButton->isEnabled() && !m_queueRetryButton->isEnabled(), "a waiting job can be moved but not retried");
	check(m_queueModel->moveDown(ida) && m_queueModel->idAt(0) == idb && m_queueModel->idAt(1) == ida, "moving a waiting job down changes the order it will run in");
	int id = 0;
	RenderJob taken;
	check(m_queueModel->takeNext(id, taken) && id == idb && taken.displayTitle == "Scene B", "the first waiting job in the new order starts");
	m_queueView->selectRow(0);
	QApplication::processEvents();
	check(!m_queueRemoveButton->isEnabled() && !m_queueUpButton->isEnabled(), "a running job can be neither removed nor moved");
	check(m_queueModel->finish(idb, State::Failed, 4.2, "boom") && m_queueModel->finishedCount() == 1, "a failed job stays in the table as a record");
	m_queueView->selectRow(0);
	QApplication::processEvents();
	check(m_queueRetryButton->isEnabled() && m_queueClearFinishedButton->isEnabled() && m_queueRemoveButton->isEnabled(), "it can be retried, removed, or cleared with the finished ones");
	const int again = m_queueModel->retry(idb);
	check(again != 0 && m_queueModel->waitingCount() == 3 && m_queueModel->queue().find(again)->job.displayTitle == "Scene B", "retrying queues it again as a new waiting job");
	check(m_queueModel->data(m_queueModel->index(0, RenderQueueModel::TimeColumn)).toString() == RenderQueueModel::tr("%1 s").arg(4.2, 0, 'f', 1), "and the old row keeps how long it ran");
	check(m_queueModel->clearWaiting() == 3 && m_queueModel->rowCount() == 1, "Clear Queue removes only the waiting jobs");
	check(m_queueModel->clearFinished() == 1 && m_queueModel->rowCount() == 0, "Clear Finished removes only the finished ones");
	(void)idc;
	refreshQueuePanel();
}

// RT_GUI_SELFTEST=queue: two real tiny CPU renders through the Render button. The second is queued behind the first; both rows must end Done with a time, stay
// in the table as a record, and Clear Finished must then empty it. Needs ray_tracer next to the GUI.
void MainWindow::runQueueSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	if (m_progressTabIndex >= 0) m_tabWidget->setCurrentIndex(m_progressTabIndex);
	resize(1100, 800);
	auto fail = [log](const QString &what) {
		log("FAIL: " + what);
		log("RESULT: FAIL");
		QApplication::exit(1);
	};
	for (int i = 0; i < m_renderModeCombo->count(); ++i)
		if (!m_renderModeCombo->itemData(i).toBool()) m_renderModeCombo->setCurrentIndex(i);   // the CPU
	m_qualityPresetCombo->setCurrentIndex(6);   // Custom: the size below
	m_widthSpinBox->setValue(64);
	m_heightSpinBox->setValue(48);
	m_samplesSpinBox->setValue(1);
	m_maxDepthSpinBox->setValue(2);
	m_aovsCheck->setChecked(true);   // the first job also writes the render passes
	onRenderClicked();
	m_aovsCheck->setChecked(false);
	onRenderClicked();   // while the first one renders: queued behind it
	if (m_queueModel->rowCount() != 2 || !m_queueModel->isRunning() || m_queueModel->waitingCount() != 1) {
		fail(QString("after two clicks the queue should be one running and one waiting (rows %1, running %2, waiting %3)")
		         .arg(m_queueModel->rowCount()).arg(m_queueModel->isRunning()).arg(m_queueModel->waitingCount()));
		return;
	}
	log("ok: the second click queued a job behind the running one");
	auto *poll = new QTimer(this);
	auto *waited = new int(0);
	connect(poll, &QTimer::timeout, this, [this, poll, waited, log, shot, fail]() {
		if (++*waited > 300) { poll->stop(); fail("the two renders did not finish in 90 s"); return; }
		if (m_queueModel->finishedCount() < 2) return;
		poll->stop();
		bool ok = true;
		auto check = [&ok, log](bool cond, const QString &what) { log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what)); ok = ok && cond; };
		check(m_queueModel->rowCount() == 2 && !m_queueModel->hasWaiting() && !m_queueModel->isRunning(), "both jobs are finished and still listed");
		for (int row = 0; row < 2; ++row) {
			check(m_queueModel->stateAt(row) == job_queue::State::Done, QString("row %1 is Done").arg(row + 1));
			const QString time = m_queueModel->data(m_queueModel->index(row, RenderQueueModel::TimeColumn)).toString();
			check(!time.isEmpty(), QString("row %1 shows how long it took: %2").arg(row + 1).arg(time));
		}
		check(m_queueClearFinishedButton->isEnabled() && !m_queueClearButton->isEnabled(), "Clear Finished is available, Clear Queue is not (nothing waits)");
		{
			// The first job asked for the render passes (--aovs): they are beside its image, in <stem>.aovs.exr; the second did not ask.
			const QFileInfo first(m_queueModel->queue().entries()[0].job.outputPath), second(m_queueModel->queue().entries()[1].job.outputPath);
			const QString firstPasses = first.absolutePath() + "/" + first.completeBaseName() + ".aovs.exr", secondPasses = second.absolutePath() + "/" + second.completeBaseName() + ".aovs.exr";
			check(QFileInfo::exists(firstPasses) && QFileInfo(firstPasses).size() > 1000, "the first job wrote its render passes: " + firstPasses);
			check(!QFileInfo::exists(secondPasses), "the second job, which did not ask, wrote none");
			QFile::remove(firstPasses);
		}
		if (m_progressTabIndex >= 0) m_tabWidget->setCurrentIndex(m_progressTabIndex);   // a finished render switches to Preview: look at the queue
		QApplication::processEvents();
		shot("queue");
		onClearFinishedJobs();
		check(m_queueModel->rowCount() == 0, "Clear Finished empties the table");
		log(ok ? "RESULT: OK" : "RESULT: FAIL");
		QApplication::exit(ok ? 0 : 1);
	});
	poll->start(300);
}

// RT_GUI_SELFTEST=tour (RT_GUI_SELFTEST_WIDTH / _HEIGHT size the window): a screenshot of every tab, for looking over the whole UI at one window size.
void MainWindow::runTourSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	const int w = qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") : 1100;
	const int h = qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") : 800;
	resize(w, h);
	if (qEnvironmentVariableIsSet("RT_GUI_SELFTEST_THEME")) switchTheme(qEnvironmentVariable("RT_GUI_SELFTEST_THEME"));   // e.g. solarized-light
	if (qEnvironmentVariableIsSet("RT_GUI_SELFTEST_MYSCENE") && m_sceneBuilder) {   // show the Settings tab with a scene of the user's own selected
		QString error;
		const QString file = m_sceneBuilder->addToSceneList(&error);
		SceneMetadataClient::refreshUserScenes();
		selectSceneById(SceneMetadataClient::sceneIdForFile(file));
	}
	auto *index = new int(0);
	auto *step = new QTimer(this);
	connect(step, &QTimer::timeout, this, [this, step, index, log, shot]() {
		if (*index >= m_tabWidget->count()) {
			step->stop();
			log("RESULT: OK");
			QApplication::exit(0);
			return;
		}
		m_tabWidget->setCurrentIndex(*index);
		QApplication::processEvents();
		if (m_tabWidget->currentWidget()->findChild<QTextEdit *>() == m_diagTextEdit)
			log(QString("diagnostics pane: empty=%1 placeholder length=%2 visible=%3").arg(m_diagTextEdit->document()->isEmpty()).arg(m_diagTextEdit->placeholderText().size()).arg(m_diagTextEdit->isVisible()));
		QString name = m_tabWidget->tabText(*index);
		name.remove(QRegularExpression("[^A-Za-z0-9]+"));
		QTimer::singleShot(350, this, [shot, name, index]() { shot(QString("tour%1_%2").arg(*index - 1).arg(name)); });
		++*index;
	});
	step->start(900);
}

// Part of the "builder" mode: Delete Scene and Delete All My Scenes (without their questions): the files go, the scenes leave the list at once, the selection
// moves to a neighbour, and the buttons follow the category.
void MainWindow::selfTestDeleteScenes(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	QString error;
	sb->setSceneName("Delete test A");
	const QString fileA = sb->addToSceneList(&error);
	sb->setSceneName("Delete test B");
	const QString fileB = sb->addToSceneList(&error);
	SceneMetadataClient::refreshUserScenes();
	const QString idA = SceneMetadataClient::sceneIdForFile(fileA), idB = SceneMetadataClient::sceneIdForFile(fileB);
	check(!idA.isEmpty() && !idB.isEmpty() && idA != idB, "two scenes were added to the list: " + idA + ", " + idB);
	selectSceneById(idA);
	check(!m_myScenesRow->isHidden() && m_deleteSceneButton->isEnabled() && m_deleteAllScenesButton->isEnabled(), "under My Scenes the delete buttons are shown and enabled");
	check(deleteUserScenes({idA}, idB) == 1 && !QFile::exists(fileA) && QFile::exists(fileB), "Delete Scene removes that scene's file and no other");
	check(SceneMetadataClient::sceneCategory(idA).isEmpty() && SceneMetadataClient::sceneCategory(idB) == QString("My Scenes"), "the deleted scene is gone from the list at once, the other stays");
	check(m_sceneCombo->currentData().toString() == idB, "the selection moved to the neighbouring scene");
	check(m_sceneCombo->findData(idA) < 0, "and the scene picker no longer offers the deleted one");
	check(!userSceneFileForId(idA).isEmpty() == false, "its file is no longer found");
	// "My Scenes" also lists scenes found elsewhere (e.g. .pbrt files a developer keeps in the pbrt_scenes folder); those are not the Scene Builder's to delete,
	// so Delete All must remove exactly the ones in the scene-list folder and leave the rest alone.
	const QStringList mine = myScenesIds();
	QStringList deletable;
	for (const QString &id : mine)
		if (!userSceneFileForId(id).isEmpty()) deletable << id;
	const int removed = deleteUserScenes(mine, QString());
	const QStringList left = myScenesIds();
	check(removed == deletable.size() && left.size() == mine.size() - deletable.size(), "Delete All My Scenes removes every scene the Scene Builder saved, and only those");
	check(!QFile::exists(fileB), "including the file");
	bool tabStillThere = false;
	for (int i = 0; m_sceneCategoryTabs && i < m_sceneCategoryTabs->count(); ++i) tabStillThere = tabStillThere || m_sceneCategoryTabs->tabData(i).toString() == QString("My Scenes");
	if (left.isEmpty()) {
		check(!tabStillThere, "the My Scenes tab is gone when none are left");
		check(!m_sceneCombo->currentData().toString().isEmpty() && m_myScenesRow->isHidden(), "a built-in scene is selected and the delete buttons are hidden");
	} else {
		check(tabStillThere, "the My Scenes tab stays while scenes that are not the Scene Builder's are listed");
		check(!m_sceneCombo->currentData().toString().isEmpty(), "and a scene is selected");
	}
}

// RT_GUI_SELFTEST=denoiser (not in the default set: it downloads ~50 MB from GitHub): installs Open Image Denoise through the real installer into a throwaway
// folder, checks it is found, and denoises a small noisy image through it. Mac only.
void MainWindow::runDenoiserSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &) {
	auto fail = [log](const QString &what) { log("FAIL: " + what); log("RESULT: FAIL"); QApplication::exit(1); };
	if (!denoiser_installer::isSupportedHere()) { log("not supported here (macOS only): skipped"); log("RESULT: OK"); QApplication::exit(0); return; }
	if (denoiser_installer::isInstalled()) { fail("the throwaway user folder already has a denoiser: " + denoiser_installer::installFolder()); return; }
	log("ok: not installed before: " + denoiser_installer::installFolder());
	auto *installer = new denoiser_installer::Installer(this);
	connect(installer, &denoiser_installer::Installer::progress, this, [log](int percent, const QString &text) {
		static int last = -10;
		if (percent >= last + 10 || percent == 100) { last = percent; log(QString("  %1%: %2").arg(percent).arg(text)); }
	});
	connect(installer, &denoiser_installer::Installer::finished, this, [log, fail](bool ok, const QString &message) {
		if (!ok) { fail("install failed: " + message); return; }
		log("ok: installed: " + message);
		if (!denoiser_installer::isInstalled()) { fail("isInstalled() is false after installing"); return; }
		log("ok: isInstalled()");
		const int w = 64, h = 64;
		std::vector<float> img(w * h * 4, 1.0f);
		std::mt19937 rng(3);
		std::normal_distribution<float> n(0.0f, 0.4f);
		for (int i = 0; i < w * h; ++i) for (int c = 0; c < 3; ++c) img[i * 4 + c] = std::max(0.0f, 0.5f + n(rng));
		auto spread = [&](const std::vector<float> &v) { double m = 0, s = 0; for (int i = 0; i < w * h; ++i) m += v[i * 4]; m /= w * h; for (int i = 0; i < w * h; ++i) s += (v[i * 4] - m) * (v[i * 4] - m); return s / (w * h); };
		const double before = spread(img);
		std::string error;
		if (!oidn_runtime::denoiseHdr(img.data(), w, h, 4, 1.0f, error)) { fail("denoise through the installed library failed: " + QString::fromStdString(error)); return; }
		const double after = spread(img);
		log(QString("ok: denoised through %1 (noise variance %2 -> %3)").arg(QString::fromStdString(oidn_runtime::libraryPath())).arg(before).arg(after));
		if (!(after < before * 0.2)) { fail("the noise did not drop enough"); return; }
		log("RESULT: OK");
		QApplication::exit(0);
	});
	installer->start();
}
