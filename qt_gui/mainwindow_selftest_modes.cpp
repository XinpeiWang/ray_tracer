// mainwindow_selftest_modes.cpp - the smaller self-test modes (options, builder, diagnostics, scenekeys, download, ui), each its own function called by
// runSelfTest() in mainwindow_selftest.cpp. A pure split of what used to be one 500-line function.

#include "mainwindow.h"
#include "scene_builder_array_dialog.h"
#include "../src/shared/oidn_runtime.h"
#include "denoiser_installer.h"
#include "live_ai_denoise.h"
#include "app_log.h"
#include <iostream>
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QAbstractSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QCheckBox>
#include <QComboBox>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QPixmap>
#include <QProcess>
#include <QSysInfo>
#include <QPainter>
#include <QList>
#include <QScrollArea>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QTextStream>
#include <QTimer>
#include <QPlainTextEdit>
#include <QRegularExpression>
#include <QSpinBox>
#include <functional>
#include <memory>
#include <cmath>
#include "scene_builder_widget.h"
#include "scene_technique_notes.h"
#include "scene_metadata_client.h"

// The Scene Builder's "Add to scene list" checks: where the copy goes (never into a .app bundle), that it is in the scene list at once under its

void MainWindow::runOptionsSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	// Every button, drop-down, line edit and spin box is the same height (46px: a 40px box and a 3px margin above and below - see the QPushButton rule in
	// mainwindow_style.cpp), on every tab. Looks at the visible ones; a control that sets its own height shows up here.
	{
		resize(1500, 1000);
		show();
		QStringList wrong;
		int checked = 0;
		for (int t = 0; t < m_tabWidget->count(); ++t) {
			m_tabWidget->setCurrentIndex(t);
			QApplication::processEvents();
			for (QWidget *w : m_tabWidget->currentWidget()->findChildren<QWidget *>()) {
				if (!w->isVisible()) continue;
				QString kind;
				if (qobject_cast<QPushButton *>(w)) kind = "button";
				else if (qobject_cast<QComboBox *>(w)) kind = "drop-down";
				else if (qobject_cast<QAbstractSpinBox *>(w)) kind = "spin box";
				else if (qobject_cast<QLineEdit *>(w) && !qobject_cast<QAbstractSpinBox *>(w->parentWidget()) && !qobject_cast<QComboBox *>(w->parentWidget())) kind = "line edit";
				else continue;
				++checked;
				if (w->height() != 46) {
					const QString text = qobject_cast<QPushButton *>(w) ? qobject_cast<QPushButton *>(w)->text() : QString();
					wrong << QString("%1 \"%2\" %3 (%4): %5px").arg(kind, text, w->objectName(), m_tabWidget->tabText(t)).arg(w->height());
				}
			}
		}
		log(QString("%1: %2 controls checked, every one 46px tall%3").arg(wrong.isEmpty() ? "ok" : "FAIL").arg(checked).arg(wrong.isEmpty() ? QString() : QString(" - not: ") + wrong.join("; ")));
		if (!wrong.isEmpty()) { QApplication::exit(1); return; }
	}
	// The Render Options tab with Live Preview selected, large enough to read: which live controls does this platform show?
	resize(1100, 1500);
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
	if (idx >= 0) m_modeCombo->setCurrentIndex(idx);
#endif
	for (int i = 0; i < m_tabWidget->count(); ++i) {
		log(QString("tab %1: %2").arg(i).arg(m_tabWidget->tabText(i)));
		if (m_tabWidget->tabText(i).contains("Render Options")) m_tabWidget->setCurrentIndex(i);
	}
	QTimer::singleShot(800, this, [this, shot, log]() {
		shot("options");
#ifdef Q_OS_MAC
		// Ticking "AI denoise" (Live Preview) with no denoiser library present offers to download it; declining must untick the box again and not leave it
		// remembered as on. (Skipped when this machine has the library, or the app cannot install it.)
		QCheckBox *aiBox = nullptr;
		for (QCheckBox *c : findChildren<QCheckBox *>())
			if (c->text() == tr("AI denoise")) aiBox = c;
		if (aiBox && !oidn_runtime::available() && denoiser_installer::isSupportedHere()) {
			QTimer::singleShot(400, this, []() {
				if (auto *question = qobject_cast<QMessageBox *>(QApplication::activeModalWidget()))
					for (QAbstractButton *b : question->buttons())
						if (question->buttonRole(b) == QMessageBox::NoRole) b->click();
			});
			aiBox->setChecked(true);   // returns once the question is answered
			const bool untickedAgain = !aiBox->isChecked() && !live_ai_denoise::savedEnabled();
			log(QString("%1: declining the denoiser download unticks \"AI denoise\" and does not remember it").arg(untickedAgain ? "ok" : "FAIL"));
			if (!untickedAgain) { QApplication::exit(1); return; }
		} else {
			log("skipped: the AI denoise download prompt (the denoiser library is present here, or not installable on this platform)");
		}
#endif
		// ...and scrolled to the bottom, where the Live Preview Settings group is.
		QList<QScrollArea *> areas = m_tabWidget->currentWidget()->findChildren<QScrollArea *>();
		if (auto *self = qobject_cast<QScrollArea *>(m_tabWidget->currentWidget())) areas.prepend(self);
		for (QScrollArea *sa : areas)
			sa->verticalScrollBar()->setValue(sa->verticalScrollBar()->maximum());
		QTimer::singleShot(500, this, [shot]() { shot("options_bottom"); QApplication::exit(0); });
	});
	return;
}

	// RT_GUI_SELFTEST=builder: drives the Scene Builder tab through an edit, undo/redo, a save and re-open, and a real preview render (needs
	// ray_tracer next to the GUI); saves screenshots <out>_builder_edit.png / _builder_preview.png and exits 0 if every step held.
// Duplicate, Array and Scatter (scene_array.h, the Array... window): each is one undo step, the copies are ordinary objects with their own names, and the window
// itself offers the same objects it would add. Leaves the document as it found it (everything is undone).
static void selfTestArray(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	const size_t n0 = sb->document().objects.size();
	sb->selectObject(2);   // the gold ball
	const std::string sourceName = sb->document().objects[2].name;
	sb->duplicateSelected();
	check(sb->document().objects.size() == n0 + 1 && sb->document().objects.back().name == sourceName + " 2", "Duplicate makes a numbered copy (" + QString::fromStdString(sb->document().objects.back().name) + ")");
	check(sb->document().objects.back().position.x > sb->document().objects[2].position.x, "the copy sits beside the original");
	check(sb->undo() && sb->document().objects.size() == n0, "one undo removes the duplicate");
	// A window with the defaults: a grid of three in a row (two new), a ring of eight (seven new), a scatter of twenty.
	ArrayDialog dialog(2, sb->document().objects, sb);
	check(dialog.copies().size() == 2, "the Array window's default grid adds 2 objects");
	dialog.setMode(ArrayDialog::Mode::Ring);
	check(dialog.copies().size() == 7, "its default ring of 8 adds 7 (the original is one of them)");
	dialog.setRingCount(5);
	check(dialog.copies().size() == 4, "a ring of 5 adds 4");
	dialog.setMode(ArrayDialog::Mode::Scatter);
	const size_t scattered = dialog.copies().size();
	check(scattered > 0 && scattered <= 20, "the scatter adds up to the 20 asked for (" + QString::number(scattered) + ")");
	// RT_GUI_SELFTEST_ARRAY_DIALOG_PNG=<prefix> saves a picture of each page of the window (<prefix>_grid.png, _ring.png, _scatter.png).
	const QString picture = qEnvironmentVariable("RT_GUI_SELFTEST_ARRAY_DIALOG_PNG");
	if (!picture.isEmpty()) {
		dialog.show();
		const std::pair<ArrayDialog::Mode, const char *> pages[] = {{ArrayDialog::Mode::Grid, "grid"}, {ArrayDialog::Mode::Ring, "ring"}, {ArrayDialog::Mode::Scatter, "scatter"}};
		for (const auto &page : pages) {
			dialog.setMode(page.first);
			QApplication::processEvents();
			dialog.grab().save(picture + "_" + page.second + ".png");
		}
		dialog.hide();
	}
	dialog.setMode(ArrayDialog::Mode::Grid);
	dialog.setGridCounts(4, 1, 3);
	const std::vector<scene_doc::Object> grid = dialog.copies();
	check(grid.size() == 11, "a 4 x 1 x 3 grid adds 11 objects");
	sb->addCopies(grid);
	check(sb->document().objects.size() == n0 + 11 && sb->document().objects[n0].name == sourceName + " 2", "the copies are added with their own names");
	check(sb->undo() && sb->document().objects.size() == n0, "all eleven are one undo step");
	check(sb->redo() && sb->document().objects.size() == n0 + 11, "and one redo");
	sb->addCopies(std::vector<scene_doc::Object>());
	check(sb->document().objects.size() == n0 + 11, "adding no copies changes nothing");
	check(sb->undo() && sb->document().objects.size() == n0, "back to the scene as it was");
	// A prop is several objects: the window ticks the other parts that belong with the one picked, and copies them as one.
	sb->addProp(scene_doc::PropKind::Tree);
	const int treeFirst = static_cast<int>(n0);
	const int treeParts = static_cast<int>(sb->document().objects.size() - n0);
	check(treeParts >= 2, "a tree prop is several objects (" + QString::number(treeParts) + ")");
	ArrayDialog treeDialog(treeFirst, sb->document().objects, sb);
	check(static_cast<int>(treeDialog.unit().size()) == treeParts, "the window ticks the tree's other parts to go with the one picked");
	treeDialog.setGridCounts(3, 1, 1);
	check(static_cast<int>(treeDialog.copies().size()) == 2 * treeParts, "a row of three trees adds two trees' worth of objects");
	treeDialog.tickPart(QString::fromStdString(sb->document().objects[n0 + 1].name), false);
	check(static_cast<int>(treeDialog.unit().size()) == treeParts - 1, "unticking a part leaves it out");
	sb->undo();
	check(sb->document().objects.size() == n0, "the tree is removed again by one undo");
	// A light is duplicated too (and numbered).
	const size_t lights0 = sb->document().lights.size();
	sb->selectLight(0);
	sb->duplicateSelected();
	check(sb->document().lights.size() == lights0 + 1, "a light can be duplicated");
	check(sb->undo() && sb->document().lights.size() == lights0, "and undone");
}

// Several items at once (scene_selection.h, scene_builder_multi.cpp): picking by list, click and box, dragging, copying, deleting and grouping them together, each
// one undo step. Leaves the document as it found it.
static void selfTestMultiSelect(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	using Sel = BuilderSelection;
	const size_t n0 = sb->document().objects.size();
	const auto objectsPicked = [sb]() { return sb->pickedItems().objects; };
	sb->pickObjects({1, 2});
	check(objectsPicked() == std::vector<int>({1, 2}) && sb->pickedRowsForTest() == 2, "two objects can be picked together (the list shows both)");
	const auto pos = [sb](int i) { return sb->document().objects[static_cast<size_t>(i)].position; };
	const scene_doc::Float3 glass0 = pos(1), gold0 = pos(2);
	check(sb->dragPickedForTest(1, QPointF(40, 0)), "dragging one of the two with the mouse moves it");
	const double dx = pos(1).x - glass0.x, dz = pos(1).z - glass0.z;
	check(dx != 0.0 && std::abs((pos(2).x - gold0.x) - dx) < 1e-9 && std::abs((pos(2).z - gold0.z) - dz) < 1e-9, "and the other moves by the same amount");
	check(sb->undo() && pos(1).x == glass0.x && pos(2).x == gold0.x, "the drag of both is one undo step");
	sb->moveSelectedBy({0.0, 1.0, 0.0});
	check(std::abs(pos(1).y - (glass0.y + 1.0)) < 1e-9 && std::abs(pos(2).y - (gold0.y + 1.0)) < 1e-9, "Move all by lifts both");
	check(sb->undo() && pos(2).y == gold0.y, "undone");

	sb->duplicateSelected();
	check(sb->document().objects.size() == n0 + 2 && objectsPicked() == std::vector<int>({static_cast<int>(n0), static_cast<int>(n0) + 1}), "Duplicate copies both and picks the copies");
	check(std::abs((pos(static_cast<int>(n0) + 1).x - pos(static_cast<int>(n0)).x) - (gold0.x - glass0.x)) < 1e-9, "the copies keep their places relative to each other");
	check(sb->undo() && sb->document().objects.size() == n0, "one undo removes both copies");

	// Click and Ctrl-click in the layout view.
	sb->selectObject(1);
	sb->clickItemForTest(Sel{Sel::Kind::Object, 2}, Qt::ControlModifier);
	check(objectsPicked() == std::vector<int>({1, 2}), "Ctrl-click adds an object to what is picked");
	sb->clickItemForTest(Sel{Sel::Kind::Object, 2}, Qt::ControlModifier);
	check(objectsPicked() == std::vector<int>({1}), "Ctrl-click on a picked object takes it out");
	sb->pickObjects({1, 2});
	sb->clickItemForTest(Sel{Sel::Kind::Object, 2});
	check(objectsPicked() == std::vector<int>({2}), "a plain click on one of several leaves just that one");

	// Turning and scaling several as one, about the middle of them.
	{
		sb->pickObjects({1, 2});
		const scene_doc::Document before = sb->document();
		const auto gap = [sb]() { const auto &a = sb->document().objects[1].position, &b = sb->document().objects[2].position; return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)); };
		const double gap0 = gap();
		const scene_doc::Float3 centre0 = scene_doc::centroidOf(sb->document(), sb->pickedItems());
		sb->turnSelectedBy(1, 90.0);
		const scene_doc::Float3 centre1 = scene_doc::centroidOf(sb->document(), sb->pickedItems());
		check(std::abs(gap() - gap0) < 1e-9 && std::abs(centre1.x - centre0.x) < 1e-9 && std::abs(centre1.z - centre0.z) < 1e-9 && std::abs(pos(1).x - before.objects[1].position.x) > 0.1 && std::abs(sb->document().objects[1].rotation.y - before.objects[1].rotation.y) > 1.0,
		      "Turn all by turns the set about its middle: the gap and the middle stay, the objects turn");
		check(sb->undo() && pos(1).x == before.objects[1].position.x && sb->document().objects[1].rotation.y == before.objects[1].rotation.y, "one undo puts them back");
		const double radius0 = sb->document().objects[2].radius;
		sb->scaleSelectedBy(2.0);
		check(std::abs(gap() - 2.0 * gap0) < 1e-9 && std::abs(sb->document().objects[2].radius - 2.0 * radius0) < 1e-9, "Scale all to 200 % doubles the gap and every size");
		check(sb->undo() && sb->document().objects[2].radius == radius0, "undone");
		// The 3D view's ring and handle do the same with the mouse, one undo step for the whole drag.
		sb->show3dView(true);
		check(sb->gizmoButtonEnabled(1) && sb->gizmoButtonEnabled(2), "Rotate and Scale are available while several are picked");
		check(sb->dragPickedRing3dForTest(1, 40.0) && std::abs(gap() - gap0) < 1e-6, "dragging the Rotate ring turns the set as one (the gap stays)");
		check(sb->undo() && pos(1).x == before.objects[1].position.x && pos(2).x == before.objects[2].position.x, "the whole drag is one undo step");
		check(sb->dragPickedScale3dForTest(0, 1.5) && gap() > gap0 * 1.2, "dragging a Scale handle stretches the set about its middle");
		check(sb->undo() && std::abs(gap() - gap0) < 1e-9, "and that is one undo step too");
		sb->show3dView(false);
	}

	// A box: whatever is inside it (computed from where the view draws things).
	const QPointF a = sb->itemScreenPosForTest(Sel{Sel::Kind::Object, 1}), b = sb->itemScreenPosForTest(Sel{Sel::Kind::Object, 2});
	const QPointF topLeft(std::min(a.x(), b.x()) - 12, std::min(a.y(), b.y()) - 12), bottomRight(std::max(a.x(), b.x()) + 12, std::max(a.y(), b.y()) + 12);
	std::vector<int> expected;
	for (int i = 0; i < static_cast<int>(n0); ++i) {
		const QPointF p = sb->itemScreenPosForTest(Sel{Sel::Kind::Object, i});
		if (p.x() >= topLeft.x() && p.x() <= bottomRight.x() && p.y() >= topLeft.y() && p.y() <= bottomRight.y()) expected.push_back(i);
	}
	sb->selectObject(3);
	sb->boxSelectForTest(topLeft, bottomRight);
	std::vector<int> want = expected;
	want.push_back(3);
	std::sort(want.begin(), want.end());
	want.erase(std::unique(want.begin(), want.end()), want.end());
	check(objectsPicked() == want && expected.size() >= 2, "a Ctrl-dragged box adds the objects inside it to what was picked (" + QString::number(expected.size()) + " inside)");

	// Groups.
	sb->pickObjects({1, 2});
	sb->groupSelected();
	const std::string group = sb->document().objects[1].group;
	check(!group.empty() && sb->document().objects[2].group == group && sb->document().objects[3].group.empty(), "Group makes the two a group");
	sb->selectObject(3);
	sb->clickItemForTest(Sel{Sel::Kind::Object, 2});
	check(objectsPicked() == std::vector<int>({1, 2}), "clicking one member picks the whole group");
	sb->duplicateSelected();
	const std::string copyGroup = sb->document().objects[n0].group;
	check(!copyGroup.empty() && copyGroup != group && sb->document().objects[n0 + 1].group == copyGroup, "a copy of a group is a group of its own");
	check(sb->undo() && sb->document().objects.size() == n0, "and one undo removes it");
	sb->pickObjects({1, 2});
	sb->deleteSelected();
	check(sb->document().objects.size() == n0 - 2 && sb->pickedItems().empty(), "Delete removes both");
	check(sb->undo() && sb->document().objects.size() == n0 && sb->document().objects[1].group == group, "one undo brings both back, still a group");
	sb->pickObjects({1, 2});
	sb->ungroupSelected();
	check(sb->document().objects[1].group.empty() && sb->document().objects[2].group.empty(), "Ungroup frees them");
	check(sb->undo() && sb->document().objects[1].group == group, "undone");
	check(sb->undo() && sb->document().objects[1].group.empty(), "and the grouping itself is one undo step");

	// The Array window takes the other picked objects along as the unit; the look of the main one goes to the rest.
	sb->pickObjects({2, 1});
	ArrayDialog dialog(2, sb->document().objects, sb, {1});
	check(dialog.unit().size() == 2 && dialog.unit()[1].name == sb->document().objects[1].name, "the Array window copies the picked objects together");
	sb->useLookOfMainObject();
	check(sb->document().objects[1].material.kind == sb->document().objects[2].material.kind && sb->document().objects[1].material.kind == scene_doc::MaterialKind::Conductor, "the main object's look is given to the other picked ones");
	check(sb->undo() && sb->document().objects[1].material.kind == scene_doc::MaterialKind::Dielectric, "undone");

	// A prop is a group: clicking its crown picks the trunk too.
	sb->addProp(scene_doc::PropKind::Tree);
	const int treeParts = static_cast<int>(sb->document().objects.size() - n0);
	check(static_cast<int>(sb->pickedItems().objects.size()) == treeParts && !sb->document().objects[n0].group.empty(), "a tree prop arrives as a group, all of it picked (" + QString::number(treeParts) + " parts)");
	sb->moveSelectedBy({-4.0, 0.0, 3.0});   // clear of the spotlight an earlier step left at the middle (a light is hit before an object)
	check(std::abs(sb->document().objects[n0].position.x - sb->document().objects[n0 + 1].position.x) < 1e-9, "the tree moves as one");
	sb->selectObject(0);
	sb->clickItemForTest(Sel{Sel::Kind::Object, static_cast<int>(n0)});
	{
		QStringList names;
		for (int i : sb->pickedItems().objects) names << QString::fromStdString(sb->document().objects[static_cast<size_t>(i)].name);
		check(static_cast<int>(sb->pickedItems().objects.size()) == treeParts, "clicking one part of the tree picks all of it (picked: " + names.join(", ") + ")");
	}
	check(sb->undo() && sb->undo() && sb->document().objects.size() == n0, "the tree and its move are two undo steps");

	sb->selectAll();
	size_t movable = 0;
	for (const scene_doc::Light &l : sb->document().lights) movable += l.kind != scene_doc::LightKind::Infinite ? 1 : 0;
	check(sb->pickedItems().objects.size() == n0 && sb->pickedItems().lights.size() == movable, "Select all picks every object and light but the sky");
	// RT_GUI_SELFTEST_MULTI_PNG=<file> saves a picture of the tab with three objects picked.
	const QString picture = qEnvironmentVariable("RT_GUI_SELFTEST_MULTI_PNG");
	if (!picture.isEmpty()) {
		sb->pickObjects({1, 2, 3});
		QApplication::processEvents();
		sb->grab().save(picture);
		sb->show3dView(true);
		sb->setGizmoToolForTest(1);   // Rotate: the rings are drawn at the middle of the three
		QApplication::processEvents();
		sb->grab().save(picture + ".3d_rotate.png");
		sb->setGizmoToolForTest(2);
		QApplication::processEvents();
		sb->grab().save(picture + ".3d_scale.png");
		sb->setGizmoToolForTest(0);
		sb->show3dView(false);
	}
	sb->selectObject(1);
}

// Saved camera views (scene_camera_views.h, scene_builder_camera_views.cpp): look through the camera, set it from the 3D view, save, use, update, rename and delete a
// view, each one undo step. Leaves the document as it found it.
static void selfTestCameraViews(SceneBuilderWidget *sb, const std::function<void(bool, const QString &)> &check) {
	const auto near3 = [](const scene_doc::Float3 &a, const scene_doc::Float3 &b) { return std::abs(a.x - b.x) < 1e-6 && std::abs(a.y - b.y) < 1e-6 && std::abs(a.z - b.z) < 1e-6; };
	const scene_doc::Camera camera0 = sb->document().camera;
	const size_t views0 = sb->document().cameraViews.size();
	sb->show3dView(true);
	sb->lookThroughCamera(true);
	check(sb->lookingThroughCameraForTest() && near3(sb->viewEyeForTest(), camera0.position), "Through camera puts the 3D view's eye at the scene's camera");
	sb->orbit3dForTest(120.0, 10.0);
	check(!sb->lookingThroughCameraForTest(), "orbiting the view leaves the camera's view");
	check(!near3(sb->viewEyeForTest(), camera0.position), "and the view is somewhere else now");

	const scene_doc::Float3 eye = sb->viewEyeForTest();
	sb->cameraFromView();
	check(near3(sb->document().camera.position, eye) && sb->document().camera.fov == camera0.fov, "Camera from view puts the camera where the view is, keeping its lens");
	sb->lookThroughCamera(true);
	check(near3(sb->viewEyeForTest(), eye), "looking through the camera now shows that place");
	check(sb->undo() && near3(sb->document().camera.position, camera0.position), "one undo puts the camera back");

	// Save, change the camera, use the view.
	sb->cameraFromView();   // (back to the orbit's place first: through the camera, this does nothing)
	sb->lookThroughCamera(false);
	sb->orbit3dForTest(40.0, 30.0);
	const scene_doc::Float3 farEye = sb->viewEyeForTest();
	const int saved = sb->saveCameraViewNamed("Bird's eye");
	check(saved == static_cast<int>(views0) && sb->document().cameraViews.size() == views0 + 1 && near3(sb->document().cameraViews.back().camera.position, farEye), "Save view remembers where the 3D view is");
	check(sb->document().cameraViews.back().name == "Bird's eye" && sb->viewComboCountForTest() == static_cast<int>(views0) + 2, "under its name, which is in the list of saved views");
	check(sb->saveCameraViewNamed("Bird's eye") >= 0 && sb->document().cameraViews.back().name == "Bird's eye 2", "a second view of the same name is numbered");
	sb->deleteCameraView(static_cast<int>(views0) + 1);
	check(sb->document().cameraViews.size() == views0 + 1, "a view can be deleted");
	check(sb->undo() && sb->document().cameraViews.size() == views0 + 2 && sb->undo() && sb->document().cameraViews.size() == views0 + 1, "and each is one undo step");

	sb->useCameraView(static_cast<int>(views0));
	check(near3(sb->document().camera.position, farEye) && sb->lookingThroughCameraForTest(), "Use puts the camera at the saved view and looks through it");
	sb->renameCameraView(static_cast<int>(views0), "Roof");
	check(sb->document().cameraViews[views0].name == "Roof", "a view can be renamed");
	sb->updateCameraViewFromNow(static_cast<int>(views0), true);
	check(near3(sb->document().cameraViews[views0].camera.position, farEye), "Update saves the camera over it");
	// RT_GUI_SELFTEST_CAMERA_PNG=<file> saves a picture of the tab looking through the camera, with the camera's properties (and its saved view) beside it.
	const QString picture = qEnvironmentVariable("RT_GUI_SELFTEST_CAMERA_PNG");
	if (!picture.isEmpty()) {
		sb->selectCameraForTest();
		sb->lookThroughCamera(true);
		QApplication::processEvents();
		sb->grab().save(picture);
	}
	// Back to how it was: three more undo steps (update changed nothing, so it took none of the log, but the edits above are still steps).
	while (sb->document().cameraViews.size() > views0 && sb->undo()) {}
	while (!near3(sb->document().camera.position, camera0.position) && sb->undo()) {}
	check(sb->document().cameraViews.size() == views0 && near3(sb->document().camera.position, camera0.position), "everything undoes back to the scene as it was");
	sb->lookThroughCamera(false);
	sb->show3dView(false);
}

void MainWindow::runBuilderSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot, const QString &outPrefix) {
	resize(qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_WIDTH") : 1500, qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") > 0 ? qEnvironmentVariableIntValue("RT_GUI_SELFTEST_HEIGHT") : 950);  // (..._WIDTH / ..._HEIGHT: a smaller window)
	if (m_sceneBuilder) m_tabWidget->setCurrentWidget(m_sceneBuilder);  // by widget, so it works in every language
	SceneBuilderWidget *sb = m_sceneBuilder;
	bool ok = sb != nullptr;
	auto check = [&ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		ok = ok && cond;
	};
	check(sb && sb->document().objects.size() == 5 && sb->document().lights.size() == 1, "starter scene has 5 objects and 1 light");
	const size_t objects0 = sb->document().objects.size();
	sb->addObject(scene_doc::ShapeKind::Cylinder);
	sb->addLight(scene_doc::LightKind::Spot);
	check(sb->document().objects.size() == objects0 + 1 && sb->document().lights.size() == 2, "added a cylinder and a spotlight");
	check(sb->isDirty(), "the scene is marked unsaved");
	check(sb->undo() && sb->document().lights.size() == 1, "undo removes the spotlight");
	check(sb->redo() && sb->document().lights.size() == 2, "redo brings it back");
	// A real mouse drag in the layout view moves the object (snapped to the 0.25 grid) as one undo step.
	const scene_doc::Float3 ball0 = sb->document().objects[1].position;
	check(sb->dragObjectForTest(1, QPointF(80, -40)), "dragging the glass ball in the layout view moves it");
	const scene_doc::Float3 ball1 = sb->document().objects[1].position;
	check(ball1.x > ball0.x + 0.3 && ball1.z < ball0.z - 0.1 && ball1.y == ball0.y, "it moved right and away from the camera, height unchanged");
	check(std::fabs(ball1.x * 4 - std::round(ball1.x * 4)) < 1e-9, "the new position is on the grid");
	check(sb->undo() && sb->document().objects[1].position.x == ball0.x && sb->document().objects[1].position.z == ball0.z, "one undo puts it back");
	check(sb->redo(), "redo moves it again");
	selfTestShapes(sb, check);
	selfTestArray(sb, check);
	selfTestMultiSelect(sb, check);
	selfTestCameraViews(sb, check);
	sb->selectObject(1);
	check(sb->problemsText().isEmpty(), "the scene has no problems or notes");
	const QString pbrt = outPrefix + "_builder.pbrt";
	check(sb->saveFile(pbrt) && !sb->isDirty(), "saved " + pbrt);
	const std::string before = scene_doc::toJson(sb->document());
	QString err;
	sb->newScene();
	check(sb->openFile(pbrt, &err) && scene_doc::toJson(sb->document()) == before, "re-opened the saved file unchanged " + err);
	selfTestSceneList(sb, check);
	selfTestDeleteScenes(sb, check);
	selfTestLog(sb, check);
	selfTestWheelGuard(check);
	selfTestWindowGeometry(check);
	selfTestCrashRecovery(check);
	selfTestRenderQueue(check);
	// The screenshots show the starter scene (the edits above are done), with the gold ball picked.
	sb->newScene();
	sb->selectObject(2);
	QTimer::singleShot(600, this, [this, shot, log, sb, ok]() mutable {
		shot("builder_edit");
		// One preview on the CPU, then (RT_GUI_SELFTEST_GPU=1, set by scripts/gui_selftest.py --live-preview on a machine with a GPU)
		// one through "Use the GPU" - Metal on a Mac - which must also give a lit picture of about the same brightness.
		const bool alsoGpu = qEnvironmentVariableIsSet("RT_GUI_SELFTEST_GPU");
		auto meanGrey = [](const QString &path) {
			const QImage img(path);
			double sum = 0;
			for (int y = 0; y < img.height(); ++y)
				for (int x = 0; x < img.width(); ++x) sum += qGray(img.pixel(x, y));
			return img.isNull() ? 0.0 : sum / (double(img.width()) * img.height());
		};
		log("starting a preview render");
		sb->startPreview([this, shot, log, sb, ok, alsoGpu, meanGrey](bool done, const QString &message) mutable {
			log(QString("preview: %1 - %2").arg(done ? "ok" : "FAIL", message));
			bool good = ok && done;
			double cpuMean = 0.0;
			if (done) {
				cpuMean = meanGrey(sb->previewImagePath());
				log(QString("preview picture, mean grey %1").arg(cpuMean));
				good = good && cpuMean > 10.0 && cpuMean < 245.0;
			}
			const auto finish = [this, shot, log](bool result) {
				QTimer::singleShot(300, this, [shot, log, result]() {
					shot("builder_preview");
					log(result ? "RESULT: OK" : "RESULT: FAIL");
					QApplication::exit(result ? 0 : 1);
				});
			};
			if (!alsoGpu || !good) { finish(good); return; }
			sb->setUseGpu(true);
			log("starting a preview render on the GPU");
			sb->startPreview([log, sb, cpuMean, meanGrey, finish](bool gpuDone, const QString &gpuMessage) {
				log(QString("GPU preview: %1 - %2").arg(gpuDone ? "ok" : "FAIL", gpuMessage));
				bool gpuGood = gpuDone;
				if (gpuDone) {
					const double gpuMean = meanGrey(sb->previewImagePath());
					log(QString("GPU preview picture, mean grey %1 (CPU %2)").arg(gpuMean).arg(cpuMean));
					// Different renderers and sample noise, but the same scene: the same order of brightness.
					gpuGood = gpuMean > 10.0 && gpuMean < 245.0 && std::abs(gpuMean - cpuMean) < 40.0;
				}
				sb->setUseGpu(false);
				finish(gpuGood);
			});
		});
	});
	return;
}

	// RT_GUI_SELFTEST=diagnostics: runs the Diagnostics tab's action (the CLI report plus the GUI's Network section) and logs the
	// finished report. RT_ASSET_BASE_URL can point the network check at a local or dead address.
void MainWindow::runDiagnosticsSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	for (int i = 0; i < m_tabWidget->count(); ++i)
		if (m_tabWidget->tabText(i).contains("Diagnostics")) m_tabWidget->setCurrentIndex(i);
	onRunDiagnosticsClicked();
	auto *poll = new QTimer(this);
	auto *waited = new int(0);
	connect(poll, &QTimer::timeout, this, [this, poll, waited, log]() {
		const bool done = !diagnosticsBusy() && m_lastDiagReport.contains("=== Photo helper");
		if (!done && ++*waited < 240) return;
		poll->stop();
		log(m_lastDiagReport);
		const bool sections = m_lastDiagReport.contains("=== Network ===") && m_lastDiagReport.contains("=== Photo helper");
		log(sections ? "RESULT: OK" : "RESULT: FAIL (no Network or Photo helper section)");
		QApplication::exit(sections ? 0 : 1);
	});
	poll->start(500);
	return;
}

	// RT_GUI_SELFTEST=scenekeys: the scene keys the GUI saves and looks up by (slugs, with ids still accepted) round-trip through the metadata library, the
	// technique notes are found by either, and the thumbnail cache is keyed by slug. Exit 0 if every check held.
void MainWindow::runSceneKeysSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	bool ok = true;
	auto check = [&ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		ok = ok && cond;
	};
	const QString slug = SceneMetadataClient::sceneSlug("A1");
	check(slug == "cornell-box", "A1's slug is cornell-box (got \"" + slug + "\")");
	check(SceneMetadataClient::sceneIdForKey("cornell-box") == "A1", "the slug resolves to A1");
	check(SceneMetadataClient::sceneIdForKey("A1") == "A1", "an id resolves to itself");
	check(SceneMetadataClient::sceneIdForKey("no-such-scene").isEmpty(), "an unknown key resolves to nothing");
	check(scene_technique_notes::hasNote("A1") && scene_technique_notes::hasNote("cornell-box"), "the Cornell Box note is found by id and by slug");
	check(scene_technique_notes::forScene("A1") == scene_technique_notes::forScene("cornell-box"), "both give the same note");
	check(thumbnailCachePath("A1").endsWith("/thumbnails/cornell-box.png"), "the thumbnail cache file is named by slug: " + thumbnailCachePath("A1"));
	// every compiled-in scene's slug is unique and resolves back to its own id
	QSet<QString> slugs;
	int count = SceneMetadataClient::sceneCount(), bad = 0;
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		const QString s = SceneMetadataClient::sceneSlug(id);
		if (s == id || slugs.contains(s) || SceneMetadataClient::sceneIdForKey(s) != id) ++bad;
		slugs.insert(s);
	}
	check(bad == 0, QString("%1 scenes: every slug is unique and resolves back to its id (%2 bad)").arg(count).arg(bad));
	// The technique notes (scene_technique_notes.h, translated GUI text, so kept out of the Qt-free registry) and the registry must describe the
	// same scenes: every self-contained built-in scene has a note, and every note belongs to one.
	QSet<QString> expected, missingNote, staleNote;
	for (int i = 0; i < count; ++i) {
		const QString id = SceneMetadataClient::sceneIdAtIndex(i);
		const QString category = SceneMetadataClient::sceneCategory(id);
		if (category == "Custom Scenes" || category == "Test Scenes" || category == "My Scenes" || SceneMetadataClient::sceneRequiresFiles(id)) continue;
		expected.insert(SceneMetadataClient::sceneSlug(id));
	}
	for (const QString &slug : expected)
		if (!scene_technique_notes::notes().contains(slug)) missingNote.insert(slug);
	for (auto it = scene_technique_notes::notes().constBegin(); it != scene_technique_notes::notes().constEnd(); ++it)
		if (!expected.contains(it.key())) staleNote.insert(it.key());
	check(missingNote.isEmpty(), QString("%1 self-contained scenes, %2 notes: missing a note: %3").arg(expected.size()).arg(scene_technique_notes::notes().size()).arg(QStringList(missingNote.begin(), missingNote.end()).join(", ")));
	check(staleNote.isEmpty(), "notes for no self-contained scene: " + QStringList(staleNote.begin(), staleNote.end()).join(", "));
	log(ok ? "RESULT: OK" : "RESULT: FAIL");
	QApplication::exit(ok ? 0 : 1);
	return;
}

	// RT_GUI_SELFTEST=download RT_GUI_SELFTEST_SCENE=<id>: selects the scene, runs the "Download missing files" action
	// without dialogs (point RT_ASSET_BASE_URL at a local server to avoid the network), and checks the files arrived and
	// the scene no longer reports them missing. Exit 0 on success.
void MainWindow::runDownloadSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	selectSceneById(qEnvironmentVariable("RT_GUI_SELFTEST_SCENE"));
	const QList<asset_downloader::Job> jobs = m_downloadableAssetJobs;
	const scene_packs::Pack *pack = m_downloadablePack;
	if (jobs.isEmpty() && pack) {
		log(QString("scene=%1, pack \"%2\": %3 file(s), %4 MB to download").arg(m_sceneCombo->currentData().toString(), pack->title).arg(pack->fileCount()).arg(pack->downloadBytes() / 1.0e6, 0, 'f', 1));
		QTimer::singleShot(1800000, this, [log]() { log("RESULT: FAIL (timed out)"); QApplication::exit(1); });
		startPackDownload(*pack, false, [this, log](bool ok, const QString &error) {
			const bool stillMissing = SceneMetadataClient::missingAssets(m_sceneCombo->currentData().toString()).any;
			log(QString("download ok=%1 error=\"%2\" scene-still-reports-missing=%3").arg(ok).arg(error).arg(stillMissing));
			log(ok && !stillMissing ? "RESULT: OK" : "RESULT: FAIL");
			QApplication::exit(ok && !stillMissing ? 0 : 1);
		});
		return;
	}
	log(QString("scene=%1, %2 downloadable missing file(s), button visible: %3")
		.arg(m_sceneCombo->currentData().toString()).arg(jobs.size()).arg(m_downloadAssetsButton && !m_downloadAssetsButton->isHidden()));
	if (jobs.isEmpty()) { log("RESULT: FAIL (nothing to download for this scene)"); QApplication::exit(1); return; }
	QTimer::singleShot(120000, this, [log]() { log("RESULT: FAIL (timed out)"); QApplication::exit(1); });
	startAssetDownload(jobs, false, [this, jobs, log](bool ok, const QString &error) {
		bool allThere = ok;
		for (const auto &j : jobs) allThere = allThere && QFileInfo(j.destination).size() == j.entry.size;
		const bool stillMissing = SceneMetadataClient::missingAssets(m_sceneCombo->currentData().toString()).any;
		log(QString("download ok=%1 error=\"%2\" files-present=%3 scene-still-reports-missing=%4").arg(ok).arg(error).arg(allThere).arg(stillMissing));
		log(allThere && !stillMissing ? "RESULT: OK" : "RESULT: FAIL");
		QApplication::exit(allThere && !stillMissing ? 0 : 1);
	});
	return;
}

void MainWindow::runUiSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	// RT_GUI_SELFTEST_SCENE=<id> selects that scene first, so the scene info (including the
	// missing-files warning) can be checked; its text is logged as well as pictured.
	const QString sceneOverride = qEnvironmentVariable("RT_GUI_SELFTEST_SCENE");
	if (!sceneOverride.isEmpty()) {
		selectSceneById(sceneOverride);
		resize(1100, 900);   // large enough to see the scene group, including the download button
	}
	log(QString("scene info: %1").arg(m_sceneInfoLabel ? m_sceneInfoLabel->text() : QString()));
	// The renderer libraries print their errors to std::cerr. This program's stderr used to be left CLOSED on Windows (the redirect into the log failed after closing
	// it), so the first such line aborted the whole program (0xC0000409). Writing one here makes that regression a crash of the self-test; when the session log says
	// stderr goes into it, the line must have arrived there.
	std::cerr << "[selftest] a line written to stderr" << std::endl;
	const QStringList logTail = AppLog::tail(400);
	const bool redirected = logTail.join('\n').contains(QStringLiteral("stderr of this process"));
	const bool arrived = logTail.join('\n').contains(QStringLiteral("[selftest] a line written to stderr"));
	log(QString("stderr: %1").arg(redirected ? (arrived ? "written to the log" : "FAIL (redirected, but the line is not in the log)") : "left where it was (a terminal)"));
	if (redirected && !arrived) { QApplication::exit(1); return; }
	QTimer::singleShot(600, this, [shot]() { shot("ui"); QApplication::exit(0); });
	return;
}
