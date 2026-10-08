// RT_GUI_SELFTEST=photo RT_GUI_SELFTEST_PHOTO=<image> RT_GUI_SELFTEST_OUT=<prefix>: the Scene Builder's "Object from a photo" without its dialogs.
// Runs the photo helper on the image (it must be set up; RAY_TRACER_PHOTO3D_PYTHON points at another install), checks the mesh object it adds, and
// renders a preview of it. Screenshots <prefix>_photo_edit.png and <prefix>_photo_preview.png; exit 0 if every step held.
#include "mainwindow.h"

#include "scene_builder_widget.h"

#include <QApplication>
#include <QFile>
#include <QTimer>

void MainWindow::runPhotoSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	if (m_sceneBuilder) m_tabWidget->setCurrentWidget(m_sceneBuilder);
	resize(1500, 950);
	SceneBuilderWidget *sb = m_sceneBuilder;
	bool ok = sb != nullptr;
	auto check = [&ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		ok = ok && cond;
	};
	QString err;
	const QString photo = qEnvironmentVariable("RT_GUI_SELFTEST_PHOTO");
	check(sb && sb->importPhoto(photo, &err), "made an object from " + photo + " " + err);
	if (sb && ok) {
		const scene_doc::Document &doc = sb->document();
		const scene_doc::Object &o = doc.objects.back();
		check(o.shape == scene_doc::ShapeKind::Mesh && QFile::exists(QString::fromStdString(o.meshFile)) && QFile::exists(QString::fromStdString(o.material.imageFile)),
		      "the new object is a mesh with its texture");
		check(!scene_doc::hasErrors(scene_doc::validate(doc)), "the scene has no errors");
	}
	QTimer::singleShot(600, this, [this, shot, log, sb, ok]() {
		shot("photo_edit");
		if (!ok) {
			log("RESULT: FAIL");
			QApplication::exit(1);
			return;
		}
		log("starting a preview render");
		sb->startPreview([this, shot, log](bool done, const QString &message) {
			log(QString("preview: %1 - %2").arg(done ? "ok" : "FAIL", message));
			QTimer::singleShot(300, this, [shot, log, done]() {
				shot("photo_preview");
				log(done ? "RESULT: OK" : "RESULT: FAIL");
				QApplication::exit(done ? 0 : 1);
			});
		});
	});
}
