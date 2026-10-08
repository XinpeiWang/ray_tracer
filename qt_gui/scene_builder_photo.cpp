// The Scene Builder's "Object from a photo" (Add menu): one photo -> a textured mesh through the optional photo helper (photo_import.h).
#include "scene_builder_widget.h"

#include "photo_import.h"
#include "scene_builder_common.h"

#include <QCheckBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QMessageBox>
#include <QProgressDialog>
#include <QSettings>

using scene_doc::Object;
using scene_doc::ShapeKind;
using namespace scene_builder_ui;

static const char *kPhotoNoticeKey = "sceneBuilder/photoNoticeSeen";

// A folder name made of the photo's own name: letters, digits, dash and underscore only.
static QString safeFolderName(const QString &base) {
	QString s;
	for (const QChar &c : base) s += (c.isLetterOrNumber() && c.unicode() < 128) || c == '-' || c == '_' ? c : QChar('_');
	return s.isEmpty() ? QStringLiteral("photo") : s.left(40);
}

void SceneBuilderWidget::addObjectFromPhoto() {
	const photo_import::Setup setup = photo_import::locate();
	if (!setup.ready) {
		QMessageBox box(QMessageBox::Information, tr("Photo helper not installed"),
		                tr("Turning a photo into a 3D object needs an optional helper: an AI model that runs on your own computer (about 5 GB to install, and an "
		                   "graphics card - NVIDIA on Windows, Apple silicon on a Mac - is strongly recommended).\n\n%1\n\n"
		                   "To set it up, use the Diagnostics tab: Run Diagnostics, then Install Photo Helper. Or run this once yourself:\n\n%2\n\n"
		                   "Then choose this again. The guide (docs/PHOTO_TO_SCENE.md) explains what it does and where its limits are.")
		                    .arg(setup.why,
#ifdef Q_OS_WIN
		                         QStringLiteral("powershell -ExecutionPolicy Bypass -File scripts\\setup_photo_to_mesh.ps1")),
#else
		                         QStringLiteral("bash scripts/setup_photo_to_mesh.sh")),
#endif
		                QMessageBox::Ok, this);
		box.exec();
		return;
	}

	QSettings settings;
	if (!settings.value(kPhotoNoticeKey, false).toBool()) {
		QMessageBox box(QMessageBox::Information, tr("Object from a photo"),
		                tr("One photo is turned into a 3D object by an AI model running on this computer; the photo is not uploaded anywhere.\n\n"
		                   "The shape is a guess: the back is invented and fine detail is soft. It works best on one object against a plain background. "
		                   "The first run downloads the model (about 1.7 GB) and can take several minutes."),
		                QMessageBox::Ok | QMessageBox::Cancel, this);
		auto *again = new QCheckBox(tr("Do not show this again"), &box);
		box.setCheckBox(again);
		if (box.exec() != QMessageBox::Ok) return;
		if (again->isChecked()) settings.setValue(kPhotoNoticeKey, true);
	}

	const QString photo = QFileDialog::getOpenFileName(this, tr("Choose a photo"), QString(), tr("Photos (*.png *.jpg *.jpeg *.bmp *.webp)"));
	if (photo.isEmpty()) return;
	QString error;
	if (!importPhoto(photo, &error) && !error.isEmpty()) QMessageBox::warning(this, tr("Could not make the object"), error);
}

// Runs the helper on `photo` behind a progress dialog and adds the result as a mesh object. Returns false with `error` set (empty when the
// user cancelled) if there is no object to add.
bool SceneBuilderWidget::importPhoto(const QString &photo, QString *error) {
	const photo_import::Setup setup = photo_import::locate();
	if (!setup.ready) {
		if (error) *error = setup.why;
		return false;
	}
	const QString baseName = QFileInfo(photo).completeBaseName();
	const QString outFolder =
	    photo_import::resultsFolder() + "/" + safeFolderName(baseName) + "-" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss");
	if (!QDir().mkpath(outFolder)) {
		if (error) *error = tr("Could not create the folder %1.").arg(outFolder);
		return false;
	}

	QProgressDialog progress(tr("Starting..."), tr("Cancel"), 0, 100, this);
	progress.setWindowTitle(tr("Making a 3D object from the photo"));
	progress.setWindowModality(Qt::WindowModal);
	progress.setMinimumDuration(0);
	progress.setMinimumWidth(420);
	progress.setValue(0);

	PhotoToMeshJob job(setup, photo, outFolder, true);
	bool ok = false, cancelled = false;
	QString message;
	QEventLoop loop;
	connect(&job, &PhotoToMeshJob::progress, &progress, [&progress](int percent, const QString &text) {
		progress.setValue(percent);
		progress.setLabelText(text);
	});
	connect(&job, &PhotoToMeshJob::finished, &loop, [&](bool good, const QString &text) {
		ok = good;
		message = text;
		loop.quit();
	});
	const QMetaObject::Connection cancelConnection = connect(&progress, &QProgressDialog::canceled, &job, [&]() {
		cancelled = true;
		job.cancel();
	});
	job.start();
	loop.exec();
	// QProgressDialog::closeEvent emits canceled(): once the job is over, closing the window is not the user cancelling (a real failure
	// would be reported as a cancel and its message dropped).
	disconnect(cancelConnection);
	progress.close();

	if (!ok) {
		QDir(outFolder).removeRecursively();
		if (error) *error = cancelled ? QString() : message;
		return false;
	}

	edit(QString(), [&]() {
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		Object o = scene_doc::makeObject(ShapeKind::Mesh, uniqueName(baseName.isEmpty() ? tr("Photo object") : baseName, names).toStdString());
		o.meshFile = QDir::toNativeSeparators(outFolder + "/mesh.obj").toStdString();
		o.material.imageFile = QDir::toNativeSeparators(outFolder + "/texture.png").toStdString();
		const scene_doc::Float3 c = dropPoint();
		o.position = {c.x, 0.0, c.z};  // the mesh stands on y = 0, about 2 units tall
		m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
	emit statusMessage(tr("Added %1 from the photo. The shape is a guess; check it from every side.").arg(baseName));
	return true;
}
