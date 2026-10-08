// RT_GUI_SELFTEST=photo RT_GUI_SELFTEST_PHOTO=<image> RT_GUI_SELFTEST_OUT=<prefix>: the Scene Builder's "Object from a photo" without its dialogs.
// Runs the photo helper on the image (it must be set up; RAY_TRACER_PHOTO3D_PYTHON points at another install), checks the mesh object it adds, and
// renders a preview of it. Screenshots <prefix>_photo_edit.png and <prefix>_photo_preview.png; exit 0 if every step held.
#include "mainwindow.h"

#include "photo_import.h"
#include "scene_builder_widget.h"

#include <QApplication>
#include <QFile>
#include <QTimer>

#include <memory>

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
	// A photo that does not exist must fail with the helper's own words, not silently (a closing progress window once looked like a cancel).
	QString badError;
	const bool badOk = sb && sb->importPhoto("C:/no/such/photo_for_the_selftest.png", &badError);
	check(sb && !badOk && !badError.isEmpty(), "a missing photo fails with a message: " + badError.left(100));
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

// RT_GUI_SELFTEST=installphoto: the Diagnostics tab's "Install Photo Helper" flow with stand-in installer scripts instead of the real multi-GB one.
// RAY_TRACER_PHOTO3D_PYTHON must name a python that does not exist (so the report lists the helper as missing) and RAY_TRACER_PHOTO3D_SETUP a
// script that prints some lines (including a \r progress update) and exits 0; RT_GUI_SELFTEST_SETUP_FAIL names one that exits 1. Checks the
// button's states, the output reaching the caller, a clean success, and a failure's message; exit 0 if every step held.
void MainWindow::runInstallPhotoSelfTest(const std::function<void(const QString &)> &log, const std::function<void(const QString &)> &shot) {
	for (int i = 0; i < m_tabWidget->count(); ++i)
		if (m_tabWidget->tabText(i).contains("Diagnostics")) m_tabWidget->setCurrentIndex(i);
	resize(1100, 800);
	auto ok = std::make_shared<bool>(true);
	auto check = [ok, log](bool cond, const QString &what) {
		log(QString("%1: %2").arg(cond ? "ok" : "FAIL", what));
		*ok = *ok && cond;
	};
	check(m_installPhotoHelperButton && !m_installPhotoHelperButton->isEnabled(), "before any diagnostics the install button is disabled");
	onRunDiagnosticsClicked();
	auto *poll = new QTimer(this);
	auto stage = std::make_shared<int>(0);
	auto waited = std::make_shared<int>(0);
	const QString failScript = qEnvironmentVariable("RT_GUI_SELFTEST_SETUP_FAIL");
	connect(poll, &QTimer::timeout, this, [=]() {
		if (++*waited > 400) { log("FAIL: timed out"); QApplication::exit(1); return; }
		if (*stage == 0 && !diagnosticsBusy() && m_lastDiagReport.contains("=== Photo helper")) {
			*stage = 1;
			const QStringList missing = photo_import::missingFacts(m_lastDiagReport);
			check(!missing.isEmpty(), "the report lists what the helper is missing (" + missing.join(" | ") + ")");
#ifdef Q_OS_WIN
			check(m_installPhotoHelperButton->isEnabled(), "so the install button is enabled");
			check(m_installPhotoHelperButton->toolTip().contains("Python Environment"), "and its tooltip says what is missing");
#endif
			shot("installphoto_report");
			startPhotoHelperInstall(false, [=](bool done, const QString &message) {
				check(done, "the stand-in installer succeeded " + message);
				*stage = 2;
			});
		} else if (*stage == 2 && !diagnosticsBusy()) {
			// After a successful install the diagnostics run again by themselves.
			*stage = 3;
			check(m_lastDiagReport.contains("=== Photo helper"), "the diagnostics ran again after the install");
			if (failScript.isEmpty()) { poll->stop(); log(*ok ? "RESULT: OK" : "RESULT: FAIL"); QApplication::exit(*ok ? 0 : 1); return; }
			qputenv("RAY_TRACER_PHOTO3D_SETUP", failScript.toLocal8Bit());
			startPhotoHelperInstall(false, [=](bool done, const QString &message) {
				check(!done && message.contains("ERROR: no network (CRLF)"), "a failing installer reports failure with the last lines of its output, CRLF or not: " + message.left(120));
				*stage = 4;
			});
		} else if (*stage == 4) {
			poll->stop();
			QTimer::singleShot(300, this, [=]() {
				check(!m_photoInstaller, "nothing is left running");
				log(*ok ? "RESULT: OK" : "RESULT: FAIL");
				QApplication::exit(*ok ? 0 : 1);
			});
			*stage = 5;
		}
	});
	poll->start(300);
}
