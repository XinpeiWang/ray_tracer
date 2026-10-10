// The Scene Builder's videos: a turntable (a camera that goes once round the point it looks at, at the camera's own height and distance, so the whole thing can be seen from
// every side) and a flythrough (a camera that goes through the saved camera views in order). The renderer already makes such videos (its --video mode, with the orbit camera
// path, or with --camera-keyframes for the views); this file runs it on the scene file and puts the finished MP4 where the user asked (see scene_builder_widget.h).
// Assembling the MP4 needs ffmpeg on Windows and Linux; a Mac has its own encoder.
#include "scene_builder_widget.h"

#include "app_log.h"
#include "scene_builder_common.h"
#include "../src/shared/camera_keyframes.h"

#include <QCheckBox>
#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QLabel>
#include <QMessageBox>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QSpinBox>
#include <QComboBox>
#include <QStandardPaths>
#include <QTextStream>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

using namespace scene_builder_ui;

namespace {

// Size and samples of the turntable's pictures: the same ladder as the preview's quality list, and "the scene's own" (the size and samples set under Camera and image).
struct TurntableQuality {
	int width;     // 0: the scene's own
	int samples;   // 0: the scene's own
};
const TurntableQuality kQualities[4] = {{480, 16}, {720, 64}, {960, 256}, {0, 0}};

}  // namespace

// The Turntable... and Flythrough... windows: how long, how fast, how good (and for a flythrough, how it moves), then where to save it.
void SceneBuilderWidget::showTurntableDialog() { showVideoDialog(false); }
void SceneBuilderWidget::showFlythroughDialog() { showVideoDialog(true); }

void SceneBuilderWidget::showVideoDialog(bool flythrough) {
	if (flythrough && m_doc.cameraViews.size() < 2) {
		emit statusMessage(tr("A flythrough needs two or more saved camera views."));
		return;
	}
	QDialog dialog(this);
	dialog.setWindowTitle(flythrough ? tr("Flythrough video") : tr("Turntable video"));
	auto *root = new QVBoxLayout(&dialog);
	QString introText = tr("Makes a video of the scene from a camera that goes once round the point the camera looks at, at the camera's own height and distance, starting where it is now. "
	                       "It renders every frame, so a long or large video takes a while; start with Draft.");
	if (flythrough) {
		QStringList names;
		for (const scene_doc::CameraView &v : m_doc.cameraViews) names << QString::fromStdString(v.name);
		introText = tr("Makes a video of the scene from a camera that flies through your saved camera views in the order they were saved: %1. The first picture is the first view and the last picture the last. "
		               "The lens (field of view, depth of field) is the first view's. It renders every frame, so a long or large video takes a while; start with Draft.").arg(names.join(QStringLiteral(" > ")));
	}
	auto *intro = new QLabel(introText, &dialog);
	intro->setWordWrap(true);
	root->addWidget(intro);
	auto *form = new QFormLayout();
	root->addLayout(form);
	auto *seconds = new QSpinBox(&dialog);
	seconds->setRange(1, 120);
	seconds->setValue(4);
	seconds->setSuffix(tr(" s"));
	auto *fps = new QComboBox(&dialog);
	for (int f : {24, 30, 60}) fps->addItem(tr("%1 frames per second").arg(f), f);
	fps->setCurrentIndex(1);
	auto *quality = new QComboBox(&dialog);
	quality->addItem(tr("Draft: 480 pixels wide, 16 samples"));
	quality->addItem(tr("Good: 720 pixels wide, 64 samples"));
	quality->addItem(tr("Best: 960 pixels wide, 256 samples"));
	quality->addItem(tr("The scene's own size and samples (%1 x %2, %3)").arg(m_doc.render.width).arg(m_doc.render.height).arg(m_doc.render.samples));
	auto *summary = new QLabel(&dialog);
	QCheckBox *ease = nullptr, *loop = nullptr;
	if (flythrough) {
		ease = new QCheckBox(tr("Start and end gently"), &dialog);
		ease->setChecked(true);
		loop = new QCheckBox(tr("Come back to the first view at the end"), &dialog);
	}
	form->addRow(tr("Length:"), seconds);
	form->addRow(tr("Speed:"), fps);
	form->addRow(tr("Quality:"), quality);
	if (flythrough) {
		form->addRow(ease);
		form->addRow(loop);
	}
	root->addWidget(summary);
	const auto refresh = [=]() { summary->setText(tr("%1 frames.").arg(seconds->value() * fps->currentData().toInt())); };
	connect(seconds, QOverload<int>::of(&QSpinBox::valueChanged), &dialog, refresh);
	connect(fps, QOverload<int>::of(&QComboBox::currentIndexChanged), &dialog, refresh);
	refresh();
	auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
	buttons->button(QDialogButtonBox::Ok)->setText(tr("Choose where to save..."));
	connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
	root->addWidget(buttons);
	if (dialog.exec() != QDialog::Accepted) return;

	const QString suffix = flythrough ? QStringLiteral("flythrough") : QStringLiteral("turntable");
	const QString start = (m_path.isEmpty() ? QDir::homePath() + "/" + suffix : QFileInfo(m_path).absolutePath() + "/" + QFileInfo(m_path).completeBaseName() + "-" + suffix) + ".mp4";
	QString mp4 = QFileDialog::getSaveFileName(this, flythrough ? tr("Save the flythrough video") : tr("Save the turntable video"), start, tr("MP4 videos (*.mp4)"));
	if (mp4.isEmpty()) return;
	if (!mp4.endsWith(".mp4", Qt::CaseInsensitive)) mp4 += ".mp4";
	const TurntableQuality q = kQualities[std::clamp(quality->currentIndex(), 0, 3)];
	const int frameCount = seconds->value() * fps->currentData().toInt();
	const int width = q.width > 0 ? q.width : m_doc.render.width, samples = q.samples > 0 ? q.samples : m_doc.render.samples;
	const auto finished = [this, flythrough](bool ok, const QString &message) {
		if (!ok) QMessageBox::warning(this, flythrough ? tr("The flythrough failed") : tr("The turntable failed"), message);
	};
	if (flythrough) startFlythrough(frameCount, fps->currentData().toInt(), width, samples, mp4, ease->isChecked(), loop->isChecked(), finished);
	else startTurntable(frameCount, fps->currentData().toInt(), width, samples, mp4, finished);
}

void SceneBuilderWidget::startTurntable(int frames, int fps, int width, int samples, const QString &outMp4, const std::function<void(bool, const QString &)> &done) {
	startCameraVideo(frames, fps, width, samples, outMp4, QString(), nullptr, done);
}

// A flythrough: the saved views in the order they were saved (and the first again at the end, if asked), as the renderer's camera keyframes.
void SceneBuilderWidget::startFlythrough(int frames, int fps, int width, int samples, const QString &outMp4, bool ease, bool returnToStart, const std::function<void(bool, const QString &)> &done) {
	if (m_doc.cameraViews.size() < 2) {
		const QString msg = tr("A flythrough needs two or more saved camera views.");
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
		return;
	}
	camera_keyframes::Path path;
	path.ease = ease;
	for (const scene_doc::CameraView &v : m_doc.cameraViews) {
		camera_keyframes::Key k;
		k.pos[0] = v.camera.position.x; k.pos[1] = v.camera.position.y; k.pos[2] = v.camera.position.z;
		k.target[0] = v.camera.target.x; k.target[1] = v.camera.target.y; k.target[2] = v.camera.target.z;
		path.keys.push_back(k);
	}
	if (returnToStart) path.keys.push_back(path.keys.front());
	// The lens is the first view's: the scene is written with that camera.
	scene_doc::Document lens = m_doc;
	lens.camera = m_doc.cameraViews.front().camera;
	startCameraVideo(frames, fps, width, samples, outMp4, QString::fromStdString(camera_keyframes::toText(path)), &lens, done);
}

// Both videos: the scene is written (with `sceneDoc`'s camera if given), the renderer is run in its video mode, and the MP4 is copied to `outMp4` when it is done.
// `keyframes`, if not empty, is the text of a camera keyframes file (otherwise the camera orbits).
void SceneBuilderWidget::startCameraVideo(int frames, int fps, int width, int samples, const QString &outMp4, const QString &keyframes, const scene_doc::Document *sceneDoc,
                                          const std::function<void(bool, const QString &)> &done) {
	flushEditLog();
	const auto fail = [&](const QString &msg) {
		AppLog::warn(QStringLiteral("builder-render"), QStringLiteral("video not started: %1").arg(msg));
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
	};
	if (m_process) return fail(tr("A render is already running."));
	if (frames < 1 || fps < 1 || width < 1) return fail(tr("A video needs at least one frame."));
	const auto problems = scene_doc::validate(m_doc);
	if (scene_doc::hasErrors(problems)) return fail(tr("Fix the problems listed under the properties first."));
	if (!QFileInfo::exists(launcherPath())) return fail(tr("The renderer (%1) was not found next to the program.").arg(launcherPath()));

	// Its own folder: the renderer clears a "frames" folder beside its output before it starts.
	const QString dir = workFolder() + "/turntable";
	QDir(dir).removeRecursively();
	QDir().mkpath(dir);
	const QString scene = dir + "/scene.pbrt";
	QString error;
	if (!writeSceneFile(scene, &error, sceneDoc)) return fail(error);
	const QString keyframeFile = dir + "/path.txt";
	if (!keyframes.isEmpty()) {
		QFile f(keyframeFile);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return fail(tr("Could not write %1.").arg(keyframeFile));
		f.write(keyframes.toUtf8());
	}

	const double aspect = m_doc.render.width > 0 ? double(m_doc.render.height) / m_doc.render.width : 0.75;
	const int height = std::max(1, static_cast<int>(std::lround(width * aspect)));
	QStringList args;
	args << (m_gpuCheck->isChecked() ? "--gpu" : "--cpu") << "--output" << dir + "/turntable.ppm" << "--video";
	if (keyframes.isEmpty()) args << "--camera-path" << "orbit";
	else args << "--camera-keyframes" << keyframeFile;
	args << "--frames" << QString::number(frames) << "--fps" << QString::number(fps) << "--height" << QString::number(height) << QString::number(width) << QString::number(samples) << QString::number(m_doc.render.maxDepth) << scene;

	m_turntableOut = outMp4;
	m_turntableFrames = frames;
	m_pendingDone = done;
	m_renderLog.clear();
	m_cancelRequested = false;
	m_renderClock.start();
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setWorkingDirectory(QCoreApplication::applicationDirPath());
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
		const QString text = QString::fromUtf8(m_process->readAll());
		m_renderLog += text;
		if (m_renderLog.size() > 20000) m_renderLog.remove(0, m_renderLog.size() - 20000);
		// "[12/120] Rendering frame_0012.ppm ..." says how far it is; the line after the last frame says it is putting the video together.
		static const QRegularExpression progress(QStringLiteral("\\[(\\d+)/(\\d+)\\] Rendering"));
		QRegularExpressionMatchIterator it = progress.globalMatch(text);
		int frame = 0;
		while (it.hasNext()) frame = it.next().captured(1).toInt();
		if (frame > 0) m_previewStatus->setText(tr("Video: frame %1 of %2...").arg(frame).arg(m_turntableFrames));
		else if (text.contains(QStringLiteral("ASSEMBLING VIDEO"))) m_previewStatus->setText(tr("Video: putting the video together..."));
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) onTurntableFinished(-1);
	});
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		onTurntableFinished(st == QProcess::NormalExit ? code : -2);
	});
	AppLog::info(QStringLiteral("builder-render"), QStringLiteral("start (%1, %2 frames at %3 fps -> %4): %5 %6").arg(keyframes.isEmpty() ? QStringLiteral("turntable") : QStringLiteral("flythrough")).arg(frames).arg(fps).arg(outMp4, launcherPath(), args.join(QLatin1Char(' '))));
	m_previewButton->setText(tr("Cancel"));
	m_previewButton->setEnabled(true);
	m_finalButton->setEnabled(false);
	if (m_turntableButton) m_turntableButton->setEnabled(false);
	if (m_flythroughButton) m_flythroughButton->setEnabled(false);
	m_previewStatus->setText(tr("Video: starting (%1 frames, %2 x %3, %4 samples)...").arg(frames).arg(width).arg(height).arg(samples));
	m_process->start(launcherPath(), args);
}

void SceneBuilderWidget::onTurntableFinished(int exitCode) {
	QProcess *p = m_process;
	if (!p) return;
	m_process = nullptr;
	p->disconnect(this);
	p->deleteLater();
	const auto done = m_pendingDone;
	m_pendingDone = nullptr;
	m_previewButton->setText(tr("Preview"));
	const QString dir = workFolder() + "/turntable";
	const QString made = dir + "/turntable_video.mp4";
	const double secs = m_renderClock.elapsed() / 1000.0;
	const bool haveVideo = exitCode == 0 && QFileInfo(made).size() > 0;
	AppLog::write(haveVideo ? log_format::Level::Info : log_format::Level::Error, QStringLiteral("builder-render"),
	              QStringLiteral("video finished: exit code %1%2, %3 s, video %4").arg(exitCode).arg(exitCode == -2 ? (m_cancelRequested ? QStringLiteral(" (cancelled)") : QStringLiteral(" (crashed or killed)")) : exitCode == -1 ? QStringLiteral(" (could not start)") : QString())
	                  .arg(secs, 0, 'f', 1).arg(haveVideo ? QStringLiteral("made") : QStringLiteral("missing")));
	QString message;
	bool ok = false;
	if (haveVideo) {
		QFile::remove(m_turntableOut);
		if (QFile::copy(made, m_turntableOut)) {
			ok = true;
			message = tr("Video done in %1 s. Saved %2.").arg(secs, 0, 'f', 1).arg(m_turntableOut);
		} else {
			message = tr("The video was made but could not be saved to %1.").arg(m_turntableOut);
		}
	} else if (exitCode == -2 && m_cancelRequested) {
		message = tr("The video was cancelled.");
	} else {
		const QStringList lines = m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts);
		for (const QString &line : lines.mid(std::max<qsizetype>(0, lines.size() - 25))) AppLog::error(QStringLiteral("builder-render"), QStringLiteral("renderer output: %1").arg(line));
		QString tail;
		for (qsizetype i = std::max<qsizetype>(0, lines.size() - 6); i < lines.size(); ++i) tail += lines[i] + "\n";
		const bool noFfmpeg = m_renderLog.contains(QStringLiteral("ffmpeg"), Qt::CaseInsensitive) && m_renderLog.contains(QStringLiteral("Could not launch"));
		message = exitCode == -2 ? tr("The renderer stopped unexpectedly.")
		        : noFfmpeg       ? tr("The frames were rendered, but putting them together needs the free program ffmpeg, which was not found. Install it (ffmpeg.org) and try again.")
		                         : tr("The renderer did not produce a video (exit code %1).\n%2").arg(exitCode).arg(tail.trimmed());
	}
	QDir(dir).removeRecursively();   // the frames are large; the video has been copied out
	m_previewStatus->setText(ok ? message : (exitCode == -2 && m_cancelRequested ? message : tr("The video failed.")));
	updateActions();
	if (done) done(ok, message);
}
