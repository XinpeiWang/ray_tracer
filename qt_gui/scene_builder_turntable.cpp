// The Scene Builder's turntable: a video of the scene seen from a camera that goes once round the point it looks at, at the camera's own height and distance, so the
// whole thing can be seen from every side. The renderer already makes such videos (its --video mode with the orbit camera path, which starts at the scene's own camera);
// this file runs it on the scene file and puts the finished MP4 where the user asked (see scene_builder_widget.h). Assembling the MP4 needs ffmpeg on Windows and Linux; a Mac
// has its own encoder.
#include "scene_builder_widget.h"

#include "app_log.h"
#include "scene_builder_common.h"

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

// The Turntable... window: how long, how fast, how good, then where to save it.
void SceneBuilderWidget::showTurntableDialog() {
	QDialog dialog(this);
	dialog.setWindowTitle(tr("Turntable video"));
	auto *root = new QVBoxLayout(&dialog);
	auto *intro = new QLabel(tr("Makes a video of the scene from a camera that goes once round the point the camera looks at, at the camera's own height and distance, starting where it is now. "
	                            "It renders every frame, so a long or large video takes a while; start with Draft."), &dialog);
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
	form->addRow(tr("Length:"), seconds);
	form->addRow(tr("Speed:"), fps);
	form->addRow(tr("Quality:"), quality);
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

	const QString start = (m_path.isEmpty() ? QDir::homePath() + "/turntable" : QFileInfo(m_path).absolutePath() + "/" + QFileInfo(m_path).completeBaseName() + "-turntable") + ".mp4";
	QString mp4 = QFileDialog::getSaveFileName(this, tr("Save the turntable video"), start, tr("MP4 videos (*.mp4)"));
	if (mp4.isEmpty()) return;
	if (!mp4.endsWith(".mp4", Qt::CaseInsensitive)) mp4 += ".mp4";
	const TurntableQuality q = kQualities[std::clamp(quality->currentIndex(), 0, 3)];
	startTurntable(seconds->value() * fps->currentData().toInt(), fps->currentData().toInt(), q.width > 0 ? q.width : m_doc.render.width, q.samples > 0 ? q.samples : m_doc.render.samples, mp4,
	               [this](bool ok, const QString &message) {
		               if (!ok) QMessageBox::warning(this, tr("The turntable failed"), message);
	               });
}

void SceneBuilderWidget::startTurntable(int frames, int fps, int width, int samples, const QString &outMp4, const std::function<void(bool, const QString &)> &done) {
	flushEditLog();
	const auto fail = [&](const QString &msg) {
		AppLog::warn(QStringLiteral("builder-render"), QStringLiteral("turntable not started: %1").arg(msg));
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
	};
	if (m_process) return fail(tr("A render is already running."));
	if (frames < 1 || fps < 1 || width < 1) return fail(tr("The turntable needs at least one frame."));
	const auto problems = scene_doc::validate(m_doc);
	if (scene_doc::hasErrors(problems)) return fail(tr("Fix the problems listed under the properties first."));
	if (!QFileInfo::exists(launcherPath())) return fail(tr("The renderer (%1) was not found next to the program.").arg(launcherPath()));

	// Its own folder: the renderer clears a "frames" folder beside its output before it starts.
	const QString dir = workFolder() + "/turntable";
	QDir(dir).removeRecursively();
	QDir().mkpath(dir);
	const QString scene = dir + "/scene.pbrt";
	QString error;
	if (!writeSceneFile(scene, &error)) return fail(error);

	const double aspect = m_doc.render.width > 0 ? double(m_doc.render.height) / m_doc.render.width : 0.75;
	const int height = std::max(1, static_cast<int>(std::lround(width * aspect)));
	QStringList args;
	args << (m_gpuCheck->isChecked() ? "--gpu" : "--cpu") << "--output" << dir + "/turntable.ppm" << "--video" << "--camera-path" << "orbit" << "--frames" << QString::number(frames)
	     << "--fps" << QString::number(fps) << "--height" << QString::number(height) << QString::number(width) << QString::number(samples) << QString::number(m_doc.render.maxDepth) << scene;

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
		if (frame > 0) m_previewStatus->setText(tr("Turntable: frame %1 of %2...").arg(frame).arg(m_turntableFrames));
		else if (text.contains(QStringLiteral("ASSEMBLING VIDEO"))) m_previewStatus->setText(tr("Turntable: putting the video together..."));
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) onTurntableFinished(-1);
	});
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		onTurntableFinished(st == QProcess::NormalExit ? code : -2);
	});
	AppLog::info(QStringLiteral("builder-render"), QStringLiteral("start (turntable, %1 frames at %2 fps -> %3): %4 %5").arg(frames).arg(fps).arg(outMp4, launcherPath(), args.join(QLatin1Char(' '))));
	m_previewButton->setText(tr("Cancel"));
	m_previewButton->setEnabled(true);
	m_finalButton->setEnabled(false);
	if (m_turntableButton) m_turntableButton->setEnabled(false);
	m_previewStatus->setText(tr("Turntable: starting (%1 frames, %2 x %3, %4 samples)...").arg(frames).arg(width).arg(height).arg(samples));
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
	              QStringLiteral("turntable finished: exit code %1%2, %3 s, video %4").arg(exitCode).arg(exitCode == -2 ? (m_cancelRequested ? QStringLiteral(" (cancelled)") : QStringLiteral(" (crashed or killed)")) : exitCode == -1 ? QStringLiteral(" (could not start)") : QString())
	                  .arg(secs, 0, 'f', 1).arg(haveVideo ? QStringLiteral("made") : QStringLiteral("missing")));
	QString message;
	bool ok = false;
	if (haveVideo) {
		QFile::remove(m_turntableOut);
		if (QFile::copy(made, m_turntableOut)) {
			ok = true;
			message = tr("Turntable done in %1 s. Saved %2.").arg(secs, 0, 'f', 1).arg(m_turntableOut);
		} else {
			message = tr("The video was made but could not be saved to %1.").arg(m_turntableOut);
		}
	} else if (exitCode == -2 && m_cancelRequested) {
		message = tr("The turntable was cancelled.");
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
	m_previewStatus->setText(ok ? message : (exitCode == -2 && m_cancelRequested ? message : tr("The turntable failed.")));
	updateActions();
	if (done) done(ok, message);
}
