// scene_builder_render.cpp - the Scene Builder's preview and final renders: running the renderer on the scene file and showing what it made (see scene_builder_widget.h).
#include "scene_builder_widget.h"

#include "scene_builder_common.h"
#include "app_log.h"
#include "window_geometry.h"
#include "atomic_file.h"
#include "../src/shared/scene_doc_diff.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

using namespace scene_builder_ui;

// ---- inspector ------------------------------------------------------------------------------------------------------------------

// ---- rendering -------------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::setUseGpu(bool on) { m_gpuCheck->setChecked(on); }

void SceneBuilderWidget::setSceneName(const QString &text) {
	const std::string name = text.toStdString();
	if (name == m_doc.title) return;
	edit(QStringLiteral("title"), [&]() { m_doc.title = name; });
}
void SceneBuilderWidget::startPreview(const std::function<void(bool, const QString &)> &done) {
	static const int widths[3] = {480, 720, 960};
	static const int spps[3] = {16, 64, 256};
	const int q = std::clamp(m_qualityCombo->currentData().toInt(), 0, 2);
	const double aspect = m_doc.render.width > 0 ? double(m_doc.render.height) / m_doc.render.width : 0.75;
	const int w = widths[q];
	const int h = std::max(1, static_cast<int>(std::lround(w * aspect)));
	runRender(w, h, spps[q], false, QString(), done);
}

void SceneBuilderWidget::onRenderFinalClicked() {
	QString start = m_path.isEmpty() ? QDir::homePath() + "/render.png" : QFileInfo(m_path).absolutePath() + "/" + QFileInfo(m_path).completeBaseName() + ".png";
	QString png = QFileDialog::getSaveFileName(this, tr("Save the rendered picture"), start, tr("PNG images (*.png)"));
	if (png.isEmpty()) return;
	if (!png.endsWith(".png", Qt::CaseInsensitive)) png += ".png";
	runRender(m_doc.render.width, m_doc.render.height, m_doc.render.samples, true, png, [this](bool ok, const QString &msg) {
		if (!ok) QMessageBox::warning(this, tr("The render failed"), msg);
	});
}

void SceneBuilderWidget::runRender(int width, int height, int samples, bool toFinalFile, const QString &finalPng,
                                   const std::function<void(bool, const QString &)> &done) {
	auto fail = [&](const QString &msg) {
		AppLog::warn(QStringLiteral("builder-render"), QStringLiteral("not started: %1").arg(msg));
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
	};
	if (m_process) return fail(tr("A render is already running."));
	const auto problems = scene_doc::validate(m_doc);
	if (scene_doc::hasErrors(problems)) return fail(tr("Fix the problems listed under the properties first."));
	if (!QFileInfo::exists(launcherPath())) return fail(tr("The renderer (%1) was not found next to the program.").arg(launcherPath()));

	const QString dir = workFolder();
	const QString scene = dir + "/scene.pbrt";
	const QString base = dir + (toFinalFile ? "/final" : "/preview");
	QFile::remove(base + ".ppm");
	QFile::remove(base + ".png");
	{
		QFile f(scene);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return fail(tr("Could not write %1.").arg(scene));
		const std::string text = scene_doc::toPbrt(m_doc);
		f.write(text.data(), static_cast<qint64>(text.size()));
	}

	QStringList args;
	args << (m_gpuCheck->isChecked() ? "--gpu" : "--cpu") << "--output" << base + ".ppm" << "--height" << QString::number(height)
	     << QString::number(width) << QString::number(samples) << QString::number(m_doc.render.maxDepth) << scene;

	m_pendingFinalPng = toFinalFile ? finalPng : QString();
	m_previewPng = base + ".png";
	m_pendingDone = done;
	m_renderLog.clear();
	m_cancelRequested = false;
	m_renderClock.start();
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setWorkingDirectory(QCoreApplication::applicationDirPath());
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
		m_renderLog += QString::fromUtf8(m_process->readAll());
		if (m_renderLog.size() > 20000) m_renderLog.remove(0, m_renderLog.size() - 20000);
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) onPreviewFinished(-1);
	});
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		onPreviewFinished(st == QProcess::NormalExit ? code : -2);
	});
	AppLog::info(QStringLiteral("builder-render"), QStringLiteral("start (%1): %2 %3").arg(toFinalFile ? QStringLiteral("final picture -> ") + finalPng : QStringLiteral("preview"), launcherPath(), args.join(QLatin1Char(' '))));
	for (const auto &problem : problems) AppLog::warn(QStringLiteral("builder-render"), QStringLiteral("scene note: %1").arg(QString::fromStdString(problem.message)));
	m_previewButton->setText(tr("Cancel"));
	m_previewButton->setEnabled(true);
	m_finalButton->setEnabled(false);
	m_previewStatus->setText(tr("Rendering %1 x %2, %3 samples...").arg(width).arg(height).arg(samples));
	m_process->start(launcherPath(), args);
}

void SceneBuilderWidget::onPreviewFinished(int exitCode) {
	QProcess *p = m_process;
	if (!p) return;
	m_process = nullptr;
	p->disconnect(this);
	p->deleteLater();
	const auto done = m_pendingDone;
	m_pendingDone = nullptr;
	m_previewButton->setText(tr("Preview"));
	updateActions();

	const double secs = m_renderClock.elapsed() / 1000.0;
	QPixmap pix(m_previewPng);
	AppLog::write(exitCode != 0 || pix.isNull() ? log_format::Level::Error : log_format::Level::Info, QStringLiteral("builder-render"),
	              QStringLiteral("finished: exit code %1%2, %3 s, picture %4").arg(exitCode).arg(exitCode == -2 ? (m_cancelRequested ? QStringLiteral(" (cancelled)") : QStringLiteral(" (crashed or killed)")) : exitCode == -1 ? QStringLiteral(" (could not start)") : QString())
	                  .arg(secs, 0, 'f', 1).arg(pix.isNull() ? QStringLiteral("missing") : QStringLiteral("%1 x %2").arg(pix.width()).arg(pix.height())));
	if (exitCode != 0 || pix.isNull()) {
		for (const QString &line : m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts).mid(std::max<qsizetype>(0, m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts).size() - 25)))
			AppLog::error(QStringLiteral("builder-render"), QStringLiteral("renderer output: %1").arg(line));
	}
	if (exitCode != 0 || pix.isNull()) {
		QString tail;
		const QStringList lines = m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts);
		for (int i = std::max(0, static_cast<int>(lines.size()) - 6); i < lines.size(); ++i) tail += lines[i] + "\n";
		const QString msg = (exitCode == -2 && m_cancelRequested) ? tr("The render was cancelled.") : exitCode == -2 ? tr("The renderer stopped unexpectedly.") : tr("The renderer did not produce a picture (exit code %1).\n%2").arg(exitCode).arg(tail.trimmed());
		m_previewStatus->setText((exitCode == -2 && m_cancelRequested) ? msg : tr("The render failed."));
		if (done) done(false, msg);
		return;
	}
	m_previewPixmap = pix;
	updatePreviewPixmap();
	QString message = tr("Done in %1 s (%2 x %3).").arg(secs, 0, 'f', 1).arg(pix.width()).arg(pix.height());
	if (!m_pendingFinalPng.isEmpty()) {
		QFile::remove(m_pendingFinalPng);
		if (QFile::copy(m_previewPng, m_pendingFinalPng)) message += " " + tr("Saved %1.").arg(m_pendingFinalPng);
		else message += " " + tr("Could not save to %1.").arg(m_pendingFinalPng);
	}
	m_previewStatus->setText(message);
	if (done) done(true, message);
}

void SceneBuilderWidget::updatePreviewPixmap() {
	if (m_previewPixmap.isNull()) return;
	m_previewLabel->setPixmap(m_previewPixmap.scaled(m_previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
