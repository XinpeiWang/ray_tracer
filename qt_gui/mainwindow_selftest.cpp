// mainwindow_selftest.cpp - an opt-in automated smoke test of the real GUI (the project had none): run the app with
//   RT_GUI_SELFTEST=livepreview RT_GUI_SELFTEST_OUT=/tmp/gui_selftest  (optionally QT_QPA_PLATFORM=offscreen)
// and it drives the actual MainWindow - selects the Live Preview output mode, starts it, lets it render, saves a screenshot
// of its own window (never the screen) and a text log, and exits with 0 on success, 1 on failure, 2 if Live Preview is not
// available in this build.
#include "mainwindow.h"

#include <QApplication>
#include <QComboBox>
#include <QFile>
#include <QPixmap>
#include <QScrollArea>
#include <QScrollBar>
#include <QStandardItemModel>
#include <QTextStream>
#include <QTimer>
#include <memory>

void MainWindow::runSelfTest(const QString &mode, const QString &outPrefix) {
	auto log = [outPrefix](const QString &line) {
		QFile f(outPrefix + ".txt");
		if (f.open(QIODevice::Append | QIODevice::Text)) QTextStream(&f) << line << "\n";
		QTextStream(stdout) << "[selftest] " << line << "\n";
	};
	auto shot = [this, outPrefix, log](const QString &tag) {
		const QString path = outPrefix + "_" + tag + ".png";
		const bool ok = grab().save(path);   // this window's own contents only
		log(QString("screenshot %1: %2").arg(path, ok ? "saved" : "FAILED"));
	};

	// Every Output Mode item and whether it can be selected.
	for (int i = 0; i < m_modeCombo->count(); ++i) {
		bool enabled = true;
		if (auto *model = qobject_cast<QStandardItemModel *>(m_modeCombo->model()))
			if (QStandardItem *item = model->item(i)) enabled = item->isEnabled();
		log(QString("output mode item %1: \"%2\" enabled=%3").arg(i).arg(m_modeCombo->itemText(i)).arg(enabled ? 1 : 0));
	}

	if (mode == "options") {
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
		QTimer::singleShot(800, this, [this, shot]() {
			shot("options");
			// ...and scrolled to the bottom, where the Live Preview Settings group is.
			QList<QScrollArea *> areas = m_tabWidget->currentWidget()->findChildren<QScrollArea *>();
			if (auto *self = qobject_cast<QScrollArea *>(m_tabWidget->currentWidget())) areas.prepend(self);
			for (QScrollArea *sa : areas)
				sa->verticalScrollBar()->setValue(sa->verticalScrollBar()->maximum());
			QTimer::singleShot(500, this, [shot]() { shot("options_bottom"); QApplication::exit(0); });
		});
		return;
	}

	if (mode == "ui") {
		shot("ui");
		QApplication::exit(0);
		return;
	}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	if (mode == "livepreview") {
		const int idx = m_modeCombo->findData(static_cast<int>(OutputMode::LivePreview));
		bool enabled = false;
		if (idx >= 0)
			if (auto *model = qobject_cast<QStandardItemModel *>(m_modeCombo->model()))
				if (QStandardItem *item = model->item(idx)) enabled = item->isEnabled();
		if (!enabled) {
			log("Live Preview is present=" + QString::number(idx >= 0) + " but NOT enabled (realtime_renderer library not found?)");
			shot("unavailable");
			QApplication::exit(2);
			return;
		}
		m_modeCombo->setCurrentIndex(idx);
		startLivePreview();
		log(QString("started: running=%1").arg(m_livePreviewRunning ? 1 : 0));
		// Let it render for a while, then look at what happened.
		// Timeline: look at 4 s, orbit the camera at 5 s (what a mouse drag does), look again at 9 s, report at 10 s.
		auto before = std::make_shared<QImage>();
		QTimer::singleShot(4000, this, [this, log, shot, before]() {
			log(QString("after 4 s: frames=%1").arg(m_livePreviewFrameCount));
			shot("live_4s");
			*before = grab().toImage();
		});
		QTimer::singleShot(5000, this, [this, log]() {
			m_orbit.azimuth += 0.7;
			updateLivePreviewCameraFromOrbit();
			log("orbited the camera by 0.7 rad");
		});
		QTimer::singleShot(10000, this, [this, log, shot, before]() {
			const qint64 frames = m_livePreviewFrameCount;
			const double secs = m_livePreviewSessionTimer.elapsed() / 1000.0;
			log(QString("after 10 s: frames=%1 (%2 fps)").arg(frames).arg(secs > 0 ? frames / secs : 0.0, 0, 'f', 1));
			shot("live_10s");
			const QImage after = grab().toImage();
			double diff = 0;
			if (!before->isNull() && before->size() == after.size()) {
				qint64 sum = 0, n = 0;
				for (int y = 0; y < after.height(); y += 2)
					for (int x = 0; x < after.width(); x += 2) {
						const QRgb a = before->pixel(x, y), b = after.pixel(x, y);
						sum += qAbs(qRed(a) - qRed(b)) + qAbs(qGreen(a) - qGreen(b)) + qAbs(qBlue(a) - qBlue(b));
						n += 3;
					}
				diff = n ? double(sum) / n : 0.0;
			}
			log(QString("picture change after the camera move: %1 (mean abs diff per channel, 0-255)").arg(diff, 0, 'f', 2));
			stopLivePreview();
			const bool ok = frames > 20 && diff > 1.0;
			log(ok ? "RESULT: OK" : "RESULT: FAIL (too few frames, or the picture did not change when the camera moved)");
			QApplication::exit(ok ? 0 : 1);
		});
		return;
	}
#endif
	log("unknown or unavailable self-test mode: " + mode);
	QApplication::exit(2);
}
