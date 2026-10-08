// mainwindow_downloads.cpp - downloading missing scene assets and asset packs (split out of mainwindow_slots.cpp; nothing changed).

#include "mainwindow.h"
#include "icon_tint.h"
#include "photo_import.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include "app_log.h"
#include "camera_math.h"
#include "../src/shared/video_preset.h"
#include "../src/shared/scene_descriptor.h"
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QLocale>
#include <QProgressDialog>
#include <QStorageInfo>
#include <QProcess>
#include <QDir>
#include <QTimer>
#include <QDateTime>
#include <QDesktopServices>
#include <QUrl>
#include <QScrollBar>
#include <QStatusBar>
#include <QCoreApplication>
#include <QSignalBlocker>
#include <QIcon>
#include <QStyle>
#include <QThread>
#include <QHash>
#include <array>
#include <cmath>
#include <optional>


// ---------------------------------------------------------------------------
// "Download missing files" (asset_downloader.h). The button appears under the scene info when the selected
// scene is missing files this project hosts; clicking it asks, downloads with a cancellable progress dialog, and
// re-checks the scene so the warning clears.
// ---------------------------------------------------------------------------

void MainWindow::onDownloadMissingAssetsClicked() {
	if (m_downloadablePack) startPackDownload(*m_downloadablePack, /*confirm=*/true);
	else startAssetDownload(m_downloadableAssetJobs, /*confirm=*/true);
}

void MainWindow::startAssetDownload(const QList<asset_downloader::Job> &jobs, bool confirm,
									std::function<void(bool, const QString &)> onDone) {
	if (jobs.isEmpty() || (m_assetDownloader && m_assetDownloader->isRunning())) return;

	qint64 bytes = 0;
	for (const auto &j : jobs) bytes += j.entry.size;
	const QString source = QUrl(asset_downloader::builtInManifest().baseUrl).host();

	if (confirm) {
		QStringList names;
		for (int i = 0; i < jobs.size() && i < 8; ++i) names << QFileInfo(jobs[i].destination).fileName();
		if (jobs.size() > 8) names << tr("… and %n more", "", jobs.size() - 8);
		const auto answer = QMessageBox::question(this, tr("Download missing files"),
			tr("Download %n file(s) (%1) from %2?\n\n%3\n\nThey will be saved in:\n%4", "", jobs.size())
				.arg(QLocale().formattedDataSize(bytes, 1), source, names.join(QStringLiteral("\n")),
					 QDir::toNativeSeparators(QFileInfo(jobs.first().destination).absolutePath())),
			QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
		if (answer != QMessageBox::Yes) return;
	}

	if (!m_assetDownloader) m_assetDownloader = new asset_downloader::Downloader(asset_downloader::builtInManifest(), this);

	QProgressDialog *dialog = nullptr;
	if (confirm) {
		dialog = new QProgressDialog(tr("Downloading…"), tr("Cancel"), 0, 100, this);
		dialog->setWindowTitle(tr("Download missing files"));
		dialog->setWindowModality(Qt::WindowModal);
		dialog->setMinimumDuration(0);
		dialog->setAutoClose(false);
		dialog->setAutoReset(false);
		dialog->setValue(0);
		connect(dialog, &QProgressDialog::canceled, m_assetDownloader, &asset_downloader::Downloader::cancel);
	}

	// One-shot connections: the Downloader outlives this call, so they must not pile up across clicks.
	auto *progressConn = new QMetaObject::Connection;
	auto *doneConn = new QMetaObject::Connection;
	*progressConn = connect(m_assetDownloader, &asset_downloader::Downloader::progress, this,
		[dialog](qint64 received, qint64 total, const QString &file) {
			if (!dialog || total <= 0) return;
			dialog->setLabelText(tr("Downloading %1…").arg(file));
			dialog->setValue(static_cast<int>(received * 100 / total));
		});
	*doneConn = connect(m_assetDownloader, &asset_downloader::Downloader::finished, this,
		[this, dialog, progressConn, doneConn, onDone](bool ok, const QString &error, int filesDone) {
			disconnect(*progressConn);
			disconnect(*doneConn);
			delete progressConn;
			delete doneConn;
			if (dialog) {
				dialog->close();
				dialog->deleteLater();
			}
			if (ok) {
				onLogMessage(tr("Downloaded %n file(s).", "", filesDone));
			} else {
				onLogMessage(tr("Download failed: %1").arg(error));
				if (dialog) QMessageBox::warning(this, tr("Download missing files"), error);
			}
			refreshSceneInfoLabel();   // the warning clears (or shrinks) now that the files are there
			if (onDone) onDone(ok, error);
		});
	onLogMessage(tr("Downloading %n file(s) (%1) from %2…", "", jobs.size())
		.arg(QLocale().formattedDataSize(bytes, 1), source));
	m_assetDownloader->start(jobs);
}

// ---------------------------------------------------------------------------
// Catalogue packs (scene_packs.h): the large third-party scenes, fetched from their ORIGINAL sites on request. Differs from the statue-model
// download above in what the confirmation must say (whose work this is and under what licence, where it comes from, how big), a disk-space
// check before anything starts (a pack can be over a gigabyte once unpacked), and a restart offer when the pack adds a scene file the
// registry only reads at startup.
// ---------------------------------------------------------------------------

void MainWindow::startPackDownload(const scene_packs::Pack &pack, bool confirm, std::function<void(bool, const QString &)> onDone) {
	if (m_packDownloader && m_packDownloader->isRunning()) return;

	QString root = asset_downloader::userAssetRoot();
	if (root.isEmpty()) root = QCoreApplication::applicationDirPath();
	QDir().mkpath(root);
	const qint64 downloadBytes = pack.downloadBytes(), diskBytes = pack.diskBytes();

	const qint64 free = QStorageInfo(root).bytesAvailable();
	const qint64 needed = diskBytes + diskBytes / 50 + (50ll << 20);
	if (free >= 0 && free < needed) {
		const QString message = tr("There is not enough free disk space for \"%1\": it needs about %2 and %3 is available in %4.")
			.arg(pack.title, QLocale().formattedDataSize(needed, 1), QLocale().formattedDataSize(free, 1), QDir::toNativeSeparators(root));
		onLogMessage(message);
		if (confirm) QMessageBox::warning(this, tr("Download missing files"), message);
		if (onDone) onDone(false, message);
		return;
	}

	if (confirm) {
		QStringList hosts;
		const auto addHost = [&hosts](const QString &url) { const QString h = QUrl(url).host(); if (!h.isEmpty() && !hosts.contains(h)) hosts << h; };
		for (const auto &a : pack.archives) addHost(a.url);
		for (const auto &f : pack.files) { addHost(f.url); break; }
		QString text = tr("Download \"%1\"?\n\n%2\n\nThis is third-party content, fetched directly from %3 (not from this project) and saved for your own use.\n\n"
						  "Download: %4 for %n file(s), about %5 on disk.\nSaved in: %6\nSource: %7",
						  "", pack.fileCount())
			.arg(pack.title, pack.credit, hosts.join(QStringLiteral(", ")), QLocale().formattedDataSize(downloadBytes, 1),
				 QLocale().formattedDataSize(diskBytes, 1), QDir::toNativeSeparators(root), pack.sourceUrl);
		if (downloadBytes > (1ll << 30)) text += tr("\n\nThis is a large download and may take a while.");
		if (QMessageBox::question(this, tr("Download missing files"), text, QMessageBox::Yes | QMessageBox::No, QMessageBox::No) != QMessageBox::Yes) return;
	}

	if (!m_packDownloader) m_packDownloader = new scene_packs::PackDownloader(this);

	QProgressDialog *dialog = nullptr;
	if (confirm) {
		dialog = new QProgressDialog(tr("Downloading…"), tr("Cancel"), 0, 1000, this);
		dialog->setWindowTitle(tr("Download missing files"));
		dialog->setWindowModality(Qt::WindowModal);
		dialog->setMinimumDuration(0);
		dialog->setAutoClose(false);
		dialog->setAutoReset(false);
		dialog->setValue(0);
		connect(dialog, &QProgressDialog::canceled, m_packDownloader, &scene_packs::PackDownloader::cancel);
	}

	auto *progressConn = new QMetaObject::Connection;
	auto *doneConn = new QMetaObject::Connection;
	const bool addsScene = pack.providesSceneFiles();
	*progressConn = connect(m_packDownloader, &scene_packs::PackDownloader::progress, this,
		[dialog](qint64 received, qint64 total, const QString &file) {
			if (!dialog || total <= 0) return;
			dialog->setLabelText(file.isEmpty() ? tr("Downloading…") : tr("Downloading %1…").arg(file));
			dialog->setValue(static_cast<int>(received * 1000 / total));
		});
	*doneConn = connect(m_packDownloader, &scene_packs::PackDownloader::finished, this,
		[this, dialog, progressConn, doneConn, onDone, addsScene](bool ok, const QString &error, int filesDone) {
			disconnect(*progressConn);
			disconnect(*doneConn);
			delete progressConn;
			delete doneConn;
			if (dialog) {
				dialog->close();
				dialog->deleteLater();
			}
			if (ok) {
				onLogMessage(tr("Downloaded %n file(s).", "", filesDone));
			} else {
				onLogMessage(tr("Download failed: %1").arg(error));
				if (dialog) QMessageBox::warning(this, tr("Download missing files"), error);
			}
			refreshSceneInfoLabel();
			if (onDone) onDone(ok, error);
			// A scene file the registry only reads at startup: its camera and settings come from that scan, so it needs a fresh start to appear right.
			if (ok && addsScene && dialog) {
				const auto answer = QMessageBox::question(this, tr("Download missing files"),
					tr("The scene was downloaded. It appears with its own camera and settings after the app restarts. Restart now?\n\n(A render in progress or queued jobs would be lost.)"),
					QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
				if (answer == QMessageBox::Yes && QProcess::startDetached(QCoreApplication::applicationFilePath(), QCoreApplication::arguments().mid(1)))
					qApp->quit();
			}
		});
	onLogMessage(tr("Downloading \"%1\" (%2)…").arg(pack.title, QLocale().formattedDataSize(downloadBytes, 1)));
	m_packDownloader->start(pack, root);
}
