#include "asset_downloader.h"

#include <QCoreApplication>
#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QUrl>

namespace asset_downloader {

const Entry *Manifest::find(const QString &appDir, const QString &absolutePath) const {
	const QString normalised = QDir::cleanPath(absolutePath);
	for (const Entry &e : entries) {
		if (QDir::cleanPath(appDir + QLatin1Char('/') + e.relativePath) == normalised) return &e;
	}
	return nullptr;
}

Manifest parseManifest(const QString &text) {
	Manifest m;
	const QStringList lines = text.split(QLatin1Char('\n'));
	for (const QString &raw : lines) {
		const QString line = raw.trimmed();
		if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
		if (line.startsWith(QLatin1String("base "))) {
			m.baseUrl = line.mid(5).trimmed();
			continue;
		}
		const QStringList parts = line.split(QLatin1Char(' '), Qt::SkipEmptyParts);
		if (parts.size() != 3) continue;
		bool sizeOk = false;
		Entry e;
		e.relativePath = parts[0];
		e.size = parts[1].toLongLong(&sizeOk);
		e.sha256 = parts[2].toLower();
		// A relative path that climbs out of the app folder would let a bad manifest write anywhere.
		if (!sizeOk || e.size <= 0 || e.sha256.size() != 64 || e.relativePath.contains(QLatin1String(".."))) continue;
		m.entries.append(e);
	}
	if (m.baseUrl.isEmpty()) m.entries.clear();
	return m;
}

const Manifest &builtInManifest() {
	static const Manifest manifest = []() {
		QFile f(QStringLiteral(":/assets/downloadable_assets.txt"));
		Manifest m;
		if (f.open(QIODevice::ReadOnly)) m = parseManifest(QString::fromUtf8(f.readAll()));
		const QString override = qEnvironmentVariable("RT_ASSET_BASE_URL");
		if (!override.isEmpty() && !m.baseUrl.isEmpty()) m.baseUrl = override.endsWith(QLatin1Char('/')) ? override : override + QLatin1Char('/');
		return m;
	}();
	return manifest;
}

Downloader::Downloader(const Manifest &manifest, QObject *parent) : QObject(parent), m_manifest(manifest) {}

void Downloader::start(const QList<Job> &jobs) {
	if (m_running) return;
	m_jobs = jobs;
	m_index = 0;
	m_doneBytes = 0;
	m_fileBytes = 0;
	m_totalBytes = 0;
	for (const Job &j : m_jobs) m_totalBytes += j.entry.size;
	m_cancelled = false;
	m_running = true;
	startNext();
}

void Downloader::cancel() {
	if (!m_running) return;
	m_cancelled = true;
	if (m_reply) m_reply->abort();   // onReplyFinished() does the cleanup and reports
	else finishWith(false, tr("Download cancelled."));
}

void Downloader::startNext() {
	if (m_index >= m_jobs.size()) {
		finishWith(true, QString());
		return;
	}
	const Job &job = m_jobs[m_index];
	if (QFileInfo::exists(job.destination) && QFileInfo(job.destination).size() == job.entry.size) {
		// Already there (another scene fetched it, or the user copied it in meanwhile).
		m_doneBytes += job.entry.size;
		++m_index;
		startNext();
		return;
	}
	QDir().mkpath(QFileInfo(job.destination).absolutePath());
	m_file = std::make_unique<QFile>(job.destination + QStringLiteral(".part"));
	if (!m_file->open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		finishWith(false, tr("Cannot write to %1: %2\n\nIf the application is in a read-only location, move it to a "
							 "folder you can write to (for example Applications) and try again.")
							  .arg(QDir::toNativeSeparators(QFileInfo(job.destination).absolutePath()), m_file->errorString()));
		return;
	}
	m_fileBytes = 0;
	QNetworkRequest request{QUrl(m_manifest.urlFor(job.entry))};
	request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
	m_reply = m_network.get(request);
	connect(m_reply, &QNetworkReply::readyRead, this, &Downloader::onReadyRead);
	connect(m_reply, &QNetworkReply::finished, this, &Downloader::onReplyFinished);
	emit progress(m_doneBytes, m_totalBytes, QFileInfo(job.destination).fileName());
}

void Downloader::onReadyRead() {
	if (!m_reply || !m_file) return;
	const QByteArray chunk = m_reply->readAll();
	if (m_file->write(chunk) != chunk.size()) {
		m_reply->abort();
		return;
	}
	m_fileBytes += chunk.size();
	emit progress(m_doneBytes + m_fileBytes, m_totalBytes, QFileInfo(m_jobs[m_index].destination).fileName());
}

void Downloader::onReplyFinished() {
	QNetworkReply *reply = m_reply;
	m_reply = nullptr;
	if (!reply) return;
	reply->deleteLater();
	const Job job = m_jobs[m_index];
	const QString partPath = job.destination + QStringLiteral(".part");
	if (reply->isOpen() && m_file && m_file->isOpen()) m_file->write(reply->readAll());
	if (m_file) m_file->close();

	const auto fail = [&](const QString &message) {
		QFile::remove(partPath);
		finishWith(false, message);
	};
	if (m_cancelled) return fail(tr("Download cancelled."));
	if (reply->error() != QNetworkReply::NoError)
		return fail(tr("Could not download %1: %2").arg(QFileInfo(job.destination).fileName(), reply->errorString()));

	QFile part(partPath);
	if (!part.open(QIODevice::ReadOnly)) return fail(tr("Could not read back the downloaded file %1.").arg(partPath));
	QCryptographicHash hash(QCryptographicHash::Sha256);
	const bool hashed = hash.addData(&part);
	const qint64 size = part.size();
	part.close();
	if (!hashed || size != job.entry.size || hash.result().toHex().toLower() != job.entry.sha256.toLatin1())
		return fail(tr("%1 downloaded incorrectly (size or checksum mismatch) and was discarded.")
						.arg(QFileInfo(job.destination).fileName()));

	QFile::remove(job.destination);
	if (!QFile::rename(partPath, job.destination))
		return fail(tr("Could not move the downloaded file into place at %1.").arg(QDir::toNativeSeparators(job.destination)));
	m_doneBytes += job.entry.size;
	++m_index;
	startNext();
}

void Downloader::finishWith(bool ok, const QString &error) {
	m_running = false;
	m_file.reset();
	emit finished(ok, error, m_index);
}

}  // namespace asset_downloader
