#include "scene_packs.h"

#include <QDir>
#include <QFileInfo>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QTimer>
#include <QRegularExpression>
#include <QUrl>

#include <algorithm>
#include <utility>
#include <vector>

// miniz renames its API to zlib's names by default (crc32, compress, ...), which would rewrite Member::crc32; the mz_* names are all used here.
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include "../src/external/miniz.h"   // raw-deflate streaming (tinfl) and CRC-32; Qt has no public zip reader

namespace scene_packs {

namespace {
constexpr int kParallel = 4;
constexpr int kStallMs = 60000;
constexpr qint64 kLocalHeaderSlack = 512;   // read this far past the central directory's idea of the local header, so one request normally covers it
constexpr quint32 kLocalHeaderSignature = 0x04034b50;

quint32 le32(const char *p) {
	const auto *u = reinterpret_cast<const unsigned char *>(p);
	return quint32(u[0]) | (quint32(u[1]) << 8) | (quint32(u[2]) << 16) | (quint32(u[3]) << 24);
}
int le16(const char *p) {
	const auto *u = reinterpret_cast<const unsigned char *>(p);
	return int(u[0]) | (int(u[1]) << 8);
}
}  // namespace

qint64 Pack::downloadBytes() const {
	qint64 total = 0;
	for (const Archive &a : archives) for (const Member &m : a.members) total += m.compressedSize;
	for (const RemoteFile &f : files) total += f.size;
	return total;
}

qint64 Pack::diskBytes() const {
	qint64 total = 0;
	for (const Archive &a : archives) for (const Member &m : a.members) total += m.size;
	for (const RemoteFile &f : files) total += f.size;
	return total;
}

int Pack::fileCount() const {
	int n = files.size();
	for (const Archive &a : archives) n += a.members.size();
	return n;
}

bool Pack::providesSceneFiles() const {
	for (const RemoteFile &f : files) if (f.dest.endsWith(QLatin1String(".pbrt"), Qt::CaseInsensitive)) return true;
	return false;
}

const Pack *Catalogue::find(const QString &relativePath) const {
	const auto it = byDest.constFind(QDir::cleanPath(relativePath));
	return it == byDest.constEnd() ? nullptr : &packs[it.value()];
}

Catalogue parseCatalogue(const QString &text) {
	Catalogue c;
	const QStringList lines = text.split(QLatin1Char('\n'));
	for (const QString &raw : lines) {
		const QString line = raw.trimmed();
		if (line.isEmpty() || line.startsWith(QLatin1Char('#'))) continue;
		const QStringList f = line.split(QLatin1Char('|'));
		if (f[0] == QLatin1String("pack") && f.size() >= 5) {
			Pack p;
			p.id = f[1]; p.title = f[2]; p.credit = f[3]; p.sourceUrl = f[4];
			c.packs.append(p);
		} else if (c.packs.isEmpty()) {
			continue;
		} else if (f[0] == QLatin1String("archive") && f.size() >= 3) {
			Archive a;
			a.url = f[1]; a.size = f[2].toLongLong();
			c.packs.last().archives.append(a);
		} else if (f[0] == QLatin1String("member") && f.size() >= 10 && !c.packs.last().archives.isEmpty()) {
			Member m;
			bool ok = true;
			m.dest = f[1]; m.name = f[2];
			m.compressedSize = f[3].toLongLong(&ok);
			m.size = f[4].toLongLong();
			m.crc32 = f[5].toUInt(nullptr, 16);
			m.headerOffset = f[6].toLongLong();
			m.method = f[7].toInt();
			m.nameLength = f[8].toInt();
			m.extraLength = f[9].toInt();
			// A path that climbs out of the destination would let a bad catalogue write anywhere.
			if (!ok || m.dest.contains(QLatin1String("..")) || (m.method != 0 && m.method != 8)) continue;
			c.packs.last().archives.last().members.append(m);
		} else if (f[0] == QLatin1String("file") && f.size() >= 5) {
			RemoteFile rf;
			rf.url = f[1]; rf.dest = f[2]; rf.size = f[3].toLongLong(); rf.gitSha1 = f[4].toLower();
			if (rf.dest.contains(QLatin1String("..")) || rf.gitSha1.size() != 40) continue;
			c.packs.last().files.append(rf);
		}
	}
	for (int i = 0; i < c.packs.size(); ++i) {
		const Pack &p = c.packs[i];
		for (const Archive &a : p.archives) for (const Member &m : a.members) c.byDest.insert(QDir::cleanPath(m.dest), i);
		for (const RemoteFile &rf : p.files) c.byDest.insert(QDir::cleanPath(rf.dest), i);
	}
	return c;
}

const Catalogue &builtInCatalogue() {
	static const Catalogue catalogue = []() {
		QFile f(QStringLiteral(":/assets/scene_packs.txt"));
		return f.open(QIODevice::ReadOnly) ? parseCatalogue(QString::fromUtf8(f.readAll())) : Catalogue();
	}();
	return catalogue;
}

// ---------------------------------------------------------------------------------------------------------------------------------

struct PackDownloader::Task {
	bool isZip = false;
	QString dest, partPath, displayName;
	const Archive *archive = nullptr;
	Member member;
	RemoteFile file;
	qint64 expectedNetBytes = 0;     // what this task will move over the network (compressed size, or the file size)
	qint64 netBytes = 0;             // moved so far
	QNetworkReply *reply = nullptr;
	QFile out;
	// zip member parsing
	int zipState = 0;                // 0 local header, 1 skipping name+extra, 2 data, 3 all data seen
	QByteArray headerBuf;
	qint64 skipRemaining = 0, dataRemaining = 0;
	tinfl_decompressor inflater;
	std::vector<mz_uint8> dictionary;
	size_t dictOffset = 0;
	bool inflateDone = false;
	quint32 crc = MZ_CRC32_INIT;
	qint64 written = 0;
	QCryptographicHash sha1{QCryptographicHash::Sha1};
	bool checkedRange = false;
};

PackDownloader::PackDownloader(QObject *parent) : QObject(parent) {
	connect(&m_stallTimer, &QTimer::timeout, this, [this]() {
		if (m_running && m_activity.elapsed() > kStallMs) finishRun(false, tr("The download stalled: no data was received for %n second(s).", "", kStallMs / 1000));
	});
}

PackDownloader::~PackDownloader() {
	finishRun(false, QString(), /*notify=*/false);
	qDeleteAll(m_pending);
	qDeleteAll(m_active);
}

void PackDownloader::start(const Pack &pack, const QString &destinationRoot) {
	if (m_running) return;
	qDeleteAll(m_pending); m_pending.clear();
	qDeleteAll(m_active); m_active.clear();
	m_totalBytes = m_doneBytes = 0;
	m_filesDone = 0;
	m_cancelled = false;
	m_firstError.clear();

	const auto addTask = [&](Task *t, const QString &dest, qint64 sizeOnDisk) {
		t->dest = QDir::cleanPath(destinationRoot + QLatin1Char('/') + dest);
		t->partPath = t->dest + QStringLiteral(".part");
		t->displayName = QFileInfo(dest).fileName();
		m_totalBytes += t->expectedNetBytes;
		if (QFileInfo(t->dest).isFile() && QFileInfo(t->dest).size() == sizeOnDisk) {   // already there: resume
			m_doneBytes += t->expectedNetBytes;
			++m_filesDone;
			delete t;
			return;
		}
		m_pending.append(t);
	};
	for (const Archive &a : pack.archives) {
		for (const Member &m : a.members) {
			auto *t = new Task;
			t->isZip = true; t->archive = &a; t->member = m;
			t->expectedNetBytes = m.compressedSize;
			addTask(t, m.dest, m.size);
		}
	}
	for (const RemoteFile &f : pack.files) {
		auto *t = new Task;
		t->file = f;
		t->expectedNetBytes = f.size;
		addTask(t, f.dest, f.size);
	}
	m_running = true;
	m_activity.restart();
	m_stallTimer.start(5000);
	emit progress(m_doneBytes, m_totalBytes, QString());
	pump();
}

void PackDownloader::cancel() {
	if (!m_running) return;
	m_cancelled = true;
	finishRun(false, tr("Download cancelled."));
}

void PackDownloader::pump() {
	if (!m_running) return;
	while (!m_pending.isEmpty() && m_active.size() < kParallel) {
		Task *t = m_pending.takeFirst();
		m_active.append(t);
		startTask(t);
		if (!m_running) return;   // startTask failed and ended the run
	}
	if (m_pending.isEmpty() && m_active.isEmpty()) finishRun(true, QString());
}

void PackDownloader::startTask(Task *t) {
	QDir().mkpath(QFileInfo(t->dest).absolutePath());
	t->out.setFileName(t->partPath);
	if (!t->out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		failTask(t, tr("Cannot write to %1: %2\n\nCheck that you have permission to write there and that the disk is not full.")
						.arg(QDir::toNativeSeparators(QFileInfo(t->dest).absolutePath()), t->out.errorString()));
		return;
	}
	QNetworkRequest request;
	request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
	request.setRawHeader("User-Agent", "RayTracerGUI");   // the McGuire archive's server refuses requests without a browser-like agent
	// No Qt transfer timeout: it bounds the WHOLE transfer, which a 45 MB member legitimately exceeds on a slow link. A stall (no bytes at all
	// for kStallMs, across every connection) is detected by m_stallTimer instead.
	if (t->isZip) {
		request.setUrl(QUrl::fromEncoded(t->archive->url.toUtf8()));
		const qint64 first = t->member.headerOffset;
		qint64 last = first + 30 + t->member.nameLength + t->member.extraLength + kLocalHeaderSlack + t->member.compressedSize - 1;
		last = std::min(last, t->archive->size - 1);
		request.setRawHeader("Range", QByteArray("bytes=") + QByteArray::number(first) + "-" + QByteArray::number(last));
		t->dataRemaining = t->member.compressedSize;
		if (t->member.method == 8) {
			tinfl_init(&t->inflater);
			t->dictionary.assign(TINFL_LZ_DICT_SIZE, 0);
		}
	} else {
		request.setUrl(QUrl::fromEncoded(t->file.url.toUtf8()));
		t->sha1.addData(QByteArray("blob ") + QByteArray::number(t->file.size) + '\0');   // git hashes "blob <size>\0<content>"
	}
	t->reply = m_network.get(request);
	connect(t->reply, &QNetworkReply::readyRead, this, [this, t]() { onData(t); });
	connect(t->reply, &QNetworkReply::finished, this, [this, t]() { onTaskFinished(t); });
}

bool PackDownloader::writeOut(Task *t, const char *data, qint64 length, QString &error) {
	if (length <= 0) return true;
	if (t->out.write(data, length) != length) { error = tr("Could not write %1: %2").arg(t->displayName, t->out.errorString()); return false; }
	t->crc = static_cast<quint32>(mz_crc32(t->crc, reinterpret_cast<const mz_uint8 *>(data), static_cast<size_t>(length)));
	t->written += length;
	if (!t->isZip) t->sha1.addData(QByteArrayView(data, length));
	return true;
}

bool PackDownloader::consumeZipBytes(Task *t, const QByteArray &chunk, QString &error) {
	const char *p = chunk.constData();
	qint64 n = chunk.size();
	while (n > 0 && t->zipState != 3) {
		if (t->zipState == 0) {
			const qint64 take = std::min<qint64>(30 - t->headerBuf.size(), n);
			t->headerBuf.append(p, take); p += take; n -= take;
			if (t->headerBuf.size() < 30) continue;
			if (le32(t->headerBuf.constData()) != kLocalHeaderSignature) { error = tr("%1: the archive's layout is not what was recorded (it has probably changed upstream).").arg(t->displayName); return false; }
			t->skipRemaining = le16(t->headerBuf.constData() + 26) + le16(t->headerBuf.constData() + 28);
			t->zipState = 1;
		} else if (t->zipState == 1) {
			const qint64 skip = std::min(t->skipRemaining, n);
			p += skip; n -= skip; t->skipRemaining -= skip;
			if (t->skipRemaining == 0) t->zipState = (t->dataRemaining > 0) ? 2 : 3;
		} else {   // 2: compressed data
			const qint64 take = std::min(t->dataRemaining, n);
			t->netBytes += take;
			if (t->member.method == 0) {
				if (!writeOut(t, p, take, error)) return false;
			} else {
				const mz_uint8 *in = reinterpret_cast<const mz_uint8 *>(p);
				size_t inLeft = static_cast<size_t>(take);
				const bool lastChunk = (t->dataRemaining - take) == 0;
				while (!t->inflateDone && (inLeft > 0 || lastChunk)) {
					size_t inBytes = inLeft;
					size_t outBytes = t->dictionary.size() - t->dictOffset;
					const mz_uint32 flags = lastChunk ? 0 : TINFL_FLAG_HAS_MORE_INPUT;
					const tinfl_status st = tinfl_decompress(&t->inflater, in, &inBytes, t->dictionary.data(), t->dictionary.data() + t->dictOffset, &outBytes, flags);
					in += inBytes; inLeft -= inBytes;
					if (!writeOut(t, reinterpret_cast<const char *>(t->dictionary.data() + t->dictOffset), static_cast<qint64>(outBytes), error)) return false;
					t->dictOffset = (t->dictOffset + outBytes) & (t->dictionary.size() - 1);
					if (st == TINFL_STATUS_DONE) { t->inflateDone = true; break; }
					if (st < 0) { error = tr("%1 could not be decompressed (corrupt download).").arg(t->displayName); return false; }
					if (st == TINFL_STATUS_NEEDS_MORE_INPUT && inLeft == 0) break;
				}
			}
			p += take; n -= take; t->dataRemaining -= take;
			if (t->dataRemaining == 0) t->zipState = 3;
		}
	}
	return true;
}

void PackDownloader::onData(Task *t) {
	if (!m_running || !t->reply) return;
	if (t->isZip && !t->checkedRange) {
		// A zip's size is recorded in the catalogue; if the server now reports another total, the recorded offsets point into a different file.
		const QByteArray cr = t->reply->rawHeader("Content-Range");   // "bytes 100-200/12345"
		const int slash = cr.lastIndexOf('/');
		if (slash >= 0) {
			t->checkedRange = true;
			if (cr.mid(slash + 1).toLongLong() != t->archive->size) {
				failTask(t, tr("%1 has changed upstream since this app's list of downloads was made, so it can no longer be fetched safely.").arg(QFileInfo(t->archive->url).fileName()));
				return;
			}
		}
	}
	m_activity.restart();
	const QByteArray chunk = t->reply->readAll();
	QString error;
	if (t->isZip) {
		if (!consumeZipBytes(t, chunk, error)) { failTask(t, error); return; }
		// Everything wanted has arrived; stop reading the slack. Queued: abort() finishes the reply synchronously, which would delete this task under us.
		if (t->zipState == 3 && t->reply) QMetaObject::invokeMethod(t->reply, &QNetworkReply::abort, Qt::QueuedConnection);
	} else {
		t->netBytes += chunk.size();
		if (t->netBytes > t->file.size) { failTask(t, tr("%1 is larger than expected and was discarded.").arg(t->displayName)); return; }
		if (!writeOut(t, chunk.constData(), chunk.size(), error)) { failTask(t, error); return; }
	}
	qint64 inFlight = 0;
	for (const Task *a : std::as_const(m_active)) inFlight += std::min(a->netBytes, a->expectedNetBytes);
	emit progress(m_doneBytes + inFlight, m_totalBytes, t->displayName);
}

void PackDownloader::onTaskFinished(Task *t) {
	if (!m_running || !m_active.contains(t) || !t->reply) return;
	QNetworkReply *reply = t->reply;
	t->reply = nullptr;
	reply->deleteLater();
	// onData() aborts the reply once a zip member is complete; that surfaces here as OperationCanceledError and is not a failure.
	const bool abortedOnPurpose = t->isZip && t->zipState == 3;
	if (reply->error() != QNetworkReply::NoError && !abortedOnPurpose) {
		failTask(t, tr("Could not download %1: %2").arg(t->displayName, reply->errorString()));
		return;
	}
	QString error;
	if (!abortedOnPurpose && reply->bytesAvailable() > 0) {   // data that arrived with the final signal
		const QByteArray rest = reply->readAll();
		if (t->isZip ? !consumeZipBytes(t, rest, error) : !writeOut(t, rest.constData(), rest.size(), error)) { failTask(t, error); return; }
		if (!t->isZip) t->netBytes += rest.size();
	}
	t->out.close();
	if (t->isZip) {
		if (t->zipState != 3 || (t->member.method == 8 && !t->inflateDone) || t->written != t->member.size) {
			failTask(t, tr("%1 downloaded incompletely and was discarded.").arg(t->displayName)); return;
		}
		if (t->crc != t->member.crc32) { failTask(t, tr("%1 failed its checksum (CRC-32) and was discarded.").arg(t->displayName)); return; }
	} else {
		if (t->written != t->file.size || QString::fromLatin1(t->sha1.result().toHex()) != t->file.gitSha1) {
			failTask(t, tr("%1 failed its checksum and was discarded.").arg(t->displayName)); return;
		}
	}
	QFile::remove(t->dest);
	if (!QFile::rename(t->partPath, t->dest)) { failTask(t, tr("Could not move the downloaded file into place at %1.").arg(QDir::toNativeSeparators(t->dest))); return; }
	m_doneBytes += t->expectedNetBytes;
	++m_filesDone;
	m_active.removeOne(t);
	delete t;
	emit progress(m_doneBytes, m_totalBytes, QString());
	pump();
}

void PackDownloader::failTask(Task *t, const QString &message) {
	Q_UNUSED(t);
	finishRun(false, message);
}

void PackDownloader::finishRun(bool ok, const QString &error, bool notify) {
	if (!m_running) return;
	m_running = false;
	m_stallTimer.stop();
	for (Task *a : std::as_const(m_active)) {
		if (a->reply) { a->reply->disconnect(this); a->reply->abort(); a->reply->deleteLater(); a->reply = nullptr; }
		a->out.close();
		QFile::remove(a->partPath);
	}
	qDeleteAll(m_active); m_active.clear();
	qDeleteAll(m_pending); m_pending.clear();
	if (notify) emit finished(ok, error, m_filesDone);
}

}  // namespace scene_packs
