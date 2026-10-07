#pragma once
// scene_packs.h - the catalogue and downloader behind "Download missing files" for scenes whose files are NOT in this repository: the large
// third-party environment scenes (Sponza, Bistro, San Miguel, ...) and the pbrt-v4-scenes examples (Zero Day, Villa, ...).
//
// Everything is fetched from the ORIGINAL upstream sites, straight to the user's machine, on the user's request; this project hosts and
// redistributes none of it. The catalogue (:/assets/scene_packs.txt, written by scripts/gen_scene_packs.py - see that script for the format
// and the reasoning) pins what may be fetched:
//   * a McGuire-archive pack lists the individual zip MEMBERS a scene references, with each member's size, CRC-32 and header offset, so only
//     those members are fetched (HTTP range requests) instead of whole multi-hundred-megabyte archives, and every extracted file is
//     CRC-checked;
//   * a pbrt-v4-scenes pack lists each file at one pinned commit with its git blob SHA-1, which is verified after download.
// Nothing outside the catalogue is ever fetched, and a file already on disk at the right size is skipped, so an interrupted download resumes.

#include <QByteArray>
#include <QCryptographicHash>
#include <QFile>
#include <QHash>
#include <QList>
#include <QNetworkAccessManager>
#include <QObject>
#include <QElapsedTimer>
#include <QTimer>
#include <QString>
#include <QStringList>

#include <memory>

class QNetworkReply;

namespace scene_packs {

struct Member {            // one file inside a remote zip
	QString dest;           // path under the destination root, e.g. "models/sponza_textures/textures/x.png"
	QString name;           // member name inside the zip
	qint64 compressedSize = 0, size = 0;
	quint32 crc32 = 0;
	qint64 headerOffset = 0;
	int method = 0;         // 0 stored, 8 deflate
	int nameLength = 0, extraLength = 0;   // from the central directory; the local header may differ, which the fetch tolerates
};

struct Archive {
	QString url;
	qint64 size = 0;        // the zip's size when the catalogue was made; a different size means upstream changed and the offsets are stale
	QList<Member> members;
};

struct RemoteFile {        // a plain file fetched whole, checked by git blob SHA-1
	QString url, dest;
	qint64 size = 0;
	QString gitSha1;
};

struct Pack {
	QString id, title, credit, sourceUrl;
	QList<Archive> archives;
	QList<RemoteFile> files;

	qint64 downloadBytes() const;   // what crosses the network
	qint64 diskBytes() const;       // what ends up on disk
	int fileCount() const;
	bool providesSceneFiles() const;   // contains .pbrt scenes (a new scene appears only after a restart)
};

struct Catalogue {
	QList<Pack> packs;
	QHash<QString, int> byDest;     // dest path -> index into packs

	// The pack that provides `relativePath` (relative to the application folder), or nullptr.
	const Pack *find(const QString &relativePath) const;
};

Catalogue parseCatalogue(const QString &text);
const Catalogue &builtInCatalogue();   // :/assets/scene_packs.txt

// Downloads one pack, up to kParallel files at a time. Files go to "<dest>.part", are verified (CRC-32 for zip members, git SHA-1 for whole
// files), then renamed into place; a failure or cancel removes the partial file and leaves finished ones, so a retry continues.
class PackDownloader : public QObject {
	Q_OBJECT
public:
	explicit PackDownloader(QObject *parent = nullptr);
	~PackDownloader() override;

	void start(const Pack &pack, const QString &destinationRoot);
	void cancel();
	bool isRunning() const { return m_running; }

signals:
	void progress(qint64 received, qint64 total, const QString &currentFile);
	void finished(bool ok, const QString &error, int filesDone);

private:
	struct Task;
	void pump();
	void startTask(Task *t);
	void onData(Task *t);
	void onTaskFinished(Task *t);
	void failTask(Task *t, const QString &message);
	void finishRun(bool ok, const QString &error, bool notify = true);
	bool consumeZipBytes(Task *t, const QByteArray &chunk, QString &error);
	bool writeOut(Task *t, const char *data, qint64 length, QString &error);

	QNetworkAccessManager m_network;
	QTimer m_stallTimer;
	QElapsedTimer m_activity;
	QList<Task *> m_pending, m_active;
	qint64 m_totalBytes = 0, m_doneBytes = 0;
	int m_filesDone = 0;
	bool m_running = false, m_cancelled = false;
	QString m_firstError;
};

}  // namespace scene_packs
