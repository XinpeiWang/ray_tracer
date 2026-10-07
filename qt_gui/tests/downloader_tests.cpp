// downloader_tests.cpp - QtTest for the GUI's two "Download missing files" back ends (asset_downloader.cpp for the files this repository
// hosts, scene_packs.cpp for the third-party scenes), against a small HTTP server inside the test, so nothing touches the network.
//
// They run on every platform the GUI builds on (see qt_gui/tests/downloader_tests.pro and .github/workflows/unit-tests.yml). What they pin
// is what a user would otherwise find out the hard way: a file with the wrong checksum is thrown away and never appears where the renderer
// looks, a finished file is not fetched twice, an upstream archive that changed is refused, and the catalogue the app ships parses.

#include "../asset_downloader.h"
#include "../scene_packs.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QHash>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QtTest>

#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES   // as scene_packs.cpp does: miniz's zlib-style crc32() would rename Member::crc32
#include <miniz.h>   // ../../src/external, for the CRC-32 of the fixtures

namespace {

// A tiny HTTP/1.1 server: GET of a registered path, with "Range: bytes=a-b" support (206 + Content-Range), 404 otherwise.
class TestServer : public QObject {
public:
	TestServer() {
		QObject::connect(&m_server, &QTcpServer::newConnection, [this]() {
			while (QTcpSocket *s = m_server.nextPendingConnection()) {
				QObject::connect(s, &QTcpSocket::readyRead, [this, s]() { handle(s); });
				QObject::connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
			}
		});
		m_listening = m_server.listen(QHostAddress::LocalHost, 0);
	}
	bool listening() const { return m_listening; }
	QString url(const QString &path) const { return QStringLiteral("http://127.0.0.1:%1%2").arg(m_server.serverPort()).arg(path); }
	QString base() const { return url(QStringLiteral("/")); }
	void add(const QString &path, const QByteArray &body) { m_files.insert(path, body); }
	int requests(const QString &path) const { return m_requests.value(path, 0); }
	int totalRequests() const { int n = 0; for (int v : m_requests) n += v; return n; }

private:
	void handle(QTcpSocket *s) {
		m_buffer[s].append(s->readAll());
		if (!m_buffer[s].contains("\r\n\r\n")) return;
		const QByteArray request = m_buffer.take(s);
		const QList<QByteArray> lines = request.split('\n');
		const QList<QByteArray> first = lines.value(0).trimmed().split(' ');
		const QString path = QString::fromLatin1(first.value(1));
		++m_requests[path];
		QByteArray range;
		for (const QByteArray &l : lines) if (l.toLower().startsWith("range:")) range = l.mid(6).trimmed();
		if (!m_files.contains(path)) {
			s->write("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
			s->disconnectFromHost();
			return;
		}
		const QByteArray &body = m_files[path];
		QByteArray head, out;
		if (range.startsWith("bytes=")) {
			const QList<QByteArray> ab = range.mid(6).split('-');
			const qint64 a = ab.value(0).toLongLong();
			const qint64 b = qMin<qint64>(ab.value(1).toLongLong(), body.size() - 1);
			out = body.mid(a, b - a + 1);
			head = "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes " + QByteArray::number(a) + "-" + QByteArray::number(b) + "/" +
				   QByteArray::number(body.size()) + "\r\n";
		} else {
			out = body;
			head = "HTTP/1.1 200 OK\r\n";
		}
		head += "Content-Length: " + QByteArray::number(out.size()) + "\r\nConnection: close\r\n\r\n";
		s->write(head + out);
		s->disconnectFromHost();
	}
	QTcpServer m_server;
	bool m_listening = false;
	QHash<QString, QByteArray> m_files;
	QHash<QString, int> m_requests;
	QHash<QTcpSocket *, QByteArray> m_buffer;
};

QByteArray sha256Hex(const QByteArray &d) { return QCryptographicHash::hash(d, QCryptographicHash::Sha256).toHex(); }
QByteArray gitBlobSha1(const QByteArray &d) {
	return QCryptographicHash::hash("blob " + QByteArray::number(d.size()) + '\0' + d, QCryptographicHash::Sha1).toHex();
}
quint32 crc32Of(const QByteArray &d) {
	return static_cast<quint32>(mz_crc32(MZ_CRC32_INIT, reinterpret_cast<const mz_uint8 *>(d.constData()), static_cast<size_t>(d.size())));
}
QByteArray le(quint32 v, int bytes) { QByteArray r; for (int i = 0; i < bytes; ++i) r.append(char((v >> (8 * i)) & 0xff)); return r; }

// Some compressible bytes, so the deflate path has real work.
QByteArray payload(int seed, int size) {
	QByteArray d;
	const QByteArray unit = "ray tracer fixture " + QByteArray::number(seed) + " - the quick brown fox jumps over the lazy dog\n";
	while (d.size() < size) d.append(unit);
	return d.left(size);
}

// A zip "archive" built from members: each is a local file header + name + data, which is all the downloader reads (it never opens the
// central directory; the catalogue carries the offsets).
struct ZipFixture {
	QByteArray bytes;
	scene_packs::Archive archive;

	void addMember(const QString &dest, const QString &name, const QByteArray &content, bool deflate, quint32 crcOverride = 0) {
		QByteArray stored = content;
		if (deflate) {
			const QByteArray z = qCompress(content, 9);   // 4-byte length, zlib header (2), raw deflate, adler32 (4)
			stored = z.mid(6, z.size() - 10);
		}
		const quint32 crc = crcOverride ? crcOverride : crc32Of(content);
		scene_packs::Member m;
		m.dest = dest; m.name = name;
		m.compressedSize = stored.size(); m.size = content.size(); m.crc32 = crc;
		m.headerOffset = bytes.size();
		m.method = deflate ? 8 : 0;
		m.nameLength = name.toUtf8().size(); m.extraLength = 0;
		bytes += le(0x04034b50, 4) + le(20, 2) + le(0, 2) + le(m.method, 2) + le(0, 2) + le(0, 2) + le(crc, 4) + le(stored.size(), 4) +
				 le(content.size(), 4) + le(m.nameLength, 2) + le(0, 2) + name.toUtf8() + stored;
		archive.members.append(m);
	}
};

QByteArray readAll(const QString &path) { QFile f(path); return f.open(QIODevice::ReadOnly) ? f.readAll() : QByteArray(); }

}  // namespace

class DownloaderTests : public QObject {
	Q_OBJECT
private slots:
	void manifestParsingKeepsOnlyWellFormedEntries();
	void catalogueParsingRejectsUnsafeRecords();
	void shippedCatalogueAndManifestParse();

	void assetDownloaderSavesAVerifiedFile();
	void assetDownloaderDiscardsAFileWithTheWrongChecksum();
	void assetDownloaderSkipsAFileAlreadyOnDisk();
	void assetDownloaderReportsAMissingFile();

	void packDownloaderFetchesStoredDeflatedAndWholeFiles();
	void packDownloaderDiscardsAMemberWithTheWrongCrc();
	void packDownloaderRefusesAnArchiveThatChangedUpstream();
	void packDownloaderDiscardsAFileWithTheWrongGitHash();
	void packDownloaderResumesBySkippingFinishedFiles();
	void packDownloaderCancelLeavesNoPartialFile();
};

void DownloaderTests::manifestParsingKeepsOnlyWellFormedEntries() {
	const QString good = QString::fromLatin1(sha256Hex("x"));
	const QString text = "# comment\nbase http://example.invalid/prefix/\n"
						 "models/a.obj 10 " + good + "\n"
						 "models/zero.obj 0 " + good + "\n"            // size must be positive
						 "models/short.obj 10 abc\n"                    // not a SHA-256
						 "models/../escape.obj 10 " + good + "\n"      // would climb out of the folder
						 "models/two words 10 " + good + " extra\n"    // wrong field count
						 "models/b.obj 20 " + good.toUpper() + "\n";    // upper-case hex is accepted and normalised
	const asset_downloader::Manifest m = asset_downloader::parseManifest(text);
	QCOMPARE(m.baseUrl, QStringLiteral("http://example.invalid/prefix/"));
	QCOMPARE(m.entries.size(), 2);
	QCOMPARE(m.entries[0].relativePath, QStringLiteral("models/a.obj"));
	QCOMPARE(m.entries[1].sha256, good);
	QCOMPARE(m.urlFor(m.entries[0]), QStringLiteral("http://example.invalid/prefix/models/a.obj"));
	QVERIFY(m.find(QStringLiteral("/app"), QStringLiteral("/app/models/a.obj")));
	QVERIFY(!m.find(QStringLiteral("/app"), QStringLiteral("/app/models/other.obj")));
	// No base line: nothing may be downloaded at all.
	QVERIFY(asset_downloader::parseManifest(QStringLiteral("models/a.obj 10 ") + good).entries.isEmpty());
}

void DownloaderTests::catalogueParsingRejectsUnsafeRecords() {
	const QString sha = QString(40, QLatin1Char('a'));
	const QString text =
		"member|orphan|o|1|1|1|0|0|1|0\n"                   // before any pack or archive: ignored
		"pack|p1|Pack One|Credit text|https://example.invalid/p1\n"
		"archive|https://example.invalid/a.zip|1000\n"
		"member|models/x.obj|x.obj|10|20|DEADBEEF|0|8|5|0\n"
		"member|models/../evil.obj|e|10|20|1|0|8|5|0\n"      // climbs out
		"member|models/bad.obj|b|10|20|1|0|9|5|0\n"          // unsupported compression method
		"file|https://example.invalid/s.pbrt|pbrt_scenes/s.pbrt|5|" + sha + "\n"
		"file|https://example.invalid/t.png|pbrt_scenes/t.png|5|tooshort\n"   // not a git SHA-1
		"file|https://example.invalid/u.png|../u.png|5|" + sha + "\n"         // climbs out
		"pack|p2|Pack Two|c|u\n";
	const scene_packs::Catalogue c = scene_packs::parseCatalogue(text);
	QCOMPARE(c.packs.size(), 2);
	const scene_packs::Pack &p = c.packs[0];
	QCOMPARE(p.archives.size(), 1);
	QCOMPARE(p.archives[0].members.size(), 1);
	QCOMPARE(p.archives[0].members[0].crc32, quint32(0xDEADBEEF));
	QCOMPARE(p.files.size(), 1);
	QCOMPARE(p.fileCount(), 2);
	QCOMPARE(p.downloadBytes(), qint64(10 + 5));
	QCOMPARE(p.diskBytes(), qint64(20 + 5));
	QVERIFY(p.providesSceneFiles());
	QVERIFY(c.find(QStringLiteral("models/x.obj")));
	QVERIFY(c.find(QStringLiteral("models//x.obj")));   // cleaned
	QVERIFY(!c.find(QStringLiteral("models/evil.obj")));
	QVERIFY(c.packs[1].archives.isEmpty() && c.packs[1].files.isEmpty());
}

void DownloaderTests::shippedCatalogueAndManifestParse() {
	// The catalogue is generated (scripts/gen_scene_packs.py) and compiled in: a syntax slip there would silently remove every pack.
	const scene_packs::Catalogue &c = scene_packs::builtInCatalogue();
	QVERIFY2(c.packs.size() >= 20, "the shipped scene catalogue lost packs");
	for (const scene_packs::Pack &p : c.packs) {
		QVERIFY2(p.fileCount() > 0, qPrintable(p.id + " has no files"));
		QVERIFY2(!p.credit.isEmpty(), qPrintable(p.id + " has no credit/licence text"));
		for (const scene_packs::RemoteFile &f : p.files) QVERIFY2(f.url.startsWith(QLatin1String("https://")), qPrintable(f.url));
		for (const scene_packs::Archive &a : p.archives) {
			QVERIFY2(a.url.startsWith(QLatin1String("https://")), qPrintable(a.url));
			QVERIFY(a.size > 0);
		}
	}
	QVERIFY(c.find(QStringLiteral("pbrt_scenes/ganesha/ganesha.pbrt")));
	const asset_downloader::Manifest &m = asset_downloader::builtInManifest();
	QVERIFY2(!m.entries.isEmpty() && m.baseUrl.startsWith(QLatin1String("https://")), "the shipped asset manifest is empty");
}

void DownloaderTests::assetDownloaderSavesAVerifiedFile() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray data = payload(1, 50000);
	server.add("/models/a.obj", data);
	asset_downloader::Manifest m; m.baseUrl = server.base();
	asset_downloader::Entry e; e.relativePath = "models/a.obj"; e.size = data.size(); e.sha256 = QString::fromLatin1(sha256Hex(data));
	m.entries.append(e);
	QTemporaryDir dir; QVERIFY(dir.isValid());
	asset_downloader::Downloader d(m);
	QSignalSpy done(&d, &asset_downloader::Downloader::finished);
	const QString dest = dir.filePath("models/a.obj");
	d.start({{e, dest}});
	QVERIFY(done.wait(15000));
	QVERIFY2(done[0][0].toBool(), qPrintable(done[0][1].toString()));
	QCOMPARE(readAll(dest), data);
	QVERIFY(!QFile::exists(dest + ".part"));
}

void DownloaderTests::assetDownloaderDiscardsAFileWithTheWrongChecksum() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray data = payload(2, 20000);
	server.add("/models/b.obj", payload(3, 20000));   // same size, different bytes: what a tampered or changed upstream file looks like
	asset_downloader::Manifest m; m.baseUrl = server.base();
	asset_downloader::Entry e; e.relativePath = "models/b.obj"; e.size = data.size(); e.sha256 = QString::fromLatin1(sha256Hex(data));
	QTemporaryDir dir;
	asset_downloader::Downloader d(m);
	QSignalSpy done(&d, &asset_downloader::Downloader::finished);
	const QString dest = dir.filePath("models/b.obj");
	d.start({{e, dest}});
	QVERIFY(done.wait(15000));
	QVERIFY(!done[0][0].toBool());
	QVERIFY2(!QFile::exists(dest) && !QFile::exists(dest + ".part"), "a bad file must not be left where the renderer would find it");
}

void DownloaderTests::assetDownloaderSkipsAFileAlreadyOnDisk() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray data = payload(4, 10000);
	server.add("/models/c.obj", data);
	asset_downloader::Manifest m; m.baseUrl = server.base();
	asset_downloader::Entry e; e.relativePath = "models/c.obj"; e.size = data.size(); e.sha256 = QString::fromLatin1(sha256Hex(data));
	QTemporaryDir dir;
	const QString dest = dir.filePath("models/c.obj");
	QDir().mkpath(QFileInfo(dest).absolutePath());
	{ QFile f(dest); QVERIFY(f.open(QIODevice::WriteOnly)); f.write(data); }
	asset_downloader::Downloader d(m);
	QSignalSpy done(&d, &asset_downloader::Downloader::finished);
	d.start({{e, dest}});
	if (done.isEmpty()) QVERIFY(done.wait(15000));   // with nothing to fetch, finished() is emitted from inside start()
	QCOMPARE(done.size(), 1);
	QVERIFY(done[0][0].toBool());
	QCOMPARE(server.totalRequests(), 0);
}

void DownloaderTests::assetDownloaderReportsAMissingFile() {
	TestServer server; QVERIFY(server.listening());
	asset_downloader::Manifest m; m.baseUrl = server.base();
	asset_downloader::Entry e; e.relativePath = "models/gone.obj"; e.size = 10; e.sha256 = QString::fromLatin1(sha256Hex("x"));
	QTemporaryDir dir;
	asset_downloader::Downloader d(m);
	QSignalSpy done(&d, &asset_downloader::Downloader::finished);
	const QString dest = dir.filePath("models/gone.obj");
	d.start({{e, dest}});
	QVERIFY(done.wait(15000));
	QVERIFY(!done[0][0].toBool());
	QVERIFY(!done[0][1].toString().isEmpty());
	QVERIFY(!QFile::exists(dest) && !QFile::exists(dest + ".part"));
}

void DownloaderTests::packDownloaderFetchesStoredDeflatedAndWholeFiles() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray stored = payload(10, 3000), deflated = payload(11, 200000), whole = payload(12, 7000);
	ZipFixture zip;
	zip.addMember("models/pack_textures/stored.png", "stored.png", stored, false);
	zip.addMember("models/pack_textures/sub/deflated.png", "sub/deflated.png", deflated, true);
	zip.archive.url = server.url("/archive.zip");
	zip.archive.size = zip.bytes.size();
	server.add("/archive.zip", zip.bytes);
	server.add("/whole.pbrt", whole);

	scene_packs::Pack pack; pack.id = "t"; pack.title = "T";
	pack.archives.append(zip.archive);
	scene_packs::RemoteFile rf; rf.url = server.url("/whole.pbrt"); rf.dest = "pbrt_scenes/t/whole.pbrt"; rf.size = whole.size();
	rf.gitSha1 = QString::fromLatin1(gitBlobSha1(whole));
	pack.files.append(rf);

	QTemporaryDir dir; QVERIFY(dir.isValid());
	scene_packs::PackDownloader d;
	QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
	d.start(pack, dir.path());
	QVERIFY(done.wait(30000));
	QVERIFY2(done[0][0].toBool(), qPrintable(done[0][1].toString()));
	QCOMPARE(done[0][2].toInt(), 3);
	QCOMPARE(readAll(dir.filePath("models/pack_textures/stored.png")), stored);
	QCOMPARE(readAll(dir.filePath("models/pack_textures/sub/deflated.png")), deflated);
	QCOMPARE(readAll(dir.filePath("pbrt_scenes/t/whole.pbrt")), whole);
	QDir root(dir.path());
	QVERIFY2(root.entryList(QStringList() << "*.part", QDir::Files | QDir::NoDotAndDotDot).isEmpty(), "no .part files may remain");
}

void DownloaderTests::packDownloaderDiscardsAMemberWithTheWrongCrc() {
	TestServer server; QVERIFY(server.listening());
	ZipFixture zip;
	zip.addMember("models/p/ok.png", "ok.png", payload(20, 4000), false);
	zip.addMember("models/p/bad.png", "bad.png", payload(21, 4000), true, /*crcOverride=*/0x12345678);   // the catalogue's CRC disagrees with the bytes
	zip.archive.url = server.url("/a.zip"); zip.archive.size = zip.bytes.size();
	server.add("/a.zip", zip.bytes);
	scene_packs::Pack pack; pack.archives.append(zip.archive);
	QTemporaryDir dir;
	scene_packs::PackDownloader d;
	QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
	d.start(pack, dir.path());
	QVERIFY(done.wait(30000));
	QVERIFY(!done[0][0].toBool());
	QVERIFY(done[0][1].toString().contains("bad.png"));
	QVERIFY2(!QFile::exists(dir.filePath("models/p/bad.png")) && !QFile::exists(dir.filePath("models/p/bad.png.part")),
			 "a file that failed its checksum must not be left behind");
}

void DownloaderTests::packDownloaderRefusesAnArchiveThatChangedUpstream() {
	TestServer server; QVERIFY(server.listening());
	ZipFixture zip;
	zip.addMember("models/q/x.png", "x.png", payload(30, 2000), false);
	zip.archive.url = server.url("/a.zip");
	zip.archive.size = zip.bytes.size() + 1000;   // the catalogue remembers a different size than the server now reports
	server.add("/a.zip", zip.bytes);
	scene_packs::Pack pack; pack.archives.append(zip.archive);
	QTemporaryDir dir;
	scene_packs::PackDownloader d;
	QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
	d.start(pack, dir.path());
	QVERIFY(done.wait(30000));
	QVERIFY(!done[0][0].toBool());
	QVERIFY(!QFile::exists(dir.filePath("models/q/x.png")));
}

void DownloaderTests::packDownloaderDiscardsAFileWithTheWrongGitHash() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray expected = payload(40, 5000);
	server.add("/f.pbrt", payload(41, 5000));
	scene_packs::Pack pack;
	scene_packs::RemoteFile rf; rf.url = server.url("/f.pbrt"); rf.dest = "pbrt_scenes/f.pbrt"; rf.size = expected.size();
	rf.gitSha1 = QString::fromLatin1(gitBlobSha1(expected));
	pack.files.append(rf);
	QTemporaryDir dir;
	scene_packs::PackDownloader d;
	QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
	d.start(pack, dir.path());
	QVERIFY(done.wait(30000));
	QVERIFY(!done[0][0].toBool());
	QVERIFY(!QFile::exists(dir.filePath("pbrt_scenes/f.pbrt")) && !QFile::exists(dir.filePath("pbrt_scenes/f.pbrt.part")));
}

void DownloaderTests::packDownloaderResumesBySkippingFinishedFiles() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray a = payload(50, 6000), b = payload(51, 6000);
	server.add("/a", a); server.add("/b", b);
	scene_packs::Pack pack;
	for (const auto &item : {std::make_pair(QString("/a"), a), std::make_pair(QString("/b"), b)}) {
		scene_packs::RemoteFile rf; rf.url = server.url(item.first); rf.dest = "pbrt_scenes/r" + item.first + ".pbrt"; rf.size = item.second.size();
		rf.gitSha1 = QString::fromLatin1(gitBlobSha1(item.second));
		pack.files.append(rf);
	}
	QTemporaryDir dir;
	{
		scene_packs::PackDownloader d;
		QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
		d.start(pack, dir.path());
		QVERIFY(done.wait(30000));
		QVERIFY(done[0][0].toBool());
	}
	QCOMPARE(server.totalRequests(), 2);
	QVERIFY(QFile::remove(dir.filePath("pbrt_scenes/r/b.pbrt")));   // the user deleted one; the other is still complete
	scene_packs::PackDownloader again;
	QSignalSpy done2(&again, &scene_packs::PackDownloader::finished);
	again.start(pack, dir.path());
	QVERIFY(done2.wait(30000));
	QVERIFY(done2[0][0].toBool());
	QCOMPARE(server.totalRequests(), 3);   // only the missing file was fetched again
	QCOMPARE(readAll(dir.filePath("pbrt_scenes/r/b.pbrt")), b);
}

void DownloaderTests::packDownloaderCancelLeavesNoPartialFile() {
	TestServer server; QVERIFY(server.listening());
	const QByteArray data = payload(60, 300000);
	server.add("/big", data);
	scene_packs::Pack pack;
	scene_packs::RemoteFile rf; rf.url = server.url("/big"); rf.dest = "pbrt_scenes/big.pbrt"; rf.size = data.size();
	rf.gitSha1 = QString::fromLatin1(gitBlobSha1(data));
	pack.files.append(rf);
	QTemporaryDir dir;
	scene_packs::PackDownloader d;
	QSignalSpy done(&d, &scene_packs::PackDownloader::finished);
	d.start(pack, dir.path());
	d.cancel();
	QVERIFY(!d.isRunning());
	QCOMPARE(done.size(), 1);
	QVERIFY(!done[0][0].toBool());
	QTest::qWait(300);   // let any late reply signal arrive: it must not recreate or complete anything
	QVERIFY(!QFile::exists(dir.filePath("pbrt_scenes/big.pbrt")) && !QFile::exists(dir.filePath("pbrt_scenes/big.pbrt.part")));
}

QTEST_GUILESS_MAIN(DownloaderTests)
#include "downloader_tests.moc"
