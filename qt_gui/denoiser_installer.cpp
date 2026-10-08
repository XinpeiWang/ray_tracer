#include "denoiser_installer.h"

#include "app_log.h"
#include "asset_downloader.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QSysInfo>

namespace denoiser_installer {
namespace {

const char *const kRelease = "v2.5.1";
const char *const kBaseUrl = "https://github.com/RenderKit/oidn/releases/download/v2.5.1/";

struct Asset {
	const char *arch;
	const char *file;
	qint64 size;
	const char *sha256;
};
// The release's own archives, pinned: a changed or tampered download is refused.
const Asset kAssets[] = {
	{"arm64", "oidn-2.5.1.arm64.macos.tar.gz", 51382095, "98e0aca8e7ab69e9f4f0191582a500fd8b0d9085d68662f18ab6469e934a6efd"},
	{"x86_64", "oidn-2.5.1.x86_64.macos.tar.gz", 51748390, "a4341639005a33ce0d944bd7c771ce34de1465be7eb3189b7c3b5b0064429832"},
};

const Asset *assetForThisMac() {
	const QString arch = architecture();
	for (const Asset &a : kAssets)
		if (arch == QLatin1String(a.arch)) return &a;
	return nullptr;
}

bool run(const QString &program, const QStringList &args, QString *error) {
	QProcess p;
	p.start(program, args);
	if (!p.waitForFinished(120000) || p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
		if (error) *error = QString::fromLocal8Bit(p.readAllStandardError()).trimmed();
		return false;
	}
	return true;
}

}  // namespace

QString installFolder() {
	const QString root = asset_downloader::userAssetRoot();
	return root.isEmpty() ? QString() : root + QStringLiteral("/denoiser");
}

bool isInstalled() {
	const QString folder = installFolder();
	return !folder.isEmpty() && QFileInfo::exists(folder + QStringLiteral("/lib/libOpenImageDenoise.dylib")) &&
	       QFileInfo::exists(folder + QStringLiteral("/lib/libOpenImageDenoise_core.2.5.1.dylib"));
}

bool isSupportedHere() {
#ifdef Q_OS_MAC
	return assetForThisMac() != nullptr;
#else
	return false;
#endif
}

QString architecture() {
	const QString arch = QSysInfo::currentCpuArchitecture();
	return arch == QLatin1String("aarch64") ? QStringLiteral("arm64") : arch;
}

QString downloadSizeText() {
	const Asset *a = assetForThisMac();
	return QCoreApplication::translate("denoiser_installer", "about %1 MB").arg(a ? (a->size + 500000) / 1000000 : 52);
}

Installer::Installer(QObject *parent) : QObject(parent) {}
Installer::~Installer() = default;

void Installer::start() {
	const Asset *asset = assetForThisMac();
	const QString folder = installFolder();
	if (!asset || folder.isEmpty()) {
		finish(false, tr("This computer cannot install the denoiser here."));
		return;
	}
	QDir().mkpath(folder);
	m_running = true;
	m_manifest = std::make_unique<asset_downloader::Manifest>();
	m_manifest->baseUrl = QString::fromLatin1(kBaseUrl);
	asset_downloader::Entry entry;
	entry.relativePath = QString::fromLatin1(asset->file);
	entry.size = asset->size;
	entry.sha256 = QString::fromLatin1(asset->sha256);
	m_manifest->entries.append(entry);
	m_downloader = std::make_unique<asset_downloader::Downloader>(*m_manifest);
	AppLog::info(QStringLiteral("denoiser"), QStringLiteral("downloading Open Image Denoise %1 (%2) from %3").arg(QLatin1String(kRelease), QLatin1String(asset->arch), m_manifest->urlFor(entry)));
	const QString archive = folder + QStringLiteral("/") + QLatin1String(asset->file);
	connect(m_downloader.get(), &asset_downloader::Downloader::progress, this, [this](qint64 received, qint64 total, const QString &) {
		emit progress(total > 0 ? static_cast<int>(received * 90 / total) : 0, tr("Downloading the denoiser..."));
	});
	connect(m_downloader.get(), &asset_downloader::Downloader::finished, this, [this, archive](bool ok, const QString &error, int) {
		if (!ok) { finish(false, error); return; }
		unpack(archive);
	});
	m_downloader->start({asset_downloader::Job{entry, archive}});
}

void Installer::cancel() {
	if (m_downloader && m_downloader->isRunning()) m_downloader->cancel();
}

// The archive holds one folder (oidn-2.5.1.<arch>.macos) with bin, doc, include and lib: keep lib/ (the loadable libraries) and the licence.
void Installer::unpack(const QString &archive) {
	emit progress(92, tr("Unpacking the denoiser..."));
	const QString folder = installFolder();
	const QString work = folder + QStringLiteral("/.unpack");
	QDir(work).removeRecursively();
	QDir().mkpath(work);
	QString error;
	if (!run(QStringLiteral("/usr/bin/tar"), {QStringLiteral("-xzf"), archive, QStringLiteral("-C"), work}, &error)) {
		QDir(work).removeRecursively();
		finish(false, tr("Could not unpack the download: %1").arg(error));
		return;
	}
	const QStringList dirs = QDir(work).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
	if (dirs.isEmpty() || !QFileInfo::exists(work + QLatin1Char('/') + dirs.first() + QStringLiteral("/lib/libOpenImageDenoise.dylib"))) {
		QDir(work).removeRecursively();
		finish(false, tr("The download did not contain the denoiser library."));
		return;
	}
	const QString inner = work + QLatin1Char('/') + dirs.first();
	QDir(folder + QStringLiteral("/lib")).removeRecursively();
	// cp -R keeps the dylib symlinks (libOpenImageDenoise.dylib -> .2.dylib -> .2.5.1.dylib) that the loader follows.
	if (!run(QStringLiteral("/bin/cp"), {QStringLiteral("-R"), inner + QStringLiteral("/lib"), folder + QStringLiteral("/lib")}, &error)) {
		QDir(work).removeRecursively();
		finish(false, tr("Could not install the denoiser: %1").arg(error));
		return;
	}
	QFile::remove(folder + QStringLiteral("/LICENSE.txt"));
	QFile::copy(inner + QStringLiteral("/doc/LICENSE.txt"), folder + QStringLiteral("/LICENSE.txt"));
	QDir(work).removeRecursively();
	QFile::remove(archive);
	if (!isInstalled()) {
		finish(false, tr("The denoiser files are not where they should be after installing."));
		return;
	}
	finish(true, tr("The denoiser is installed."));
}

void Installer::finish(bool ok, const QString &message) {
	m_running = false;
	AppLog::write(ok ? log_format::Level::Info : log_format::Level::Error, QStringLiteral("denoiser"), QStringLiteral("install %1: %2").arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
	emit progress(100, message);
	emit finished(ok, message);
}

}  // namespace denoiser_installer
