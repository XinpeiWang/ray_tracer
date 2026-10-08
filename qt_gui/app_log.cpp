#include "app_log.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLocale>
#include <QMutex>
#include <QMutexLocker>
#include <QSettings>
#include <QStandardPaths>
#include <QSysInfo>
#include <QThread>
#include <QtGlobal>

#ifdef Q_OS_WIN
#include <io.h>
#include <stdio.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace AppLog {
namespace {

constexpr qint64 kMaxBytes = 5 * 1024 * 1024;
constexpr int kKeepOld = 3;

QMutex g_mutex;
QFile g_file;
QString g_path;
QString g_folder;
bool g_unclean = false;
bool g_inited = false;
bool g_stderrRedirected = false;
int g_writesSinceRotateCheck = 0;
QtMessageHandler g_previousHandler = nullptr;

QString markerPath() { return g_folder + QStringLiteral("/.session_running"); }

QString pickFolder() {
	const QString env = qEnvironmentVariable("RAY_TRACER_LOG_DIR");
	if (!env.isEmpty()) return env;
#ifdef Q_OS_MAC
	return QDir::homePath() + QStringLiteral("/Library/Logs/Ray Tracer");
#else
	return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/logs");
#endif
}

bool processAlive(qint64 pid) {
#ifdef Q_OS_WIN
	Q_UNUSED(pid);
	return false;   // no cheap check; treat a leftover marker as a stale one
#else
	return pid > 0 && (::kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM);
#endif
}

// Caller holds the mutex.
void writeLocked(log_format::Level level, const QString &category, const QString &message) {
	if (!g_file.isOpen()) return;
	if (++g_writesSinceRotateCheck >= 200) {   // a long session: keep the file bounded without checking the size on every line
		g_writesSinceRotateCheck = 0;
		if (g_file.size() > kMaxBytes) {
			g_file.close();
			log_format::rotateIfLarge(g_path.toStdString(), static_cast<std::uintmax_t>(kMaxBytes), kKeepOld);
			g_file.open(QIODevice::WriteOnly | QIODevice::Append);
		}
	}
	const std::string line = log_format::formatLine(QDateTime::currentDateTime().toString(QStringLiteral("yyyy-MM-dd HH:mm:ss.zzz")).toStdString(), level,
	                                                category.toStdString(), message.toStdString());
	g_file.write(line.c_str(), static_cast<qint64>(line.size()));
	g_file.write("\n", 1);
	g_file.flush();
}

void qtMessageHandler(QtMsgType type, const QMessageLogContext &context, const QString &message) {
	log_format::Level level = log_format::Level::Info;
	switch (type) {
		case QtDebugMsg: level = log_format::Level::Debug; break;
		case QtInfoMsg: level = log_format::Level::Info; break;
		case QtWarningMsg: level = log_format::Level::Warn; break;
		case QtCriticalMsg:
		case QtFatalMsg: level = log_format::Level::Error; break;
	}
	QString where = context.category && qstrcmp(context.category, "default") != 0 ? QString::fromLatin1(context.category) : QStringLiteral("qt");
	if (type == QtDebugMsg && where == QLatin1String("qt")) where = QStringLiteral("qt-debug");
	write(level, where, message);
	// Once stderr goes to the file the default handler would write the same line a second time; from a terminal it is still wanted there.
	if (g_previousHandler && !g_stderrRedirected) g_previousHandler(type, context, message);
}

void redirectStderr() {
#ifdef Q_OS_WIN
	FILE *f = nullptr;
	if (_wfreopen_s(&f, reinterpret_cast<const wchar_t *>(g_path.utf16()), L"a", stderr) == 0 && f) g_stderrRedirected = true;
#else
	if (::isatty(2)) return;   // started from a terminal: leave stderr there
	const int fd = ::open(QFile::encodeName(g_path).constData(), O_WRONLY | O_APPEND);
	if (fd >= 0 && ::dup2(fd, 2) >= 0) g_stderrRedirected = true;
	if (fd >= 0) ::close(fd);
#endif
}

}  // namespace

void init() {
	QMutexLocker lock(&g_mutex);
	if (g_inited) return;
	g_inited = true;
	g_folder = pickFolder();
	if (!QDir().mkpath(g_folder)) { g_folder.clear(); return; }
	g_path = g_folder + QStringLiteral("/ray_tracer_gui.log");
	log_format::rotateIfLarge(g_path.toStdString(), static_cast<std::uintmax_t>(kMaxBytes), kKeepOld);

	// A marker left by a session that never reached shutdown(): it crashed, was killed, or lost power.
	QFile marker(markerPath());
	qint64 oldPid = 0;
	if (marker.exists() && marker.open(QIODevice::ReadOnly)) {
		oldPid = marker.readAll().trimmed().toLongLong();
		marker.close();
		g_unclean = !processAlive(oldPid) || oldPid == QCoreApplication::applicationPid();
	}
	if (marker.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		marker.write(QByteArray::number(QCoreApplication::applicationPid()));
		marker.close();
	}

	g_file.setFileName(g_path);
	if (!g_file.open(QIODevice::WriteOnly | QIODevice::Append)) { g_folder.clear(); g_path.clear(); return; }

	const QString sep = QStringLiteral("=======================================================================");
	writeLocked(log_format::Level::Info, QStringLiteral("session"), sep);
	writeLocked(log_format::Level::Info, QStringLiteral("session"),
	            QStringLiteral("started: %1 %2 (pid %3)").arg(QCoreApplication::applicationName(), QCoreApplication::applicationVersion()).arg(QCoreApplication::applicationPid()));
	writeLocked(log_format::Level::Info, QStringLiteral("session"),
	            QStringLiteral("system: %1, kernel %2, %3, %4 logical CPUs, locale %5")
	                .arg(QSysInfo::prettyProductName(), QSysInfo::kernelVersion(), QSysInfo::currentCpuArchitecture())
	                .arg(QThread::idealThreadCount())
	                .arg(QLocale::system().name()));
	writeLocked(log_format::Level::Info, QStringLiteral("session"),
	            QStringLiteral("qt: built with %1, running with %2").arg(QStringLiteral(QT_VERSION_STR), QString::fromLatin1(qVersion())));
	writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("program: %1").arg(QCoreApplication::applicationFilePath()));
	writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("working folder: %1").arg(QDir::currentPath()));
	writeLocked(log_format::Level::Info, QStringLiteral("session"),
	            QStringLiteral("user assets: %1; scenes folder override: %2")
	                .arg(qEnvironmentVariable("RAY_TRACER_USER_ASSETS", QStringLiteral("(not set)")), qEnvironmentVariable("RAY_TRACER_PBRT_DIR", QStringLiteral("(not set)"))));
	writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("settings file: %1").arg(QSettings().fileName()));
	writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("log file: %1").arg(g_path));
	if (g_unclean)
		writeLocked(log_format::Level::Warn, QStringLiteral("session"),
		            QStringLiteral("the previous session (pid %1) did not exit cleanly: it crashed or was killed. The lines above this session's header show its last actions.%2")
		                .arg(oldPid)
#ifdef Q_OS_MAC
		                .arg(QStringLiteral(" A crash report may be in ~/Library/Logs/DiagnosticReports.")));
#else
		                .arg(QString()));
#endif

	g_previousHandler = qInstallMessageHandler(qtMessageHandler);
	redirectStderr();
	if (g_stderrRedirected) writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("stderr of this process (renderer libraries included) is written to this file"));
	if (QCoreApplication::instance()) QObject::connect(QCoreApplication::instance(), &QCoreApplication::aboutToQuit, []() { shutdown(); });
}

void write(log_format::Level level, const QString &category, const QString &message) {
	QMutexLocker lock(&g_mutex);
	writeLocked(level, category, message);
}

QString filePath() { return g_path; }
QString folderPath() { return g_folder; }
bool previousSessionEndedUnexpectedly() { return g_unclean; }

QStringList tail(int lines) {
	QStringList out;
	QMutexLocker lock(&g_mutex);
	if (g_path.isEmpty()) return out;
	for (const std::string &line : log_format::tailLines(g_path.toStdString(), static_cast<std::size_t>(lines))) out << QString::fromStdString(line);
	return out;
}

void shutdown() {
	QMutexLocker lock(&g_mutex);
	if (g_folder.isEmpty() || !g_file.isOpen()) return;
	writeLocked(log_format::Level::Info, QStringLiteral("session"), QStringLiteral("ended normally"));
	g_file.close();
	QFile::remove(markerPath());
}

}  // namespace AppLog
