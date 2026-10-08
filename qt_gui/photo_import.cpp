#include "photo_import.h"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QStandardPaths>
#include <QStringList>
#include <QTimer>

namespace {

// <relative path> in the program's own folder or the one above it (three above for a macOS .app, which keeps its files beside the bundle), else "".
QString nextToProgram(const QString &relative) {
	const QString app = QCoreApplication::applicationDirPath();
	QStringList up{"", "../"};
	if (app.contains(".app/Contents/")) up << "../../../";
	for (const QString &u : up) {
		const QString candidate = app + "/" + u + relative;
		if (QFileInfo::exists(candidate)) return QDir::cleanPath(candidate);
	}
	return QString();
}

// A process whose output (stdout and stderr) arrives together; `python` also makes Python's output UTF-8 and unbuffered so progress is live.
QProcess *newProcess(QObject *parent, bool python) {
	auto *p = new QProcess(parent);
	p->setProcessChannelMode(QProcess::MergedChannels);
	if (python) {
		QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
		env.insert("PYTHONIOENCODING", "utf-8");
		env.insert("PYTHONUNBUFFERED", "1");
		p->setProcessEnvironment(env);
	}
	return p;
}

std::string toBytes(const QByteArray &b) { return std::string(b.constData(), static_cast<size_t>(b.size())); }

}  // namespace

namespace photo_import {

QString environmentFolder() {
	// A short path on purpose: PyTorch and transformers have deep folders, and Windows stops at 260 characters.
	return QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/RayTracerPhoto";
}

QString resultsFolder() {
	return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/photo_meshes";
}

QString resultsSummary() {
	const QDir root(resultsFolder());
	if (!root.exists()) return QObject::tr("none yet");
	const int folders = root.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size();
	qint64 bytes = 0;
	QDirIterator it(root.absolutePath(), QDir::Files, QDirIterator::Subdirectories);
	while (it.hasNext()) bytes += QFileInfo(it.next()).size();
	if (folders == 0) return QObject::tr("none yet");
	return QObject::tr("%1 folders, %2 MB").arg(folders).arg(bytes / 1000000);
}

static QString environmentPython() {
#ifdef Q_OS_WIN
	return environmentFolder() + "/venv/Scripts/python.exe";
#else
	return environmentFolder() + "/venv/bin/python";
#endif
}

Setup locate() {
	Setup s;
	s.script = nextToProgram("tools/photo_to_mesh/photo_to_mesh.py");
	const QString overridePython = qEnvironmentVariable("RAY_TRACER_PHOTO3D_PYTHON");
	s.python = overridePython.isEmpty() ? environmentPython() : overridePython;
	if (s.script.isEmpty())
		s.why = QObject::tr("The helper script (tools/photo_to_mesh/photo_to_mesh.py) was not found next to the program.");
	else if (!QFileInfo::exists(s.python))
		s.why = QObject::tr("The photo helper has not been set up on this computer yet.");
	else
		s.ready = true;
	return s;
}

QString setupScript() {
	const QString overridePath = qEnvironmentVariable("RAY_TRACER_PHOTO3D_SETUP");
	if (!overridePath.isEmpty() && qEnvironmentVariableIsSet("RT_GUI_SELFTEST")) return QFileInfo::exists(overridePath) ? overridePath : QString();
	return nextToProgram("scripts/setup_photo_to_mesh.ps1");
}

QStringList missingFacts(const QString &report) {
	QStringList out;
	for (const std::string &fact : photo_report::missingFacts(report.toStdString())) out << QString::fromStdString(fact);
	return out;
}

}  // namespace photo_import

// ---------------------------------------------------------------------------------------------------------------------------------
// PhotoHelperInstaller
// ---------------------------------------------------------------------------------------------------------------------------------
PhotoHelperInstaller::~PhotoHelperInstaller() {
	if (m_process) {
		m_process->disconnect(this);
		killTree();
		m_process->waitForFinished(3000);
	}
}

void PhotoHelperInstaller::start() {
	if (m_process) return;
	const QString script = photo_import::setupScript();
	if (script.isEmpty()) {
		m_done = true;
		QTimer::singleShot(0, this, [this]() { emit finished(false, tr("The installer script (scripts/setup_photo_to_mesh.ps1) was not found next to the program.")); });
		return;
	}
	m_process = newProcess(this, false);
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() { onOutput(); });
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		if (m_done) return;
		m_done = true;
		onOutput();
		const QString rest = QString::fromLocal8Bit(m_splitter.flush().c_str());
		if (!rest.isEmpty()) emit line(rest);
		const bool ok = st == QProcess::NormalExit && code == 0 && !m_cancelled;
		QString message;
		if (m_cancelled) message = tr("Cancelled.");
		else if (!ok) message = m_tail.isEmpty() ? tr("The installer stopped (exit code %1).").arg(code) : m_tail.join('\n');
		QProcess *p = m_process;
		m_process = nullptr;
		p->deleteLater();
		emit finished(ok, message);
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e != QProcess::FailedToStart || m_done) return;
		m_done = true;
		QProcess *p = m_process;
		m_process = nullptr;
		p->deleteLater();
		emit finished(false, tr("Could not start PowerShell to run the installer."));
	});
	m_process->start("powershell.exe", {"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script});
}

void PhotoHelperInstaller::cancel() {
	m_cancelled = true;
	killTree();
}

void PhotoHelperInstaller::killTree() {
	if (!m_process) return;
#ifdef Q_OS_WIN
	// PowerShell started pip, git and python: stop the whole tree, not just the shell.
	if (m_process->processId() > 0) QProcess::execute("taskkill", {"/PID", QString::number(m_process->processId()), "/T", "/F"});
#endif
	m_process->kill();
}

void PhotoHelperInstaller::onOutput() {
	if (!m_process) return;
	for (const auto &piece : m_splitter.feed(toBytes(m_process->readAllStandardOutput()))) {
		const QString text = QString::fromLocal8Bit(piece.text.c_str());
		if (piece.endsLine) {
			emit line(text);
			m_tail << text;
			while (m_tail.size() > 8) m_tail.removeFirst();
		}
		emit status(text);
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------
// PhotoHelperCheck
// ---------------------------------------------------------------------------------------------------------------------------------
PhotoHelperCheck::PhotoHelperCheck(QObject *parent) : QObject(parent) {
	m_timeout = new QTimer(this);
	m_timeout->setSingleShot(true);
	m_timeout->setInterval(60000);  // importing PyTorch takes a few seconds; a hung or broken install must not hold the Diagnostics button forever
	connect(m_timeout, &QTimer::timeout, this, [this]() {
		if (!m_running) return;
		dropProcess();
		finish(QStringLiteral("Python Check: not usable (no answer after 60 seconds)\n"));
	});
}

PhotoHelperCheck::~PhotoHelperCheck() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
}

void PhotoHelperCheck::dropProcess() {
	if (!m_process) return;
	m_process->disconnect(this);
	m_process->kill();
	m_process->deleteLater();
	m_process = nullptr;
}

void PhotoHelperCheck::start() {
	if (m_running) return;
	m_running = true;
	dropProcess();
	const photo_import::Setup setup = photo_import::locate();
	m_head = QStringLiteral("=== Photo helper (Scene Builder, Add > Object from a photo; optional) ===\n");
	m_head += setup.script.isEmpty() ? QStringLiteral("Helper Script: missing (tools/photo_to_mesh/photo_to_mesh.py was not found next to the program)\n")
	                                 : QStringLiteral("Helper Script: present (%1)\n").arg(QDir::toNativeSeparators(setup.script));
	if (!QFileInfo::exists(setup.python)) {
		m_head += QStringLiteral("Python Environment: not available (not set up; run scripts\\setup_photo_to_mesh.ps1 once to enable the feature, about 5 GB)\n");
		m_head += QStringLiteral("Expected At: %1\n").arg(QDir::toNativeSeparators(setup.python));
		finish(QString());
		return;
	}
	m_head += QStringLiteral("Python Environment: present (%1)\n").arg(QDir::toNativeSeparators(setup.python));
	if (setup.script.isEmpty()) {
		finish(QString());
		return;
	}
	m_process = newProcess(this, true);
	QProcess *process = m_process;
	connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process](int code, QProcess::ExitStatus st) {
		if (process != m_process) return;
		const QString out = QString::fromUtf8(process->readAll());
		if (st == QProcess::NormalExit && code == 0) {
			finish(out);
		} else {
			// A broken install: say it as a fact, with the last line of Python's complaint.
			const QStringList lines = out.trimmed().split('\n');
			finish(QStringLiteral("Python Check: not usable (%1)\n").arg(lines.isEmpty() ? QStringLiteral("exit code %1").arg(code) : lines.last().trimmed().left(200)));
		}
	});
	connect(process, &QProcess::errorOccurred, this, [this, process](QProcess::ProcessError e) {
		if (process == m_process && e == QProcess::FailedToStart) finish(QStringLiteral("Python Check: not usable (could not start python)\n"));
	});
	m_timeout->start();
	process->start(setup.python, {setup.script, "--check"});
}

void PhotoHelperCheck::finish(const QString &facts) {
	if (!m_running) return;
	m_running = false;
	m_timeout->stop();
	QString clean = facts;
	clean.remove('\r');  // Python writes \r\n to a pipe on Windows; the report uses \n like the rest of it
	QString section = m_head + clean;
	if (!section.endsWith('\n')) section += '\n';
	section += QStringLiteral("Saved Photo Objects: %1 (%2; safe to delete when no saved scene uses them)\n")
	               .arg(photo_import::resultsSummary(), QDir::toNativeSeparators(photo_import::resultsFolder()));
	emit finished(section);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// PhotoToMeshJob
// ---------------------------------------------------------------------------------------------------------------------------------
PhotoToMeshJob::PhotoToMeshJob(const photo_import::Setup &setup, const QString &image, const QString &outFolder, bool removeBackground, QObject *parent)
    : QObject(parent), m_setup(setup), m_image(image), m_out(outFolder), m_removeBackground(removeBackground) {}

PhotoToMeshJob::~PhotoToMeshJob() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
}

void PhotoToMeshJob::start() {
	m_process = newProcess(this, true);
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() { onOutput(); });
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) { onFinished(code, st); });
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart && !m_done) {
			m_done = true;
			emit finished(false, tr("Could not start the photo helper (%1).").arg(m_setup.python));
		}
	});
	QStringList args{m_setup.script, m_image, "--out", m_out};
	if (!m_removeBackground) args << "--no-remove-bg";
	m_process->start(m_setup.python, args);
}

void PhotoToMeshJob::cancel() {
	m_cancelled = true;
	if (m_process) m_process->kill();
}

void PhotoToMeshJob::onOutput() {
	for (const auto &piece : m_splitter.feed(toBytes(m_process->readAllStandardOutput()))) {
		const QString text = QString::fromUtf8(piece.text.c_str());
		int percent = 0;
		std::string message;
		if (!piece.endsLine) continue;  // the helper only ever ends its lines
		if (photo_report::parseProgressLine(piece.text, percent, message)) {
			emit progress(percent, QString::fromUtf8(message.c_str()));
		} else if (text.startsWith("ERROR ")) {
			m_lastError = text.mid(6);
		} else if (text.startsWith("NOTE ")) {
			m_note = text.mid(5);
		} else {
			m_tail << text;  // the end of a Python traceback, when it crashes
			while (m_tail.size() > 6) m_tail.removeFirst();
		}
	}
}

void PhotoToMeshJob::onFinished(int exitCode, QProcess::ExitStatus status) {
	if (m_done) return;
	m_done = true;
	onOutput();
	if (m_cancelled) {
		emit finished(false, tr("Cancelled."));
		return;
	}
	const bool made = QFileInfo::exists(m_out + "/mesh.obj") && QFileInfo::exists(m_out + "/texture.png");
	if (status == QProcess::NormalExit && exitCode == 0 && made) {
		emit finished(true, m_note);
		return;
	}
	QString why = m_lastError;
	if (why.isEmpty()) why = tr("The helper stopped unexpectedly (exit code %1).").arg(exitCode) + (m_tail.isEmpty() ? QString() : "\n\n" + m_tail.join('\n'));
	emit finished(false, why);
}
