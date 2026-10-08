#include "photo_import.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QStringList>
#include <QTimer>

namespace photo_import {

QString environmentFolder() {
	// A short path on purpose: PyTorch and transformers have deep folders, and Windows stops at 260 characters.
	return QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/RayTracerPhoto";
}

QString resultsFolder() {
	return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/photo_meshes";
}

static QString environmentPython() {
#ifdef Q_OS_WIN
	return environmentFolder() + "/venv/Scripts/python.exe";
#else
	return environmentFolder() + "/venv/bin/python";
#endif
}

static QString helperScript() {
	const QString app = QCoreApplication::applicationDirPath();
	for (const QString &c : {app + "/tools/photo_to_mesh/photo_to_mesh.py", app + "/../tools/photo_to_mesh/photo_to_mesh.py",
	                         app + "/../../tools/photo_to_mesh/photo_to_mesh.py", app + "/../../../tools/photo_to_mesh/photo_to_mesh.py"})
		if (QFileInfo::exists(c)) return QDir::cleanPath(c);
	return QString();
}

Setup locate() {
	Setup s;
	s.script = helperScript();
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

bool parseProgressLine(const QString &line, int *percent, QString *message) {
	static const QString tag = QStringLiteral("PROGRESS ");
	if (!line.startsWith(tag)) return false;
	const QString rest = line.mid(tag.size());
	const int space = rest.indexOf(' ');
	bool ok = false;
	const int p = (space < 0 ? rest : rest.left(space)).toInt(&ok);
	if (!ok) return false;
	if (percent) *percent = qBound(0, p, 100);
	if (message) *message = space < 0 ? QString() : rest.mid(space + 1).trimmed();
	return true;
}

QString setupScript() {
	const QString overridePath = qEnvironmentVariable("RAY_TRACER_PHOTO3D_SETUP");
	if (!overridePath.isEmpty()) return QFileInfo::exists(overridePath) ? overridePath : QString();
	const QString app = QCoreApplication::applicationDirPath();
	for (const QString &c : {app + "/scripts/setup_photo_to_mesh.ps1", app + "/../scripts/setup_photo_to_mesh.ps1", app + "/../../scripts/setup_photo_to_mesh.ps1",
	                         app + "/../../../scripts/setup_photo_to_mesh.ps1"})
		if (QFileInfo::exists(c)) return QDir::cleanPath(c);
	return QString();
}

QStringList missingFacts(const QString &report) {
	QStringList out;
	const int at = report.indexOf(QStringLiteral("=== Photo helper"));
	if (at < 0) return out;
	QString section = report.mid(at);
	const int next = section.indexOf(QStringLiteral("\n==="));  // the next section's banner (this one's own banner is on the first line)
	if (next > 0) section = section.left(next);
	for (const QString &raw : section.split('\n')) {
		const QString line = raw.trimmed();
		const int colon = line.indexOf(':');
		if (line.startsWith("===") || colon < 0) continue;
		const QString key = line.left(colon);
		if (key == "Helper Script" || key == "Expected At" || key.startsWith("Graphics Card") || key == "Python") continue;  // not something setup fixes
		const QString lower = line.toLower();
		if (lower.contains("missing") || lower.contains("not usable") || lower.contains("not available")) out << line;
	}
	return out;
}

}  // namespace photo_import

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
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() { onOutput(); });
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		if (m_done) return;
		m_done = true;
		onOutput();
		if (!m_pending.trimmed().isEmpty()) emit line(m_pending.trimmed());
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
	if (script.isEmpty()) {
		m_done = true;
		QProcess *p = m_process;
		m_process = nullptr;
		p->deleteLater();
		QTimer::singleShot(0, this, [this]() { emit finished(false, tr("The installer script (scripts/setup_photo_to_mesh.ps1) was not found next to the program.")); });
		return;
	}
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
	m_pending += QString::fromLocal8Bit(m_process->readAllStandardOutput());
	// pip redraws its progress bar with \r; a bare \r replaces the status text, \n ends a line.
	int i = 0;
	QString current;
	for (; i < m_pending.size(); ++i) {
		const QChar c = m_pending[i];
		if (c == '\n') {
			const QString text = current.trimmed();
			if (!text.isEmpty()) {
				emit line(text);
				emit status(text);
				m_tail << text;
				while (m_tail.size() > 8) m_tail.removeFirst();
			}
			current.clear();
		} else if (c == '\r') {
			if (!current.trimmed().isEmpty()) emit status(current.trimmed());
			current.clear();
		} else {
			current += c;
		}
	}
	m_pending = current;
}

PhotoHelperCheck::~PhotoHelperCheck() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
}

void PhotoHelperCheck::start() {
	if (m_running) return;
	m_running = true;
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
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
	env.insert("PYTHONIOENCODING", "utf-8");
	m_process->setProcessEnvironment(env);
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		const QString out = QString::fromUtf8(m_process->readAll());
		if (st == QProcess::NormalExit && code == 0) {
			finish(out);
		} else {
			// A broken install: say it as a fact, with the last line of Python's complaint.
			const QStringList lines = out.trimmed().split('\n');
			finish(QStringLiteral("Python Check: not usable (%1)\n").arg(lines.isEmpty() ? QStringLiteral("exit code %1").arg(code) : lines.last().trimmed().left(200)));
		}
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) finish(QStringLiteral("Python Check: not usable (could not start python)\n"));
	});
	// Importing PyTorch takes a few seconds; a hung or broken install must not hold the Diagnostics button forever.
	QTimer::singleShot(60000, this, [this]() {
		if (m_running && m_process) {
			m_process->disconnect(this);
			m_process->kill();
			finish(QStringLiteral("Python Check: not usable (no answer after 60 seconds)\n"));
		}
	});
	m_process->start(setup.python, {setup.script, "--check"});
}

void PhotoHelperCheck::finish(const QString &facts) {
	if (!m_running) return;
	m_running = false;
	QString section = m_head + facts;
	if (!section.endsWith('\n')) section += '\n';
	emit finished(section);
}

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
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
	env.insert("PYTHONIOENCODING", "utf-8");
	env.insert("PYTHONUNBUFFERED", "1");
	m_process->setProcessEnvironment(env);
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
	m_pending += QString::fromUtf8(m_process->readAllStandardOutput());
	int nl;
	while ((nl = m_pending.indexOf('\n')) >= 0) {
		const QString line = m_pending.left(nl).trimmed();
		m_pending.remove(0, nl + 1);
		int percent = 0;
		QString message;
		if (photo_import::parseProgressLine(line, &percent, &message))
			emit progress(percent, message);
		else if (line.startsWith("ERROR "))
			m_lastError = line.mid(6);
		else if (line.startsWith("NOTE "))
			m_note = line.mid(5);
		else if (!line.isEmpty()) {
			m_tail << line;  // the end of a Python traceback, when it crashes
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
