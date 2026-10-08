// The Diagnostics tab's "Install Photo Helper" button: after Run Diagnostics shows the Photo helper section with something missing, this runs
// scripts/setup_photo_to_mesh.ps1 (photo_import.h's PhotoHelperInstaller) behind a confirmation and a progress window, then runs the
// diagnostics again so the report shows the result.
#include "app_log.h"
#include "mainwindow.h"

#include "photo_import.h"

#include <QApplication>
#include <QDialog>
#include <QDir>
#include <QFontDatabase>
#include <QLabel>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

void MainWindow::updateInstallPhotoButton() {
	if (!m_installPhotoHelperButton) return;
	const bool installing = m_photoInstaller && m_photoInstaller->isRunning();
	const QStringList missing = diagnosticsBusy() ? QStringList() : photo_import::missingFacts(m_lastDiagReport);
	const bool canInstall = !missing.isEmpty() && !diagnosticsBusy() && !installing;
	if (installing) m_installPhotoHelperButton->setToolTip(tr("The photo helper is being installed."));
	else if (diagnosticsBusy()) m_installPhotoHelperButton->setToolTip(tr("Wait for the diagnostics to finish."));
	else if (!m_lastDiagReport.contains("=== Photo helper")) m_installPhotoHelperButton->setToolTip(tr("Run Diagnostics first: it shows what the photo helper is missing."));
	else if (missing.isEmpty()) m_installPhotoHelperButton->setToolTip(tr("Nothing is missing: the photo helper is installed."));
	else m_installPhotoHelperButton->setToolTip(tr("Download and install what the photo helper is missing:\n%1").arg(missing.join('\n')));
	m_installPhotoHelperButton->setEnabled(canInstall);
}

void MainWindow::onInstallPhotoHelperClicked() {
	startPhotoHelperInstall(true);
}

// `confirm` false skips the question (the self-test). `onDone` gets the result after the window has closed.
void MainWindow::startPhotoHelperInstall(bool confirm, const std::function<void(bool, const QString &)> &onDone) {
	if (m_photoInstaller && m_photoInstaller->isRunning()) return;
	if (confirm) {
		const QString script = photo_import::setupScript();
		if (script.isEmpty()) {
			QMessageBox::warning(this, tr("Install the photo helper"), tr("The installer script (scripts/setup_photo_to_mesh) was not found next to the program."));
			return;
		}
		const QStringList missing = photo_import::missingFacts(m_lastDiagReport);
#ifdef Q_OS_WIN
		const QString runner = tr("The script that will run (PowerShell, with the execution policy bypassed):");
		const QString pytorchFrom = tr("PyTorch (from download.pytorch.org)");
		const QString pythonNeeds = tr("It needs Python 3.10 to 3.12 and git on your PATH");
#else
		const QString runner = tr("The script that will run (bash):");
		const QString pytorchFrom = tr("PyTorch (from PyPI)");
		const QString pythonNeeds = tr("It needs Python 3.10 to 3.12 (for example brew install python@3.12) and git (xcode-select --install)");
#endif
		const QString question =
		    tr("Install the photo helper?\n\nThis downloads about 5 GB and installs it for your user only, in %1:\n"
		       "  - %4 and the Python packages the helper needs (from PyPI)\n"
		       "  - the TripoSR code (from GitHub) and its model weights, about 1.7 GB (from Hugging Face)\n"
		       "  - the background-removal model, about 176 MB (from GitHub)\n\n"
		       "%5 and takes several minutes. You can keep using the program meanwhile.\n\n"
		       "Missing now:\n%2\n\n%6\n%3")
		        .arg(QDir::toNativeSeparators(photo_import::environmentFolder()), missing.join('\n'), QDir::toNativeSeparators(script), pytorchFrom, pythonNeeds, runner);
		if (QMessageBox::question(this, tr("Install the photo helper"), question, QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel) != QMessageBox::Yes) return;
	}

	auto *dialog = new QDialog(this);
	dialog->setAttribute(Qt::WA_DeleteOnClose);
	dialog->setWindowTitle(tr("Installing the photo helper"));
	dialog->resize(760, 420);
	auto *layout = new QVBoxLayout(dialog);
	auto *statusLabel = new QLabel(tr("Starting..."), dialog);
	statusLabel->setWordWrap(true);
	auto *bar = new QProgressBar(dialog);
	bar->setRange(0, 0);  // busy: the script does not report a percentage
	bar->setTextVisible(false);
	auto *log = new QPlainTextEdit(dialog);
	log->setReadOnly(true);
	log->setMaximumBlockCount(3000);
	log->setLineWrapMode(QPlainTextEdit::NoWrap);
	#ifdef Q_OS_WIN
	log->setFont(QFont("Consolas", 9));
#else
	log->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
#endif
	auto *button = new QPushButton(tr("Cancel"), dialog);
	button->setAutoDefault(false);
	layout->addWidget(statusLabel);
	layout->addWidget(bar);
	layout->addWidget(log, 1);
	layout->addWidget(button, 0, Qt::AlignRight);

	auto *installer = new PhotoHelperInstaller(dialog);
	m_photoInstaller = installer;
	connect(installer, &QObject::destroyed, this, [this]() { m_photoInstaller = nullptr; });
	connect(installer, &PhotoHelperInstaller::line, log, &QPlainTextEdit::appendPlainText);
	connect(installer, &PhotoHelperInstaller::status, statusLabel, &QLabel::setText);
	// The installer's output is kept in the log file too: a failed 5 GB install is hard to reproduce.
	connect(installer, &PhotoHelperInstaller::line, this, [](const QString &line) { AppLog::info(QStringLiteral("photo-install"), line); });
	connect(installer, &PhotoHelperInstaller::status, this, [](const QString &text) { AppLog::info(QStringLiteral("photo-install"), QStringLiteral("status: %1").arg(text)); });
	connect(button, &QPushButton::clicked, dialog, [installer, dialog, button]() {
		if (installer->isRunning()) {
			button->setEnabled(false);
			installer->cancel();
		} else {
			dialog->close();
		}
	});
	connect(dialog, &QDialog::rejected, installer, [installer]() { if (installer->isRunning()) installer->cancel(); });
	connect(installer, &PhotoHelperInstaller::finished, this, [](bool ok, const QString &message) {
		AppLog::write(ok ? log_format::Level::Info : log_format::Level::Error, QStringLiteral("photo-install"), QStringLiteral("finished: %1: %2").arg(ok ? QStringLiteral("ok") : QStringLiteral("failed"), message));
	});
	connect(installer, &PhotoHelperInstaller::finished, this, [this, dialog, statusLabel, bar, button, confirm, onDone](bool ok, const QString &message) {
		m_photoInstaller = nullptr;
		bar->setRange(0, 1);
		bar->setValue(ok ? 1 : 0);
		statusLabel->setText(ok ? tr("Installed. Running the diagnostics again to check it...") : tr("The installation did not finish: %1").arg(message));
		button->setText(tr("Close"));
		button->setEnabled(true);
		updateInstallPhotoButton();
		if (ok) onRunDiagnosticsClicked();
		if (onDone) onDone(ok, message);
		if (ok || !confirm) dialog->close();  // a failure stays on screen so its output can be read (the self-test closes it)
	});
	connect(dialog, &QObject::destroyed, this, [this]() { updateInstallPhotoButton(); });
	dialog->show();
	updateInstallPhotoButton();
	installer->start();
}
