// The AI denoiser's install-on-first-use (Mac, Metal renderer): ticking "AI denoiser" asks to download Intel Open Image Denoise (~50 MB, once), shows the
// download and unpacking, and unticks the box again if it was declined, cancelled or failed - so a render never starts with a denoiser that is not there.
#include "mainwindow.h"

#include "app_log.h"
#include "denoiser_installer.h"

#include <QMessageBox>
#include <QProgressDialog>
#include <QSignalBlocker>
#include <QStatusBar>

void MainWindow::onDenoiseToggled(bool on) {
	if (!on || !denoiser_installer::isSupportedHere() || denoiser_installer::isInstalled()) return;
	if (m_denoiserInstaller && m_denoiserInstaller->isRunning()) return;
	auto untick = [this]() {
		const QSignalBlocker blocker(m_denoiseCheck);
		m_denoiseCheck->setChecked(false);
		m_denoiseBlendSpin->setEnabled(false);
	};
	const auto choice = QMessageBox::question(this, tr("Install the denoiser"),
		tr("The AI denoiser is Intel's Open Image Denoise (open source, Apache-2.0). It is not part of this app: it is downloaded once (%1) from its own release page "
		   "on GitHub (github.com/RenderKit/oidn), checked against a known checksum, and kept in your user folder.\n\nDownload it now?")
			.arg(denoiser_installer::downloadSizeText()),
		QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
	if (choice != QMessageBox::Yes) { untick(); return; }

	if (!m_denoiserInstaller) m_denoiserInstaller = new denoiser_installer::Installer(this);
	auto *dialog = new QProgressDialog(tr("Starting..."), tr("Cancel"), 0, 100, this);
	dialog->setWindowTitle(tr("Installing the denoiser"));
	dialog->setWindowModality(Qt::WindowModal);
	dialog->setMinimumDuration(0);
	dialog->setAutoClose(false);
	dialog->setAutoReset(false);
	dialog->setAttribute(Qt::WA_DeleteOnClose);
	connect(dialog, &QProgressDialog::canceled, m_denoiserInstaller, &denoiser_installer::Installer::cancel);
	connect(m_denoiserInstaller, &denoiser_installer::Installer::progress, dialog, [dialog](int percent, const QString &text) {
		dialog->setValue(percent);
		dialog->setLabelText(text);
	});
	connect(m_denoiserInstaller, &denoiser_installer::Installer::finished, dialog, [this, dialog, untick](bool ok, const QString &message) {
		disconnect(m_denoiserInstaller, nullptr, dialog, nullptr);
		dialog->close();
		if (ok) {
			statusBar()->showMessage(message, 6000);
			updateRenderOptionsEnabled();
			return;
		}
		untick();
		if (!message.contains(QLatin1String("cancel"), Qt::CaseInsensitive))
			QMessageBox::warning(this, tr("Install the denoiser"), tr("The denoiser could not be installed: %1").arg(message));
	});
	m_denoiserInstaller->start();
}
