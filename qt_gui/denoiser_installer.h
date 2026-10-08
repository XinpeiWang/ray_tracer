#ifndef DENOISER_INSTALLER_H
#define DENOISER_INSTALLER_H

// Installs Intel Open Image Denoise (OIDN, Apache-2.0) on request: for the Metal renderer on a Mac, and for the CPU renderer on Windows. The library is ~50 MB (the network weights), so the app does not
// ship it: when a user first ticks the AI denoiser, they are asked, and the release for their Mac's architecture is downloaded from OIDN's own GitHub release page
// (checked against a SHA-256 pinned here), unpacked, and its lib/ folder kept under <user assets>/denoiser/ - where src/shared/oidn_runtime.h, and so the
// ray_tracer the GUI starts, finds it. See docs/DENOISING.md.

#include <QObject>
#include <QString>

#include <memory>

namespace asset_downloader { class Downloader; struct Manifest; }

namespace denoiser_installer {

// Where it is installed ("" if the per-user folder is not set); the loadable library is <that>/lib/libOpenImageDenoise.dylib.
QString installFolder();
bool isInstalled();
// The installer exists for macOS (the Metal renderer) and 64-bit Windows (the CPU renderer; a Windows GPU render uses the OptiX denoiser).
bool isSupportedHere();
// "arm64" or "x86_64": the architecture of THIS program, which the library must match to be loaded.
QString architecture();
// Human-readable size of the download, e.g. "about 51 MB".
QString downloadSizeText();

class Installer : public QObject {
	Q_OBJECT
public:
	explicit Installer(QObject *parent = nullptr);
	~Installer() override;
	void start();
	void cancel();
	bool isRunning() const { return m_running; }

signals:
	void progress(int percent, const QString &text);
	void finished(bool ok, const QString &message);

private:
	void unpack(const QString &archive);
	void finish(bool ok, const QString &message);

	std::unique_ptr<asset_downloader::Manifest> m_manifest;
	std::unique_ptr<asset_downloader::Downloader> m_downloader;
	bool m_running = false;
};

}  // namespace denoiser_installer

#endif  // DENOISER_INSTALLER_H
