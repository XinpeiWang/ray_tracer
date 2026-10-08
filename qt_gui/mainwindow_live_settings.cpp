// mainwindow_live_settings.cpp - the Live Preview controls remembered between runs (QSettings), one load/save pair each (split out of
// mainwindow_tabs_render.cpp; nothing changed). Compiled only where Live Preview exists.

#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"
#include "settings_keys.h"

#include "../src/shared/scene_descriptor.h"

#include <cmath>

#include <QTabBar>
#include <QStatusBar>
#include "scene_builder_widget.h"
#include "scene_metadata_client.h"
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
#include "realtime_preview_session.h"
#endif
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QStyleFactory>
#include <QPalette>
#include <QProcess>
#include <QDir>
#include <QDateTime>
#include <QScrollArea>
#include <QScrollBar>
#include <QScreen>
#include <QTimer>
#include <QAbstractItemView>
#include <QIcon>
#include <QDesktopServices>
#include <QUrl>
#include <QSplitter>
#include <QStackedWidget>
#include <QSlider>
#include <QStandardPaths>
#include <QFile>
#include <QToolButton>
#include <QSettings>
#include <cmath>
#include <algorithm>


#ifdef RT_GUI_HAVE_LIVE_PREVIEW

// Live Preview mouse/keyboard sensitivity persistence - same
// QSettings(kOrg, kApp) location and per-call-instance shape as
// loadSavedThemeId()/saveThemeId() (theme_switch.cpp).
double MainWindow::loadSavedMouseSensitivity() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewMouseSensitivityKey, 1.0).toDouble();
}

void MainWindow::saveMouseSensitivity(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewMouseSensitivityKey, value);
}

double MainWindow::loadSavedKeyboardSensitivity() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewKeyboardSensitivityKey, 1.0).toDouble();
}

void MainWindow::saveKeyboardSensitivity(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewKeyboardSensitivityKey, value);
}

bool MainWindow::loadSavedLiveDenoiseEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewDenoiseEnabledKey, false).toBool();
}

void MainWindow::saveLiveDenoiseEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewDenoiseEnabledKey, value);
}

double MainWindow::loadSavedLiveDenoiseBlend() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewDenoiseBlendKey, 0.0).toDouble();
}

void MainWindow::saveLiveDenoiseBlend(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewDenoiseBlendKey, value);
}

bool MainWindow::loadSavedLiveDenoiseShowLatest() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewDenoiseShowLatestKey, false).toBool();
}

void MainWindow::saveLiveDenoiseShowLatest(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewDenoiseShowLatestKey, value);
}

bool MainWindow::loadSavedLiveSvgfEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfEnabledKey, false).toBool();
}

void MainWindow::saveLiveSvgfEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfEnabledKey, value);
}

bool MainWindow::loadSavedLiveRestirGiEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewRestirGiEnabledKey, true).toBool();
}

void MainWindow::saveLiveRestirGiEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewRestirGiEnabledKey, value);
}

bool MainWindow::loadSavedLiveRestirDiEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewRestirDiEnabledKey, true).toBool();
}

void MainWindow::saveLiveRestirDiEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewRestirDiEnabledKey, value);
}

bool MainWindow::loadSavedLiveProbeCacheEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewProbeCacheEnabledKey, false).toBool();
}

void MainWindow::saveLiveProbeCacheEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewProbeCacheEnabledKey, value);
}

bool MainWindow::loadSavedLivePathGuidingEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewPathGuidingEnabledKey, false).toBool();
}

void MainWindow::saveLivePathGuidingEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewPathGuidingEnabledKey, value);
}

bool MainWindow::loadSavedLiveNrcEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewNrcEnabledKey, false).toBool();
}

void MainWindow::saveLiveNrcEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewNrcEnabledKey, value);
}

bool MainWindow::loadSavedLiveNeuralUpscaleEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewNeuralUpscaleEnabledKey, false).toBool();
}

void MainWindow::saveLiveNeuralUpscaleEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewNeuralUpscaleEnabledKey, value);
}

bool MainWindow::loadSavedLiveAutoExposure() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
#ifdef Q_OS_MAC
	const bool defaultValue = true;    // dim scenes should not come up black
#else
	const bool defaultValue = false;   // no control for it on this platform
#endif
	return settings.value(settings_keys::kLivePreviewAutoExposureKey, defaultValue).toBool();
}

void MainWindow::saveLiveAutoExposure(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewAutoExposureKey, value);
}

bool MainWindow::loadSavedLiveSmoothNoise() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
#ifdef Q_OS_MAC
	const bool defaultValue = true;    // the Metal backend has no denoiser of its own
#else
	const bool defaultValue = false;   // the OptiX build has its own denoisers, and this setting has no control there
#endif
	return settings.value(settings_keys::kLivePreviewSmoothNoiseKey, defaultValue).toBool();
}

void MainWindow::saveLiveSmoothNoise(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSmoothNoiseKey, value);
}

bool MainWindow::loadSavedLiveDofEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewDofEnabledKey, false).toBool();
}

void MainWindow::saveLiveDofEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewDofEnabledKey, value);
}

double MainWindow::loadSavedLiveAperture() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewApertureKey, 1.0).toDouble();
}

void MainWindow::saveLiveAperture(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewApertureKey, value);
}

double MainWindow::loadSavedLiveFocusDistance() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewFocusDistanceKey, 10.0).toDouble();
}

void MainWindow::saveLiveFocusDistance(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewFocusDistanceKey, value);
}

int MainWindow::loadSavedLiveTemporalUpscaleFactor() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewTemporalUpscaleFactorKey, 1).toInt();
}

void MainWindow::saveLiveTemporalUpscaleFactor(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewTemporalUpscaleFactorKey, value);
}

double MainWindow::loadSavedLiveExposure() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewExposureKey, 1.0).toDouble();
}

void MainWindow::saveLiveExposure(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewExposureKey, value);
}

int MainWindow::loadSavedLiveSamples() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSamplesKey, 1).toInt();
}

void MainWindow::saveLiveSamples(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSamplesKey, value);
}

int MainWindow::loadSavedLiveMaxDepth() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewMaxDepthKey, 8).toInt();
}

void MainWindow::saveLiveMaxDepth(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewMaxDepthKey, value);
}

double MainWindow::loadSavedLiveFireflyClamp() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewFireflyClampKey, 50.0).toDouble();
}

void MainWindow::saveLiveFireflyClamp(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewFireflyClampKey, value);
}

bool MainWindow::loadSavedLiveAdaptiveSamplingEnabled() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewAdaptiveSamplingEnabledKey, false).toBool();
}

void MainWindow::saveLiveAdaptiveSamplingEnabled(bool value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewAdaptiveSamplingEnabledKey, value);
}

double MainWindow::loadSavedLiveAdaptiveSamplingThreshold() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewAdaptiveSamplingThresholdKey, 0.01).toDouble();
}

void MainWindow::saveLiveAdaptiveSamplingThreshold(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewAdaptiveSamplingThresholdKey, value);
}

double MainWindow::loadSavedLiveSvgfTemporalAlpha() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfTemporalAlphaKey, 0.2).toDouble();
}

void MainWindow::saveLiveSvgfTemporalAlpha(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfTemporalAlphaKey, value);
}

double MainWindow::loadSavedLiveSvgfMaxHistoryLength() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfMaxHistoryLengthKey, 32.0).toDouble();
}

void MainWindow::saveLiveSvgfMaxHistoryLength(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfMaxHistoryLengthKey, value);
}

double MainWindow::loadSavedLiveSvgfVarianceBootstrapFrames() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfVarianceBootstrapFramesKey, 4.0).toDouble();
}

void MainWindow::saveLiveSvgfVarianceBootstrapFrames(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfVarianceBootstrapFramesKey, value);
}

int MainWindow::loadSavedLiveSvgfVarianceBootstrapRadius() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfVarianceBootstrapRadiusKey, 3).toInt();
}

void MainWindow::saveLiveSvgfVarianceBootstrapRadius(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfVarianceBootstrapRadiusKey, value);
}

double MainWindow::loadSavedLiveSvgfSigmaNormal() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfSigmaNormalKey, 128.0).toDouble();
}

void MainWindow::saveLiveSvgfSigmaNormal(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfSigmaNormalKey, value);
}

double MainWindow::loadSavedLiveSvgfSigmaDepth() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfSigmaDepthKey, 1.0).toDouble();
}

void MainWindow::saveLiveSvgfSigmaDepth(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfSigmaDepthKey, value);
}

double MainWindow::loadSavedLiveSvgfSigmaLuminance() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfSigmaLuminanceKey, 4.0).toDouble();
}

void MainWindow::saveLiveSvgfSigmaLuminance(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfSigmaLuminanceKey, value);
}

int MainWindow::loadSavedLiveSvgfAtrousRadius() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfAtrousRadiusKey, 2).toInt();
}

void MainWindow::saveLiveSvgfAtrousRadius(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfAtrousRadiusKey, value);
}

double MainWindow::loadSavedLiveSvgfMinAlbedo() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfMinAlbedoKey, 0.02).toDouble();
}

void MainWindow::saveLiveSvgfMinAlbedo(double value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfMinAlbedoKey, value);
}

int MainWindow::loadSavedLiveSvgfAtrousPasses() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	return settings.value(settings_keys::kLivePreviewSvgfAtrousPassesKey, 4).toInt();
}

void MainWindow::saveLiveSvgfAtrousPasses(int value) const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.setValue(settings_keys::kLivePreviewSvgfAtrousPassesKey, value);
}

#endif
