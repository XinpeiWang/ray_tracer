// mainwindow_scene_info.cpp - what the GUI shows about the chosen scene: camera distance, the info label, recommended settings, and onSceneChanged (split
// out of mainwindow_slots.cpp; nothing changed).

#include "mainwindow.h"
#include "icon_tint.h"
#include "photo_import.h"
#include "scene_metadata_client.h"
#include "win_taskbar.h"
#include "render_output_parser.h"
#include "app_log.h"
#include "camera_math.h"
#include "../src/shared/video_preset.h"
#include "../src/shared/scene_descriptor.h"
#include <QApplication>
#include <QFileDialog>
#include <QMessageBox>
#include <QFileInfo>
#include <QLocale>
#include <QProgressDialog>
#include <QStorageInfo>
#include <QProcess>
#include <QDir>
#include <QTimer>
#include <QDateTime>
#include <QDesktopServices>
#include <QUrl>
#include <QScrollBar>
#include <QStatusBar>
#include <QCoreApplication>
#include <QSignalBlocker>
#include <QIcon>
#include <QStyle>
#include <QThread>
#include <QHash>
#include <array>
#include <cmath>
#include <optional>


// ============================================================================
// MainWindow::refreshCameraDistanceDisplay
// ============================================================================
// Recomputes the Distance spinbox's displayed value from the current X/Y/Z
// spinboxes and m_currentLookatX/Y/Z, without re-triggering
// onCameraDistanceChanged (which would otherwise try to reposition X/Y/Z
// right back, a harmless but wasteful no-op loop).
// ============================================================================
void MainWindow::refreshCameraDistanceDisplay() {
	const double dist = camera_math::distanceFromTarget(currentCameraPosition(),
														currentLookAt());
	const QSignalBlocker blocker(m_cameraDistance);
	m_cameraDistance->setValue(dist);
}

// Shared by refreshSceneInfoLabel() and updateSceneRecommendedSettingsHint()
// below: use `preloaded` if the caller already fetched one (onSceneChanged()/
// applyRecommendedSettings() do, to share a single sceneMetadata() call
// across several functions for the same scene), otherwise fetch into
// `local` and return that instead. The fetch's own success/failure is
// deliberately not surfaced here - both callers already treat a "not
// found" SceneMetadata's default-constructed empty/neutral fields
// (description=="", recommendedIntegrator=="", etc.) as "nothing to show"
// on their own, the same way every individual accessor's own "" fallback
// always meant before this helper existed.
static const SceneMetadataClient::SceneMetadata* resolveSceneMeta(
		const SceneMetadataClient::SceneMetadata* preloaded, const QString& sceneId,
		SceneMetadataClient::SceneMetadata& local) {
	if (preloaded) return preloaded;
	SceneMetadataClient::sceneMetadata(sceneId, local);
	return &local;
}

void MainWindow::refreshSceneInfoLabel(const SceneMetadataClient::SceneMetadata* preloaded) {
	if (!m_sceneCombo || !m_sceneInfoLabel) return;
	const int index = m_sceneCombo->currentIndex();
	if (index < 0) return;
	const QString scene_id = m_sceneCombo->itemData(index).toString();
	updateSceneTechInfoIcon(scene_id);
	m_downloadableAssetJobs.clear();
	m_downloadablePack = nullptr;
	if (m_downloadAssetsButton) m_downloadAssetsButton->setVisible(false);

	SceneMetadataClient::SceneMetadata local;
	const SceneMetadataClient::SceneMetadata* meta = resolveSceneMeta(preloaded, scene_id, local);
	if (meta->description.isEmpty()) return;

	// Which compatibility field actually reflects "GPU Support" here depends
	// on the backend this platform/build can even offer - same
	// Q_OS_MAC/m_metalGpuAvailable-aware choice as the auto-switch-to-CPU
	// check below (see that block's own comment for why gpuCompatible and
	// metalCompatible aren't interchangeable).
#ifdef Q_OS_MAC
	// No OptiX fallback on macOS: when Metal isn't usable at runtime (or
	// wasn't compiled in) there is NO GPU path at all, so falling back to
	// gpuCompatible (the OptiX/Windows flag) would show "GPU Support: Yes"
	// on a machine that can only render on the CPU.
	const bool gpuSupported = m_metalGpuAvailable && meta->metalCompatible;
#else
	const bool gpuSupported = meta->gpuCompatible;
#endif

	QString infoText = tr("<b>Description:</b> %1<br>").arg(meta->description);
	// What the word means: one 400 x 400 picture at the recommended samples on a 16-core CPU (docs/SCENE_SELECTION.md).
	QString performanceRange;
	if (meta->performance == "Fast") performanceRange = tr("under 10 s");
	else if (meta->performance == "Medium") performanceRange = tr("10 to 30 s");
	else if (meta->performance == "Slow") performanceRange = tr("30 s to 2 min");
	else if (meta->performance == "Very Slow") performanceRange = tr("over 2 min");
	else performanceRange = tr("not measured");
	infoText += tr("<b>Performance:</b> %1 (%2)<br>").arg(SceneMetadataClient::displayPerformance(meta->performance), performanceRange);
	infoText += tr("<b>Recommended SPP:</b> %1<br>").arg(meta->recommendedSpp);
	infoText += tr("<b>GPU Support:</b> %1<br>").arg(gpuSupported ? tr("Yes") : tr("CPU only"));
	// These two warnings are the only coloured text in the label, so they take
	// their colours from the theme's log severities rather than fixed hex - a
	// gold-on-cream warning is unreadable on the light schemes. Rebuilt fresh
	// on every call (rather than cached) specifically so restyleThemedWidgets()
	// calling this on a theme switch picks up the new theme's colours instead
	// of leaving an already-shown badge stuck in the old one.
	if (meta->requiresFiles) {
		// Checked on every selection, not cached: the user may copy the missing
		// folder in while the app is open and re-select the scene to see it clear.
		const SceneMetadataClient::MissingAssets missing = SceneMetadataClient::missingAssets(scene_id);
		const QString folder = QDir::toNativeSeparators(missing.folder).toHtmlEscaped();
		// Which of the missing files this project can fetch for the user (the manifest's statue meshes).
		if (missing.any && m_downloadAssetsButton) {
			const QString appDir = QCoreApplication::applicationDirPath();
			const asset_downloader::Manifest &manifest = asset_downloader::builtInManifest();
			for (const QString &path : missing.missingPaths) {
				if (const asset_downloader::Entry *e = manifest.find(appDir, path)) {
					// Into the per-user folder when there is one (works from a read-only disk image), else where the scene looks.
					const QString root = asset_downloader::userAssetRoot();
					m_downloadableAssetJobs.append({*e, root.isEmpty() ? QDir::cleanPath(path) : QDir::cleanPath(root + QLatin1Char('/') + e->relativePath)});
				}
			}
			if (!m_downloadableAssetJobs.isEmpty()) {
				qint64 bytes = 0;
				for (const auto &j : m_downloadableAssetJobs) bytes += j.entry.size;
				m_downloadAssetsButton->setText(tr("Download %n missing file(s) (%1)", "", m_downloadableAssetJobs.size())
					.arg(QLocale().formattedDataSize(bytes, 1)));
				m_downloadAssetsButton->setVisible(true);
			}
			// Not one of the statue models: perhaps a large third-party scene this project knows how to fetch from its original site.
			if (m_downloadableAssetJobs.isEmpty()) {
				for (const QString &path : missing.missingPaths) {
					if (const scene_packs::Pack *pack = scene_packs::builtInCatalogue().find(QDir(appDir).relativeFilePath(path))) { m_downloadablePack = pack; break; }
				}
				if (m_downloadablePack) {
					m_downloadAssetsButton->setText(tr("Download \"%1\" (%2)").arg(m_downloadablePack->title, QLocale().formattedDataSize(m_downloadablePack->downloadBytes(), 1)));
					m_downloadAssetsButton->setVisible(true);
				}
			}
		}
		if (missing.any && missing.referenced == 0) {
			// The scene's own .pbrt is absent (a scene from a collection that has not been downloaded).
			infoText += tr("<br><b style='color: %1;'>&#9888; This scene's file was not found: %2</b>"
						   "<br>Expected in: %3<br>Rendering this scene will fail until it is installed.")
				.arg(m_activeTheme.logError.name(), QDir::toNativeSeparators(missing.example).toHtmlEscaped(), folder);
		} else if (missing.any) {
			infoText += tr("<br><b style='color: %1;'>&#9888; Missing external files: %2 of %3 not found (first: %4)</b>"
						   "<br>Put them in: %5<br>Rendering this scene will fail until they are installed.")
				.arg(m_activeTheme.logError.name())
				.arg(missing.missing).arg(missing.referenced)
				.arg(missing.example.toHtmlEscaped(), folder);
		} else {
			infoText += tr("<br><b style='color: %1;'>&#9888; Requires external files</b>")
				.arg(m_activeTheme.logWarning.name());
		}
	}
	if (!gpuSupported)
		infoText += tr("<br><b style='color: %1;'>&#9888; CPU renderer only</b>")
			.arg(m_activeTheme.logError.name());
	m_sceneInfoLabel->setText(infoText);
}

// Mirrors cpu_render_main()'s own recommended-settings mismatch check
// (cpu_interface.cpp) - same skip values ("volpath"/"sobol"/"bvh", each
// integrator/sampler/light-sampler's own real default) and the same "only
// when the default/no-flag value is actually in effect" scope (a recommendation
// the user already matched, or already overrode with a DIFFERENT explicit
// choice, is not a mismatch either way - same as the CLI only warning when
// no --sampler/--lightsampler was passed at all, and only when running the
// plain default path tracer for the integrator check), so this never claims
// a mismatch the CLI itself wouldn't warn about. Purely informational - like
// the CLI's own warning, this never changes the user's current selection.
void MainWindow::updateSceneRecommendedSettingsHint(const QString &sceneId,
		const SceneMetadataClient::SceneMetadata* preloaded) {
	if (!m_sceneRecommendedSettingsHint) return;
	if (sceneId.isEmpty() || !m_integratorCombo || !m_samplerCombo || !m_lightSamplerCombo) {
		m_sceneRecommendedSettingsHint->setVisible(false);
		if (m_applyRecommendedSettingsButton) m_applyRecommendedSettingsButton->setVisible(false);
		return;
	}

	// Not found (or DLL unavailable) leaves `local` at its default-
	// constructed empty recommendedIntegrator/Sampler/LightSampler - the
	// same "" every mismatch check below already treats as "no
	// recommendation, so no mismatch", matching what the old individual
	// sceneRecommended*() calls did on a failed lookup.
	SceneMetadataClient::SceneMetadata local;
	const SceneMetadataClient::SceneMetadata* meta = resolveSceneMeta(preloaded, sceneId, local);

	QStringList mismatches;

	const auto integrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	if (integrator == IntegratorMode::Default) {
		if (!meta->recommendedIntegrator.isEmpty() && meta->recommendedIntegrator != QLatin1String("volpath"))
			mismatches << tr("Integrator \"%1\"").arg(meta->recommendedIntegrator);
	}

	if (m_samplerCombo->currentData().toString().isEmpty()) {
		if (!meta->recommendedSampler.isEmpty() && meta->recommendedSampler != QLatin1String("sobol"))
			mismatches << tr("Sampler \"%1\"").arg(meta->recommendedSampler);
	}

	if (m_lightSamplerCombo->currentData().toString().isEmpty()) {
		if (!meta->recommendedLightSampler.isEmpty() && meta->recommendedLightSampler != QLatin1String("bvh"))
			mismatches << tr("Light Sampler \"%1\"").arg(meta->recommendedLightSampler);
	}

	if (mismatches.isEmpty()) {
		m_sceneRecommendedSettingsHint->setVisible(false);
		if (m_applyRecommendedSettingsButton) m_applyRecommendedSettingsButton->setVisible(false);
		return;
	}

	m_sceneRecommendedSettingsHint->setText(
		tr("⚠ This scene's file recommends %1, but the Render Options tab "
		   "is currently set to the default(s) instead - click Apply, or "
		   "change it there yourself, to match the scene's own settings.")
			.arg(mismatches.join(tr(", "))));
	m_sceneRecommendedSettingsHint->setVisible(true);
	if (m_applyRecommendedSettingsButton) m_applyRecommendedSettingsButton->setVisible(true);
}

// integratorModeForPbrtName(): the one place this GUI maps a pbrt-v4
// Integrator directive TYPE STRING (as loaded verbatim from a .pbrt file's
// own Integrator directive, SceneDescriptor::recommended_integrator) onto
// this GUI's own IntegratorMode enum - deliberately local to
// applyRecommendedSettings() rather than a shared table, since nothing
// else in this codebase needs the reverse of this mapping (a loaded pbrt
// file's Integrator directive is never auto-applied to an actual render -
// see docs/PBRT_SUPPORT.md's own note that only --bdpt/--sppm/--mlt/
// default CLI flags ever decide that). "path" is accepted as a synonym
// for "volpath" - pre-pbrt-v4 scenes and some hand-written ones still use
// the old name. Returns false (mode left unchanged) for an unrecognized
// string - includes this project's own "lightpath" (light-tracer)
// extension, not a real pbrt-v4 integrator type name, alongside the 8
// real ones.
static bool integratorModeForPbrtName(const QString &name, IntegratorMode &outMode) {
	static const QHash<QString, IntegratorMode> kByName = {
		{QStringLiteral("volpath"), IntegratorMode::Default},
		{QStringLiteral("path"), IntegratorMode::Default},
		{QStringLiteral("sppm"), IntegratorMode::Sppm},
		{QStringLiteral("bdpt"), IntegratorMode::Bdpt},
		{QStringLiteral("mlt"), IntegratorMode::Mlt},
		{QStringLiteral("randomwalk"), IntegratorMode::RandomWalk},
		{QStringLiteral("ambientocclusion"), IntegratorMode::Ao},
		{QStringLiteral("simplepath"), IntegratorMode::SimplePath},
		{QStringLiteral("simplevolpath"), IntegratorMode::SimpleVolPath},
		{QStringLiteral("lightpath"), IntegratorMode::LightPath},
	};
	const auto it = kByName.constFind(name);
	if (it == kByName.constEnd()) return false;
	outMode = it.value();
	return true;
}

void MainWindow::applyRecommendedSettings() {
	if (!m_sceneCombo || !m_integratorCombo || !m_samplerCombo || !m_lightSamplerCombo) return;
	const QString sceneId = m_sceneCombo->currentData().toString();
	if (sceneId.isEmpty()) return;

	// One fetch for all three recommended-setting lookups below AND the
	// updateSceneRecommendedSettingsHint() call at the bottom, instead of
	// four separate scene_metadata.dll round-trips (each its own
	// find_scene() scan) for the same sceneId.
	SceneMetadataClient::SceneMetadata meta;
	SceneMetadataClient::sceneMetadata(sceneId, meta); // failure leaves meta at its empty/neutral defaults, same as each individual accessor's own "not found" fallback

	IntegratorMode mode;
	if (!meta.recommendedIntegrator.isEmpty() && integratorModeForPbrtName(meta.recommendedIntegrator, mode)) {
		const int idx = m_integratorCombo->findData(static_cast<int>(mode));
		if (idx >= 0) m_integratorCombo->setCurrentIndex(idx);
	}

	if (!meta.recommendedSampler.isEmpty()) {
		const int idx = m_samplerCombo->findData(meta.recommendedSampler);
		if (idx >= 0) m_samplerCombo->setCurrentIndex(idx);
	}

	if (!meta.recommendedLightSampler.isEmpty()) {
		const int idx = m_lightSamplerCombo->findData(meta.recommendedLightSampler);
		if (idx >= 0) m_lightSamplerCombo->setCurrentIndex(idx);
	}

	// Each setCurrentIndex() above already re-ran
	// updateSceneRecommendedSettingsHint() via its own change handler (the
	// Integrator combo's onIntegratorChanged(), the Sampler/Light Sampler
	// combos' own lambdas - mainwindow_tabs_render.cpp), but calling it once
	// more here is cheap and guarantees the hint reflects the FINAL state
	// of all three rather than whatever it was after just the first change.
	// Passes `meta` through rather than letting it self-fetch a 5th time.
	updateSceneRecommendedSettingsHint(sceneId, &meta);
}

void MainWindow::onSceneChanged(int index) {
	if (index < 0) {
		// The only realistic way to reach this once scenes have loaded is
		// m_sceneSearchBox narrowing the current category to zero matches -
		// every category tab always holds at least one scene otherwise
		// (rebuildCategoryTabs() excludes empty categories outright).
		// Leaving the description panel showing the PREVIOUS scene made a
		// zero-match search look like nothing had happened.
		if (m_sceneInfoLabel && m_sceneCombo && m_sceneCombo->count() == 0) {
			const QString term = m_sceneSearchBox ? m_sceneSearchBox->text().trimmed() : QString();
			m_sceneInfoLabel->setText(term.isEmpty()
				? tr("No scenes in this category.")
				: tr("No scenes match \"%1\" in this category.").arg(term));
			updateSceneTechInfoIcon(QString());
		}
		return;
	}
	// A raw combo-row index is no longer a valid id on its own (ids are
	// category letter + number now - see scene_registry.h's
	// SceneDescriptor::id comment), so the m_sceneCombo-null fallback
	// resolves the id via the registry position instead.
	QString scene_id = m_sceneCombo ? m_sceneCombo->itemData(index).toString()
									 : SceneMetadataClient::sceneIdAtIndex(index);

	// Keeps m_sceneGrid's highlighted tile in sync with whatever scene just
	// became current, regardless of which view (combo, grid, search, tab
	// switch, selectSceneById) drove the change - every one of those paths
	// already funnels through here. Blocked so this never re-enters via the
	// grid's own currentItemChanged handler (mainwindow_tabs.cpp).
	if (m_sceneGrid) {
		const QSignalBlocker blocker(m_sceneGrid);
		bool found = false;
		for (int i = 0; i < m_sceneGrid->count(); ++i) {
			if (m_sceneGrid->item(i)->data(Qt::UserRole).toString() == scene_id) {
				m_sceneGrid->setCurrentRow(i);
				found = true;
				break;
			}
		}
		if (!found) m_sceneGrid->setCurrentRow(-1);
	}

	// One call fetches everything below (description, GPU support,
	// recommended SPP/exposure, recommended camera) instead of the ~7
	// separate scene_metadata.dll round-trips (each its own registry scan)
	// this used to make - see SceneMetadataClient::SceneMetadata's own
	// comment. Still the single source of truth scene_metadata.dll always
	// was; this just stops asking it the same question several times over.
	// Not found (DLL missing, or scene_id doesn't exist) means there's
	// nothing to show - leave the UI exactly as it was for the previous
	// scene rather than blanking it out.
	SceneMetadataClient::SceneMetadata meta;
	if (!SceneMetadataClient::sceneMetadata(scene_id, meta)) return;
	if (meta.description.isEmpty()) return;

	refreshSceneInfoLabel(&meta);
	updateSceneRecommendedSettingsHint(scene_id, &meta);
	// Same description text, shown in the Preview tab's sidebar too - see
	// createPreviewTab()'s own comment on why this is kept in sync here
	// rather than only refreshed on render completion.
	m_previewSceneDescLabel->setText(meta.description);

	// Unconditionally reset Samples/Exposure to this scene's curated
	// recommendation - a stale value carried over from a previous scene is
	// actively wrong, not just suboptimal (an unrelated scene's exposure
	// left in place can wash a normal render out to solid white; its SPP
	// left in place under- or over-samples a scene of very different
	// complexity). Same reasoning as the camera-position reset below, which
	// was already unconditional for exactly this reason - Samples used to
	// be the odd one out, gated behind a "still probably untouched"
	// 100/200/500 sentinel check that could both wrongly clobber a
	// deliberately-chosen 200 and wrongly leave a stale 512 in place;
	// unifying it here removes that inconsistency rather than adding a new
	// one.
	m_samplesSpinBox->setValue(meta.recommendedSpp);
	m_exposureSpin->setValue(meta.recommendedExposure);

	// Same unconditional-reset reasoning for the Video Generation Settings'
	// camera-path combo - meta.recommendedCameraPath (scene_registry.h's
	// recommended_camera_path_for()) is curated per scene/category (e.g.
	// Large Scenes default to a lateral "linear" flythrough rather than an
	// "orbit" that could circle straight through a room's walls), so a
	// stale path left over from a previous scene is equally as wrong as a
	// stale exposure would be.
	if (m_cameraPathCombo) {
		const int pathIdx = m_cameraPathCombo->findData(meta.recommendedCameraPath);
		if (pathIdx >= 0) m_cameraPathCombo->setCurrentIndex(pathIdx);
	}

	// Auto-switch to CPU when scene doesn't support GPU. Which flag
	// actually applies depends on which GPU backend is in play -
	// meta.gpuCompatible means "OptiX's own scene_builder.cpp reproduces
	// this scene" (Windows), meta.metalCompatible means "this scene has a
	// real .pbrt file backing it" (gpu/metal/'s own criterion, macOS) -
	// genuinely different scene sets, not one reinterpreted as the other
	// (see SceneMetadata::metalCompatible's own comment,
	// scene_metadata_client.h). kGpuOptionAvailable/m_metalGpuAvailable
	// are never both true (one is Windows-only, the other Q_OS_MAC-only),
	// so this always resolves to exactly one of the two checks.
#ifdef Q_OS_MAC
	const bool sceneSupportsSelectedGpuBackend = m_metalGpuAvailable ? meta.metalCompatible : meta.gpuCompatible;
#else
	const bool sceneSupportsSelectedGpuBackend = meta.gpuCompatible;
#endif
	if (!sceneSupportsSelectedGpuBackend && m_renderModeCombo->currentData().toBool()) {
		m_renderModeCombo->setCurrentIndex(1); // index 1 = CPU
	}

	// Reset the camera position to this scene's own recommended default -
	// every preset in m_cameraPresetCombo is now a direction*ratio relative
	// to m_currentSceneCamDistance rather than an absolute Cornell-Box-scale
	// position (see the combo's setup comment in mainwindow_tabs.cpp), but a
	// stale position from a previously viewed scene could still be wildly
	// wrong-scale for a much smaller scene's actual geometry (e.g. scene 1's
	// spheres sit within roughly +-15 units of the origin) until
	// m_currentSceneCamDistance itself is refreshed below. Switches to
	// "Custom" (index 7) first so onCameraPresetChanged() enables the
	// spinboxes for editing without also overwriting the values we're about
	// to set (Custom is specifically exempted from that overwrite - see its
	// own comment). The user can still freely adjust the camera afterward,
	// same as for every other scene.
	//
	// meta's camera fields are unconditionally valid here (unlike the old
	// separate SceneMetadataClient::recommendedCamera() call, which could
	// fail independently of the description lookup above) - the
	// sceneMetadata() call already returned true, and every SceneDescriptor
	// has a real camera, never an optional one.
	{
		m_cameraPresetCombo->setCurrentIndex(7);
		m_currentLookatX = meta.camLookatX;
		m_currentLookatY = meta.camLookatY;
		m_currentLookatZ = meta.camLookatZ;
		// This scene's own default viewing distance - the reference every
		// named preset in m_cameraPresetCombo scales against (see its setup
		// comment). Falls back to the Cornell-scale default (1078) if the
		// recommended camera sits exactly on its own lookat (distance 0),
		// which would otherwise collapse every preset to the lookat point.
		double dx = meta.camLookfromX - meta.camLookatX;
		double dy = meta.camLookfromY - meta.camLookatY;
		double dz = meta.camLookfromZ - meta.camLookatZ;
		double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
		m_currentSceneCamDistance = (dist > 1e-6) ? dist : 1078.0;
		m_cameraPosX->setValue(meta.camLookfromX);
		m_cameraPosY->setValue(meta.camLookfromY);
		m_cameraPosZ->setValue(meta.camLookfromZ);
		refreshCameraDistanceDisplay();

		// Scale the spinboxes' arrow-key/scroll step to this scene's scale
		// too - a fixed 10-unit step (Cornell Box's own scale) is unusably
		// coarse for e.g. scene 39's Stanford Armadillo, whose whole ~3-unit
		// model would be crossed in well under one click. 1/100th of the
		// scene's own default viewing distance keeps Cornell Box's step at
		// its original ~10.8, matching prior behavior there.
		const double step = m_currentSceneCamDistance / 100.0;
		m_cameraPosX->setSingleStep(step);
		m_cameraPosY->setSingleStep(step);
		m_cameraPosZ->setSingleStep(step);
		m_cameraDistance->setSingleStep(step);
	}
}
