// Render Options tab, Preview tab (and its sub-tab management), and Video
// Settings tab - split out of mainwindow_tabs.cpp to keep that file to the
// Basic/Advanced settings tabs and their scene-list helpers. Video Settings
// lives here too even though it sat between createRenderOptionsTab() and
// createPreviewTab() in the original file's function order - between
// Render Options and Preview, not after Progress/Log/Diagnostics as an
// earlier version of this comment incorrectly claimed (verified against
// mainwindow.cpp's actual setupUI() call order: Basic, Advanced, Render
// Options, Video Settings, Preview, Progress, Log, Diagnostics) - it's
// grouped with Render Options/Preview here by KIND (render-configuration
// tabs), which happens to match its original physical position too; see
// mainwindow_tabs_output.cpp for the Progress/Log/Diagnostics tabs.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"

#include "../src/shared/scene_descriptor.h"
#include "../src/shared/video_preset.h"

#include <QTabBar>
#include "scene_metadata_client.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
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
#include <cmath>
#include <algorithm>

// ============================================================================
// Render Options Tab
// ============================================================================
// Exposes CLI flags RenderController::start() (mainwindow.cpp) already knows
// how to emit but that no earlier tab surfaced: --sampler, --spectral,
// --exposure, --tonemap, --stats, --denoise, --optix-validate. Deliberately
// a separate tab from "Advanced Settings" above (resolution/samples/depth/
// camera) rather than folded into it, since that name already means
// something else and this tab is exclusively about render BEHAVIOR flags,
// not image/camera parameters.
void MainWindow::createRenderOptionsTab() {
	QWidget *optionsTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(optionsTab);
	layout->setSpacing(14);
	layout->setContentsMargins(12, 12, 12, 12);

	// ------------------------------------------------------------------
	// Integrator group - the algorithm selector itself, plus sub-flags for
	// whichever alternate integrator it selects. Comes first (rather than
	// after Sampling & Spectral/Output) since it's this tab's primary
	// choice, not a separate axis - everything below only applies to
	// whichever integrator is picked here.
	// ------------------------------------------------------------------
	m_integratorOptionsGroup = new QGroupBox(tr("Integrator"), optionsTab);
	styleGroupBox(m_integratorOptionsGroup);
	QVBoxLayout *integratorGroupLayout = new QVBoxLayout(m_integratorOptionsGroup);
	integratorGroupLayout->setContentsMargins(15, 22, 15, 12);

	// Which render algorithm to use instead of the default path tracer.
	// See IntegratorMode's own comment (mainwindow.h) for why this is a
	// combo (structural mutual exclusion) where the CLI itself uses 8
	// independent bool flags with hand-written guards.
	m_integratorCombo = new QComboBox(optionsTab);
	{
		// Same "(i)" mark + per-row tooltip as populateSceneCombo() gives
		// each of its own rows - icon_tint::addItem() (not a plain
		// combo->addItem()) so a theme switch's restyleThemedWidgets() ->
		// retintItems() sweep recolours these the same way every other
		// combo's icons already do, and setItemData(..., Qt::ToolTipRole)
		// so hovering a row in the OPEN dropdown shows what that specific
		// integrator does. Qt also shows the current item's icon natively
		// inside the closed combo box, so this single mechanism covers
		// both "browsing the list" and "at a glance, what's selected".
		const IntegratorMode modes[] = {
			IntegratorMode::Default, IntegratorMode::Sppm, IntegratorMode::Bdpt,
			IntegratorMode::Mlt, IntegratorMode::RandomWalk, IntegratorMode::Ao,
			IntegratorMode::SimplePath, IntegratorMode::SimpleVolPath, IntegratorMode::LightPath,
		};
		const QString labels[] = {
			tr("Path Tracer (default)"), tr("SPPM (Photon Mapping)"), tr("BDPT (Bidirectional)"),
			tr("MLT (Metropolis Light Transport)"), tr("RandomWalk (reference, unbiased)"),
			tr("Ambient Occlusion (debug)"), tr("SimplePath (reference)"),
			tr("SimpleVolPath (reference, volumetric)"), tr("LightPath (light tracer)"),
		};
		for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
			icon_tint::addItem(m_integratorCombo, ":/icons/info.svg", labels[i],
				static_cast<int>(modes[i]), m_activeTheme.textBody);
			m_integratorCombo->setItemData(m_integratorCombo->count() - 1,
				wrapTooltipHtml(integratorDescription(modes[i])), Qt::ToolTipRole);
		}
	}
	styleComboBox(m_integratorCombo);
	m_integratorCombo->setToolTip(
		tr("Which rendering algorithm to use. Path Tracer (the default) is the\n"
		"well-tested, general-purpose choice - the alternates below trade\n"
		"generality for a specific technique (photon mapping, bidirectional/\n"
		"Metropolis light transport, or a handful of unbiased reference and\n"
		"debug integrators). All alternates are CPU-only except SPPM, and none\n"
		"can be combined with Generate Video mode.\n\n"
		"Sampler/Spectral/Exposure/Tonemap/Stats below only affect the\n"
		"default Path Tracer - see each control's own tooltip."));
	connect(m_integratorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onIntegratorChanged);
	QFormLayout *integratorSelectorLayout = new QFormLayout();
	integratorSelectorLayout->setVerticalSpacing(10);
	integratorSelectorLayout->setHorizontalSpacing(10);
	// Same two-column labelWithInfo() row shape every sibling control on
	// the Basic Settings tab uses (Renderer/GPU Backend/Quality/
	// Resolution), so the label column stays aligned across all of them. A
	// SEPARATE dynamic icon here (an earlier version of this row) turned
	// out to duplicate the per-item icon Qt already shows natively inside
	// the closed combo box (every item now carries its own "(i)" +
	// tooltip, added above via icon_tint::addItem/setItemData(Qt::ToolTipRole))
	// - that already covers "what does the currently-selected integrator
	// do" without a second, differently-aligned icon widget.
	integratorSelectorLayout->addRow(labelWithInfo(tr("Integrator:"),
		tr("The rendering algorithm itself, not just how fast it runs. Path "
		"Tracer (the default) is the general-purpose, well-tested choice "
		"used everywhere else in this app.\n\n"
		"SPPM (Stochastic Progressive Photon Mapping) handles hard caustics/"
		"glass scenes path tracing struggles with. BDPT and MLT (built on "
		"BDPT) trace light paths from both the camera and the light source "
		"and connect them - better for some difficult lighting, area lights "
		"only. RandomWalk, Ambient Occlusion, SimplePath, SimpleVolPath, "
		"and LightPath are reference/debug integrators - simpler, often "
		"noisier or narrower in scope (e.g. Ambient Occlusion isn't a lit "
		"render at all), useful for isolating what a specific technique "
		"contributes.\n\nHover any item in the dropdown for details on that "
		"specific integrator.")),
		m_integratorCombo);
	integratorGroupLayout->addLayout(integratorSelectorLayout);

	m_integratorVideoWarningLabel = new QLabel(
		tr("⚠ Generate Video cannot be combined with an alternate integrator - "
		"switch back to Path Tracer, or to Single Image output."), optionsTab);
	m_integratorVideoWarningLabel->setObjectName("statusWarning");
	m_integratorVideoWarningLabel->setWordWrap(true);
	m_integratorVideoWarningLabel->setVisible(false);
	integratorGroupLayout->addWidget(m_integratorVideoWarningLabel);

	m_integratorOptionsStack = new CurrentPageSizedStackedWidget(m_integratorOptionsGroup);

	// Prepends a word-wrapped description of `mode` as the first (full-
	// width) row of `pageLayout` - reuses the exact same text the per-item
	// combo tooltips already show (integratorDescription(),
	// mainwindow_style.cpp), so SPPM/BDPT/MLT/AO/SimplePath's pages get the
	// same persistent, always-visible explanation the shared placeholder
	// page (Default/RandomWalk/SimpleVolPath/LightPath, below) already has
	// via m_integratorNoOptionsLabel - not just a tooltip you'd only see by
	// opening the dropdown and hovering.
	auto addIntegratorDescription = [this](QFormLayout *pageLayout, IntegratorMode mode) {
		QLabel *desc = new QLabel(integratorDescription(mode));
		desc->setWordWrap(true);
		pageLayout->addRow(desc);
	};

	// Page 0: shared placeholder for Default/RandomWalk/SimpleVolPath/
	// LightPath - none of these four have any sub-flags, so they share
	// one page instead of each getting a near-duplicate empty one. Text
	// is swapped per-mode in onIntegratorChanged() (mainwindow_slots.cpp).
	m_integratorNoOptionsLabel = new QLabel(tr("The default Path Tracer has no integrator-specific options here - see the Render Options above."), m_integratorOptionsStack);
	m_integratorNoOptionsLabel->setWordWrap(true);
	m_integratorOptionsStack->addWidget(m_integratorNoOptionsLabel);

	// Page 1: SPPM
	QWidget *sppmPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *sppmLayout = new QFormLayout(sppmPage);
	sppmLayout->setVerticalSpacing(10);
	sppmLayout->setHorizontalSpacing(10);
	addIntegratorDescription(sppmLayout, IntegratorMode::Sppm);
	m_sppmIterationsSpin = new QSpinBox(sppmPage);
	m_sppmIterationsSpin->setRange(1, 1000000);
	m_sppmIterationsSpin->setValue(100);
	styleSpinBox(m_sppmIterationsSpin);
	sppmLayout->addRow(labelWithInfo(tr("Iterations:"),
		tr("How many camera-pass + photon-pass rounds SPPM runs. More "
		"iterations converge to a cleaner result, at a roughly linear "
		"cost in render time.")),
		m_sppmIterationsSpin);
	m_sppmPhotonsSpin = new QSpinBox(sppmPage);
	m_sppmPhotonsSpin->setRange(1, 100000000);
	m_sppmPhotonsSpin->setValue(5000);
	styleSpinBox(m_sppmPhotonsSpin);
	sppmLayout->addRow(labelWithInfo(tr("Photons per iteration:"),
		tr("How many photons are shot from the lights each iteration. "
		"More photons reduce noise in indirect/caustic lighting at the "
		"cost of a slower photon pass.")),
		m_sppmPhotonsSpin);
	m_integratorOptionsStack->addWidget(sppmPage);

	// Page 2: BDPT
	QWidget *bdptPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *bdptLayout = new QFormLayout(bdptPage);
	bdptLayout->setVerticalSpacing(10);
	bdptLayout->setHorizontalSpacing(10);
	addIntegratorDescription(bdptLayout, IntegratorMode::Bdpt);
	m_bdptMaxDepthSpin = new QSpinBox(bdptPage);
	m_bdptMaxDepthSpin->setRange(1, 100);
	m_bdptMaxDepthSpin->setValue(5);
	styleSpinBox(m_bdptMaxDepthSpin);
	bdptLayout->addRow(labelWithInfo(tr("Max path depth:"),
		tr("Maximum bounces for each of the two subpaths (camera side and "
		"light side) that BDPT connects together.")),
		m_bdptMaxDepthSpin);
	m_integratorOptionsStack->addWidget(bdptPage);

	// Page 3: MLT
	QWidget *mltPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *mltLayout = new QFormLayout(mltPage);
	mltLayout->setVerticalSpacing(10);
	mltLayout->setHorizontalSpacing(10);
	addIntegratorDescription(mltLayout, IntegratorMode::Mlt);
	m_mltBootstrapSpin = new QSpinBox(mltPage);
	m_mltBootstrapSpin->setRange(1, 10000000);
	m_mltBootstrapSpin->setValue(100000);
	styleSpinBox(m_mltBootstrapSpin);
	mltLayout->addRow(labelWithInfo(tr("Bootstrap samples:"),
		tr("How many candidate light paths MLT samples up front, per "
		"depth, to seed its Markov chains - more gives a better-informed "
		"starting distribution.")),
		m_mltBootstrapSpin);
	m_mltMutationsSpin = new QSpinBox(mltPage);
	m_mltMutationsSpin->setRange(1, 1000000000);
	m_mltMutationsSpin->setValue(4000000);
	styleSpinBox(m_mltMutationsSpin);
	mltLayout->addRow(labelWithInfo(tr("Mutations:"),
		tr("Total Metropolis mutations across all chains combined - the "
		"main knob for render time/quality, analogous to samples per "
		"pixel in the default path tracer.")),
		m_mltMutationsSpin);
	m_mltMaxDepthSpin = new QSpinBox(mltPage);
	m_mltMaxDepthSpin->setRange(1, 100);
	m_mltMaxDepthSpin->setValue(5);
	styleSpinBox(m_mltMaxDepthSpin);
	mltLayout->addRow(labelWithInfo(tr("Max path depth:"),
		tr("Same meaning as BDPT's max path depth (MLT is built directly "
		"on BDPT's subpath machinery).")),
		m_mltMaxDepthSpin);
	m_integratorOptionsStack->addWidget(mltPage);

	// Page 4: Ambient Occlusion
	QWidget *aoPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *aoLayout = new QFormLayout(aoPage);
	aoLayout->setVerticalSpacing(10);
	aoLayout->setHorizontalSpacing(10);
	addIntegratorDescription(aoLayout, IntegratorMode::Ao);
	m_aoMaxDistSpin = new QDoubleSpinBox(aoPage);
	m_aoMaxDistSpin->setRange(0.01, 1.0e12);
	m_aoMaxDistSpin->setDecimals(2);
	m_aoMaxDistSpin->setValue(1.0e10);
	styleSpinBox(m_aoMaxDistSpin);
	aoLayout->addRow(labelWithInfo(tr("Max occlusion distance:"),
		tr("How far an occlusion test ray can reach before counting as "
		"unoccluded. The default (10 billion) is effectively unbounded - "
		"lower it to only count nearby geometry as occluding.")),
		m_aoMaxDistSpin);
	m_aoUniformCheck = new QCheckBox(tr("Uniform-hemisphere sampling (instead of cosine)"), aoPage);
	styleCheckBox(m_aoUniformCheck);
	aoLayout->addRow(checkboxWithInfo(m_aoUniformCheck,
		tr("The default samples occlusion rays weighted toward the "
		"surface normal (cosine-hemisphere), matching how a Lambertian "
		"surface would actually be lit. Uniform-hemisphere spreads "
		"samples evenly instead - a different, unweighted estimator.")));
	m_aoIllumScaleSpin = new QDoubleSpinBox(aoPage);
	m_aoIllumScaleSpin->setRange(0.0, 1000.0);
	m_aoIllumScaleSpin->setValue(1.0);
	styleSpinBox(m_aoIllumScaleSpin);
	aoLayout->addRow(labelWithInfo(tr("Illumination scale:"),
		tr("Flat multiplier on the occlusion color below.")),
		m_aoIllumScaleSpin);
	QWidget *aoIllumRgbRow = new QWidget(aoPage);
	QHBoxLayout *aoIllumRgbLayout = new QHBoxLayout(aoIllumRgbRow);
	aoIllumRgbLayout->setContentsMargins(0, 0, 0, 0);
	aoIllumRgbLayout->setSpacing(6);
	m_aoIllumRSpin = new QDoubleSpinBox(aoIllumRgbRow);
	m_aoIllumGSpin = new QDoubleSpinBox(aoIllumRgbRow);
	m_aoIllumBSpin = new QDoubleSpinBox(aoIllumRgbRow);
	for (QDoubleSpinBox *spin : {m_aoIllumRSpin, m_aoIllumGSpin, m_aoIllumBSpin}) {
		spin->setRange(0.0, 1.0);
		spin->setSingleStep(0.05);
		spin->setValue(1.0);
		styleSpinBox(spin);
		aoIllumRgbLayout->addWidget(spin);
	}
	aoLayout->addRow(labelWithInfo(tr("Occlusion color (R, G, B):"),
		tr("The color ambient occlusion is visualized in - not a lit "
		"render, so this is a visualization choice, not a light color. "
		"Default is white (1, 1, 1).")),
		aoIllumRgbRow);
	m_integratorOptionsStack->addWidget(aoPage);

	// Page 5: SimplePath
	QWidget *simplepathPage = new QWidget(m_integratorOptionsStack);
	QFormLayout *simplepathLayout = new QFormLayout(simplepathPage);
	simplepathLayout->setVerticalSpacing(10);
	simplepathLayout->setHorizontalSpacing(10);
	addIntegratorDescription(simplepathLayout, IntegratorMode::SimplePath);
	m_simplepathNoLightsCheck = new QCheckBox(tr("Disable next-event estimation (direct light sampling)"), simplepathPage);
	styleCheckBox(m_simplepathNoLightsCheck);
	simplepathLayout->addRow(checkboxWithInfo(m_simplepathNoLightsCheck,
		tr("On by default. Direct light sampling explicitly aims shadow "
		"rays at lights each bounce, sharply reducing noise on scenes "
		"with small/bright lights. Disabling it falls back to finding "
		"lights only by chance, the way a purely unbiased path tracer "
		"would.")));
	m_simplepathNoBsdfCheck = new QCheckBox(tr("Disable BSDF importance sampling"), simplepathPage);
	styleCheckBox(m_simplepathNoBsdfCheck);
	simplepathLayout->addRow(checkboxWithInfo(m_simplepathNoBsdfCheck,
		tr("On by default. Samples each bounce's new direction weighted "
		"toward where the surface's material actually reflects light. "
		"Disabling it falls back to uniform hemisphere sampling.")));
	m_integratorOptionsStack->addWidget(simplepathPage);

	integratorGroupLayout->addWidget(m_integratorOptionsStack);
	layout->addWidget(m_integratorOptionsGroup);

	// ------------------------------------------------------------------
	// Sampling & Spectral group - CPU default path tracer only
	// ------------------------------------------------------------------
	// "&&" (not "&") - a single "&" is a Qt mnemonic-accelerator marker,
	// which would eat the "&" and underline the next letter instead of
	// showing a literal ampersand.
	QGroupBox *samplingGroup = new QGroupBox(tr("Sampling && Spectral"), optionsTab);
	styleGroupBox(samplingGroup);
	QFormLayout *samplingLayout = new QFormLayout(samplingGroup);
	samplingLayout->setVerticalSpacing(10);
	samplingLayout->setHorizontalSpacing(10);
	samplingLayout->setContentsMargins(15, 22, 15, 12);

	m_samplerCombo = new QComboBox(optionsTab);
	// Same "(i)" mark + per-row tooltip as m_integratorCombo's own items -
	// see that combo's construction comment for why icon_tint::addItem()
	// (not a plain addItem()) plus setItemData(Qt::ToolTipRole).
	{
		struct SamplerEntry { QString label; QString value; QString tooltip; };
		const SamplerEntry entries[] = {
			{tr("Sobol (default)"), QString(), tr(
				"A low-discrepancy sequence based on Sobol sequences, "
				"scrambled per pixel. The best general-purpose default - "
				"fast convergence with no visible structure.")},
			{tr("Z-Sobol"), QStringLiteral("zsobol"), tr(
				"A variant of Sobol reordered along a Morton (Z-order) "
				"curve. Converges at least as well as plain Sobol, with "
				"better behavior under adaptive/progressive sampling.")},
			{tr("Padded Sobol"), QStringLiteral("paddedsobol"), tr(
				"Sobol sequence with extra padding dimensions, avoiding "
				"correlation artifacts when a pixel needs more random "
				"dimensions than base Sobol comfortably covers (e.g. "
				"paths with many bounces).")},
			{tr("Stratified"), QStringLiteral("stratified"), tr(
				"Splits each pixel into a grid of sub-cells and takes one "
				"sample per cell. Simple, predictable coverage - less "
				"sophisticated than Sobol/Halton, but useful as a "
				"reference/comparison sampler.")},
			{tr("PMJ02BN"), QStringLiteral("pmj02bn"), tr(
				"Progressive multi-jittered (0,2) sequence with blue-noise "
				"ordering. Especially even spatial (blue-noise) "
				"distribution of samples across neighboring pixels.")},
			{tr("Halton"), QStringLiteral("halton"), tr(
				"A classic low-discrepancy sequence built from a "
				"different prime base per dimension. Well-tested, avoids "
				"the axis-aligned clustering plain stratified sampling "
				"can show.")},
			{tr("Independent (no stratification)"), QStringLiteral("independent"), tr(
				"Plain uncorrelated pseudo-random numbers, no low-"
				"discrepancy structure at all. Included for fidelity to a "
				"loaded .pbrt scene's own Sampler directive, not a "
				"recommended choice for its own sake.")},
		};
		for (const SamplerEntry &entry : entries) {
			icon_tint::addItem(m_samplerCombo, ":/icons/info.svg", entry.label, entry.value, m_activeTheme.textBody);
			m_samplerCombo->setItemData(m_samplerCombo->count() - 1, wrapTooltipHtml(entry.tooltip), Qt::ToolTipRole);
		}
	}
	m_samplerCombo->setToolTip(
		tr("Which sampler drives random decisions (all but Independent are\n"
		"low-discrepancy). CPU default path tracer only - no effect on GPU\n"
		"or under BDPT/MLT/SPPM/the debug integrators."));
	styleComboBox(m_samplerCombo);
	samplingLayout->addRow(labelWithInfo(tr("Sampler:"),
		tr("Ray tracing needs a lot of random numbers - which direction to "
		"bounce a ray, which point on a light to sample, and so on - and "
		"HOW those \"random\" numbers are generated changes how quickly "
		"the image converges to a clean result.\n\n"
		"A naive random-number generator clusters and leaves gaps; most "
		"samplers here (Sobol, Halton, etc.) are low-discrepancy sequences, "
		"deliberately spread out to cover the sampling space more evenly, "
		"which converges to a clean image faster than true randomness "
		"would for the same sample count. Independent is the exception - "
		"plain uncorrelated random numbers, included for fidelity to a "
		"loaded .pbrt scene's own Sampler directive rather than as a "
		"recommended choice.\n\n"
		"Grayed out? This only affects the CPU renderer's default path "
		"tracer - switch Renderer to CPU on the Basic Settings tab to "
		"use it.")),
		m_samplerCombo);
	connect(m_samplerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		if (m_sceneCombo) updateSceneRecommendedSettingsHint(m_sceneCombo->currentData().toString());
	});

	// Same "(i)" mark + per-row tooltip pattern as m_samplerCombo just
	// above - see that combo's construction comment.
	m_lightSamplerCombo = new QComboBox(optionsTab);
	{
		struct LightSamplerEntry { QString label; QString value; QString tooltip; };
		const LightSamplerEntry entries[] = {
			{tr("BVH (default)"), QString(), tr(
				"Builds a spatial hierarchy over the scene's lights and "
				"weighs each one by both power and proximity to the "
				"shading point, adapting per bounce rather than using one "
				"global weighting. pbrt-v4's own default - generally the "
				"best convergence, at a small extra bookkeeping cost.")},
			{tr("Power"), QStringLiteral("power"), tr(
				"Picks a light with probability weighted by its total "
				"emitted power - bright lights get sampled more often "
				"than dim ones. Converges faster than uniform in scenes "
				"with a wide range of light brightness, but ignores "
				"distance and occlusion.")},
			{tr("Uniform"), QStringLiteral("uniform"), tr(
				"Picks a light uniformly at random from every light in "
				"the scene, regardless of how bright or how far away it "
				"is. Simple and unbiased, but converges slowly in scenes "
				"with many lights of very different brightness - a dim "
				"light gets sampled just as often as a bright one.")},
		};
		for (const LightSamplerEntry &entry : entries) {
			icon_tint::addItem(m_lightSamplerCombo, ":/icons/info.svg", entry.label, entry.value, m_activeTheme.textBody);
			m_lightSamplerCombo->setItemData(m_lightSamplerCombo->count() - 1, wrapTooltipHtml(entry.tooltip), Qt::ToolTipRole);
		}
	}
	m_lightSamplerCombo->setToolTip(
		tr("Which strategy picks the light to sample at each next-event-\n"
		"estimation bounce. Affects noise/convergence speed, not the\n"
		"converged image. CPU default path tracer only - no effect on GPU\n"
		"or under BDPT/MLT/SPPM/the debug integrators."));
	styleComboBox(m_lightSamplerCombo);
	samplingLayout->addRow(labelWithInfo(tr("Light Sampler:"),
		tr("Every diffuse/glossy bounce needs to pick ONE light (out of "
		"potentially many) to sample directly for next-event estimation "
		"- which light gets picked, and how fairly, changes how quickly "
		"the image converges, though never what it converges TO.\n\n"
		"BVH (the default, matching pbrt-v4 itself) builds a spatial "
		"hierarchy over the scene's lights and adapts its weighting per "
		"shading point - both bright AND nearby lights get preferred. "
		"Power picks by brightness alone, ignoring position - simpler, "
		"worse in scenes where light distance varies a lot. Uniform "
		"ignores both - every light equally likely regardless of "
		"brightness or distance, included mainly for comparison/"
		"debugging.\n\nGrayed out? This only affects the CPU renderer's "
		"default path tracer - switch Renderer to CPU on the Basic "
		"Settings tab to use it.")),
		m_lightSamplerCombo);
	connect(m_lightSamplerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		if (m_sceneCombo) updateSceneRecommendedSettingsHint(m_sceneCombo->currentData().toString());
	});

	m_spectralCheck = new QCheckBox(tr("Spectral rendering (--spectral)"), optionsTab);
	m_spectralCheck->setToolTip(
		tr("Real hero-wavelength spectral rendering instead of flat RGB.\n"
		"CPU default path tracer only. Only lambertian, metal, dielectric,\n"
		"rough_dielectric, conductor, and diffuse_light materials are\n"
		"supported - a scene using anything else fails to render rather\n"
		"than silently rendering wrong colors. Noticeably slower per-sample."));
	styleCheckBox(m_spectralCheck);
	samplingLayout->addRow(checkboxWithInfo(m_spectralCheck,
		tr("Ordinary rendering tracks light as three numbers - red, "
		"green, blue - the same way a screen displays color.\n\n"
		"Real light is a continuous spectrum of wavelengths, and a "
		"few physical effects (like a prism splitting white light "
		"into a rainbow) only happen because different wavelengths "
		"refract by different amounts - RGB alone can't represent "
		"that. Spectral rendering tracks a handful of actual "
		"wavelengths per ray instead of just RGB, at the cost of "
		"being noisier and slower per sample.\n\n"
		"Grayed out? This only exists on the CPU renderer's default "
		"path tracer - switch Renderer to CPU on the Basic Settings "
		"tab to use it.")));

	m_exposureSpin = new QDoubleSpinBox(optionsTab);
	m_exposureSpin->setRange(0.01, 100.0);
	m_exposureSpin->setValue(1.0);
	m_exposureSpin->setSingleStep(0.1);
	m_exposureSpin->setToolTip(
		tr("Flat multiplier on linear color before tone-mapping (1.0 = no-op).\n"
		"Both CPU and GPU default path tracer only."));
	styleSpinBox(m_exposureSpin);
	samplingLayout->addRow(labelWithInfo(tr("Exposure:"),
		tr("A flat brightness multiplier applied to the whole image, the "
		"same knob a camera's exposure setting is.\n\n"
		"1.0 leaves the image unchanged; below 1.0 darkens it, above 1.0 "
		"brightens it - useful for a scene that's rendering correctly "
		"but is just too dark or too bright to see clearly, without "
		"changing any actual light in the scene.")),
		m_exposureSpin);

	m_regularizeCheck = new QCheckBox(tr("Path regularization (--regularize)"), optionsTab);
	m_regularizeCheck->setToolTip(
		tr("Widens a rough BSDF's GGX alpha after the path's first\n"
		"non-specular bounce - tames fireflies from hard caustic paths,\n"
		"at the cost of some blur. Both CPU and GPU default path tracer\n"
		"only. A scene that already requests this itself is unaffected -\n"
		"this checkbox only ever adds the request, never removes it."));
	styleCheckBox(m_regularizeCheck);
	samplingLayout->addRow(checkboxWithInfo(m_regularizeCheck,
		tr("Some light paths are genuinely hard for a path tracer to find "
		"cleanly - light that bounces off a rough (but not perfectly "
		"specular) surface, through another rough surface, into a small "
		"bright light. Those paths show up as bright, isolated speckle "
		"(\"fireflies\") that take a very long time to average away.\n\n"
		"Path regularization deliberately blurs a surface's roughness a "
		"little more with each non-specular bounce a path has already "
		"taken - it introduces a small bias (technically a wrong "
		"answer), but in exchange fireflies convergence dramatically "
		"faster, which is usually the better trade for how a render "
		"actually looks.\n\n"
		"Off by default, matching pbrt-v4's own default. If a loaded "
		".pbrt scene's file already requests this itself, it's applied "
		"either way - this checkbox can only add the request on top, "
		"never take it away.")));

	m_maxComponentValueCheck = new QCheckBox(tr("Firefly clamp (--maxcomponentvalue)"), optionsTab);
	m_maxComponentValueCheck->setToolTip(
		tr("Clamps any pixel sample whose brightest channel exceeds the\n"
		"value below, scaling all channels down together to preserve hue.\n"
		"CPU and both GPU backends - recursive matches CPU exactly,\n"
		"wavefront approximates it per-contribution rather than\n"
		"per-sample-total."));
	styleCheckBox(m_maxComponentValueCheck);
	m_maxComponentValueSpin = new QDoubleSpinBox(optionsTab);
	// Range/step/default chosen for a typical 0-a-few-dozen linear-light
	// scene, not the CLI's own "1e9 = unbounded" sentinel - showing that
	// literal value in a spinbox would read as a bug, not "off". 10.0 is a
	// commonly-cited reasonable starting clamp for a first attempt; the
	// checkbox itself (not a magic spinbox value) is what actually decides
	// whether --maxcomponentvalue is emitted at all - see the connect()
	// lambda just below, which enables/disables this spinbox with the
	// checkbox.
	m_maxComponentValueSpin->setRange(0.01, 10000.0);
	m_maxComponentValueSpin->setValue(10.0);
	m_maxComponentValueSpin->setSingleStep(1.0);
	m_maxComponentValueSpin->setEnabled(false);
	m_maxComponentValueSpin->setToolTip(m_maxComponentValueCheck->toolTip());
	styleSpinBox(m_maxComponentValueSpin);
	connect(m_maxComponentValueCheck, &QCheckBox::toggled, m_maxComponentValueSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *clampRow = new QWidget(optionsTab);
		QHBoxLayout *clampRowLayout = new QHBoxLayout(clampRow);
		clampRowLayout->setContentsMargins(0, 0, 0, 0);
		clampRowLayout->addWidget(m_maxComponentValueCheck);
		clampRowLayout->addWidget(m_maxComponentValueSpin, 1);
		samplingLayout->addRow(checkboxWithInfo(m_maxComponentValueCheck,
			tr("A Monte Carlo path tracer occasionally samples a path that's "
			"individually correct but extremely bright - a ray that happens "
			"to graze a small, intense light at just the right angle - and "
			"one such sample can dominate a pixel's average for a long "
			"time before enough other samples arrive to smooth it out. "
			"These show up as bright, isolated speckle (\"fireflies\").\n\n"
			"This clamp caps how bright any single sample's brightest "
			"channel is allowed to be before it's averaged in, trading a "
			"small, controlled bias for a dramatically cleaner-looking "
			"image at the same sample count - lower values clean up more "
			"aggressively but risk visibly dimming genuinely bright small "
			"lights, not just outlier noise.\n\n"
			"Off by default (effectively unbounded, matching pbrt-v4's own "
			"default). CPU default path tracer only.")), clampRow);
	}

	layout->addWidget(samplingGroup);

	// ------------------------------------------------------------------
	// Output group
	// ------------------------------------------------------------------
	QGroupBox *outputGroup = new QGroupBox(tr("Output"), optionsTab);
	styleGroupBox(outputGroup);
	QFormLayout *outputLayout = new QFormLayout(outputGroup);
	outputLayout->setVerticalSpacing(10);
	outputLayout->setHorizontalSpacing(10);
	outputLayout->setContentsMargins(15, 22, 15, 12);

	m_tonemapCombo = new QComboBox(optionsTab);
	m_tonemapCombo->addItem(tr("ACES (default)"), QString());
	m_tonemapCombo->addItem(tr("Reinhard"), QStringLiteral("reinhard"));
	m_tonemapCombo->addItem(tr("None"), QStringLiteral("none"));
	m_tonemapCombo->setToolTip(
		tr("Which tone-mapping operator to apply before the sRGB curve.\n"
		"Applies to both CPU and GPU (recursive and wavefront) - no\n"
		"effect under BDPT/MLT/SPPM/the debug integrators."));
	styleComboBox(m_tonemapCombo);
	outputLayout->addRow(labelWithInfo(tr("Tone mapping:"),
		tr("A raytraced scene's true brightness values are unbounded - a "
		"light bulb might be a hundred times brighter than a wall - but "
		"a screen can only display a fixed range. Tone mapping is the "
		"curve that compresses that huge range down into something "
		"displayable.\n\n"
		"ACES rolls off bright highlights gently, the way film does; "
		"Reinhard is a simpler, older compression; None just clips "
		"anything too bright to flat white, which can look harsh.")),
		m_tonemapCombo);

	m_statsCheck = new QCheckBox(tr("Print render stats"), optionsTab);
	m_statsCheck->setToolTip(
		tr("Print a small end-of-render stats block (rays cast, bounces,\n"
		"shadow rays, samples/sec) to the Log tab. Observation-only -\n"
		"never changes the rendered image."));
	styleCheckBox(m_statsCheck);
	outputLayout->addRow(checkboxWithInfo(m_statsCheck,
		tr("Prints a short summary after the render finishes - how many "
		"rays were cast, how many bounces happened, how many shadow "
		"rays were traced, and samples per second.\n\n"
		"Purely informational: it never changes the rendered image, "
		"just tells you what the renderer actually did.")));

	m_denoiseCheck = new QCheckBox(tr("OptiX AI denoiser (GPU only)"), optionsTab);
	m_denoiseCheck->setToolTip(
		tr("Run the OptiX AI denoiser on the finished render, guided by\n"
		"albedo + normal buffers. GPU only, both backends (recursive\n"
		"and wavefront each have their own denoiser)."));
	styleCheckBox(m_denoiseCheck);
	outputLayout->addRow(checkboxWithInfo(m_denoiseCheck,
		tr("Ray tracing is noisy by nature - low sample counts leave a "
		"grainy, speckled image, which is why more samples usually "
		"means a cleaner picture.\n\n"
		"A denoiser is a machine-learning model trained to recognize "
		"that speckle pattern and smooth it away after the fact, "
		"without needing to trace additional rays - a way to get a "
		"clean-looking image faster, at some cost in fine detail.\n\n"
		"Grayed out? This needs the GPU recursive backend - switch "
		"Renderer to GPU (and GPU Backend to Recursive) on the Basic "
		"Settings tab to use it.")));

	m_optixValidateCheck = new QCheckBox(tr("OptiX validation mode (slower, debugging only)"), optionsTab);
	m_optixValidateCheck->setToolTip(
		tr("Enable OptiX validation mode - extra device-side checks with a\n"
		"real per-launch cost. GPU only, for debugging, not routine use."));
	styleCheckBox(m_optixValidateCheck);
	outputLayout->addRow(checkboxWithInfo(m_optixValidateCheck,
		tr("Turns on extra correctness checks inside the GPU ray-tracing "
		"pipeline itself, catching certain classes of bugs that would "
		"otherwise silently produce a wrong image or crash "
		"unpredictably.\n\n"
		"It's a debugging aid for people working on the renderer's "
		"own GPU code, not something a normal render benefits from - "
		"it has a real performance cost and doesn't change what a "
		"correct render looks like.\n\n"
		"Grayed out? This is GPU-only - switch Renderer to GPU on "
		"the Basic Settings tab to use it.")));

	layout->addWidget(outputGroup);

	// ------------------------------------------------------------------
	// Crop Window group
	// ------------------------------------------------------------------
	QGroupBox *cropGroup = new QGroupBox(tr("Crop Window"), optionsTab);
	styleGroupBox(cropGroup);
	QFormLayout *cropLayout = new QFormLayout(cropGroup);
	cropLayout->setVerticalSpacing(10);
	cropLayout->setHorizontalSpacing(10);
	cropLayout->setContentsMargins(15, 22, 15, 12);

	m_cropCheck = new QCheckBox(tr("Render only part of the frame (--crop)"), optionsTab);
	m_cropCheck->setToolTip(
		tr("Restricts rendering to a rectangle of the frame, given as\n"
		"fractions of the full image in [0,1]. Both CPU and GPU default\n"
		"path tracer only."));
	styleCheckBox(m_cropCheck);
	cropLayout->addRow(checkboxWithInfo(m_cropCheck,
		tr("Renders only a rectangular slice of the full frame - "
		"everything outside it is left black - instead of the whole "
		"image. The rectangle is given as four fractions of the full "
		"frame width/height, from 0 (left/top edge) to 1 (right/bottom "
		"edge), so it stays the same shape regardless of resolution.\n\n"
		"Useful for iterating faster on one troublesome part of a large, "
		"slow scene - the same total sample count converges much faster "
		"when it only has to cover a corner of the frame instead of the "
		"whole thing.\n\n"
		"Off by default (the full frame). If a loaded .pbrt scene's file "
		"already requests its own cropwindow/pixelbounds, checking this "
		"overrides it with the rectangle below; leaving it unchecked "
		"lets the scene's own request (if any) stand.")));

	const auto makeCropSpin = [this, optionsTab]() {
		QDoubleSpinBox *spin = new QDoubleSpinBox(optionsTab);
		spin->setRange(0.0, 1.0);
		spin->setSingleStep(0.05);
		spin->setDecimals(2);
		spin->setEnabled(false);
		styleSpinBox(spin);
		return spin;
	};
	m_cropX0Spin = makeCropSpin();
	m_cropX0Spin->setValue(0.0);
	m_cropY0Spin = makeCropSpin();
	m_cropY0Spin->setValue(0.0);
	m_cropX1Spin = makeCropSpin();
	m_cropX1Spin->setValue(1.0);
	m_cropY1Spin = makeCropSpin();
	m_cropY1Spin->setValue(1.0);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropX0Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropY0Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropX1Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropY1Spin, &QDoubleSpinBox::setEnabled);
	cropLayout->addRow(tr("Left (X0):"), m_cropX0Spin);
	cropLayout->addRow(tr("Top (Y0):"), m_cropY0Spin);
	cropLayout->addRow(tr("Right (X1):"), m_cropX1Spin);
	cropLayout->addRow(tr("Bottom (Y1):"), m_cropY1Spin);

	layout->addWidget(cropGroup);

	// ------------------------------------------------------------------
	// Seed group
	// ------------------------------------------------------------------
	QGroupBox *seedGroup = new QGroupBox(tr("Reproducibility"), optionsTab);
	styleGroupBox(seedGroup);
	QFormLayout *seedLayout = new QFormLayout(seedGroup);
	seedLayout->setVerticalSpacing(10);
	seedLayout->setHorizontalSpacing(10);
	seedLayout->setContentsMargins(15, 22, 15, 12);

	m_seedCheck = new QCheckBox(tr("Reproducible render (--seed)"), optionsTab);
	m_seedCheck->setToolTip(
		tr("Makes this render reproduce byte-for-byte on a rerun with the\n"
		"same seed. Both CPU and GPU default path tracer only."));
	styleCheckBox(m_seedCheck);
	seedLayout->addRow(checkboxWithInfo(m_seedCheck,
		tr("Renders normally use a different random sequence every time, "
		"so two runs of the same scene never match pixel-for-pixel even "
		"with identical settings. Checking this fixes the random seed, so "
		"the same seed value always reproduces the exact same image - "
		"useful for comparing before/after a scene edit, or for isolating "
		"whether a visual difference came from a code change or just "
		"random noise.\n\n"
		"Off by default (genuinely random every render).")));

	m_seedSpin = new QSpinBox(optionsTab);
	m_seedSpin->setRange(0, 2147483647);
	m_seedSpin->setValue(0);
	m_seedSpin->setEnabled(false);
	m_seedSpin->setToolTip(m_seedCheck->toolTip());
	styleSpinBox(m_seedSpin);
	connect(m_seedCheck, &QCheckBox::toggled, m_seedSpin, &QSpinBox::setEnabled);
	seedLayout->addRow(tr("Seed:"), m_seedSpin);

	layout->addWidget(seedGroup);
	layout->addStretch();

	// Initial enabled state matches whatever m_renderModeCombo/
	// m_gpuBackendCombo/m_integratorCombo already hold at this point in
	// construction - updateRenderOptionsEnabled() is the single source of
	// truth for this, also called live from each of those combos' own
	// change handlers (mainwindow.cpp/mainwindow_slots.cpp).
	updateRenderOptionsEnabled();

	ThemedScrollArea *scrollArea = new ThemedScrollArea();  // theme motif support - see that class's own comment
	scrollArea->setWidget(optionsTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	scrollArea->setObjectName("tabScroll");
	scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
	scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

	m_tabWidget->addTab(scrollArea, tr("Render Options"));
}

// Single source of truth for m_samplerCombo/m_lightSamplerCombo/
// m_spectralCheck/m_exposureSpin/
// m_tonemapCombo/m_statsCheck/m_denoiseCheck/m_optixValidateCheck/
// m_seedCheck/m_gpuBackendCombo's enabled state, replacing what used to be a
// hand-duplicated 2-input (GPU/CPU x wavefront/recursive) condition
// copy-pasted at construction and in two separate connect() lambdas -
// adding Integrator as a third input would have made that a 3-site
// hand-copy of an increasingly complex condition, so it's factored here
// instead and called from all four controls' own change handlers
// (construction, m_renderModeCombo's lambda, m_gpuBackendCombo's lambda,
// onIntegratorChanged() - mainwindow.cpp/mainwindow_slots.cpp).
void MainWindow::updateRenderOptionsEnabled() {
	const bool gpuSelected = m_renderModeCombo->currentData().toBool();
	const auto integrator = static_cast<IntegratorMode>(m_integratorCombo->currentData().toInt());
	const bool isDefault = (integrator == IntegratorMode::Default);

	if (m_gpuBackendCombo) m_gpuBackendCombo->setEnabled(isDefault && gpuSelected);
	m_samplerCombo->setEnabled(isDefault && !gpuSelected);
	m_lightSamplerCombo->setEnabled(isDefault && !gpuSelected);
	m_spectralCheck->setEnabled(isDefault && !gpuSelected);
	m_exposureSpin->setEnabled(isDefault);
	m_tonemapCombo->setEnabled(isDefault);
	m_statsCheck->setEnabled(isDefault);
	// Both GPU backends have their own real denoiser now (WavefrontPathTracer::
	// denoise(), gpu/optix/wavefront_path_tracer.cpp) - no longer gated on
	// !wavefrontSelected.
	m_denoiseCheck->setEnabled(isDefault && gpuSelected);
	m_optixValidateCheck->setEnabled(isDefault && gpuSelected);
	m_regularizeCheck->setEnabled(isDefault);
	// Both GPU backends have real maxComponentValue support now too
	// (GpuCameraParams::maxComponentValue's own comment, optix_types.h) -
	// no longer gated on !gpuSelected, same fix denoise/regularize already
	// got above/just above when their own GPU support landed.
	m_maxComponentValueCheck->setEnabled(isDefault);
	m_maxComponentValueSpin->setEnabled(isDefault && m_maxComponentValueCheck->isChecked());
	m_cropCheck->setEnabled(isDefault);
	const bool cropSpinsEnabled = isDefault && m_cropCheck->isChecked();
	m_cropX0Spin->setEnabled(cropSpinsEnabled);
	m_cropY0Spin->setEnabled(cropSpinsEnabled);
	m_cropX1Spin->setEnabled(cropSpinsEnabled);
	m_cropY1Spin->setEnabled(cropSpinsEnabled);
	m_seedCheck->setEnabled(isDefault);
	m_seedSpin->setEnabled(isDefault && m_seedCheck->isChecked());
}

void MainWindow::createPreviewTab() {
	QWidget *previewWidget = new QWidget();
	QHBoxLayout *outerLayout = new QHBoxLayout(previewWidget);
	outerLayout->setContentsMargins(12, 12, 12, 12);
	outerLayout->setSpacing(0);

	// Sub-tabs on the left in a large, dominant pane; info/buttons in a
	// narrow sidebar on the right, so the render gets most of the tab's
	// space instead of splitting height with a full-width info/button strip
	// underneath it. A QSplitter (not a fixed QHBoxLayout split) so the
	// user can still drag the sidebar narrower/wider if they want even more
	// render space.
	QSplitter *splitter = new QSplitter(Qt::Horizontal, previewWidget);
	splitter->setChildrenCollapsible(false);
	outerLayout->addWidget(splitter);

	// Each completed render gets its own closable sub-tab (see
	// addImagePreviewTab()/addVideoPreviewTab()) rather than a single
	// shared pane the next render overwrites - switching between sub-tabs
	// keeps every past render's image/video around. Not movable: tab order
	// (render order) is itself informative, and reordering would also
	// complicate m_previewTitleCounts' de-duplication.
	m_previewSubTabs = new SplitPreviewTabs();
	m_previewSubTabs->setMinimumSize(200, 200);
	// Qt's own auto-managed close button is deliberately left off - its
	// placement follows a style hint (SH_TabBar_CloseButtonPosition) that
	// this app's active stylesheet does not reliably let a per-widget style
	// override, so it always landed on the wrong side regardless.
	// HorizontalTabBar instead hand-paints and hand-hit-tests its own close
	// glyph directly on the left of each tab's label (see paintEvent()/
	// mousePressEvent() in mainwindow.h), which sidesteps that negotiation
	// entirely. Not movable either (QTabBar's own default): tab order
	// (render order) is itself informative, and reordering would also
	// complicate m_previewTitleCounts' de-duplication.
	//
	// Left side rather than across the top - a render can accumulate many
	// of these over a session, and a vertical list scales far better than
	// a widening horizontal strip. setUsesScrollButtons() is what keeps the
	// list reachable once it overflows the pane's height (Qt's tab bar has
	// no literal scrollbar, but this is its own equivalent: small up/down
	// arrow buttons appear once the tabs no longer fit). objectName'd so
	// its QSS rule (mainwindow_style.cpp) can give it left-rounded corners
	// and a right-edge accent border instead of the main tab strip's
	// top-rounded/bottom-underline styling, which is built for a
	// horizontal bar and would land on the wrong edges here. The West
	// shape itself is set in SplitPreviewTabs's own constructor
	// (mainwindow.h), since that's a QTabBar property independent of
	// QTabWidget - SplitPreviewTabs doesn't use one.
	m_previewSubTabs->tabBar()->setObjectName("previewSubTabsBar");
	m_previewSubTabs->tabBar()->setUsesScrollButtons(true);
	m_previewSubTabs->setElideMode(Qt::ElideRight);
	connect(m_previewSubTabs, &SplitPreviewTabs::currentChanged, this, [this](int) {
		updatePreviewSidebarForActiveTab();
	});
	connect(m_previewSubTabs->tabBar(), &HorizontalTabBar::closeRequested,
	        this, &MainWindow::closePreviewSubTab);
	splitter->addWidget(m_previewSubTabs);

	// Recent Renders: past renders' output files are never deleted when
	// their tab is closed (see closePreviewSubTab()), just no longer
	// reachable from the UI once the session that created them ends - this
	// list, inserted into the empty-state prompt (visible in exactly the
	// state where recovering a past render matters), reopens one with a
	// double-click via the same addImagePreviewTab()/addVideoPreviewTab()
	// calls a fresh render itself uses. Built here (see
	// refreshRecentRendersList(), recent_renders.cpp) and rebuilt again
	// after every save and every tab close, so renders from earlier this
	// session show up without an app restart.
	refreshRecentRendersList();

	QWidget *sidebar = new QWidget();
	m_previewSidebar = sidebar;
	sidebar->setMinimumWidth(200);
	sidebar->setMaximumWidth(320);
	QVBoxLayout *sideLayout = new QVBoxLayout(sidebar);
	sideLayout->setContentsMargins(12, 4, 0, 0);
	sideLayout->setSpacing(10);

	m_previewInfoLabel = new QLabel(sidebar);
	m_previewInfoLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewInfoLabel->setWordWrap(true);
	m_previewInfoLabel->setObjectName("previewInfo");
	sideLayout->addWidget(m_previewInfoLabel);

	// Selected scene's description - same text/source as the Basic Settings
	// tab's #sceneInfo box (see onSceneChanged()), kept in sync with the
	// scene combo rather than tied to a completed render, so it's already
	// showing what you're about to render before the first click.
	m_previewSceneDescLabel = new QLabel(sidebar);
	m_previewSceneDescLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewSceneDescLabel->setWordWrap(true);
	m_previewSceneDescLabel->setObjectName("previewSceneDesc");
	sideLayout->addWidget(m_previewSceneDescLabel);

	// The active render's own technique note (what the scene demonstrates
	// and why it looks the way it does, plus the render's own technique/
	// settings summary) - see updatePreviewSidebarForActiveTab(), which
	// populates this from the tab's "sceneId"/"techniqueHtml" properties
	// and hides the scroll wrapper when there's nothing to show at all.
	// Scrollable (rather than a plain QLabel like previewInfo/
	// previewSceneDesc above) since the combined content can run long -
	// stretch factor 1 so it absorbs whatever vertical space is left in
	// the sidebar instead of the whole sidebar growing past the visible
	// area and pushing the buttons below out of view.
	m_previewTechniqueLabel = new QLabel();
	m_previewTechniqueLabel->setAlignment(Qt::AlignLeft | Qt::AlignTop);
	m_previewTechniqueLabel->setWordWrap(true);
	m_previewTechniqueLabel->setObjectName("previewTechniqueNote");

	m_previewTechniqueScroll = new QScrollArea(sidebar);
	m_previewTechniqueScroll->setWidget(m_previewTechniqueLabel);
	m_previewTechniqueScroll->setWidgetResizable(true);
	m_previewTechniqueScroll->setFrameShape(QFrame::NoFrame);
	m_previewTechniqueScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_previewTechniqueScroll->setObjectName("previewTechniqueScroll");
	m_previewTechniqueScroll->setVisible(false);
	sideLayout->addWidget(m_previewTechniqueScroll, /*stretch=*/1);

	// Geometry only - colour and hover/focus states come from the global
	// theme so every secondary button behaves identically. Full sidebar
	// width and stacked vertically now that they're beside the image, not
	// centered in a horizontal strip underneath it. Both act on whichever
	// sub-tab is currently active, not just the most recent render - see
	// currentPreviewProperty().
	QString previewBtnStyle =
		"QPushButton { min-height: 28px; max-height: 28px; padding: 0px 20px; font-size: 11pt; }";

	QPushButton *openFolderButton = new QPushButton(tr("Open Output &Folder"));
	icon_tint::apply(openFolderButton, ":/icons/folder.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openFolderButton->setStyleSheet(previewBtnStyle);
	openFolderButton->setToolTip(tr("Show the folder containing the active tab's render in Explorer"));
	connect(openFolderButton, &QPushButton::clicked, this, [this]() {
		const QString path = currentPreviewProperty("outputPath");
		if (path.isEmpty()) return;
		QFileInfo fileInfo(path);
		QDesktopServices::openUrl(QUrl::fromLocalFile(fileInfo.absolutePath()));
	});
	sideLayout->addWidget(openFolderButton);

	QPushButton *openViewerButton = new QPushButton(tr("Open in Default &Viewer"));
	icon_tint::apply(openViewerButton, ":/icons/image.svg", icon_tint::Role::Body, m_activeTheme.textBody);
	openViewerButton->setStyleSheet(previewBtnStyle);
	openViewerButton->setToolTip(tr("Open the active tab's render in the system viewer"));
	connect(openViewerButton, &QPushButton::clicked, this, [this]() {
		const QString path = currentPreviewProperty("previewPath");
		if (path.isEmpty()) return;
		QDesktopServices::openUrl(QUrl::fromLocalFile(path));
	});
	sideLayout->addWidget(openViewerButton);

	// No trailing addStretch() here - m_previewTechniqueScroll's own
	// stretch factor (set above) already absorbs whatever vertical space
	// is left when it's visible; when it's hidden (nothing to show), the
	// buttons simply sit right after the info/scene-desc labels instead
	// of being pushed toward the bottom of an otherwise-empty sidebar.
	splitter->addWidget(sidebar);

	// Bias initial space toward the render - the sidebar only needs enough
	// width for its buttons/info text, everything else goes to the render.
	splitter->setStretchFactor(0, 1);
	splitter->setStretchFactor(1, 0);
	splitter->setSizes({700, 220});

	// Starts hidden - no tabs exist yet at construction time, so there is
	// nothing for the sidebar to describe. updatePreviewSidebarForActiveTab()
	// takes over from here once real tabs come and go.
	sidebar->setVisible(false);

	m_previewTabIndex = m_tabWidget->addTab(previewWidget, tr("Preview"));
}

QString MainWindow::uniquePreviewTabTitle(const QString &baseTitle) {
	int &count = m_previewTitleCounts[baseTitle];
	++count;
	return count == 1 ? baseTitle : QString("%1 (%2)").arg(baseTitle).arg(count);
}

QString MainWindow::currentPreviewProperty(const char *name) const {
	if (!m_previewSubTabs) return QString();
	QWidget *page = m_previewSubTabs->currentWidget();
	if (!page) return QString();
	return page->property(name).toString();
}

void MainWindow::updatePreviewSidebarForActiveTab() {
	if (m_previewInfoLabel) m_previewInfoLabel->setText(currentPreviewProperty("infoText"));
	if (m_previewTechniqueLabel && m_previewTechniqueScroll) {
		const QString sceneId = currentPreviewProperty("sceneId");
		// Scene note first: it's what actually answers "what's special
		// about this image / why does it look this way", so it leads:
		// the generic integrator/settings text below is useful background,
		// not the headline. techniqueHtml is precomputed once at
		// tab-creation time (see PreviewTechniqueInfo/
		// MainWindow::renderTechniqueHtml()) since a completed render's
		// own settings never change; the scene note stays a live lookup
		// instead, since scene_technique_notes.h is the single source of
		// truth for that text.
		QString html;
		if (!sceneId.isEmpty() && scene_technique_notes::hasNote(sceneId)) {
			html = tr("<b>Why it looks this way</b><br>%1")
						.arg(plainTextToHtmlParagraphs(scene_technique_notes::forScene(sceneId)));
		}
		const QString techniqueHtml = currentPreviewProperty("techniqueHtml");
		if (!techniqueHtml.isEmpty()) {
			if (!html.isEmpty()) html += "<br><br>";
			html += techniqueHtml;
		}
		m_previewTechniqueScroll->setVisible(!html.isEmpty());
		if (!html.isEmpty()) {
			m_previewTechniqueLabel->setText(html);
			// QScrollArea only clamps an out-of-range scroll value to the
			// new content's height on relayout, it doesn't zero it - without
			// this, switching from a long-scrolled tab to a shorter one
			// would open already scrolled toward the bottom instead of at
			// the top.
			m_previewTechniqueScroll->verticalScrollBar()->setValue(0);
		}
	}
	// Nothing in the sidebar (render info, Open Folder/Viewer) means anything
	// without an active render tab to point at - hidden rather than left
	// showing stale info/dead buttons alongside the empty-state prompt (see
	// SplitPreviewTabs's own empty-state widget in mainwindow.h), and it also
	// lets that prompt use the pane's full width instead of being squeezed
	// down to whatever the splitter left it.
	if (m_previewSidebar && m_previewSubTabs) {
		m_previewSidebar->setVisible(m_previewSubTabs->tabBar()->count() > 0);
	}
	updateActionStates();
}

void MainWindow::closePreviewSubTab(int index) {
	if (!m_previewSubTabs) return;
	QWidget *page = m_previewSubTabs->widget(index);
	m_previewSubTabs->removeTab(index);
	// Any QMediaPlayer/QVideoWidget a video tab owns is a CHILD of `page`
	// (see addVideoPreviewTab()), so deleting it tears those down too
	// rather than leaking a player per closed tab.
	if (page) page->deleteLater();
	updatePreviewSidebarForActiveTab();
	// The file this tab pointed at is still on disk (only the tab itself
	// closed) - if closing this was the last tab, the empty state (and its
	// Recent Renders list) is about to show, so make sure it includes this
	// one rather than waiting for the next render or app restart.
	refreshRecentRendersList();
}

void MainWindow::addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
									 const QString &infoText, const QString &outputPath, const QString &previewPath,
									 const PreviewTechniqueInfo &technique) {
	if (!m_previewSubTabs) return;

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(0, 0, 0, 0);

	ScaledImageLabel *label = new ScaledImageLabel();
	label->setMinimumSize(200, 200);
	label->setPreviewPixmap(pixmap);
	// Styled globally by class name - see the ScaledImageLabel rule.
	layout->addWidget(label);

	page->setProperty("outputPath", outputPath);
	page->setProperty("previewPath", previewPath);
	page->setProperty("infoText", infoText);
	page->setProperty("sceneId", technique.sceneId);
	page->setProperty("techniqueHtml", technique.techniqueHtml);

	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);
}

void MainWindow::addVideoPreviewTab(const QString &title, const QString &tooltip,
									 const QString &videoPath, const QString &infoText,
									 const PreviewTechniqueInfo &technique) {
	if (!m_previewSubTabs) return;

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(6);

	QVideoWidget *videoWidget = new QVideoWidget();
	videoWidget->setMinimumSize(200, 200);
	layout->addWidget(videoWidget, 1);

	// Parented to `page`, not `this` - a self-contained player per tab that
	// only ever loads ONE file, once. Besides being the natural way to give
	// each tab independent playback state, this sidesteps the QMediaPlayer/
	// FFmpeg-backend failure mode a single shared, reused player used to
	// hit (reloading a DIFFERENT file at a path it had already opened
	// before could come back "Invalid data found when processing input"
	// even though the file itself was perfectly valid) - with one player
	// per unique file, that scenario can no longer arise.
	QMediaPlayer *player = new QMediaPlayer(page);
	QAudioOutput *audioOutput = new QAudioOutput(page);
	player->setAudioOutput(audioOutput);
	player->setVideoOutput(videoWidget);

	QWidget *controls = new QWidget();
	QHBoxLayout *controlsLayout = new QHBoxLayout(controls);
	controlsLayout->setContentsMargins(0, 0, 0, 0);
	controlsLayout->setSpacing(8);

	QPushButton *playPauseButton = new QPushButton(tr("Pause"));
	playPauseButton->setFixedWidth(70);
	connect(playPauseButton, &QPushButton::clicked, page, [player]() {
		if (player->playbackState() == QMediaPlayer::PlayingState) player->pause();
		else player->play();
	});
	controlsLayout->addWidget(playPauseButton);

	QSlider *positionSlider = new QSlider(Qt::Horizontal);
	controlsLayout->addWidget(positionSlider, 1);
	layout->addWidget(controls);

	connect(player, &QMediaPlayer::playbackStateChanged, page, [playPauseButton](QMediaPlayer::PlaybackState state) {
		playPauseButton->setText(state == QMediaPlayer::PlayingState ? tr("Pause") : tr("Play"));
	});
	connect(player, &QMediaPlayer::durationChanged, page, [positionSlider](qint64 duration) {
		positionSlider->setRange(0, static_cast<int>(duration));
	});
	connect(player, &QMediaPlayer::positionChanged, page, [positionSlider](qint64 position) {
		if (!positionSlider->isSliderDown())
			positionSlider->setValue(static_cast<int>(position));
	});
	connect(player, &QMediaPlayer::errorOccurred, page, [this](QMediaPlayer::Error error, const QString &errorString) {
		onLogMessage(tr("Video playback error (%1): %2").arg(static_cast<int>(error)).arg(errorString));
	});
	// A small value bubble that follows the handle while scrubbing, showing
	// the position being dragged to as M:SS - Fluent-style sliders do this
	// by default; QSlider has no built-in equivalent. A plain child widget
	// rather than a real QToolTip, which auto-hides on mouse movement -
	// exactly what a drag never stops doing. Parented to positionSlider so
	// it's destroyed with it automatically; each video preview tab gets its
	// own slider (and so its own bubble), never a shared one.
	QLabel *scrubBubble = new QLabel(positionSlider);
	scrubBubble->setObjectName("scrubBubble");
	scrubBubble->setAlignment(Qt::AlignCenter);
	scrubBubble->hide();

	const auto formatMs = [](qint64 ms) {
		const qint64 totalSeconds = ms / 1000;
		return QString("%1:%2").arg(totalSeconds / 60).arg(totalSeconds % 60, 2, 10, QChar('0'));
	};
	// Horizontal position only (handle height doesn't vary), placed just
	// above the groove. Proportional to (value-min)/(max-min) across the
	// slider's own current width - not a hand-tuned pixel offset - so it
	// tracks correctly regardless of the tab's width or DPI.
	const auto moveBubbleTo = [positionSlider, scrubBubble](int value) {
		scrubBubble->adjustSize();
		const int span = std::max(0, positionSlider->width() - scrubBubble->width());
		const int range = positionSlider->maximum() - positionSlider->minimum();
		const double t = range > 0
			? double(value - positionSlider->minimum()) / range : 0.0;
		scrubBubble->move(static_cast<int>(t * span), -scrubBubble->height() - 4);
	};

	// Pausing before setPosition() (and resuming after, if it was playing)
	// is the standard fix for scrubbing that "doesn't seem to do anything":
	// while playing, the player's own clock keeps advancing on its own
	// timeline in parallel with each setPosition() call from the drag, so
	// the seek and normal playback fight over what frame gets shown next -
	// on some backends the dragged-to frame never visibly lands at all.
	// Seeking while paused is a single deterministic jump with nothing
	// racing it. "Was playing" rides as a property on the slider itself
	// rather than a captured variable, since this tab's own connections are
	// the only thing that needs it.
	connect(positionSlider, &QSlider::sliderPressed, page,
			[player, positionSlider, scrubBubble, formatMs, moveBubbleTo]() {
		positionSlider->setProperty("wasPlaying", player->playbackState() == QMediaPlayer::PlayingState);
		player->pause();
		scrubBubble->setText(formatMs(positionSlider->value()));
		moveBubbleTo(positionSlider->value());
		scrubBubble->show();
		scrubBubble->raise();
	});
	connect(positionSlider, &QSlider::sliderMoved, page,
			[player, scrubBubble, formatMs, moveBubbleTo](int position) {
		player->setPosition(position);
		scrubBubble->setText(formatMs(position));
		moveBubbleTo(position);
	});
	connect(positionSlider, &QSlider::sliderReleased, page, [player, positionSlider, scrubBubble]() {
		if (positionSlider->property("wasPlaying").toBool()) player->play();
		scrubBubble->hide();
	});

	page->setProperty("outputPath", videoPath);
	page->setProperty("previewPath", videoPath);
	page->setProperty("infoText", infoText);
	page->setProperty("sceneId", technique.sceneId);
	page->setProperty("techniqueHtml", technique.techniqueHtml);

	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);

	player->setSource(QUrl::fromLocalFile(videoPath));
	player->play();
}

void MainWindow::createVideoTab() {
	QWidget *videoTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(videoTab);
	layout->setSpacing(12);
	layout->setContentsMargins(12, 12, 12, 12);

	// The tab stays fully interactive regardless of Output Mode (a prior
	// version disabled it outright when Output Mode wasn't "Generate Video",
	// but that blocked browsing/configuring these settings ahead of
	// switching modes) - this banner is the substitute warning, so a
	// configured preset/camera path/frame count never gets silently ignored
	// without the user having seen why. onModeChanged() toggles its
	// visibility; the initial state matches m_videoMode's own default
	// (false - see the MainWindow constructor), so no extra sync call is
	// needed here.
	m_videoModeWarningLabel = new QLabel(
		tr("⚠ These settings only take effect when Output Mode (Basic Settings tab) is set to \"Generate Video\"."),
		videoTab);
	m_videoModeWarningLabel->setObjectName("videoModeWarning");
	m_videoModeWarningLabel->setWordWrap(true);
	m_videoModeWarningLabel->setVisible(!m_videoMode);
	layout->addWidget(m_videoModeWarningLabel);

	// Video parameters group
	QGroupBox *videoGroup = new QGroupBox(tr("Video Generation Settings"), videoTab);
	styleGroupBox(videoGroup);
	QFormLayout *videoLayout = new QFormLayout(videoGroup);
	videoLayout->setVerticalSpacing(10);
	videoLayout->setHorizontalSpacing(10);
	videoLayout->setContentsMargins(15, 22, 15, 12);

	// Preset selector - sets the scene picker (on the Basic tab), camera
	// path, and the three spinboxes below all at once from one of
	// video_preset.h's named bundles. First row, above Camera Path, since
	// picking one is meant to replace tuning the other four controls, not
	// sit alongside them as a fifth independent setting.
	m_videoPresetCombo = new QComboBox();
	m_videoPresetCombo->addItem(tr("(custom - choose settings below)"), QString());
	for (const video_preset::VideoPreset& p : video_preset::kAll)
		m_videoPresetCombo->addItem(
			QString("[%1] %2").arg(QString::fromUtf8(p.id), QString::fromUtf8(p.name)),
			QString::fromUtf8(p.id));
	m_videoPresetCombo->setToolTip(
		tr("Famous ray-tracing reference scenes and motions, pre-tuned so you don't\n"
		"have to set the scene, camera path, frame count, fps, and speed by hand.\n"
		"Selecting one changes the scene on the Basic tab too. Choosing any of the\n"
		"other controls on this tab afterward is fine - they simply stop matching\n"
		"the preset, the same as if you had built the same settings by hand."));
	styleComboBox(m_videoPresetCombo);
	connect(m_videoPresetCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
			this, &MainWindow::onVideoPresetChanged);
	videoLayout->addRow(labelWithInfo(tr("Preset:"),
		tr("A ready-made bundle of scene + camera path + frame count + fps "
		"+ speed, tuned so the resulting video actually looks good "
		"without hand-picking every setting yourself.\n\n"
		"Picking one fills in every field below (and the scene on the "
		"Basic tab) - you can still change anything afterward, it just "
		"stops matching the preset once you do.")),
		m_videoPresetCombo);

	// Camera path selector
	m_cameraPathCombo = new QComboBox();
	m_cameraPathCombo->addItem(tr("Orbit (Circular rotation)"), "orbit");
	m_cameraPathCombo->addItem(tr("Linear (Straight path)"), "linear");
	m_cameraPathCombo->addItem(tr("Figure-8 (Lemniscate)"), "figure8");
	m_cameraPathCombo->addItem(tr("Spiral (Zoom-in)"), "spiral");
	m_cameraPathCombo->setToolTip(
		tr("How the camera moves over the frame sequence:\n"
		"  Orbit     — full circle around the scene, always looking at its centre\n"
		"  Linear    — straight sweep past the scene\n"
		"  Figure-8  — lemniscate, crossing back through the middle\n"
		"  Spiral    — orbits while moving steadily closer\n"
		"Every path starts from the camera position on the Advanced tab."));
	m_cameraPathCombo->setCurrentIndex(0);
	styleComboBox(m_cameraPathCombo);
	videoLayout->addRow(labelWithInfo(tr("Camera Path:"),
		tr("How the camera moves across the sequence of frames.\n\n"
		"Orbit circles fully around the scene, always facing its center "
		"- the classic \"turntable\" shot. Linear sweeps past in a "
		"straight line. Figure-8 traces a lemniscate, crossing back "
		"through the middle. Spiral orbits while steadily moving closer. "
		"Every path starts from wherever the camera is positioned on "
		"the Advanced tab.")),
		m_cameraPathCombo);

	// Frame count
	m_videoFramesSpinBox = new QSpinBox();
	m_videoFramesSpinBox->setRange(10, 1000);
	m_videoFramesSpinBox->setValue(60);
	m_videoFramesSpinBox->setSuffix(tr(" frames"));
	styleSpinBox(m_videoFramesSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Frame Count:"),
		tr("How many individual images make up the video - each one is a "
		"full, independent render, so this multiplies total render time "
		"directly (100 frames takes roughly 100x as long as one image "
		"at the same settings).\n\n"
		"Paired with Frames Per Second below to determine the video's "
		"total length in seconds.")),
		m_videoFramesSpinBox);

	// FPS (frames per second)
	m_videoFPSSpinBox = new QSpinBox();
	m_videoFPSSpinBox->setRange(15, 120);
	m_videoFPSSpinBox->setValue(30);
	m_videoFPSSpinBox->setSuffix(tr(" fps"));
	styleSpinBox(m_videoFPSSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Frames Per Second:"),
		tr("How many of the rendered frames play per second of video.\n\n"
		"Doesn't change how many frames get rendered (that's Frame "
		"Count above) - only how fast they play back, and therefore how "
		"many seconds long the finished video is (Frame Count divided "
		"by FPS).")),
		m_videoFPSSpinBox);

	// Movement speed multiplier - does NOT change the camera path itself (it
	// always completes the exact same full sweep: 1 rotation for
	// orbit/figure8, 2 for spiral, the whole start->end traversal for
	// linear). Instead it expands the actual number of rendered frames:
	// speed 0.5x renders 2x the Frame Count above, spreading the same
	// journey over more frames (and more real video time at the same fps),
	// so it looks slower without ever cutting the path short. speed 2x
	// renders half as many frames, covering the same journey faster.
	m_videoSpeedSpinBox = new QDoubleSpinBox();
	m_videoSpeedSpinBox->setRange(0.1, 5.0);
	m_videoSpeedSpinBox->setSingleStep(0.1);
	m_videoSpeedSpinBox->setDecimals(2);
	m_videoSpeedSpinBox->setValue(1.0);
	m_videoSpeedSpinBox->setSuffix(tr("x"));
	styleSpinBox(m_videoSpeedSpinBox);
	videoLayout->addRow(labelWithInfo(tr("Movement Speed:"),
		tr("A multiplier on how many frames the camera's full path is "
		"spread across - not a change to the path itself, which always "
		"completes the same full sweep.\n\n"
		"Speed 0.5x renders twice as many frames to cover the same "
		"journey more slowly and smoothly; speed 2x renders half as "
		"many frames, covering the same journey faster.")),
		m_videoSpeedSpinBox);

	// Video duration info (calculated from frames/fps)
	m_videoInfoLabel = new QLabel();
	m_videoInfoLabel->setWordWrap(true);
	m_videoInfoLabel->setObjectName("videoInfo");

	// Update duration display when frames, FPS, speed, or path changes
	auto updateVideoDuration = [this]() {
		int baseFrames = m_videoFramesSpinBox->value();
		int fps = m_videoFPSSpinBox->value();
		double speed = m_videoSpeedSpinBox->value();
		QString cameraPath = m_cameraPathCombo->currentData().toString();

		// Mirrors main.cpp's render_frame_count derivation exactly, so this
		// preview matches what will actually be rendered.
		int actualFrames = qMax(1, static_cast<int>(std::llround(baseFrames / speed)));
		const int kMaxVideoFrames = 5000;
		bool capped = actualFrames > kMaxVideoFrames;
		if (capped) actualFrames = kMaxVideoFrames;
		double duration = static_cast<double>(actualFrames) / fps;

		QString framesLine = (actualFrames == baseFrames)
			? tr("%1 frames").arg(actualFrames)
			: tr("%1 frames (base %2 × 1/%3x speed)%4")
				.arg(actualFrames).arg(baseFrames).arg(QString::number(speed, 'f', 2))
				.arg(capped ? tr(" - capped at 5000") : QString());

		m_videoInfoLabel->setText(tr(
			"<b>Video Duration:</b> %1 seconds (%2)<br>"
			"<b>Camera Path:</b> %3, always completes its full sweep regardless of speed<br>"
			"<b>Output:</b> Frames will be saved to <code>output/frames/</code>"
		).arg(QString::number(duration, 'f', 1), framesLine, cameraPath));
	};

	connect(m_videoFramesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoFPSSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), updateVideoDuration);
	connect(m_videoSpeedSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), updateVideoDuration);
	connect(m_cameraPathCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), updateVideoDuration);
	updateVideoDuration();

	videoLayout->addRow("", m_videoInfoLabel);

	layout->addWidget(videoGroup);

	// Requirements info
	QGroupBox *requirementsGroup = new QGroupBox(tr("ℹ️ Requirements"), videoTab);
	styleGroupBox(requirementsGroup);
	QVBoxLayout *requirementsLayout = new QVBoxLayout(requirementsGroup);

	QLabel *requirementsInfo = new QLabel(
		tr("<b>Requires ffmpeg:</b> Video encoding uses ffmpeg (libx264), which must be installed and on your PATH.<br>"
		"<small>Get it from <a href=\"https://ffmpeg.org/download.html\">ffmpeg.org</a> if the render log reports it's missing.</small><br><br>"
		"<b>Automatic Assembly:</b> After rendering all frames, the video will be automatically assembled and opened.")
	);
	requirementsInfo->setOpenExternalLinks(true);
	requirementsInfo->setWordWrap(true);
	requirementsInfo->setObjectName("mutedInfo");
	requirementsLayout->addWidget(requirementsInfo);

	layout->addWidget(requirementsGroup);

	// Usage instructions
	QGroupBox *usageGroup = new QGroupBox(tr("Usage Instructions"), videoTab);
	styleGroupBox(usageGroup);
	QVBoxLayout *usageLayout = new QVBoxLayout(usageGroup);

	QLabel *usageText = new QLabel(
		tr("<b>Step 1:</b> Configure video settings (camera path, frames, FPS)<br>"
		"<b>Step 2:</b> Configure quality settings in Basic/Advanced tabs<br>"
		"<b>Step 3:</b> Click START VIDEO RENDER and wait<br>"
		"<b>Step 4:</b> Video automatically assembles and opens when done!<br><br>"
		"<b>Tips:</b><br>"
		"• Use GPU mode for faster rendering<br>"
		"• Lower samples/pixel for quick previews (10-50)<br>"
		"• Higher samples/pixel for production quality (100-500)<br>"
		"• Typical render time: 1-5 minutes (GPU), 15-60 minutes (CPU)")
	);
	usageText->setWordWrap(true);
	usageText->setObjectName("mutedInfo");
	usageLayout->addWidget(usageText);

	layout->addWidget(usageGroup);

	layout->addStretch();

	// Scroll area for video tab
	ThemedScrollArea *scrollArea = new ThemedScrollArea();  // theme motif support - see that class's own comment
	scrollArea->setWidget(videoTab);
	scrollArea->setWidgetResizable(true);
	scrollArea->setFrameShape(QFrame::NoFrame);
	// Named so the global stylesheet can paint a theme's decorative motif here.
	// It has to be the scroll area rather than QTabWidget::pane: the pane is
	// covered edge to edge by this widget, so a background set on it is never
	// seen. QAbstractScrollArea is also the one thing Qt documents as
	// supporting background-attachment, which is what keeps the motif still
	// while the settings scroll past.
	scrollArea->setObjectName("tabScroll");

	m_videoTabIndex = m_tabWidget->addTab(scrollArea, tr("Video Settings"));
}
