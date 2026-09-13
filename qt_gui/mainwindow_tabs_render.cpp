// Render Options tab and the Preview tab (and its sub-tab management) -
// split out of mainwindow_tabs.cpp to keep that file to the Settings tab
// (formerly separate Basic/Advanced settings tabs, and since then Video
// Settings too - all three merged into one scrollable tab, see that file's
// own header comment) and its scene-list helpers; see
// mainwindow_tabs_output.cpp for the Progress/Log/Diagnostics tabs.
#include "mainwindow.h"
#include "icon_tint.h"
#include "scene_technique_notes.h"
#include "settings_keys.h"

#include "../src/shared/scene_descriptor.h"

#include <cmath>

#include <QTabBar>
#include "scene_metadata_client.h"
#ifdef RT_GUI_HAVE_GPU
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

// The "(i)" mark + per-row tooltip pattern every enum-valued combo on the
// Render Options tab uses (Sampler/LightSampler/Accelerator/SplitMethod) -
// was 4 independently hand-copied local struct+loop pairs before being
// factored into this one shared method (see ComboEntry's own comment,
// mainwindow.h).
void MainWindow::populateComboEntries(QComboBox *combo, std::initializer_list<ComboEntry> entries) {
	for (const ComboEntry &entry : entries) {
		icon_tint::addItem(combo, ":/icons/info.svg", entry.label, entry.value, m_activeTheme.textBody);
		combo->setItemData(combo->count() - 1, wrapTooltipHtml(entry.tooltip), Qt::ToolTipRole);
	}
}

// ============================================================================
// Render Options Tab
// ============================================================================
// Exposes CLI flags RenderController::start() (mainwindow.cpp) already knows
// how to emit but that no earlier tab surfaced: --sampler, --spectral,
// --exposure, --tonemap, --stats, --denoise, --optix-validate. Deliberately
// a separate tab from Settings (resolution/samples/depth/camera, among
// other things) rather than folded into it, since this tab is exclusively
// about render BEHAVIOR flags, not image/camera parameters.
void MainWindow::createRenderOptionsTab() {
	QWidget *optionsTab = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(optionsTab);
	layout->setSpacing(14);
	layout->setContentsMargins(12, 12, 12, 12);

#ifdef RT_GUI_HAVE_GPU
	// Same "banner, don't hide" convention as the Settings tab's own
	// m_videoModeWarningLabel/m_liveModeWarningLabel - see their comments
	// (mainwindow.h). Live Preview always renders via the GPU wavefront path
	// tracer directly, so almost nothing on this tab applies to it - except
	// the Denoiser group's own "Live Preview" subsection further down,
	// which is specifically FOR it (and is dimmed/undimmed the same way
	// this banner is shown/hidden).
	m_liveModeOptionsWarningLabel = makeModeWarningBanner(optionsTab,
		tr("⚠ Live Preview uses the GPU progressive path tracer directly - none of the "
		"settings on this tab apply to it, except the Denoiser section's own "
		"\"Live Preview\" subsection below."));
	layout->addWidget(m_liveModeOptionsWarningLabel);
#endif

	// ------------------------------------------------------------------
	// Integrator group - the algorithm selector itself, plus sub-flags for
	// whichever alternate integrator it selects. Comes first (rather than
	// after Sampling & Spectral/Output) since it's this tab's primary
	// choice, not a separate axis - everything below only applies to
	// whichever integrator is picked here.
	// ------------------------------------------------------------------
	m_integratorOptionsGroup = new InfoGroupBox(tr("Integrator"), optionsTab);
	styleGroupBox(m_integratorOptionsGroup);
	m_integratorOptionsGroup->setInfoIcon(createInfoIcon(
		tr("Choose the light-transport algorithm - the default path tracer, "
		"or an alternate like SPPM/BDPT/MLT/AO with its own sub-options "
		"shown below once picked. Alternate integrators are CPU-only and "
		"can't be combined with Video mode.")));
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
	// the Settings tab uses (Renderer/GPU Backend/Quality/
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
	InfoGroupBox *samplingGroup = new InfoGroupBox(tr("Sampling && Spectral"), optionsTab);
	styleGroupBox(samplingGroup);
	samplingGroup->setInfoIcon(createInfoIcon(
		tr("Sampler algorithm, spectral rendering, adaptive sampling, and a "
		"time-limit alternative to a fixed sample count. These control HOW "
		"samples are drawn, separately from HOW MANY (Settings tab).")));
	QFormLayout *samplingLayout = new QFormLayout(samplingGroup);
	samplingLayout->setVerticalSpacing(10);
	samplingLayout->setHorizontalSpacing(10);
	samplingLayout->setContentsMargins(15, 22, 15, 12);

	m_samplerCombo = new QComboBox(optionsTab);
	// Same "(i)" mark + per-row tooltip as m_integratorCombo's own items -
	// see that combo's construction comment for why icon_tint::addItem()
	// (not a plain addItem()) plus setItemData(Qt::ToolTipRole).
	populateComboEntries(m_samplerCombo, {
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
	});
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
		"tracer - switch Renderer to CPU on the Settings tab to "
		"use it.")),
		m_samplerCombo);
	connect(m_samplerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		if (m_sceneCombo) updateSceneRecommendedSettingsHint(m_sceneCombo->currentData().toString());
	});

	// Same "(i)" mark + per-row tooltip pattern as m_samplerCombo just
	// above - see that combo's construction comment.
	m_lightSamplerCombo = new QComboBox(optionsTab);
	populateComboEntries(m_lightSamplerCombo, {
		{tr("BVH (default)"), QString(), tr(
			"Builds a spatial hierarchy over the scene's lights and "
			"weighs each one by both power and proximity to the "
			"shading point, adapting per bounce rather than using one "
			"global weighting. pbrt-v4's own default - generally the "
			"best convergence, at a small extra bookkeeping cost.")},
		{tr("Auto (use scene's own request)"), QStringLiteral("auto"), tr(
			"Uses whatever this scene's own Integrator \"string "
			"lightsampler\" parameter requested (BVH if it made no "
			"request, or requested something this project doesn't "
			"implement) instead of a fixed choice - matches the CLI's "
			"own --lightsampler auto. Picking this once and leaving it "
			"is the one choice here that stays correct as you switch "
			"between scenes with different recommendations.")},
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
	});
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
		"Auto instead uses whatever the loaded scene's own Integrator "
		"parameter requested (BVH if it made no request). "
		"Power picks by brightness alone, ignoring position - simpler, "
		"worse in scenes where light distance varies a lot. Uniform "
		"ignores both - every light equally likely regardless of "
		"brightness or distance, included mainly for comparison/"
		"debugging.\n\nGrayed out? This only affects the CPU renderer's "
		"default path tracer - switch Renderer to CPU on the Settings "
		"tab to use it.")),
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
		"path tracer - switch Renderer to CPU on the Settings "
		"tab to use it.")));

	m_adaptiveSamplingCheck = new QCheckBox(tr("Adaptive sampling (--adaptive)"), optionsTab);
	m_adaptiveSamplingCheck->setToolTip(
		tr("Stops sampling a pixel early once it's converged, instead of\n"
		"always spending the full Samples budget on every pixel - Samples\n"
		"becomes a ceiling, not a fixed count. CPU default path tracer only."));
	styleCheckBox(m_adaptiveSamplingCheck);
	m_adaptiveThresholdSpin = new QDoubleSpinBox(optionsTab);
	m_adaptiveThresholdSpin->setRange(0.0001, 1.0);
	m_adaptiveThresholdSpin->setDecimals(4);
	m_adaptiveThresholdSpin->setValue(0.01);
	m_adaptiveThresholdSpin->setSingleStep(0.005);
	m_adaptiveThresholdSpin->setEnabled(false);
	m_adaptiveThresholdSpin->setToolTip(
		tr("Target relative noise level to consider a pixel converged.\n"
		"Lower = cleaner but slower. 0.01 matches Blender Cycles' own default."));
	styleSpinBox(m_adaptiveThresholdSpin);
	connect(m_adaptiveSamplingCheck, &QCheckBox::toggled, m_adaptiveThresholdSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *adaptiveRow = new QWidget(optionsTab);
		QHBoxLayout *adaptiveRowLayout = new QHBoxLayout(adaptiveRow);
		adaptiveRowLayout->setContentsMargins(0, 0, 0, 0);
		adaptiveRowLayout->addWidget(m_adaptiveSamplingCheck);
		adaptiveRowLayout->addWidget(m_adaptiveThresholdSpin, 1);
		samplingLayout->addRow(checkboxWithInfo(m_adaptiveSamplingCheck,
			tr("A Monte Carlo path tracer's noise comes from randomness - some "
			"pixels (a bright, evenly-lit wall) converge to a clean estimate "
			"in just a few samples, while others (a dim corner lit only by a "
			"small window) need far more before the noise settles down. "
			"Spending the same fixed sample count on both wastes time on the "
			"pixels that were already done.\n\n"
			"Adaptive sampling tracks each pixel's own running noise estimate "
			"and stops early once it drops below the threshold below, "
			"letting Samples act as a ceiling rather than a flat quota - "
			"the same idea as Blender Cycles' own adaptive sampling.\n\n"
			"Grayed out? This only affects the CPU renderer's default path "
			"tracer - switch Renderer to CPU on the Settings tab to "
			"use it.")), adaptiveRow);
	}

	m_timeLimitCheck = new QCheckBox(tr("Time limit (--time-limit)"), optionsTab);
	m_timeLimitCheck->setToolTip(
		tr("Stop rendering once this many seconds have elapsed, instead of\n"
		"always running until every scanline is done. CPU default path\n"
		"tracer only."));
	styleCheckBox(m_timeLimitCheck);
	m_timeLimitSpin = new QDoubleSpinBox(optionsTab);
	m_timeLimitSpin->setRange(1.0, 86400.0);
	m_timeLimitSpin->setDecimals(0);
	m_timeLimitSpin->setValue(60.0);
	m_timeLimitSpin->setSingleStep(10.0);
	m_timeLimitSpin->setSuffix(tr(" s"));
	m_timeLimitSpin->setEnabled(false);
	m_timeLimitSpin->setToolTip(
		tr("Seconds to render before stopping, regardless of the Samples\n"
		"budget above."));
	styleSpinBox(m_timeLimitSpin);
	connect(m_timeLimitCheck, &QCheckBox::toggled, m_timeLimitSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *timeLimitRow = new QWidget(optionsTab);
		QHBoxLayout *timeLimitRowLayout = new QHBoxLayout(timeLimitRow);
		timeLimitRowLayout->setContentsMargins(0, 0, 0, 0);
		timeLimitRowLayout->addWidget(m_timeLimitCheck);
		timeLimitRowLayout->addWidget(m_timeLimitSpin, 1);
		samplingLayout->addRow(checkboxWithInfo(m_timeLimitCheck,
			tr("Useful for a fixed preview or render-farm time budget instead "
			"of guessing a sample count that happens to finish in time - "
			"lets Samples above stay a generous ceiling while this decides "
			"when to actually stop.\n\n"
			"Scanline-granular: whatever rows were already being worked on "
			"when the deadline passes finish normally; any row that never "
			"got started is written black rather than left out, so the "
			"image stays a valid (if incomplete) render instead of a "
			"corrupted file.\n\n"
			"Generating a video? This is a budget for the WHOLE video, not "
			"each frame - later frames get whatever's left of it, and any "
			"frames still remaining once it runs out are skipped.\n\n"
			"Grayed out? This only affects the CPU renderer's default path "
			"tracer - switch Renderer to CPU on the Settings tab to "
			"use it.")), timeLimitRow);
	}

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
	// Accelerator group - CPU only, but (unlike Sampling & Spectral above)
	// shared by every integrator, not just the default path tracer - see
	// updateRenderOptionsEnabled()'s own comment for why this combo pair's
	// enabled condition differs from every other CPU-only control here.
	// ------------------------------------------------------------------
	InfoGroupBox *acceleratorGroup = new InfoGroupBox(tr("Accelerator"), optionsTab);
	styleGroupBox(acceleratorGroup);
	acceleratorGroup->setInfoIcon(createInfoIcon(
		tr("CPU-only spatial acceleration structure and split heuristic "
		"used to speed up ray-scene intersection tests. Applies to every "
		"integrator, not just the default path tracer - the defaults work "
		"well for almost every scene.")));
	QFormLayout *acceleratorLayout = new QFormLayout(acceleratorGroup);
	acceleratorLayout->setVerticalSpacing(10);
	acceleratorLayout->setHorizontalSpacing(10);
	acceleratorLayout->setContentsMargins(15, 22, 15, 12);

	m_acceleratorCombo = new QComboBox(optionsTab);
	populateComboEntries(m_acceleratorCombo, {
		{tr("Scene's own choice (default)"), QString(), tr(
			"Leaves a loaded .pbrt scene's own Accelerator directive "
			"alone (bvh if it named none, real pbrt-v4's own default). "
			"No effect on a native (non-.pbrt) scene either way.")},
		{tr("BVH"), QStringLiteral("bvh"), tr(
			"A bounding volume hierarchy, built by the split method "
			"chosen below - this project's pre-existing default "
			"accelerator, matching pbrt-v4 itself.")},
		{tr("Kd-tree"), QStringLiteral("kdtree"), tr(
			"A k-d tree instead of a BVH - pbrt-v4's other real "
			"accelerator option. Has no split-method concept of its "
			"own (the combo below is ignored when this is chosen). "
			"Falls back to BVH/sah on a scene with object motion blur "
			"(this project's kd-tree wrapper has no per-ray-time "
			"channel) - a warning is printed when that happens.")},
	});
	m_acceleratorCombo->setToolTip(
		tr("Which acceleration structure holds the scene's geometry. Every\n"
		"choice renders the identical converged image - a build-strategy/\n"
		"perf knob, not a quality one. CPU only, every integrator - no\n"
		"effect on GPU (always its own fixed BVH) or a native (non-.pbrt)\n"
		"scene (has no Accelerator directive to override - see the log)."));
	styleComboBox(m_acceleratorCombo);
	acceleratorLayout->addRow(labelWithInfo(tr("Accelerator:"),
		tr("A flat list of triangles/spheres would make every ray test "
		"every single primitive in the scene - the acceleration structure "
		"is what lets a ray skip most of the scene and only test the "
		"handful of primitives near where it actually travels.\n\n"
		"BVH (bounding volume hierarchy) and Kd-tree are two different "
		"real strategies for organizing the same geometry - both produce "
		"the exact same rendered image, just at different build/traversal "
		"speeds depending on the scene's shape.\n\n"
		"Only a loaded .pbrt scene has an Accelerator directive to "
		"override at all - a native (non-.pbrt) scene always uses its own "
		"fixed BVH regardless of this choice (a warning is printed if you "
		"pick something else anyway).")),
		m_acceleratorCombo);
	connect(m_acceleratorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		updateRenderOptionsEnabled();
	});

	m_splitMethodCombo = new QComboBox(optionsTab);
	populateComboEntries(m_splitMethodCombo, {
		{tr("SAH (default)"), QString(), tr(
			"Surface Area Heuristic - estimates the traversal cost of "
			"several candidate splits and picks the cheapest. Slower "
			"to build than Middle/Equal, but the best-traversing tree "
			"in most scenes - this project's pre-existing BVH build.")},
		{tr("Middle"), QStringLiteral("middle"), tr(
			"Splits each node at the midpoint of its bounding box's "
			"longest axis. Cheap to build, no cost estimation at all - "
			"can traverse poorly on unevenly-distributed geometry.")},
		{tr("Equal counts"), QStringLiteral("equal"), tr(
			"Splits each node so an equal number of primitives fall on "
			"each side, regardless of their spatial extent. Cheap to "
			"build; can produce badly-shaped nodes for clustered "
			"geometry.")},
		{tr("HLBVH"), QStringLiteral("hlbvh"), tr(
			"Hierarchical Linear BVH - builds bottom-up from Morton "
			"codes, the fastest of these four to build for very large "
			"triangle counts, at some traversal-quality cost versus "
			"SAH.")},
	});
	m_splitMethodCombo->setToolTip(
		tr("How the BVH above is built (ignored when Accelerator is set to\n"
		"Kd-tree). Every choice renders the identical converged image.\n"
		"Falls back to sah on a scene with object motion blur (this\n"
		"project's non-sah BVH build has no per-ray-time channel) - a\n"
		"warning is printed when that happens."));
	styleComboBox(m_splitMethodCombo);
	acceleratorLayout->addRow(labelWithInfo(tr("BVH split method:"),
		tr("Only consulted when the accelerator above resolves to BVH "
		"(ignored for Kd-tree, which has no split-method concept). All "
		"four build the same tree shape family from a different "
		"strategy - SAH spends more time building in exchange for a "
		"better-traversing tree; Middle/Equal are cheap, simple "
		"fallbacks; HLBVH trades some traversal quality for the fastest "
		"build on very large scenes.")),
		m_splitMethodCombo);
	connect(m_splitMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		updateRenderOptionsEnabled();
	});

	layout->addWidget(acceleratorGroup);

	// ------------------------------------------------------------------
	// Post-Processing & Diagnostics group
	// ------------------------------------------------------------------
	// Named to be unambiguous next to the Settings tab's own "Output" group
	// (file path/format, mainwindow_tabs.cpp) - the two used to share the
	// same title despite covering entirely different things, a real
	// name collision for anyone searching for "the Output settings" by
	// title alone. Also gives m_optixValidateCheck (a debugging flag) a
	// title that actually covers it, instead of the old "Output" name
	// needing its own tooltip caveat to explain why a debug-only control
	// lived there.
	InfoGroupBox *outputGroup = new InfoGroupBox(tr("Post-Processing && Diagnostics"), optionsTab);
	styleGroupBox(outputGroup);
	outputGroup->setInfoIcon(createInfoIcon(
		tr("Tone mapping curve, whether to print render statistics, and "
		"OptiX validation mode - behavior flags for how the final image is "
		"processed and reported, not what to render. Denoising has its own "
		"dedicated section further down this tab.")));
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
		"the Settings tab to use it.")));

	layout->addWidget(outputGroup);

	// ------------------------------------------------------------------
	// Denoiser group
	// ------------------------------------------------------------------
	// One central place for every mode's own denoiser controls, instead of
	// batch/video's own being one row buried inside Output above while Live
	// Preview's own lived entirely on a different tab (Settings' own Live
	// Preview Settings group) - two nested subsections, each independently
	// dimmed by Output Mode via setGroupDimmed() in onModeChanged()
	// (mainwindow_slots.cpp), same pattern as the top-level m_videoGroupBox/
	// m_advancedParamsGroupBox/m_liveModeSettingsGroupBox just one level
	// deeper. This outer group itself is never dimmed - one of its two
	// children is always relevant regardless of which mode is selected.
	m_denoiserGroupBox = new InfoGroupBox(tr("Denoiser"), optionsTab);
	styleGroupBox(m_denoiserGroupBox);
	m_denoiserGroupBox->setInfoIcon(createInfoIcon(
		tr("Every render mode's own denoiser settings, gathered in one "
		"place. Image/Video and Live Preview each have independent "
		"controls below - only the subsection for the currently selected "
		"Output Mode is active.")));
	QVBoxLayout *denoiserGroupLayout = new QVBoxLayout(m_denoiserGroupBox);
	denoiserGroupLayout->setSpacing(10);
	denoiserGroupLayout->setContentsMargins(15, 22, 15, 12);

	// --- Image & Video subsection ---
	m_denoiserImageVideoGroupBox = new QGroupBox(tr("Image & Video"), optionsTab);
	styleGroupBox(m_denoiserImageVideoGroupBox);
	QFormLayout *denoiserImageVideoLayout = new QFormLayout(m_denoiserImageVideoGroupBox);
	denoiserImageVideoLayout->setVerticalSpacing(10);
	denoiserImageVideoLayout->setHorizontalSpacing(10);
	denoiserImageVideoLayout->setContentsMargins(15, 22, 15, 12);

	m_denoiseCheck = new QCheckBox(tr("OptiX AI denoiser (GPU only)"), optionsTab);
	m_denoiseCheck->setToolTip(
		tr("Run the OptiX AI denoiser on the finished render, guided by\n"
		"albedo + normal buffers. GPU only, both backends (recursive\n"
		"and wavefront each have their own denoiser)."));
	styleCheckBox(m_denoiseCheck);
	m_denoiseBlendSpin = new QDoubleSpinBox(optionsTab);
	m_denoiseBlendSpin->setRange(0.0, 1.0);
	m_denoiseBlendSpin->setDecimals(2);
	m_denoiseBlendSpin->setValue(0.0);
	m_denoiseBlendSpin->setSingleStep(0.05);
	m_denoiseBlendSpin->setEnabled(false);
	m_denoiseBlendSpin->setToolTip(
		tr("Blend between the noisy input and the fully denoised output\n"
		"(0.0 = 100% denoised, 1.0 = original noisy image). Lower this to\n"
		"preserve more fine texture/grain that full-strength denoising\n"
		"can over-smooth."));
	styleSpinBox(m_denoiseBlendSpin);
	connect(m_denoiseCheck, &QCheckBox::toggled, m_denoiseBlendSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *denoiseRow = new QWidget(optionsTab);
		QHBoxLayout *denoiseRowLayout = new QHBoxLayout(denoiseRow);
		denoiseRowLayout->setContentsMargins(0, 0, 0, 0);
		denoiseRowLayout->addWidget(m_denoiseCheck);
		denoiseRowLayout->addWidget(m_denoiseBlendSpin, 1);
		denoiserImageVideoLayout->addRow(checkboxWithInfo(m_denoiseCheck,
			tr("Ray tracing is noisy by nature - low sample counts leave a "
			"grainy, speckled image, which is why more samples usually "
			"means a cleaner picture.\n\n"
			"A denoiser is a machine-learning model trained to recognize "
			"that speckle pattern and smooth it away after the fact, "
			"without needing to trace additional rays - a way to get a "
			"clean-looking image faster, at some cost in fine detail. The "
			"number to its right blends between the noisy original and "
			"the fully denoised result - 0 is fully denoised (the "
			"default); raising it keeps back some of the original grain, "
			"useful when full-strength denoising smooths away texture "
			"you wanted to keep.\n\n"
			"Grayed out? This needs the GPU recursive backend - switch "
			"Renderer to GPU (and GPU Backend to Recursive) on the Settings "
			"tab to use it.")), denoiseRow);
	}

#ifdef RT_GUI_HAVE_GPU
	// --- Live Preview subsection - RT_GUI_HAVE_GPU-only, like every widget
	// it contains (m_liveDenoiserModeCombo etc., mainwindow.h) and like the
	// "Live Preview Settings" group this block used to live inside. ---
	m_denoiserLivePreviewGroupBox = new QGroupBox(tr("Live Preview"), optionsTab);
	styleGroupBox(m_denoiserLivePreviewGroupBox);
	QFormLayout *denoiserLivePreviewLayout = new QFormLayout(m_denoiserLivePreviewGroupBox);
	denoiserLivePreviewLayout->setVerticalSpacing(10);
	denoiserLivePreviewLayout->setHorizontalSpacing(10);
	denoiserLivePreviewLayout->setContentsMargins(15, 22, 15, 12);

	// OptiX AI denoiser and SVGF are mutually exclusive alternatives (never
	// both at once - this project's own SVGF plan), so a single 3-way
	// dropdown (None/OptiX AI Denoiser/SVGF Denoiser) replaces what used to
	// be two independently-toggled-but-kept-in-sync checkboxes - the
	// dropdown makes "exactly one of these, or neither" structural instead
	// of enforced by a QButtonGroup after the fact. "Show latest frame" and
	// Blend only apply to the OptiX AI denoiser, so they get their own row
	// below the mode dropdown, enabled only in that mode.
	QWidget *liveDenoiseRow = new QWidget();
	QHBoxLayout *liveDenoiseRowLayout = new QHBoxLayout(liveDenoiseRow);
	liveDenoiseRowLayout->setContentsMargins(0, 0, 0, 0);
	liveDenoiseRowLayout->setSpacing(10);

	QWidget *liveDenoiseOptionsRow = new QWidget();
	QHBoxLayout *liveDenoiseOptionsRowLayout = new QHBoxLayout(liveDenoiseOptionsRow);
	liveDenoiseOptionsRowLayout->setContentsMargins(0, 0, 0, 0);
	liveDenoiseOptionsRowLayout->setSpacing(10);

	// Index 0 = None, 1 = OptiX AI Denoiser, 2 = SVGF Denoiser - every place
	// that reads/sets the selection uses these same three literal indices.
	m_liveDenoiserModeCombo = new QComboBox();
	m_liveDenoiserModeCombo->addItem(tr("None"));
	m_liveDenoiserModeCombo->addItem(tr("OptiX AI Denoiser"));
	m_liveDenoiserModeCombo->addItem(tr("SVGF Denoiser (experimental)"));
	m_liveDenoiserModeCombo->setCurrentIndex(m_liveSvgfEnabled ? 2 : (m_liveDenoiseEnabled ? 1 : 0));
	// No maximumWidth cap (unlike the numeric spinboxes elsewhere on this
	// tab) - a combo box's natural sizeHint already fits its longest item
	// ("SVGF Denoiser (experimental)") plus the dropdown arrow; capping it
	// to an arbitrary pixel width just truncates that text instead.
	styleComboBox(m_liveDenoiserModeCombo);

	m_liveDenoiseBlendSpin = new QDoubleSpinBox();
	m_liveDenoiseBlendSpin->setRange(0.0, 1.0);
	m_liveDenoiseBlendSpin->setDecimals(2);
	m_liveDenoiseBlendSpin->setSingleStep(0.05);
	m_liveDenoiseBlendSpin->setValue(m_liveDenoiseBlend);
	m_liveDenoiseBlendSpin->setEnabled(m_liveDenoiseEnabled);
	styleSpinBox(m_liveDenoiseBlendSpin);
	connect(m_liveDenoiseBlendSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveDenoiseBlend = value;
		saveLiveDenoiseBlend(value);
		pushLiveDenoiseToSession();
	});
	// Built here (not inline at its addWidget() call site further down) so
	// the mode-combo's own connect() below can capture it and keep it in
	// sync with m_liveDenoiseBlendSpin's enabled state - this row isn't a
	// QFormLayout, so FormLabelEnabledSync can't do it automatically the
	// way it does for Sampler:/Light Sampler:/etc.
	QWidget *liveDenoiseBlendLabel = labelWithInfo(tr("Blend:"),
		tr("OptiX AI Denoiser only. Blend between the noisy input and the "
		"fully denoised output (0.0 = 100% denoised, 1.0 = original noisy "
		"image), same meaning as the Image & Video subsection's own blend "
		"control."));
	liveDenoiseBlendLabel->setEnabled(m_liveDenoiseEnabled);

	m_liveDenoiseShowLatestCheck = new QCheckBox(tr("Show latest frame instead of accumulating"));
	m_liveDenoiseShowLatestCheck->setChecked(m_liveDenoiseShowLatest);
	m_liveDenoiseShowLatestCheck->setEnabled(m_liveDenoiseEnabled);
	styleCheckBox(m_liveDenoiseShowLatestCheck);
	connect(m_liveDenoiseShowLatestCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveDenoiseShowLatest = checked;
		saveLiveDenoiseShowLatest(checked);
		pushLiveDenoiseToSession();
	});

	// SVGF Advanced Tuning - nested group, dimmed (not disabled - see
	// setGroupDimmed()'s own comment) except in SVGF mode, so the ten knobs
	// below can still be pre-adjusted before switching to SVGF, the same
	// "browse ahead of switching" behavior every OTHER mode-gated group on
	// this app already gets. Previously used plain setEnabled() instead -
	// the one group nested INSIDE a setGroupDimmed()'d parent
	// (m_denoiserLivePreviewGroupBox above) that dimmed differently from
	// its own parent's mechanism, for conceptually the same "not relevant
	// right now" state. Ten controls mirroring SvgfTuningParams
	// (gpu/optix/svgf_tuning_params.h) field-for-field; every range/default
	// below matches that struct's own comments. Declared here (before the
	// combo's own connect() below, which references it) even though its
	// full contents are built further down.
	m_liveSvgfTuningGroupBox = new QGroupBox(tr("SVGF Advanced Tuning"));
	styleGroupBox(m_liveSvgfTuningGroupBox);
	setGroupDimmed(m_liveSvgfTuningGroupBox, !m_liveSvgfEnabled);

	// One handler drives everything the mode selection affects: the two
	// backing bools (pushed to the session independently, exactly as two
	// checkboxes would have), persistence, and which sub-controls are
	// enabled - replaces the QButtonGroup's native mutual exclusion with an
	// equivalent "at most one of these two bools is ever true" invariant
	// that's now structurally guaranteed by construction (one selected
	// index) rather than enforced after the fact.
	connect(m_liveDenoiserModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this, liveDenoiseBlendLabel](int index) {
		m_liveDenoiseEnabled = (index == 1);
		m_liveSvgfEnabled = (index == 2);
		saveLiveDenoiseEnabled(m_liveDenoiseEnabled);
		saveLiveSvgfEnabled(m_liveSvgfEnabled);
		pushLiveDenoiseToSession();
		pushLiveSvgfToSession();
		m_liveDenoiseBlendSpin->setEnabled(m_liveDenoiseEnabled);
		liveDenoiseBlendLabel->setEnabled(m_liveDenoiseEnabled);
		m_liveDenoiseShowLatestCheck->setEnabled(m_liveDenoiseEnabled);
		setGroupDimmed(m_liveSvgfTuningGroupBox, !m_liveSvgfEnabled);
	});

	liveDenoiseRowLayout->addWidget(labelWithInfo(tr("Denoiser:"),
		tr("None: raw accumulated samples, no denoising.\n\n"
		"OptiX AI Denoiser: cleans up Live Preview's noisy low-sample image "
		"using the same OptiX AI denoiser the Image & Video subsection "
		"above runs for finished renders - lets the view look reasonable "
		"almost immediately instead of waiting many frames to converge. "
		"Costs a small amount of GPU time per frame.\n\n"
		"SVGF Denoiser: an alternative, experimental spatiotemporal filter - "
		"tracks per-pixel variance over time and uses it to drive an "
		"edge-aware spatial filter, which holds up better during camera "
		"movement than the AI denoiser + running-mean combination. Always "
		"shows the latest filtered frame rather than accumulating (see the "
		"SVGF Advanced Tuning group below for its own tunable constants).")));
	// Stretch factor on the combo itself (not a trailing addStretch()) so it
	// grows to fill the row's full width, matching every other row's
	// full-width fields on this tab.
	liveDenoiseRowLayout->addWidget(m_liveDenoiserModeCombo, 1);

	liveDenoiseOptionsRowLayout->addWidget(liveDenoiseBlendLabel);
	liveDenoiseOptionsRowLayout->addWidget(m_liveDenoiseBlendSpin);
	// Stretch factor on the checkbox's own container (not a trailing
	// addStretch()) so the row fills the full line width, matching the
	// Denoiser combo row above it.
	liveDenoiseOptionsRowLayout->addWidget(checkboxWithInfo(m_liveDenoiseShowLatestCheck,
		tr("OptiX AI Denoiser only. Displays each denoised frame as-is "
		"instead of averaging it into a running mean with earlier frames. "
		"Trades away the extra quality accumulating more samples would "
		"eventually reach, in exchange for a view that always reflects only "
		"the most recent frame - useful while flying around with WASD, "
		"where older accumulated frames are from a camera position you've "
		"already left.")), 1);

	denoiserLivePreviewLayout->addRow(liveDenoiseRow);
	denoiserLivePreviewLayout->addRow(liveDenoiseOptionsRow);

	QGridLayout *svgfTuningGrid = new QGridLayout(m_liveSvgfTuningGroupBox);
	svgfTuningGrid->setHorizontalSpacing(10);
	svgfTuningGrid->setVerticalSpacing(8);
	svgfTuningGrid->setColumnStretch(1, 1);
	svgfTuningGrid->setColumnStretch(3, 1);

	auto addSvgfDoubleSpin = [&](int row, int col, const QString &label, const QString &tooltip,
								   double lo, double hi, double step, int decimals, double value) {
		QDoubleSpinBox *spin = new QDoubleSpinBox();
		spin->setRange(lo, hi);
		spin->setSingleStep(step);
		spin->setDecimals(decimals);
		spin->setValue(value);
		styleSpinBox(spin);
		svgfTuningGrid->addWidget(labelWithInfo(label, tooltip), row, col * 2);
		svgfTuningGrid->addWidget(spin, row, col * 2 + 1);
		return spin;
	};
	auto addSvgfIntSpin = [&](int row, int col, const QString &label, const QString &tooltip,
							   int lo, int hi, int value) {
		QSpinBox *spin = new QSpinBox();
		spin->setRange(lo, hi);
		spin->setValue(value);
		styleSpinBox(spin);
		svgfTuningGrid->addWidget(labelWithInfo(label, tooltip), row, col * 2);
		svgfTuningGrid->addWidget(spin, row, col * 2 + 1);
		return spin;
	};

	m_liveSvgfTemporalAlphaSpin = addSvgfDoubleSpin(0, 0, tr("Temporal Alpha:"),
		tr("Floor on the temporal blend rate - lower holds onto history "
		"longer (less noise, more lag on a changing scene), higher adapts "
		"faster (more noise, less lag)."),
		0.01, 1.0, 0.01, 2, m_liveSvgfTemporalAlpha);
	m_liveSvgfMaxHistoryLengthSpin = addSvgfDoubleSpin(0, 1, tr("Max History Length:"),
		tr("Caps how many frames of history a converged pixel can accumulate "
		"- bounds how \"sticky\" it gets."),
		1.0, 256.0, 1.0, 0, m_liveSvgfMaxHistoryLength);
	m_liveSvgfVarianceBootstrapFramesSpin = addSvgfDoubleSpin(1, 0, tr("Variance Bootstrap Frames:"),
		tr("Below this history length, variance is spatially prefiltered "
		"from neighboring pixels instead of trusted alone - helps a fresh "
		"or disoccluded pixel's edge-stopping weights before it has enough "
		"of its own temporal history."),
		0.0, 32.0, 1.0, 0, m_liveSvgfVarianceBootstrapFrames);
	m_liveSvgfVarianceBootstrapRadiusSpin = addSvgfIntSpin(1, 1, tr("Variance Bootstrap Radius:"),
		tr("Box radius (in pixels) used for the variance prefilter above - "
		"radius 3 means a 7x7 box."),
		0, 8, m_liveSvgfVarianceBootstrapRadius);
	m_liveSvgfSigmaNormalSpin = addSvgfDoubleSpin(2, 0, tr("Sigma Normal:"),
		tr("Edge-stopping sensitivity to shading-normal differences - "
		"higher rejects a smaller normal difference, preventing blur across "
		"curved surfaces or silhouettes."),
		1.0, 1024.0, 1.0, 0, m_liveSvgfSigmaNormal);
	m_liveSvgfSigmaDepthSpin = addSvgfDoubleSpin(2, 1, tr("Sigma Depth:"),
		tr("Edge-stopping sensitivity to depth differences, relative to the "
		"local depth gradient - higher tolerates more depth variation "
		"before rejecting a neighbor as a different surface."),
		0.01, 16.0, 0.1, 2, m_liveSvgfSigmaDepth);
	m_liveSvgfSigmaLuminanceSpin = addSvgfDoubleSpin(3, 0, tr("Sigma Luminance:"),
		tr("Edge-stopping sensitivity to luminance differences, relative to "
		"the pixel's own estimated noise level - higher blurs across a "
		"larger brightness difference."),
		0.1, 32.0, 0.1, 1, m_liveSvgfSigmaLuminance);
	m_liveSvgfAtrousRadiusSpin = addSvgfIntSpin(3, 1, tr("A-trous Radius:"),
		tr("Filter footprint radius per A-trous pass - clamped to [0,2] "
		"(radius 2 = 5x5) since the filter's own kernel weight table only "
		"has 3 entries."),
		0, 2, m_liveSvgfAtrousRadius);
	m_liveSvgfMinAlbedoSpin = addSvgfDoubleSpin(4, 0, tr("Min Albedo:"),
		tr("Floor applied before dividing color by albedo (demodulation) - "
		"prevents a near-zero-albedo pixel from blowing up or round-tripping "
		"to black."),
		0.001, 1.0, 0.001, 3, m_liveSvgfMinAlbedo);
	m_liveSvgfAtrousPassesSpin = addSvgfIntSpin(4, 1, tr("A-trous Passes:"),
		tr("Number of A-trous filter passes (step sizes double each pass: "
		"1,2,4,8,...) - more passes cover a larger effective radius at "
		"proportionally higher GPU cost."),
		0, 8, m_liveSvgfAtrousPasses);

	QPushButton *svgfTuningResetButton = new QPushButton(tr("Reset to Defaults"));
	connect(svgfTuningResetButton, &QPushButton::clicked, this, [this]() {
		m_liveSvgfTemporalAlphaSpin->setValue(0.2);
		m_liveSvgfMaxHistoryLengthSpin->setValue(32.0);
		m_liveSvgfVarianceBootstrapFramesSpin->setValue(4.0);
		m_liveSvgfVarianceBootstrapRadiusSpin->setValue(3);
		m_liveSvgfSigmaNormalSpin->setValue(128.0);
		m_liveSvgfSigmaDepthSpin->setValue(1.0);
		m_liveSvgfSigmaLuminanceSpin->setValue(4.0);
		m_liveSvgfAtrousRadiusSpin->setValue(2);
		m_liveSvgfMinAlbedoSpin->setValue(0.02);
		m_liveSvgfAtrousPassesSpin->setValue(4);
	});
	svgfTuningGrid->addWidget(svgfTuningResetButton, 5, 0, 1, 4);

	// Every SVGF tuning spinbox pushes the WHOLE bundle (not just its own
	// field) via pushLiveSvgfTuningToSession() - see that method's own
	// comment for why they're always sent together as one SvgfTuningParams.
	auto pushSvgfTuning = [this]() {
		m_liveSvgfTemporalAlpha = m_liveSvgfTemporalAlphaSpin->value();
		m_liveSvgfMaxHistoryLength = m_liveSvgfMaxHistoryLengthSpin->value();
		m_liveSvgfVarianceBootstrapFrames = m_liveSvgfVarianceBootstrapFramesSpin->value();
		m_liveSvgfVarianceBootstrapRadius = m_liveSvgfVarianceBootstrapRadiusSpin->value();
		m_liveSvgfSigmaNormal = m_liveSvgfSigmaNormalSpin->value();
		m_liveSvgfSigmaDepth = m_liveSvgfSigmaDepthSpin->value();
		m_liveSvgfSigmaLuminance = m_liveSvgfSigmaLuminanceSpin->value();
		m_liveSvgfAtrousRadius = m_liveSvgfAtrousRadiusSpin->value();
		m_liveSvgfMinAlbedo = m_liveSvgfMinAlbedoSpin->value();
		m_liveSvgfAtrousPasses = m_liveSvgfAtrousPassesSpin->value();
		saveLiveSvgfTemporalAlpha(m_liveSvgfTemporalAlpha);
		saveLiveSvgfMaxHistoryLength(m_liveSvgfMaxHistoryLength);
		saveLiveSvgfVarianceBootstrapFrames(m_liveSvgfVarianceBootstrapFrames);
		saveLiveSvgfVarianceBootstrapRadius(m_liveSvgfVarianceBootstrapRadius);
		saveLiveSvgfSigmaNormal(m_liveSvgfSigmaNormal);
		saveLiveSvgfSigmaDepth(m_liveSvgfSigmaDepth);
		saveLiveSvgfSigmaLuminance(m_liveSvgfSigmaLuminance);
		saveLiveSvgfAtrousRadius(m_liveSvgfAtrousRadius);
		saveLiveSvgfMinAlbedo(m_liveSvgfMinAlbedo);
		saveLiveSvgfAtrousPasses(m_liveSvgfAtrousPasses);
		pushLiveSvgfTuningToSession();
	};
	connect(m_liveSvgfTemporalAlphaSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfMaxHistoryLengthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfVarianceBootstrapFramesSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfVarianceBootstrapRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfSigmaNormalSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfSigmaDepthSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfSigmaLuminanceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfAtrousRadiusSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfMinAlbedoSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, pushSvgfTuning);
	connect(m_liveSvgfAtrousPassesSpin, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSvgfTuning);

	denoiserLivePreviewLayout->addRow(m_liveSvgfTuningGroupBox);
#endif

	denoiserGroupLayout->addWidget(m_denoiserImageVideoGroupBox);
	setGroupDimmed(m_denoiserImageVideoGroupBox, isLiveMode());
#ifdef RT_GUI_HAVE_GPU
	denoiserGroupLayout->addWidget(m_denoiserLivePreviewGroupBox);
	setGroupDimmed(m_denoiserLivePreviewGroupBox, !isLiveMode());
#endif

	layout->addWidget(m_denoiserGroupBox);

#ifdef RT_GUI_HAVE_GPU
	// ------------------------------------------------------------------
	// Live Preview Settings group - render-BEHAVIOR knobs moved here from
	// the Settings tab's own former "Live Preview Settings" group (renamed
	// "Live Preview Controls", now sensitivity-only - see
	// m_liveModeSettingsGroupBox's own comment, mainwindow.h) so every
	// render setting lives on this tab, next to the Denoiser section right
	// above. RT_GUI_HAVE_GPU-only, like every widget it contains and like
	// the Denoiser group's own "Live Preview" subsection.
	// ------------------------------------------------------------------
	m_liveRenderSettingsGroupBox = new InfoGroupBox(tr("Live Preview Settings"), optionsTab);
	styleGroupBox(m_liveRenderSettingsGroupBox);
	m_liveRenderSettingsGroupBox->setInfoIcon(createInfoIcon(
		tr("ReSTIR DI/GI, the radiance cache, path guiding, exposure, "
		"samples/max-bounces per frame, and the firefly clamp - all "
		"independent of the Advanced Parameters group below (which only "
		"applies to Image/Video) and of the Denoiser section above. Only "
		"takes effect when Output Mode above is \"Live Preview "
		"(interactive)\", but stays editable in any mode.")));
	setGroupDimmed(m_liveRenderSettingsGroupBox, !isLiveMode());
	QFormLayout *liveRenderSettingsLayout = new QFormLayout(m_liveRenderSettingsGroupBox);
	liveRenderSettingsLayout->setVerticalSpacing(10);
	liveRenderSettingsLayout->setHorizontalSpacing(10);
	liveRenderSettingsLayout->setContentsMargins(15, 22, 15, 12);

	// ReSTIR GI on/off, exposure, samples/max-bounces per frame, and the
	// firefly clamp - each independent of everything else on this tab,
	// grouped into one row purely for layout compactness.
	QWidget *liveRenderSettingsRow = new QWidget();
	QGridLayout *liveRenderSettingsGrid = new QGridLayout(liveRenderSettingsRow);
	liveRenderSettingsGrid->setContentsMargins(0, 0, 0, 0);
	liveRenderSettingsGrid->setHorizontalSpacing(10);
	liveRenderSettingsGrid->setVerticalSpacing(8);
	liveRenderSettingsGrid->setColumnStretch(1, 1);
	liveRenderSettingsGrid->setColumnStretch(3, 1);

	m_liveRestirGiCheck = new QCheckBox(tr("ReSTIR GI"));
	m_liveRestirGiCheck->setChecked(m_liveRestirGiEnabled);
	styleCheckBox(m_liveRestirGiCheck);
	connect(m_liveRestirGiCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveRestirGiEnabled = checked;
		saveLiveRestirGiEnabled(checked);
		pushLiveRestirGiToSession();
	});
	// Own row, spanning every column - a checkbox has nothing to pair with
	// the way the label+field values below do, so it doesn't belong forced
	// into the same 2-per-row grouping.
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveRestirGiCheck,
		tr("Resampled one-bounce indirect lighting (ReSTIR GI) - independent "
		"of which denoiser is active above. Disabling it falls back to the "
		"classic single-sample indirect estimate, which is noisier but "
		"cheaper per frame.")),
		0, 0, 1, 4);

	m_liveRestirDiCheck = new QCheckBox(tr("ReSTIR DI"));
	m_liveRestirDiCheck->setChecked(m_liveRestirDiEnabled);
	styleCheckBox(m_liveRestirDiCheck);
	connect(m_liveRestirDiCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveRestirDiEnabled = checked;
		saveLiveRestirDiEnabled(checked);
		pushLiveRestirDiToSession();
	});
	// Own row too, same reasoning as ReSTIR GI's row above.
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveRestirDiCheck,
		tr("Resampled direct-light sampling (ReSTIR DI) - independent of "
		"ReSTIR GI above (that resamples one-bounce INDIRECT lighting; this "
		"resamples the direct-light draw classic next-event estimation would "
		"otherwise make from a single global alias-table sample). Disabling "
		"it falls back to that classic single-sample draw, which is noisier "
		"in scenes with many lights but cheaper per frame.")),
		1, 0, 1, 4);

	m_liveProbeCacheCheck = new QCheckBox(tr("Radiance Cache"));
	m_liveProbeCacheCheck->setChecked(m_liveProbeCacheEnabled);
	styleCheckBox(m_liveProbeCacheCheck);
	connect(m_liveProbeCacheCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_liveProbeCacheEnabled = checked;
		saveLiveProbeCacheEnabled(checked);
		pushLiveProbeCacheToSession();
	});
	// Own row too, same reasoning as ReSTIR GI/DI's rows above.
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveProbeCacheCheck,
		tr("World-space irradiance probe cache - caches and resamples "
		"diffuse lighting two or more bounces deep, independent of ReSTIR "
		"GI above (that only resamples the FIRST indirect bounce; this "
		"covers every bounce beyond it). Converges over many frames, so "
		"expect it to look patchy or noisy for the first few seconds after "
		"enabling it or moving the camera into a new area. Disabling it "
		"falls back to classic next-event estimation for every bounce, "
		"which is noisier in scenes with a lot of deep indirect light but "
		"has no convergence delay.")),
		2, 0, 1, 4);

	m_livePathGuidingCheck = new QCheckBox(tr("Path Guiding"));
	m_livePathGuidingCheck->setChecked(m_livePathGuidingEnabled);
	styleCheckBox(m_livePathGuidingCheck);
	connect(m_livePathGuidingCheck, &QCheckBox::toggled, this, [this](bool checked) {
		m_livePathGuidingEnabled = checked;
		saveLivePathGuidingEnabled(checked);
		pushLivePathGuidingToSession();
	});
	// Own row too, same reasoning as the checkboxes above.
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_livePathGuidingCheck,
		tr("Importance-samples the bounce direction for glossy/metal "
		"surfaces against a coarse, incrementally-learned estimate of where "
		"incident light actually is, instead of relying purely on the "
		"material's own reflection-lobe sampling. Requires the Radiance "
		"Cache above to also be on - it reuses that cache's own grid and "
		"update pipeline, and is a no-op without it. Like the Radiance "
		"Cache, converges over several seconds; disabling it falls back to "
		"the material's own unbiased reflection sampling.")),
		3, 0, 1, 4);

	// The four numeric values grouped into their own clean 2-per-row grid
	// (rows 4-5), separate from the checkboxes above.
	m_liveExposureSpin = new QDoubleSpinBox();
	m_liveExposureSpin->setRange(0.01, 100.0);
	m_liveExposureSpin->setSingleStep(0.1);
	m_liveExposureSpin->setValue(m_liveExposure);
	styleSpinBox(m_liveExposureSpin);
	connect(m_liveExposureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveExposure = value;
		saveLiveExposure(value);
		pushLiveExposureToSession();
	});
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Exposure:"),
		tr("A flat brightness multiplier applied before tone-mapping, same "
		"meaning as this tab's own Output-group Exposure control but "
		"independently set for Live Preview.")),
		4, 0);
	liveRenderSettingsGrid->addWidget(m_liveExposureSpin, 4, 1);

	m_liveSamplesSpinBox = new QSpinBox();
	m_liveSamplesSpinBox->setRange(1, 16);
	m_liveSamplesSpinBox->setValue(m_liveSamples);
	styleSpinBox(m_liveSamplesSpinBox);
	auto pushSppMaxDepth = [this]() {
		m_liveSamples = m_liveSamplesSpinBox->value();
		m_liveMaxDepth = m_liveMaxDepthSpinBox->value();
		saveLiveSamples(m_liveSamples);
		saveLiveMaxDepth(m_liveMaxDepth);
		pushLiveSppMaxDepthToSession();
	};
	connect(m_liveSamplesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSppMaxDepth);
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Samples/Frame:"),
		tr("Samples per pixel rendered on each Live Preview call - Live "
		"Preview has its own independent value from the Advanced Parameters "
		"group below, which only applies to Image/Video.")),
		4, 2);
	liveRenderSettingsGrid->addWidget(m_liveSamplesSpinBox, 4, 3);

	m_liveMaxDepthSpinBox = new QSpinBox();
	m_liveMaxDepthSpinBox->setRange(1, 32);
	m_liveMaxDepthSpinBox->setValue(m_liveMaxDepth);
	styleSpinBox(m_liveMaxDepthSpinBox);
	connect(m_liveMaxDepthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSppMaxDepth);
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Max Bounces:"),
		tr("Maximum ray depth for Live Preview - independent from the "
		"Advanced Parameters group below, which only applies to Image/Video.")),
		5, 0);
	liveRenderSettingsGrid->addWidget(m_liveMaxDepthSpinBox, 5, 1);

	m_liveFireflyClampSpin = new QDoubleSpinBox();
	m_liveFireflyClampSpin->setRange(1.0, 10000.0);
	m_liveFireflyClampSpin->setSingleStep(5.0);
	m_liveFireflyClampSpin->setDecimals(1);
	m_liveFireflyClampSpin->setValue(m_liveFireflyClamp);
	styleSpinBox(m_liveFireflyClampSpin);
	connect(m_liveFireflyClampSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveFireflyClamp = value;
		saveLiveFireflyClamp(value);
		pushLiveFireflyClampToSession();
	});
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Firefly Clamp:"),
		tr("Caps the brightest possible sample value to suppress fireflies, "
		"at the cost of clipping genuinely bright highlights. Lower values "
		"clamp more aggressively.")),
		5, 2);
	liveRenderSettingsGrid->addWidget(m_liveFireflyClampSpin, 5, 3);

	liveRenderSettingsLayout->addRow(liveRenderSettingsRow);

	layout->addWidget(m_liveRenderSettingsGroupBox);
#endif

	// ------------------------------------------------------------------
	// Crop Window group
	// ------------------------------------------------------------------
	InfoGroupBox *cropGroup = new InfoGroupBox(tr("Crop Window"), optionsTab);
	styleGroupBox(cropGroup);
	cropGroup->setInfoIcon(createInfoIcon(
		tr("Render only a rectangular sub-region of the full frame, given "
		"as normalized 0-1 coordinates - useful for quickly test-rendering "
		"one area of a scene without paying for the whole image.")));
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
	// A 4-column grid (label, field, label, field) instead of QFormLayout's
	// one-pair-per-row - the top-left corner (X0/Y0) and bottom-right
	// corner (X1/Y1) pack two fields per row, halving this group's height.
	QGridLayout *cropCornersGrid = new QGridLayout();
	cropCornersGrid->setHorizontalSpacing(10);
	cropCornersGrid->setColumnStretch(1, 1);
	cropCornersGrid->setColumnStretch(3, 1);
	// labelWithInfo() (not a bare QLabel) for the same reason every other
	// field label on this tab uses it - and captured into a local, rather
	// than discarded like this grid used to, so the connect() calls below
	// can dim them alongside their spinbox. A plain QGridLayout has no
	// QFormLayout::labelForField()-style API FormLabelEnabledSync could use
	// instead, so these need this explicit wiring.
	QWidget *cropX0Label = labelWithInfo(tr("Left (X0):"),
		tr("Left edge of the crop rectangle, as a fraction of the full "
		"frame width (0 = left edge, 1 = right edge)."));
	QWidget *cropY0Label = labelWithInfo(tr("Top (Y0):"),
		tr("Top edge of the crop rectangle, as a fraction of the full "
		"frame height (0 = top edge, 1 = bottom edge)."));
	QWidget *cropX1Label = labelWithInfo(tr("Right (X1):"),
		tr("Right edge of the crop rectangle, as a fraction of the full "
		"frame width - must be greater than Left (X0) to render anything."));
	QWidget *cropY1Label = labelWithInfo(tr("Bottom (Y1):"),
		tr("Bottom edge of the crop rectangle, as a fraction of the full "
		"frame height - must be greater than Top (Y0) to render anything."));
	cropCornersGrid->addWidget(cropX0Label, 0, 0);
	cropCornersGrid->addWidget(m_cropX0Spin, 0, 1);
	cropCornersGrid->addWidget(cropY0Label, 0, 2);
	cropCornersGrid->addWidget(m_cropY0Spin, 0, 3);
	cropCornersGrid->addWidget(cropX1Label, 1, 0);
	cropCornersGrid->addWidget(m_cropX1Spin, 1, 1);
	cropCornersGrid->addWidget(cropY1Label, 1, 2);
	cropCornersGrid->addWidget(m_cropY1Spin, 1, 3);
	cropLayout->addRow(cropCornersGrid);

	connect(m_cropCheck, &QCheckBox::toggled, m_cropX0Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropY0Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropX1Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, m_cropY1Spin, &QDoubleSpinBox::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, cropX0Label, &QWidget::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, cropY0Label, &QWidget::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, cropX1Label, &QWidget::setEnabled);
	connect(m_cropCheck, &QCheckBox::toggled, cropY1Label, &QWidget::setEnabled);
	cropX0Label->setEnabled(m_cropCheck->isChecked());
	cropY0Label->setEnabled(m_cropCheck->isChecked());
	cropX1Label->setEnabled(m_cropCheck->isChecked());
	cropY1Label->setEnabled(m_cropCheck->isChecked());

	layout->addWidget(cropGroup);

	// ------------------------------------------------------------------
	// Seed group
	// ------------------------------------------------------------------
	InfoGroupBox *seedGroup = new InfoGroupBox(tr("Reproducibility"), optionsTab);
	styleGroupBox(seedGroup);
	seedGroup->setInfoIcon(createInfoIcon(
		tr("Fix the random seed so a render can be reproduced exactly, "
		"pixel-for-pixel, on a later run - useful for comparing settings "
		"changes without random noise differences confusing the comparison.")));
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
	// labelWithInfo() (not a bare string label) for consistency with every
	// other field on this tab - this row was the one exception, a plain
	// QFormLayout-generated label with no info affordance next to its own
	// sibling checkbox row, which already gets one via checkboxWithInfo().
	seedLayout->addRow(labelWithInfo(tr("Seed:"),
		tr("The specific integer used to seed the render's random number "
		"generator. Only takes effect when Reproducible Render above is "
		"checked - the same seed on the same scene/settings always "
		"produces pixel-identical noise.")),
		m_seedSpin);

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
// m_spectralCheck/m_timeLimitCheck/m_timeLimitSpin/m_exposureSpin/
// m_tonemapCombo/m_statsCheck/m_denoiseCheck/m_denoiseBlendSpin/m_optixValidateCheck/
// m_seedCheck/m_gpuBackendCombo/m_acceleratorCombo/m_splitMethodCombo's
// enabled state, replacing what used to be a
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
	m_adaptiveSamplingCheck->setEnabled(isDefault && !gpuSelected);
	m_adaptiveThresholdSpin->setEnabled(isDefault && !gpuSelected && m_adaptiveSamplingCheck->isChecked());
	m_timeLimitCheck->setEnabled(isDefault && !gpuSelected);
	m_timeLimitSpin->setEnabled(isDefault && !gpuSelected && m_timeLimitCheck->isChecked());
	m_exposureSpin->setEnabled(isDefault);
	m_tonemapCombo->setEnabled(isDefault);
	m_statsCheck->setEnabled(isDefault);
	// Both GPU backends have their own real denoiser now (WavefrontPathTracer::
	// denoise(), gpu/optix/wavefront_path_tracer.cpp) - no longer gated on
	// !wavefrontSelected.
	m_denoiseCheck->setEnabled(isDefault && gpuSelected);
	m_denoiseBlendSpin->setEnabled(isDefault && gpuSelected && m_denoiseCheck->isChecked());
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
	// Unlike every field above, NOT gated on isDefault - accelerator/
	// splitmethod affect scene construction, shared by every integrator
	// (default path tracer, BDPT/MLT, SPPM), not one integrator's own logic.
	// Still CPU-only (GPU always builds its own fixed BVH).
	m_acceleratorCombo->setEnabled(!gpuSelected);
	const bool splitMethodMeaningful =
		!gpuSelected && m_acceleratorCombo->currentData().toString() != QStringLiteral("kdtree");
	m_splitMethodCombo->setEnabled(splitMethodMeaningful);
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
		stopLivePreviewIfNavigatedAway();
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

	// Selected scene's description - same text/source as the Settings
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

	// Stop the running live preview (rather than let it keep burning GPU
	// cycles unseen) whenever the live sub-tab stops being visibly active -
	// leaving the Preview tab entirely fires here; leaving the live
	// sub-tab for a different one while staying on Preview fires via the
	// m_previewSubTabs::currentChanged connection just above instead. A
	// no-op on a non-GPU build - see stopLivePreviewIfNavigatedAway()'s own
	// comment (mainwindow.h).
	connect(m_tabWidget, &QTabWidget::currentChanged, this, &MainWindow::stopLivePreviewIfNavigatedAway);
}

#ifdef RT_GUI_HAVE_GPU
// Live Preview - GPU progressive-refinement preview (see this project's own
// real-time-preview plan), just another Output Mode (see that enum's own
// comment, mainwindow_jobtypes.h) driven by the same pinned Render/Stop
// button pair every other mode uses - no button of its own. Its running
// image shows up as an ordinary sub-tab under the Preview tab, exactly like
// a finished Image/Video render (addLivePreviewTab() mirrors
// addImagePreviewTab()/addVideoPreviewTab()'s shape), rather than a
// dedicated top-level tab of its own. Camera comes from the existing
// m_cameraPosX/Y/Z spinboxes (read at Start time, and live-forwarded via
// onLivePreviewCameraChanged() while running) as a starting point, PLUS
// click-drag-to-orbit/wheel-to-zoom directly in the preview image
// (OrbitPreviewLabel, mainwindow_widgets.h) - see m_orbit's own comment
// (mainwindow.h) for how those two camera representations stay in sync.
// The actual spherical-coordinate math lives in camera_math.h (Qt-free,
// unit-tested - see tests/unit/camera_math_tests.cpp), matching this
// codebase's own existing convention for camera arithmetic
// (onCameraDistanceChanged()'s own comment, mainwindow_slots.cpp); the
// functions here are just plumbing that reads/writes m_orbit and forwards
// the result to RealtimePreviewSession.
void MainWindow::initLivePreviewSession() {
	if (!RealtimePreviewSession::isAvailable()) {
		// Same "fail quiet, explain why" pattern as every scene_metadata.dll
		// query - realtime_renderer.dll missing/wrong-arch/etc. shouldn't
		// crash the GUI, just leave the session null. The Output Mode
		// combo's own "Live Preview" item is separately disabled with a
		// matching tooltip - see createSettingsTab()'s own comment - so
		// startLivePreview()'s own null guard is unreachable through
		// normal UI in this case.
		return;
	}

	m_livePreviewSession = new RealtimePreviewSession(this);
	connect(m_livePreviewSession, &RealtimePreviewSession::frameReady,
	        this, &MainWindow::onLivePreviewFrameReady);
	connect(m_livePreviewSession, &RealtimePreviewSession::statusChanged,
	        this, &MainWindow::onLivePreviewStatus);
}

void MainWindow::addLivePreviewTab(const QString &sceneId, const QString &sceneName) {
	if (!m_previewSubTabs) return;

	// A previous session's sub-tab, if still open (Stop leaves it in place,
	// showing the frozen last frame, until closed - see stopLivePreview()'s
	// own comment), would otherwise pile up indefinitely across repeated
	// Start/Stop cycles instead of being replaced by this one. Reuses
	// closePreviewSubTab()'s own stop-before-clear teardown rather than
	// duplicating it here; its stopLivePreview() call is a harmless no-op
	// since the old session is already stopped by this point (startLivePreview()'s
	// own guard wouldn't have let a second Start through otherwise).
	if (m_livePreviewPage) {
		const int oldIndex = m_previewSubTabs->indexOf(m_livePreviewPage);
		if (oldIndex >= 0) closePreviewSubTab(oldIndex);
	}

	QWidget *page = new QWidget();
	QVBoxLayout *layout = new QVBoxLayout(page);
	layout->setContentsMargins(12, 12, 12, 12);

	m_livePreviewLabel = new OrbitPreviewLabel(page);
	m_livePreviewLabel->setMinimumSize(200, 200);
	m_livePreviewLabel->setPlaceholderText(
		tr("Waiting for first frame...\n\nDrag to orbit, scroll or +/- to zoom, "
		   "WASD to move, Up/Down to fly"));
	connect(m_livePreviewLabel, &OrbitPreviewLabel::orbitDragged, this, &MainWindow::onLivePreviewOrbitDragged);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::zoomRequested, this, &MainWindow::onLivePreviewZoomRequested);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::keyOrbitRequested, this, &MainWindow::onLivePreviewKeyOrbit);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::keyZoomRequested, this, &MainWindow::onLivePreviewKeyZoom);
	connect(m_livePreviewLabel, &OrbitPreviewLabel::translateRequested, this, &MainWindow::onLivePreviewTranslate);
	layout->addWidget(m_livePreviewLabel, /*stretch=*/1);
	// Grabs keyboard focus immediately so arrow-key/+/- navigation works
	// without an extra click first (which would itself start an orbit
	// drag) - safe to call before the page is even shown, since a fresh
	// label is created on every Start (see this function's own guard
	// above), so there's always exactly one live-preview label at a time
	// that legitimately wants focus.
	m_livePreviewLabel->setFocus();

	m_livePreviewStatusLabel = new QLabel(page);
	m_livePreviewStatusLabel->setAlignment(Qt::AlignCenter);
	layout->addWidget(m_livePreviewStatusLabel);

	// No outputPath/previewPath/techniqueHtml - there's no file on disk and
	// no completed-render settings summary, so Open Folder/Open Viewer
	// correctly stay disabled (currentPreviewProperty() on an unset
	// property just yields an empty string) rather than needing special-
	// casing. sceneId IS set, so updatePreviewSidebarForActiveTab()'s
	// existing scene_technique_notes lookup works for this page too.
	page->setProperty("infoText", tr("Live Preview — %1").arg(sceneName));
	page->setProperty("sceneId", sceneId);

	m_livePreviewPage = page;
	addPreviewSubTabPage(page, tr("Live Preview"),
		tr("Interactive GPU preview - drag to orbit, scroll or +/- to zoom, "
		"WASD to move, Up/Down to fly, Left/Right to orbit"));
}

bool MainWindow::isLivePreviewSubTabVisible() const {
	return m_tabWidget->currentIndex() == m_previewTabIndex
		&& m_previewSubTabs && m_livePreviewPage
		&& m_previewSubTabs->currentWidget() == m_livePreviewPage;
}

void MainWindow::stopLivePreviewIfNavigatedAway() {
	if (isLivePreviewSubTabVisible()) return;
	if (m_livePreviewLabel) m_livePreviewLabel->cancelDrag();
	if (m_livePreviewRunning) stopLivePreview();
}

void MainWindow::startLivePreview() {
	if (!m_livePreviewSession || m_livePreviewRunning) return;

	const QString sceneId = m_sceneCombo->currentData().toString();
	if (sceneId.isEmpty()) {
		// No live sub-tab exists yet at this point (it's only created
		// below, once a preview actually starts) - m_statusLabel is the
		// one status surface that's always available regardless of which
		// tab is open, matching how every other "can't start" message
		// (Stopping/Abandoning/Paused/etc.) already reports through it.
		m_statusLabel->setText(tr("Select a scene first"));
		return;
	}
	// Fixed, modest resolution - keeps per-frame cost low regardless of the
	// Settings tab's own width/height fields (this preview is about
	// interactive feedback, not a final-quality render at the requested
	// output size). Matches this pass's "progressive refinement only, no
	// per-resolution controls yet" scope.
	constexpr int kPreviewWidth = 400;
	constexpr int kPreviewHeight = 300;
	// Seed both the orbit state AND the free-fly pivot from wherever the
	// camera spinboxes currently point, around the CURRENT scene's own
	// lookAt point (currentLookAt()) - a fresh starting pivot every
	// session, exactly like m_orbit itself. From here on, m_livePreviewLookAt
	// (not currentLookAt()) is what orbiting/zooming/translating actually
	// revolves around - see its own comment (mainwindow.h).
	const camera_math::Vec3 camera = currentCameraPosition();
	m_livePreviewLookAt = currentLookAt();
	m_orbit = camera_math::cartesianToOrbit(camera, m_livePreviewLookAt);
	// Exposure and SVGF tuning aren't part of start()'s own parameter list
	// (neither has an accumulation-structure side effect - see setExposure()'s
	// own comment - so it's simplest to push them separately); pushed BEFORE
	// start() (not after) so the very first rendered frame already uses them,
	// not just frame 2 onward - start()'s own queued call synchronously
	// renders frame 1 as part of the SAME worker-thread event, so a push
	// queued AFTER start() would only take effect starting with frame 2.
	m_livePreviewSession->setExposure(m_liveExposure);
	pushLiveSvgfTuningToSession();
	m_livePreviewSession->start(sceneId, kPreviewWidth, kPreviewHeight, camera.x, camera.y, camera.z,
								 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z,
								 m_liveDenoiseEnabled, m_liveDenoiseBlend, m_liveDenoiseShowLatest,
								 m_liveSvgfEnabled, m_liveRestirGiEnabled, m_liveRestirDiEnabled,
								 m_liveProbeCacheEnabled, m_livePathGuidingEnabled,
								 m_liveSamples, m_liveMaxDepth, m_liveFireflyClamp);
	// m_livePreviewRunning stays false until BOTH tab switches below have
	// happened. addLivePreviewTab()'s own m_previewSubTabs->setCurrentIndex()
	// call (and the m_tabWidget switch after it) synchronously re-emit
	// currentChanged - QTabBar's own signal is direct, not queued - which
	// reaches stopLivePreviewIfNavigatedAway() reentrantly, inside this very
	// function call, before either switch has fully landed. Setting this
	// flag only once both are done means that reentrant call's own
	// "if (m_livePreviewRunning) stopLivePreview();" guard is still false
	// and a no-op, instead of killing the session this function just
	// started before it ever got to run.
	addLivePreviewTab(sceneId, SceneMetadataClient::sceneName(sceneId));
	// Same "click Render -> land where you watch it happen" behavior every
	// other Output Mode already gets from startRenderJob()'s own switch to
	// the Progress tab - just a different destination tab for this mode.
	if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
	m_livePreviewRunning = true;
	m_livePreviewStatusLabel->setText(tr("Starting..."));
	updateTransportButtons();
	updateActionStates();  // Escape (m_actStop) becomes enabled - see its own comment
}

void MainWindow::stopLivePreview() {
	if (!m_livePreviewSession || !m_livePreviewRunning) return;
	m_livePreviewSession->stop();
	m_livePreviewRunning = false;
	// Defensively ends an in-progress orbit drag the same way
	// stopLivePreviewIfNavigatedAway() already does - Escape (m_actStop's
	// shortcut) and the Stop button both route here directly, and neither
	// waits for a mouseReleaseEvent that may never arrive if the mouse
	// button is still held when Stop fires. Also covers closePreviewSubTab(),
	// which calls this function before tearing the page down.
	if (m_livePreviewLabel) m_livePreviewLabel->cancelDrag();
	m_livePreviewStatusLabel->setText(tr("Stopped"));
	updateTransportButtons();
	updateActionStates();  // Escape (m_actStop) becomes disabled again
}

void MainWindow::onLivePreviewFrameReady(QImage image, int sampleCount) {
	// m_livePreviewLabel/m_livePreviewStatusLabel are only ever set together
	// (addLivePreviewTab()) or cleared together (closePreviewSubTab()), but
	// guarding both here rather than relying on that invariant costs
	// nothing and doesn't assume a future edit can't decouple them.
	if (!m_livePreviewLabel || !m_livePreviewStatusLabel) return;
	m_livePreviewLabel->setPreviewPixmap(QPixmap::fromImage(image));
	// While effectively showing the latest frame instead of accumulating
	// (RealtimePreviewWorker::renderLoop()'s own gating condition), sampleCount
	// is really just a frame counter - no real accumulation is happening, so
	// "N samples" would misleadingly imply ongoing convergence.
	if (m_liveDenoiseEnabled && m_liveDenoiseShowLatest) {
		m_livePreviewStatusLabel->setText(tr("Live (denoised, not accumulating)"));
	} else {
		m_livePreviewStatusLabel->setText(tr("%1 samples").arg(sampleCount));
	}
}

void MainWindow::onLivePreviewStatus(QString text) {
	// Both statusChanged() call sites (RealtimePreviewWorker::renderLoop())
	// are genuine failures, not routine progress - unlike onLivePreviewFrameReady()'s
	// own per-frame sample-count text just above, which is never routed to
	// the Log tab. Logged only when the text actually CHANGES (compared
	// against what the label is already showing, BEFORE overwriting it) -
	// renderLoop() reposts itself every frame even after a failure and would
	// otherwise re-emit the identical message every frame indefinitely,
	// flooding the Log tab with duplicate lines for one single underlying
	// problem.
	if (m_livePreviewStatusLabel) {
		if (m_livePreviewStatusLabel->text() != text) {
			onLogMessage(tr("Live Preview: %1").arg(text));
		}
		m_livePreviewStatusLabel->setText(text);
	}
}

void MainWindow::onLivePreviewCameraChanged() {
	if (!m_livePreviewRunning || !m_livePreviewSession) return;
	const camera_math::Vec3 camera = currentCameraPosition();
	m_orbit = camera_math::cartesianToOrbit(camera, m_livePreviewLookAt);
	m_livePreviewSession->setCamera(camera.x, camera.y, camera.z,
									 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z);
}

void MainWindow::updateLivePreviewCameraFromOrbit() {
	if (!m_livePreviewRunning || !m_livePreviewSession) return;
	const camera_math::Vec3 camera = camera_math::orbitToCartesian(m_orbit, m_livePreviewLookAt);
	m_livePreviewSession->setCamera(camera.x, camera.y, camera.z,
									 m_livePreviewLookAt.x, m_livePreviewLookAt.y, m_livePreviewLookAt.z);
}

void MainWindow::pushLiveDenoiseToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setDenoise(m_liveDenoiseEnabled, m_liveDenoiseBlend, m_liveDenoiseShowLatest);
}

void MainWindow::pushLiveSvgfToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSvgf(m_liveSvgfEnabled);
}

void MainWindow::pushLiveRestirGiToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setRestirGi(m_liveRestirGiEnabled);
}

void MainWindow::pushLiveRestirDiToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setRestirDi(m_liveRestirDiEnabled);
}

void MainWindow::pushLiveProbeCacheToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setProbeCache(m_liveProbeCacheEnabled);
}

void MainWindow::pushLivePathGuidingToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setPathGuiding(m_livePathGuidingEnabled);
}

void MainWindow::pushLiveExposureToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setExposure(m_liveExposure);
}

void MainWindow::pushLiveSppMaxDepthToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSppAndMaxDepth(m_liveSamples, m_liveMaxDepth);
}

void MainWindow::pushLiveFireflyClampToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setFireflyClamp(m_liveFireflyClamp);
}

void MainWindow::pushLiveSvgfTuningToSession() {
	if (!m_livePreviewSession) return;
	m_livePreviewSession->setSvgfTuning(
		m_liveSvgfTemporalAlpha, m_liveSvgfMaxHistoryLength, m_liveSvgfVarianceBootstrapFrames,
		m_liveSvgfVarianceBootstrapRadius, m_liveSvgfSigmaNormal, m_liveSvgfSigmaDepth,
		m_liveSvgfSigmaLuminance, m_liveSvgfAtrousRadius, m_liveSvgfMinAlbedo, m_liveSvgfAtrousPasses);
}

// Applies a rotation to m_orbit and clamps elevation short of the true
// poles (+-90deg): AT a pole, azimuth becomes meaningless (every azimuth
// points the same direction), which would make the very next step's
// horizontal component do nothing/jump - matches the standard orbit-camera
// convention (Blender, Maya, etc.). Shared by the mouse-drag and keyboard
// arrow-key paths, which differ only in how they convert their own raw
// input (pixel deltas vs. discrete steps) into radians before calling this.
void MainWindow::applyOrbitDelta(double azimuthDelta, double elevationDelta) {
	m_orbit.azimuth += azimuthDelta;
	m_orbit.elevation += elevationDelta;
	constexpr double kMaxElevation = 1.5533;  // 89 degrees in radians
	if (m_orbit.elevation > kMaxElevation) m_orbit.elevation = kMaxElevation;
	if (m_orbit.elevation < -kMaxElevation) m_orbit.elevation = -kMaxElevation;
	updateLivePreviewCameraFromOrbit();
}

// Applies a zoom factor to m_orbit.radius and clamps it away from the
// lookAt point, where the view direction becomes degenerate. Shared by the
// mouse-wheel and keyboard +/- paths.
void MainWindow::applyZoomDelta(double factor) {
	m_orbit.radius *= factor;
	constexpr double kMinRadius = 1.0;
	if (m_orbit.radius < kMinRadius) m_orbit.radius = kMinRadius;
	updateLivePreviewCameraFromOrbit();
}

// Free-fly WASD/Up-Down translation - unlike applyOrbitDelta()/
// applyZoomDelta() above (which rotate/scale around a FIXED
// m_livePreviewLookAt), this MOVES the pivot together with the camera by
// the same world-space delta, so the camera keeps facing the same
// direction it already was rather than snapping to re-aim at wherever the
// pivot used to be. Basis is derived from the camera's CURRENT facing
// direction (forward = toward the pivot; right = perpendicular to forward
// in the horizontal plane; up = world +Y, confirmed the vertical axis
// throughout camera_math.h and the GPU renderer's own vup convention -
// scene_builder.cpp hardcodes make_float3(0,1,0) everywhere), not a
// separately-tracked orientation, so there's no drift between this and
// what orbitToCartesian() would compute from m_orbit right now.
//
// Matches applyOrbitDelta()/applyZoomDelta()'s own division of labor: the
// caller (onLivePreviewTranslate()) has already turned raw step counts into
// real world-space distances via kUnitsPerStep/m_keyboardSensitivity, so
// forwardDelta/rightDelta/upDelta here are plain distances along each basis
// vector, not step counts this function would need to scale itself.
void MainWindow::applyTranslateDelta(double forwardDelta, double rightDelta, double upDelta) {
	const camera_math::Vec3 camera = camera_math::orbitToCartesian(m_orbit, m_livePreviewLookAt);
	const camera_math::Vec3 forward = camera_math::normalized(m_livePreviewLookAt - camera);
	constexpr camera_math::Vec3 kWorldUp{0.0, 1.0, 0.0};
	// Derived directly from azimuth rather than normalized(cross(forward,
	// kWorldUp)): that cross product is forward's horizontal component
	// scaled by cos(elevation), which shrinks to the zero vector (silently
	// killing A/D strafing - normalized() of a near-zero vector is defined
	// to return zero, see camera_math.h) once the camera looks near-straight
	// up or down. The formula below is that same horizontal direction with
	// the cos(elevation) factor divided back out, so it stays unit-length
	// and well-defined at every elevation, including both poles, while
	// matching cross(forward, kWorldUp)'s own direction everywhere else.
	const camera_math::Vec3 right{std::cos(m_orbit.azimuth), 0.0, -std::sin(m_orbit.azimuth)};
	const camera_math::Vec3 delta = forward * forwardDelta + right * rightDelta + kWorldUp * upDelta;
	// Camera and pivot translate by the IDENTICAL delta, preserving
	// distance/orientation between them - re-deriving m_orbit via
	// cartesianToOrbit() rather than assuming it stays numerically
	// unchanged matches this file's own existing practice (e.g.
	// onLivePreviewCameraChanged()) of never assuming float-exact
	// preservation across a recomputation.
	m_livePreviewLookAt = m_livePreviewLookAt + delta;
	const camera_math::Vec3 newCamera = camera + delta;
	m_orbit = camera_math::cartesianToOrbit(newCamera, m_livePreviewLookAt);
	updateLivePreviewCameraFromOrbit();
}

void MainWindow::onLivePreviewOrbitDragged(int dxPixels, int dyPixels) {
	if (!m_livePreviewRunning) return;
	// Radians per pixel of drag - chosen so a full 180-degree turn takes
	// roughly the preview panel's own width in drag distance (~400px, this
	// pass's fixed preview resolution), a comfortable, not-too-twitchy feel
	// for a panel this size.
	constexpr double kRadiansPerPixel = 0.008;
	// Screen Y grows downward, so dragging UP (dyPixels negative) should
	// raise the camera (increase elevation) - hence the negation.
	applyOrbitDelta(dxPixels * kRadiansPerPixel * m_mouseSensitivity,
					 -dyPixels * kRadiansPerPixel * m_mouseSensitivity);
}

void MainWindow::onLivePreviewZoomRequested(int angleDeltaY) {
	if (!m_livePreviewRunning) return;
	// ~5.8% radius change per standard wheel notch (angleDelta of +-120) -
	// a smooth, moderate zoom step; std::pow with a negative exponent
	// (scrolling the other way) naturally inverts it. Scaling the EXPONENT
	// (not the base) by sensitivity keeps 1.0x exactly today's behavior and
	// preserves "higher sensitivity = bigger effect, same direction" -
	// scaling the base instead would need a second formula to keep values
	// above/below 1.0 behaving symmetrically.
	constexpr double kZoomFactorPerUnit = 0.9995;
	applyZoomDelta(std::pow(kZoomFactorPerUnit, static_cast<double>(angleDeltaY) * m_mouseSensitivity));
}

// Keyboard equivalents of the two mouse handlers above - see
// OrbitPreviewLabel::keyOrbitRequested()/keyZoomRequested()'s own
// comments for why these are separate slots rather than routed through
// the mouse ones, and mainwindow.h's comment on m_keyboardSensitivity
// for why it's one multiplier per device rather than per axis.
void MainWindow::onLivePreviewKeyOrbit(int azimuthSteps, int elevationSteps) {
	if (!m_livePreviewRunning) return;
	// Deliberately coarser per-press than the mouse's per-pixel rate (which
	// fires many times over one drag) - a single key press should be a
	// noticeable, discrete nudge, not an imperceptible fraction of one.
	constexpr double kRadiansPerKeyStep = 0.05;
	applyOrbitDelta(azimuthSteps * kRadiansPerKeyStep * m_keyboardSensitivity,
					 elevationSteps * kRadiansPerKeyStep * m_keyboardSensitivity);
}

void MainWindow::onLivePreviewKeyZoom(int radiusSteps) {
	if (!m_livePreviewRunning) return;
	// ~15% radius change per press at 1.0x - a single press should read as
	// a deliberate zoom step, matching kRadiansPerKeyStep's own "noticeable
	// per press" intent above rather than the mouse wheel's much finer
	// per-notch granularity.
	constexpr double kZoomFactorPerKeyStep = 0.85;
	applyZoomDelta(std::pow(kZoomFactorPerKeyStep, radiusSteps * m_keyboardSensitivity));
}

void MainWindow::onLivePreviewTranslate(int forwardSteps, int rightSteps, int upSteps) {
	if (!m_livePreviewRunning) return;
	// World-space distance per step - tuned against the Cornell Box's own
	// ~555-unit scale (a comfortable walking pace across the room takes a
	// handful of presses, not one giant leap or an imperceptible creep).
	// Scaled here rather than inside applyTranslateDelta(), matching
	// onLivePreviewKeyOrbit()/onLivePreviewKeyZoom()'s own "resolve steps to
	// a real delta before calling the shared apply* helper" convention.
	constexpr double kUnitsPerStep = 20.0;
	const double scale = kUnitsPerStep * m_keyboardSensitivity;
	applyTranslateDelta(forwardSteps * scale, rightSteps * scale, upSteps * scale);
}

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
#ifdef RT_GUI_HAVE_GPU
	if (page == m_livePreviewPage) {
		// Stop BEFORE clearing the tracking pointers below - stopLivePreview()
		// itself dereferences m_livePreviewStatusLabel, so it must run while
		// that pointer is still valid rather than relying on removeTab()'s
		// currentChanged -> stopLivePreviewIfNavigatedAway() to catch this
		// indirectly (which would fire only after the pointers were already null).
		stopLivePreview();
		m_livePreviewPage = nullptr;
		m_livePreviewLabel = nullptr;
		m_livePreviewStatusLabel = nullptr;
	}
#endif
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

// Trailing three lines shared by every addXPreviewTab() - the page's own
// contents (image label / video player / orbit label) genuinely differ per
// caller and stay in each function, but registering the finished page and
// making it current is identical for all three.
void MainWindow::addPreviewSubTabPage(QWidget *page, const QString &title, const QString &tooltip) {
	const int index = m_previewSubTabs->addTab(page, uniquePreviewTabTitle(title));
	m_previewSubTabs->setTabToolTip(index, tooltip);
	m_previewSubTabs->setCurrentIndex(index);
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

	addPreviewSubTabPage(page, title, tooltip);
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

	addPreviewSubTabPage(page, title, tooltip);

	player->setSource(QUrl::fromLocalFile(videoPath));
	player->play();
}
