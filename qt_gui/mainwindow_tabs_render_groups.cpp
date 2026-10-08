// mainwindow_tabs_render_groups.cpp - the option groups of the Render Options tab (Integrator, Sampling, Accelerator, Post-processing, Denoiser, Crop,
// Depth of Field, Seed), built by createRenderOptionsTab() in mainwindow_tabs_render.cpp. A pure split: nothing here changed.

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


// The "(i)" mark + per-row tooltip pattern every enum-valued combo on the
// Render Options tab uses (Sampler/LightSampler/Accelerator/SplitMethod) -
// was 4 independently hand-copied local struct+loop pairs before being
// factored into this one shared method (see ComboEntry's own comment,
// mainwindow.h).
void MainWindow::populateComboEntries(QComboBox *combo, std::initializer_list<ComboEntry> entries) {
	for (const ComboEntry &entry : entries) {
		icon_tint::addItem(combo, ":/icons/info.svg", entry.label, entry.value, m_activeTheme.textBody);
		setRichItemTooltip(combo, combo->count() - 1, entry.tooltip);
	}
}

// The Integrator group: the algorithm selector, and a page of sub-flags for whichever alternate integrator it selects.
void MainWindow::buildIntegratorGroup(QWidget *optionsTab, QVBoxLayout *layout) {
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
		tr("Choose which rendering method to use - the default path tracer, "
		"or one of several alternate methods (with names like SPPM, BDPT, "
		"MLT, or AO) that each have their own extra options shown below "
		"once picked. These alternates only run on the CPU and can't be "
		"used together with Video mode.")));
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
			setRichItemTooltip(m_integratorCombo, m_integratorCombo->count() - 1,
				integratorDescription(modes[i]));
		}
	}
	styleComboBox(m_integratorCombo);
	m_integratorCombo->setToolTip(
		tr("Which rendering method to use. Path Tracer (the default) is the\n"
		"well-tested, general-purpose choice - the alternates below trade\n"
		"that generality for a specific technique (like photon mapping, or\n"
		"tracing light from both the camera and the light source and\n"
		"connecting them), or are simplified versions used for testing and\n"
		"comparison. All alternates run on the CPU only except SPPM, and none\n"
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
		tr("This picks the rendering method itself, not just how fast it "
		"runs. Path Tracer (the default) is the general-purpose, "
		"well-tested choice used everywhere else in this app.\n\n"
		"SPPM (a photon-mapping technique) handles tricky glass and "
		"focused-light effects that regular path tracing struggles with. "
		"BDPT and MLT (built on top of BDPT) trace rays starting from both "
		"the camera and the light source and connect them together in the "
		"middle - this helps with some difficult lighting setups, but only "
		"works with area lights. RandomWalk, Ambient Occlusion, "
		"SimplePath, SimpleVolPath, and LightPath are reference and "
		"debugging modes - simpler, often noisier or narrower in what they "
		"show (for example, Ambient Occlusion doesn't produce a real lit "
		"image at all), useful for isolating what one specific technique "
		"contributes to the final picture.\n\nHover any item in the "
		"dropdown for details on that specific mode.")),
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
		tr("How many rounds of the SPPM technique to run (each round traces "
		"rays from the camera, then traces simulated light particles from "
		"the lights). More rounds build up a cleaner result, at a render "
		"time cost that grows roughly in proportion.")),
		m_sppmIterationsSpin);
	m_sppmPhotonsSpin = new QSpinBox(sppmPage);
	m_sppmPhotonsSpin->setRange(1, 100000000);
	m_sppmPhotonsSpin->setValue(5000);
	styleSpinBox(m_sppmPhotonsSpin);
	sppmLayout->addRow(labelWithInfo(tr("Photons per iteration:"),
		tr("How many simulated light particles (photons) are sent out from "
		"the lights during each round. More photons reduce graininess in "
		"bounced and focused lighting (like light through glass), at the "
		"cost of a slower round.")),
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
		tr("The most bounces allowed on each of the two ray paths - one "
		"starting from the camera, one from the light - that this method "
		"traces and then joins together.")),
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
		tr("How many candidate light paths this method tries out up "
		"front, for each bounce depth, before it starts refining from "
		"them - more gives it a better-informed starting point.")),
		m_mltBootstrapSpin);
	m_mltMutationsSpin = new QSpinBox(mltPage);
	m_mltMutationsSpin->setRange(1, 1000000000);
	m_mltMutationsSpin->setValue(4000000);
	styleSpinBox(m_mltMutationsSpin);
	mltLayout->addRow(labelWithInfo(tr("Mutations:"),
		tr("The total number of small random tweaks this method tries "
		"while refining its light paths, added up across all of its "
		"parallel search chains - the main render time/quality knob "
		"here, similar to what samples per pixel controls in the default "
		"path tracer.")),
		m_mltMutationsSpin);
	m_mltMaxDepthSpin = new QSpinBox(mltPage);
	m_mltMaxDepthSpin->setRange(1, 100);
	m_mltMaxDepthSpin->setValue(5);
	styleSpinBox(m_mltMaxDepthSpin);
	mltLayout->addRow(labelWithInfo(tr("Max path depth:"),
		tr("Same meaning as BDPT's max path depth above - this method is "
		"built directly on top of that same two-sided path-tracing "
		"machinery.")),
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
		tr("How far a test ray is allowed to travel before it's "
		"considered to have found open sky (nothing blocking it). The "
		"default (10 billion) is effectively unlimited - lower it if you "
		"only want nearby objects to count as blocking.")),
		m_aoMaxDistSpin);
	m_aoUniformCheck = new QCheckBox(tr("Uniform-hemisphere sampling (instead of cosine)"), aoPage);
	styleCheckBox(m_aoUniformCheck);
	aoLayout->addRow(checkboxWithInfo(m_aoUniformCheck,
		tr("By default, test rays are aimed more toward straight-up-"
		"from-the-surface directions, matching how a plain matte surface "
		"is actually lit in real life. Turning this on spreads the test "
		"rays out evenly in every direction instead - a different, "
		"unweighted way of measuring the same thing.")));
	m_aoIllumScaleSpin = new QDoubleSpinBox(aoPage);
	m_aoIllumScaleSpin->setRange(0.0, 1000.0);
	m_aoIllumScaleSpin->setValue(1.0);
	styleSpinBox(m_aoIllumScaleSpin);
	aoLayout->addRow(labelWithInfo(tr("Illumination scale:"),
		tr("A flat brightness multiplier applied to the occlusion color "
		"below.")),
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
		tr("The color used to visualize how occluded (blocked-off) each "
		"point is - since this mode isn't a real lit render, this is just "
		"a display choice, not an actual light color. Default is white "
		"(1, 1, 1).")),
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
		tr("On by default. This aims a ray directly at a light source on "
		"every bounce, instead of hoping a random bounce happens to hit "
		"one - it sharply cuts down graininess in scenes with small, "
		"bright lights. Turning it off falls back to finding lights only "
		"by chance, the way a bare-bones path tracer would.")));
	m_simplepathNoBsdfCheck = new QCheckBox(tr("Disable BSDF importance sampling"), simplepathPage);
	styleCheckBox(m_simplepathNoBsdfCheck);
	simplepathLayout->addRow(checkboxWithInfo(m_simplepathNoBsdfCheck,
		tr("On by default. Picks each bounce's new direction weighted "
		"toward the directions the surface's material actually reflects "
		"light in, rather than guessing blindly. Turning it off falls "
		"back to picking directions evenly at random, which is less "
		"efficient.")));
	m_integratorOptionsStack->addWidget(simplepathPage);

	integratorGroupLayout->addWidget(m_integratorOptionsStack);
	layout->addWidget(m_integratorOptionsGroup);
}

// The Sampling & Spectral group (CPU default path tracer only).
void MainWindow::buildSamplingGroup(QWidget *optionsTab, QVBoxLayout *layout) {
	// ------------------------------------------------------------------
	// Sampling & Spectral group - CPU default path tracer only
	// ------------------------------------------------------------------
	// "&&" (not "&") - a single "&" is a Qt mnemonic-accelerator marker,
	// which would eat the "&" and underline the next letter instead of
	// showing a literal ampersand.
	InfoGroupBox *samplingGroup = new InfoGroupBox(tr("Sampling && Spectral"), optionsTab);
	styleGroupBox(samplingGroup);
	samplingGroup->setInfoIcon(createInfoIcon(
		tr("Which random-sampling method is used, whether to simulate light "
		"by individual wavelength, whether to stop early on pixels that "
		"already look clean, and a time-limit alternative to a fixed sample "
		"count. These control HOW samples are gathered, separately from HOW "
		"MANY (set on the Settings tab).")));
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
			"A well-spread pattern of sample points (mixed up "
			"differently in each pixel) that fills in gaps more evenly "
			"than pure randomness. The best general-purpose default - "
			"it cleans up the image quickly without leaving visible "
			"patterns.")},
		{tr("Z-Sobol"), QStringLiteral("zsobol"), tr(
			"A variant of the Sobol pattern above, reordered to work "
			"better when samples are taken progressively (a few at a "
			"time) rather than all at once. Cleans up the image at "
			"least as well as plain Sobol, with better behavior when "
			"combined with adaptive sampling.")},
		{tr("Padded Sobol"), QStringLiteral("paddedsobol"), tr(
			"The Sobol pattern above, extended with extra random "
			"dimensions - this avoids repeating patterns showing up "
			"when a pixel needs more random choices than plain Sobol "
			"comfortably covers (for example, light paths with many "
			"bounces).")},
		{tr("Stratified"), QStringLiteral("stratified"), tr(
			"Splits each pixel into a small grid of sub-cells and "
			"takes one sample from each cell. Simple, predictable "
			"coverage - less refined than Sobol/Halton above, but "
			"useful as a plain reference to compare against.")},
		{tr("PMJ02BN"), QStringLiteral("pmj02bn"), tr(
			"A sampling pattern designed to spread samples especially "
			"evenly between neighboring pixels (what's called "
			"\"blue-noise\" distribution), avoiding clumps of similar "
			"samples landing next to each other.")},
		{tr("Halton"), QStringLiteral("halton"), tr(
			"A classic, well-spread sampling pattern built from a "
			"well-established mathematical formula. Well-tested, and "
			"avoids the grid-like clustering that plain stratified "
			"sampling above can show.")},
		{tr("Independent (no stratification)"), QStringLiteral("independent"), tr(
			"Plain, ordinary random numbers, with none of the "
			"deliberate even-spacing the other options use. Included "
			"so a loaded .pbrt scene file that specifically asks for "
			"this can be reproduced faithfully - not a recommended "
			"choice otherwise.")},
	});
	m_samplerCombo->setToolTip(
		tr("Which method generates the random decisions used while\n"
		"rendering (all but Independent spread samples out more evenly\n"
		"than pure randomness). CPU default path tracer only - no effect\n"
		"on GPU or under the alternate rendering methods above."));
	styleComboBox(m_samplerCombo);
	samplingLayout->addRow(labelWithInfo(tr("Sampler:"),
		tr("Rendering needs a lot of random numbers - which direction to "
		"bounce a ray, which point on a light to aim at, and so on - and "
		"HOW those \"random\" choices are generated changes how quickly "
		"the image builds up into a clean result.\n\n"
		"Ordinary random numbers tend to clump together in some spots and "
		"leave gaps in others. Most samplers here (Sobol, Halton, etc.) "
		"instead use patterns deliberately spread out to cover all the "
		"possibilities more evenly, which cleans up the image faster than "
		"true randomness would for the same number of samples. Independent "
		"is the exception - plain, uncorrelated random numbers, included "
		"so a loaded .pbrt scene file that specifically asks for it can be "
		"reproduced faithfully, not as a recommended choice.\n\n"
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
			"Organizes the scene's lights into a quick-lookup index and "
			"weighs each one by both its brightness and how close it is "
			"to the point being shaded, adjusting per bounce instead of "
			"using one fixed weighting for the whole scene. The "
			"underlying renderer's own default - generally converges to "
			"a clean image fastest, at a small extra bookkeeping cost.")},
		{tr("Auto (use scene's own request)"), QStringLiteral("auto"), tr(
			"Uses whatever light-sampling method the loaded scene file "
			"itself asks for (falling back to BVH above if it doesn't "
			"ask for anything, or asks for something this app doesn't "
			"support) instead of a fixed choice. Picking this once and "
			"leaving it is the one choice here that stays correct as "
			"you switch between scenes with different recommendations.")},
		{tr("Power"), QStringLiteral("power"), tr(
			"Picks a light to sample with odds weighted by its overall "
			"brightness - brighter lights get picked more often than "
			"dim ones. Cleans up the image faster than picking lights "
			"with equal odds in scenes with a wide range of light "
			"brightness, but it ignores how far away or how blocked-off "
			"a light is.")},
		{tr("Uniform"), QStringLiteral("uniform"), tr(
			"Picks a light completely at random from everything in the "
			"scene, with equal odds regardless of brightness or "
			"distance. Simple, but cleans up slowly in scenes with many "
			"lights of very different brightness - a dim light gets "
			"picked just as often as a bright one.")},
	});
	m_lightSamplerCombo->setToolTip(
		tr("Which strategy picks the light to aim a ray at directly, each\n"
		"time a bounce tries to sample light straight from a source.\n"
		"Affects how grainy the image looks along the way and how fast it\n"
		"cleans up, not what it eventually converges to. CPU default path\n"
		"tracer only - no effect on GPU or under the alternate rendering\n"
		"methods above."));
	styleComboBox(m_lightSamplerCombo);
	samplingLayout->addRow(labelWithInfo(tr("Light Sampler:"),
		tr("Every bounce off a matte or semi-glossy surface needs to "
		"pick ONE light (out of potentially many in the scene) to aim a "
		"ray directly at - which light gets picked, and how fairly, "
		"changes how quickly the image cleans up, though never what it "
		"eventually looks like.\n\n"
		"BVH (the default) organizes the scene's lights into a "
		"quick-lookup index and adjusts its weighting for each point "
		"being shaded - both bright AND nearby lights get preferred. "
		"Auto instead uses whatever the loaded scene file itself asks "
		"for (falling back to BVH if it doesn't ask for anything). "
		"Power picks by brightness alone, ignoring position - simpler, "
		"and worse in scenes where lights are at very different "
		"distances. Uniform ignores both - every light is equally "
		"likely regardless of brightness or distance, included mainly "
		"for comparison and debugging.\n\nGrayed out? This only affects "
		"the CPU renderer's default path tracer - switch Renderer to "
		"CPU on the Settings tab to use it.")),
		m_lightSamplerCombo);
	connect(m_lightSamplerCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		if (m_sceneCombo) updateSceneRecommendedSettingsHint(m_sceneCombo->currentData().toString());
	});

	m_spectralCheck = new QCheckBox(tr("Spectral rendering (--spectral)"), optionsTab);
	m_spectralCheck->setToolTip(
		tr("Simulates individual wavelengths of light (like a rainbow)\n"
		"instead of simplifying everything to red/green/blue. CPU default\n"
		"path tracer only. Only lambertian, metal, dielectric,\n"
		"rough_dielectric, conductor, and diffuse_light materials are\n"
		"supported - a scene using anything else fails to render rather\n"
		"than silently rendering wrong colors. Noticeably slower per-sample."));
	styleCheckBox(m_spectralCheck);
	samplingLayout->addRow(checkboxWithInfo(m_spectralCheck,
		tr("Ordinary rendering tracks light as just three numbers - red, "
		"green, blue - the same way a screen displays color.\n\n"
		"Real light is actually a continuous spectrum of wavelengths, "
		"and a few physical effects (like a prism splitting white light "
		"into a rainbow) only happen because different wavelengths bend "
		"by different amounts - plain red/green/blue can't represent "
		"that. This option tracks a handful of actual individual "
		"wavelengths per ray instead, at the cost of a grainier, slower "
		"render.\n\n"
		"Grayed out? This only exists on the CPU renderer's default "
		"path tracer - switch Renderer to CPU on the Settings "
		"tab to use it.")));

	m_adaptiveSamplingCheck = new QCheckBox(tr("Adaptive sampling (--adaptive)"), optionsTab);
	m_adaptiveSamplingCheck->setToolTip(
		tr("Stops adding more samples to a pixel once it already looks\n"
		"clean, instead of always spending the full Samples budget on\n"
		"every pixel - Samples becomes a ceiling, not a fixed amount every\n"
		"pixel must use. CPU default path tracer only."));
	styleCheckBox(m_adaptiveSamplingCheck);
	m_adaptiveThresholdSpin = new QDoubleSpinBox(optionsTab);
	m_adaptiveThresholdSpin->setRange(0.0001, 1.0);
	m_adaptiveThresholdSpin->setDecimals(4);
	m_adaptiveThresholdSpin->setValue(0.01);
	m_adaptiveThresholdSpin->setSingleStep(0.005);
	m_adaptiveThresholdSpin->setEnabled(false);
	m_adaptiveThresholdSpin->setToolTip(
		tr("How clean a pixel must look before it's considered done.\n"
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
			tr("Rendering builds up an image from many random samples, so "
			"it looks grainy at first and gradually cleans up - but "
			"different pixels clean up at different speeds. A pixel on a "
			"bright, evenly-lit wall might look clean after just a few "
			"samples, while a dim corner lit only by a small window can "
			"take far more before its graininess settles down. Spending "
			"the same fixed number of samples on both wastes time on "
			"pixels that were already done.\n\n"
			"Adaptive sampling keeps track of how grainy each pixel still "
			"looks and stops adding samples to it early once that drops "
			"below the threshold below, letting Samples act as a ceiling "
			"rather than a flat amount every pixel must use - the same "
			"idea as Blender Cycles' own adaptive sampling.\n\n"
			"Grayed out? This only affects the CPU renderer's default path "
			"tracer - switch Renderer to CPU on the Settings tab to "
			"use it.")), adaptiveRow);
	}

	m_timeLimitCheck = new QCheckBox(tr("Time limit (--time-limit)"), optionsTab);
	m_timeLimitCheck->setToolTip(
		tr("Stop rendering once this many seconds have passed, instead of\n"
		"always running until the whole image is finished. CPU default\n"
		"path tracer only."));
	styleCheckBox(m_timeLimitCheck);
	m_timeLimitSpin = new QDoubleSpinBox(optionsTab);
	m_timeLimitSpin->setRange(1.0, 86400.0);
	m_timeLimitSpin->setDecimals(0);
	m_timeLimitSpin->setValue(60.0);
	m_timeLimitSpin->setSingleStep(10.0);
	m_timeLimitSpin->setSuffix(tr(" s"));
	m_timeLimitSpin->setEnabled(false);
	m_timeLimitSpin->setToolTip(
		tr("How many seconds to render before stopping, regardless of the\n"
		"Samples budget above."));
	styleSpinBox(m_timeLimitSpin);
	connect(m_timeLimitCheck, &QCheckBox::toggled, m_timeLimitSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *timeLimitRow = new QWidget(optionsTab);
		QHBoxLayout *timeLimitRowLayout = new QHBoxLayout(timeLimitRow);
		timeLimitRowLayout->setContentsMargins(0, 0, 0, 0);
		timeLimitRowLayout->addWidget(m_timeLimitCheck);
		timeLimitRowLayout->addWidget(m_timeLimitSpin, 1);
		samplingLayout->addRow(checkboxWithInfo(m_timeLimitCheck,
			tr("Useful for a fixed preview or a shared-computer time budget, "
			"instead of guessing a sample count that happens to finish in "
			"time - lets Samples above stay a generous ceiling while this "
			"decides when to actually stop.\n\n"
			"It stops row by row: whichever rows of the image were already "
			"being worked on when time runs out still finish normally; any "
			"row that was never started is left black instead of skipped "
			"over, so you still get a valid (if incomplete) image rather "
			"than a broken file.\n\n"
			"Generating a video? This is a budget for the WHOLE video, not "
			"each frame - later frames get whatever time is left, and any "
			"frames still remaining once it runs out are skipped entirely.\n\n"
			"Grayed out? This only affects the CPU renderer's default path "
			"tracer - switch Renderer to CPU on the Settings tab to "
			"use it.")), timeLimitRow);
	}

	m_exposureSpin = new QDoubleSpinBox(optionsTab);
	m_exposureSpin->setRange(0.01, 100.0);
	m_exposureSpin->setValue(1.0);
	m_exposureSpin->setSingleStep(0.1);
	m_exposureSpin->setToolTip(
		tr("A flat brightness multiplier applied before the final\n"
		"brightness/contrast adjustment (1.0 = no change). Both CPU and\n"
		"GPU default path tracer only."));
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
		tr("Slightly softens a rough, reflective, or glassy surface's\n"
		"sharpness after the path's first non-mirror-like bounce - this\n"
		"calms down fireflies (isolated bright speckles) from hard-to-\n"
		"trace light paths, at the cost of a little extra blur. CPU and\n"
		"OptiX GPU default path tracer only - not implemented under Metal\n"
		"(macOS GPU rendering). A scene that already asks for this itself\n"
		"is unaffected - this checkbox only ever adds the request, never\n"
		"removes it."));
	styleCheckBox(m_regularizeCheck);
	samplingLayout->addRow(checkboxWithInfo(m_regularizeCheck,
		tr("Some light paths are genuinely hard for a path tracer to find "
		"cleanly - light that bounces off a rough (but not mirror-"
		"perfect) surface, through another rough surface, into a small "
		"bright light. Those paths show up as fireflies: single rays "
		"that happen to catch a very bright, small light at just the "
		"right angle, appearing as an isolated bright speckle that "
		"takes a very long time to average away.\n\n"
		"This setting deliberately softens a surface's roughness a "
		"little more with each non-mirror-like bounce a light path has "
		"already taken - it introduces a small, technically-incorrect "
		"bias, but in exchange the fireflies clean up dramatically "
		"faster, which usually looks better in the final image.\n\n"
		"Off by default. If a loaded .pbrt scene file already asks for "
		"this itself, it's applied either way - this checkbox can only "
		"add the request on top, never take it away.")));

	m_maxComponentValueCheck = new QCheckBox(tr("Firefly clamp (--maxcomponentvalue)"), optionsTab);
	m_maxComponentValueCheck->setToolTip(
		tr("Caps any single sample whose brightest color channel exceeds\n"
		"the value below, scaling all its channels down together so the\n"
		"color/hue stays the same. CPU and both OptiX GPU backends - the\n"
		"recursive GPU mode matches the CPU exactly, while the wavefront\n"
		"GPU mode applies it slightly differently (per light bounce\n"
		"rather than per whole sample). Not implemented under Metal\n"
		"(macOS GPU rendering)."));
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
			tr("Ray-traced rendering occasionally comes up with a sample "
			"that's technically correct but extremely bright - a ray that "
			"happens to catch a small, intense light at just the right "
			"angle - and one such sample can dominate a pixel's average "
			"for a long time before enough other samples arrive to "
			"balance it out. These show up as fireflies: single bright "
			"speckles standing out against the rest of the image.\n\n"
			"This clamp caps how bright any single sample's brightest "
			"color channel is allowed to be before it gets averaged in, "
			"trading a small, controlled inaccuracy for a much "
			"cleaner-looking image at the same sample count - lower "
			"values clean up more aggressively but risk visibly dimming "
			"genuinely bright small lights, not just stray noise.\n\n"
			"Off by default (effectively unlimited). CPU default path "
			"tracer only.")), clampRow);
	}

	layout->addWidget(samplingGroup);
}

// The Accelerator group (CPU, shared by every integrator).
void MainWindow::buildAcceleratorGroup(QWidget *optionsTab, QVBoxLayout *layout) {
	// ------------------------------------------------------------------
	// Accelerator group - CPU only, but (unlike Sampling & Spectral above)
	// shared by every integrator, not just the default path tracer - see
	// updateRenderOptionsEnabled()'s own comment for why this combo pair's
	// enabled condition differs from every other CPU-only control here.
	// ------------------------------------------------------------------
	InfoGroupBox *acceleratorGroup = new InfoGroupBox(tr("Accelerator"), optionsTab);
	styleGroupBox(acceleratorGroup);
	acceleratorGroup->setInfoIcon(createInfoIcon(
		tr("A CPU-only setting for the index the renderer builds to quickly "
		"skip past objects a ray obviously can't hit, instead of checking "
		"every single object in the scene, plus the strategy used to build "
		"that index. Applies to every rendering method, not just the "
		"default path tracer - the defaults work well for almost every "
		"scene.")));
	QFormLayout *acceleratorLayout = new QFormLayout(acceleratorGroup);
	acceleratorLayout->setVerticalSpacing(10);
	acceleratorLayout->setHorizontalSpacing(10);
	acceleratorLayout->setContentsMargins(15, 22, 15, 12);

	m_acceleratorCombo = new QComboBox(optionsTab);
	populateComboEntries(m_acceleratorCombo, {
		{tr("Scene's own choice (default)"), QString(), tr(
			"Leaves a loaded .pbrt scene file's own accelerator choice "
			"alone (falling back to BVH below if it didn't name one). "
			"Has no effect either way on a scene that isn't loaded from "
			"a .pbrt file.")},
		{tr("BVH"), QStringLiteral("bvh"), tr(
			"An index built by grouping nearby objects inside a "
			"hierarchy of bounding boxes, using the split strategy "
			"chosen below - this app's default index type.")},
		{tr("Kd-tree"), QStringLiteral("kdtree"), tr(
			"A different indexing structure that divides up space "
			"itself into regions, instead of grouping objects into "
			"boxes. It has no split-strategy option of its own (the "
			"combo below is ignored when this is chosen). Falls back "
			"to BVH above on a scene with moving-object motion blur "
			"(this app's version of this index type can't handle "
			"that) - a warning is printed when that happens.")},
	});
	m_acceleratorCombo->setToolTip(
		tr("Which indexing structure organizes the scene's geometry for\n"
		"fast ray tests. Every choice renders the identical final image -\n"
		"this only affects build time and render speed, not quality. CPU\n"
		"only, every rendering method - no effect on GPU (which always\n"
		"uses its own fixed index) or a scene that wasn't loaded from a\n"
		".pbrt file (see the log)."));
	styleComboBox(m_acceleratorCombo);
	acceleratorLayout->addRow(labelWithInfo(tr("Accelerator:"),
		tr("If the renderer had to check every single object in the "
		"scene for every ray, even simple scenes would be painfully "
		"slow. This index lets a ray quickly skip past objects it "
		"obviously can't hit, and only test the handful of objects "
		"actually near where it travels.\n\n"
		"BVH (grouping nearby objects into a hierarchy of boxes) and "
		"Kd-tree (dividing up space itself) are two different real "
		"strategies for organizing the same geometry - both produce "
		"the exact same rendered image, just at different build and "
		"render speeds depending on the scene's shape.\n\n"
		"Only a scene loaded from a .pbrt file has its own accelerator "
		"choice to override at all - any other scene always uses its "
		"own fixed BVH-style index regardless of this setting (a "
		"warning is printed if you pick something else anyway).")),
		m_acceleratorCombo);
	connect(m_acceleratorCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		updateRenderOptionsEnabled();
	});

	m_splitMethodCombo = new QComboBox(optionsTab);
	populateComboEntries(m_splitMethodCombo, {
		{tr("SAH (default)"), QString(), tr(
			"Tries out several ways of splitting the index and "
			"estimates which one will be fastest to search later, then "
			"picks the cheapest. Slower to build than Middle/Equal "
			"Counts below, but produces the best-performing index for "
			"most scenes - this app's long-standing default.")},
		{tr("Middle"), QStringLiteral("middle"), tr(
			"Splits each group of objects right down the middle of its "
			"longest side. Cheap and fast to build, with no cost "
			"estimation at all - but can perform poorly on "
			"unevenly-spread-out geometry.")},
		{tr("Equal counts"), QStringLiteral("equal"), tr(
			"Splits each group so an equal number of objects fall on "
			"each side, regardless of how they're actually spread out "
			"in space. Cheap to build, but can produce badly-shaped "
			"groups when objects are clustered together.")},
		{tr("HLBVH"), QStringLiteral("hlbvh"), tr(
			"Builds the index from the bottom up using a fast "
			"spatial-sorting trick, making it the fastest of these "
			"four strategies to build for very large numbers of "
			"triangles - at some cost to how well the finished index "
			"performs later, compared to SAH above.")},
	});
	m_splitMethodCombo->setToolTip(
		tr("How the BVH index above is built (ignored when Accelerator is\n"
		"set to Kd-tree). Every choice renders the identical final image.\n"
		"Falls back to SAH above on a scene with moving-object motion blur\n"
		"(the other build strategies here can't handle that) - a warning\n"
		"is printed when that happens."));
	styleComboBox(m_splitMethodCombo);
	acceleratorLayout->addRow(labelWithInfo(tr("BVH split method:"),
		tr("Only used when the accelerator above is set to BVH (ignored "
		"for Kd-tree, which has no split-strategy option). All four "
		"strategies build the same general kind of index in a "
		"different way - SAH above spends more time building in "
		"exchange for a better-performing index; Middle and Equal "
		"Counts are cheap, simple fallbacks; HLBVH trades a little "
		"performance for the fastest build on very large scenes.")),
		m_splitMethodCombo);
	connect(m_splitMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int) {
		updateRenderOptionsEnabled();
	});

	layout->addWidget(acceleratorGroup);
}

// The Post-Processing & Diagnostics group.
void MainWindow::buildPostProcessingGroup(QWidget *optionsTab, QVBoxLayout *layout) {
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
		tr("How the image's brightness/contrast is adjusted for display, "
		"whether to print render statistics, and a GPU debugging mode - "
		"these control how the final image is processed and reported, not "
		"what to render. Denoising (cleaning up graininess) has its own "
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
		tr("Which brightness/contrast curve to apply before converting to\n"
		"the final display colors. Applies to both CPU and GPU (both GPU\n"
		"modes) - no effect under the alternate rendering methods above."));
	styleComboBox(m_tonemapCombo);
	outputLayout->addRow(labelWithInfo(tr("Tone mapping:"),
		tr("A rendered scene's true brightness values have no upper "
		"limit - a light bulb might be a hundred times brighter than a "
		"wall - but a screen can only show a fixed range of brightness. "
		"Tone mapping is the curve used to compress that huge range "
		"down into something a screen can actually display.\n\n"
		"ACES rolls off bright highlights gently, the way film does, "
		"giving a soft, filmic look; Reinhard is a simpler, older way "
		"of compressing brightness; None just clips anything too "
		"bright straight to flat white, which can look harsh.")),
		m_tonemapCombo);

	m_statsCheck = new QCheckBox(tr("Print render stats"), optionsTab);
	m_statsCheck->setToolTip(
		tr("Print a small end-of-render statistics summary (rays cast,\n"
		"bounces, shadow rays, samples/sec) to the Log tab. Purely\n"
		"informational - it never changes the rendered image."));
	styleCheckBox(m_statsCheck);
	outputLayout->addRow(checkboxWithInfo(m_statsCheck,
		tr("Prints a short summary after the render finishes - how many "
		"rays were cast, how many bounces happened, how many shadow "
		"rays were traced (rays checking whether a point can see a "
		"light), and samples per second.\n\n"
		"Purely informational: it never changes the rendered image, "
		"it just tells you what the renderer actually did.")));

	m_optixValidateCheck = new QCheckBox(tr("OptiX validation mode (slower, debugging only)"), optionsTab);
	m_optixValidateCheck->setToolTip(
		tr("Turns on extra GPU-side correctness checks, which have a real\n"
		"performance cost each time the GPU runs. OptiX GPU only (not\n"
		"available under Metal, macOS GPU rendering), meant for\n"
		"debugging, not routine use."));
	styleCheckBox(m_optixValidateCheck);
	outputLayout->addRow(checkboxWithInfo(m_optixValidateCheck,
		tr("Turns on extra correctness checks inside the GPU rendering "
		"process itself, catching certain kinds of bugs that would "
		"otherwise silently produce a wrong image or crash "
		"unpredictably.\n\n"
		"It's a debugging aid for people working on the renderer's "
		"own GPU code, not something a normal render benefits from - "
		"it has a real performance cost and doesn't change what a "
		"correct render looks like.\n\n"
		"This is specific to the OptiX GPU backend (Windows); it has no "
		"Metal equivalent, so it stays grayed out even with Renderer "
		"set to GPU on macOS.")));

	layout->addWidget(outputGroup);
}

// The Denoiser group, with its Image & Video and Live Preview subsections.
void MainWindow::buildDenoiserGroup(QWidget *optionsTab, QVBoxLayout *layout) {
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
		tr("Runs an AI denoiser on the finished render to smooth out\n"
		"graininess, using extra information about each pixel's base\n"
		"color and surface direction to do a better job than a plain\n"
		"blur. OptiX GPU only, both GPU modes (recursive and wavefront\n"
		"each have their own denoiser) - not available under Metal\n"
		"(macOS GPU rendering)."));
	styleCheckBox(m_denoiseCheck);
	m_denoiseBlendSpin = new QDoubleSpinBox(optionsTab);
	m_denoiseBlendSpin->setRange(0.0, 1.0);
	m_denoiseBlendSpin->setDecimals(2);
	m_denoiseBlendSpin->setValue(0.0);
	m_denoiseBlendSpin->setSingleStep(0.05);
	m_denoiseBlendSpin->setEnabled(false);
	m_denoiseBlendSpin->setToolTip(
		tr("Blends between the grainy original and the fully denoised\n"
		"result (0.0 = fully denoised, 1.0 = original grainy image).\n"
		"Raise this toward 1.0 to keep back more fine texture/grain that\n"
		"full-strength denoising can smooth away."));
	styleSpinBox(m_denoiseBlendSpin);
	connect(m_denoiseCheck, &QCheckBox::toggled, m_denoiseBlendSpin, &QDoubleSpinBox::setEnabled);
	{
		QWidget *denoiseRow = new QWidget(optionsTab);
		QHBoxLayout *denoiseRowLayout = new QHBoxLayout(denoiseRow);
		denoiseRowLayout->setContentsMargins(0, 0, 0, 0);
		denoiseRowLayout->addWidget(m_denoiseCheck);
		denoiseRowLayout->addWidget(m_denoiseBlendSpin, 1);
		denoiserImageVideoLayout->addRow(checkboxWithInfo(m_denoiseCheck,
			tr("Rendering is grainy by nature when only a few samples are "
			"used, which is why more samples usually means a cleaner "
			"picture (but also a slower render).\n\n"
			"A denoiser is an AI model trained to recognize that "
			"graininess and smooth it away after the fact, without "
			"needing to trace additional rays - a way to get a "
			"clean-looking image faster, at some cost in fine detail. The "
			"number to its right blends between the noisy original and "
			"the fully denoised result - 0 is fully denoised (the "
			"default); raising it keeps back some of the original grain, "
			"useful when full-strength denoising smooths away texture "
			"you wanted to keep.\n\n"
			"Grayed out? This needs the OptiX GPU backend (Windows) - "
			"switch Renderer to GPU on the Settings tab. Both the "
			"recursive and wavefront GPU modes support it; it has no "
			"Metal equivalent, so it stays grayed out on macOS.")), denoiseRow);
	}

#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	buildDenoiserLivePreviewSubsection(optionsTab);
#endif

	denoiserGroupLayout->addWidget(m_denoiserImageVideoGroupBox);
	setGroupDimmed(m_denoiserImageVideoGroupBox, isLiveMode());
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	denoiserGroupLayout->addWidget(m_denoiserLivePreviewGroupBox);
	setGroupDimmed(m_denoiserLivePreviewGroupBox, !isLiveMode());
#endif

	layout->addWidget(m_denoiserGroupBox);
#ifdef RT_GUI_HAVE_LIVE_PREVIEW
	buildLivePreviewSettingsSection(optionsTab, layout);
#endif
}

// The Crop Window group.
void MainWindow::buildCropGroup(QWidget *optionsTab, QVBoxLayout *layout) {
	// ------------------------------------------------------------------
	// Crop Window group
	// ------------------------------------------------------------------
	InfoGroupBox *cropGroup = new InfoGroupBox(tr("Crop Window"), optionsTab);
	styleGroupBox(cropGroup);
	cropGroup->setInfoIcon(createInfoIcon(
		tr("Render only a rectangular slice of the full frame, given as "
		"fractions from 0 to 1 of the image's width and height - useful "
		"for quickly test-rendering one area of a scene without paying "
		"for the whole image.")));
	QFormLayout *cropLayout = new QFormLayout(cropGroup);
	cropLayout->setVerticalSpacing(10);
	cropLayout->setHorizontalSpacing(10);
	cropLayout->setContentsMargins(15, 22, 15, 12);

	m_cropCheck = new QCheckBox(tr("Render only part of the frame (--crop)"), optionsTab);
	m_cropCheck->setToolTip(
		tr("Restricts rendering to a rectangle of the frame, given as\n"
		"fractions of the full image from 0 to 1. Default path tracer\n"
		"only; works on the CPU and both GPU backends (OptiX and\n"
		"Metal). Pixels outside the rectangle are left black."));
	styleCheckBox(m_cropCheck);
	cropLayout->addRow(checkboxWithInfo(m_cropCheck,
		tr("Renders only a rectangular slice of the full frame - "
		"everything outside it is left black - instead of the whole "
		"image. The rectangle is given as four fractions of the full "
		"frame's width/height, from 0 (left/top edge) to 1 (right/bottom "
		"edge), so it stays the same shape regardless of resolution.\n\n"
		"Useful for iterating faster on one troublesome part of a large, "
		"slow scene - the same total number of samples cleans up much "
		"faster when it only has to cover a corner of the frame instead "
		"of the whole thing.\n\n"
		"Off by default (the full frame). If a loaded .pbrt scene file "
		"already requests its own crop region, checking this overrides "
		"it with the rectangle below; leaving it unchecked lets the "
		"scene's own request (if any) stand.")));

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
}

// The Depth of Field group.
void MainWindow::buildDepthOfFieldGroup(QWidget *optionsTab, QVBoxLayout *layout) {
	// ------------------------------------------------------------------
	// Depth of Field group
	// ------------------------------------------------------------------
	// Not gated on isDefault/sppmSelected below (unlike maxComponentValue/
	// crop/seed) - camera ray generation, and therefore this override, is
	// shared by every integrator INCLUDING SPPM (see the accelerator/
	// splitmethod comment further down for the identical reasoning) with
	// one real exception: cpu_render_main_sppm()/optix_render_main_sppm()
	// don't currently receive RenderOptions at all, so this override is
	// silently a no-op under --sppm specifically, not just on the native
	// demo gallery - a known, narrower-than-ideal limitation left for a
	// follow-up rather than threading RenderOptions into the SPPM entry
	// points too.
	InfoGroupBox *dofGroup = new InfoGroupBox(tr("Depth of Field"), optionsTab);
	styleGroupBox(dofGroup);
	dofGroup->setInfoIcon(createInfoIcon(
		tr("Override the active scene's own camera lens diameter/focus "
		"distance without editing its scene file - only affects scenes "
		"loaded from a scene file; built-in demo-gallery scenes keep their "
		"own fixed camera.")));
	QFormLayout *dofLayout = new QFormLayout(dofGroup);
	dofLayout->setVerticalSpacing(10);
	dofLayout->setHorizontalSpacing(10);
	dofLayout->setContentsMargins(15, 22, 15, 12);

	m_dofOverrideCheck = new QCheckBox(tr("Override depth of field (--aperture/--focus-distance)"), optionsTab);
	m_dofOverrideCheck->setToolTip(
		tr("Sets the camera's lens diameter and focus distance, overriding\n"
		"whatever the scene's own Camera directive requests. Only affects\n"
		"scenes loaded from a scene file - has no effect on the built-in\n"
		"demo gallery, which keeps its own fixed camera."));
	styleCheckBox(m_dofOverrideCheck);
	dofLayout->addRow(checkboxWithInfo(m_dofOverrideCheck,
		tr("Thin-lens depth-of-field blur is already fully supported for any "
		"scene loaded from a scene file - a \"lensradius\"/\"focaldistance\" "
		"Camera directive in the file is all it takes. This lets you set or "
		"change that without hand-editing the file: Aperture is the lens "
		"diameter in world units (0 = pinhole-sharp, no blur), and Focus "
		"Distance is how far away the plane of sharp focus sits.\n\n"
		"Off by default (the scene's own camera, unchanged). Only affects "
		"scenes loaded from a scene file - the built-in demo gallery's "
		"scenes keep their own author-chosen fixed camera regardless of "
		"this setting.")));

	m_apertureSpin = new QDoubleSpinBox(optionsTab);
	m_apertureSpin->setRange(0.0, 100.0);
	m_apertureSpin->setSingleStep(0.1);
	m_apertureSpin->setValue(1.0);
	m_apertureSpin->setEnabled(false);
	m_apertureSpin->setToolTip(m_dofOverrideCheck->toolTip());
	styleSpinBox(m_apertureSpin);
	QWidget *apertureLabel = labelWithInfo(tr("Aperture:"),
		tr("Lens diameter in world units - larger values blur more. 0 means "
		"pinhole-sharp (no blur)."));
	dofLayout->addRow(apertureLabel, m_apertureSpin);

	m_focusDistanceSpin = new QDoubleSpinBox(optionsTab);
	m_focusDistanceSpin->setRange(0.01, 100000.0);
	m_focusDistanceSpin->setSingleStep(1.0);
	m_focusDistanceSpin->setValue(10.0);
	m_focusDistanceSpin->setEnabled(false);
	m_focusDistanceSpin->setToolTip(m_dofOverrideCheck->toolTip());
	styleSpinBox(m_focusDistanceSpin);
	QWidget *focusDistanceLabel = labelWithInfo(tr("Focus Distance:"),
		tr("Distance from the camera to the plane of sharp focus, in world "
		"units."));
	dofLayout->addRow(focusDistanceLabel, m_focusDistanceSpin);

	connect(m_dofOverrideCheck, &QCheckBox::toggled, m_apertureSpin, &QDoubleSpinBox::setEnabled);
	connect(m_dofOverrideCheck, &QCheckBox::toggled, m_focusDistanceSpin, &QDoubleSpinBox::setEnabled);
	connect(m_dofOverrideCheck, &QCheckBox::toggled, apertureLabel, &QWidget::setEnabled);
	connect(m_dofOverrideCheck, &QCheckBox::toggled, focusDistanceLabel, &QWidget::setEnabled);
	apertureLabel->setEnabled(m_dofOverrideCheck->isChecked());
	focusDistanceLabel->setEnabled(m_dofOverrideCheck->isChecked());

	layout->addWidget(dofGroup);
}

// The Reproducibility (seed) group.
void MainWindow::buildSeedGroup(QWidget *optionsTab, QVBoxLayout *layout) {
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
		"same seed. Default path tracer only. On the GPU (both OptiX and\n"
		"Metal) renders are already repeatable by default; the seed picks a\n"
		"different, but equally repeatable, random sequence."));
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
}
