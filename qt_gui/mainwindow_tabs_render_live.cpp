// mainwindow_tabs_render_live.cpp - the Live Preview-specific portions of
// the Render Options tab (see mainwindow_tabs_render.cpp's own header
// comment for that tab's overall scope). Split out of
// createRenderOptionsTab() (mainwindow_tabs_render.cpp) as a code-health
// pass: that function had grown to ~1779 lines as every Live Preview
// feature added this session (Neural Radiance Cache, Neural Temporal
// Upscale, Depth of Field) got appended to it. Both methods here are
// called from createRenderOptionsTab() at the exact positions their
// inline code used to occupy - no behavior change, purely a
// file-organization split. RT_GUI_HAVE_GPU-only, like every widget either
// one builds (this whole file compiles to nothing without it).
#include "mainwindow.h"
#include "settings_keys.h"

#ifdef RT_GUI_HAVE_GPU
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QPushButton>

// Collapses the construct+setChecked+styleCheckBox+connect(toggled,...)
// sequence hand-repeated for every Live Preview toggle checkbox
// (ReSTIR GI/DI, Radiance Cache, Path Guiding, NRC, DOF, Neural
// Reconstruction) below into one call - see its own declaration comment,
// mainwindow.h, for why it returns the raw QCheckBox* rather than an
// already-wrapped widget.
QCheckBox *MainWindow::createLiveToggleCheckbox(const QString &label, bool initialChecked,
												  std::function<void(bool)> onToggled) {
	QCheckBox *check = new QCheckBox(label);
	check->setChecked(initialChecked);
	styleCheckBox(check);
	connect(check, &QCheckBox::toggled, this, onToggled);
	return check;
}

void MainWindow::buildDenoiserLivePreviewSubsection(QWidget *optionsTab) {
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
		tr("Only matters when the OptiX AI Denoiser is selected above. "
		"Controls how much of the smoothing you actually see: 0.0 shows the "
		"fully smoothed image, 1.0 shows the original grainy image with no "
		"smoothing at all. Same control as the Image & Video subsection's "
		"own Blend setting above, just set separately for Live Preview."));
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
		tr("None: shows the image exactly as it's rendered, with all its "
		"natural graininess - no smoothing applied.\n\n"
		"OptiX AI Denoiser: cleans up the grainy, low-detail look Live "
		"Preview has while you're moving around, using the same AI-powered "
		"smoothing the Image & Video subsection above applies to finished "
		"renders. This makes the preview look reasonably clean right away "
		"instead of waiting for it to gradually clear up on its own. Costs "
		"a small amount of extra GPU time per frame.\n\n"
		"SVGF Denoiser: an alternative, experimental noise-reduction filter. "
		"Instead of blending many frames together, it tracks how much each "
		"pixel's brightness has been changing over time and smooths it out "
		"along natural edges - it holds up better than the AI Denoiser "
		"while you're actively moving the camera. Always shows the latest "
		"smoothed frame rather than gradually sharpening over time (see the "
		"SVGF Advanced Tuning group below for its own fine-tuning options).")));
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
		tr("Only matters when the OptiX AI Denoiser is selected above. "
		"Shows each freshly smoothed frame on its own, instead of blending "
		"it together with earlier frames into a running average. You give "
		"up the extra quality that blending more frames together would "
		"eventually reach, in exchange for a view that always reflects only "
		"the most recent frame - useful while flying around with WASD, "
		"since older blended-in frames were rendered from a camera position "
		"you've already left.")), 1);

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
		tr("Controls how quickly the filter forgets older frames. Lower "
		"values hold onto history longer, which gives smoother results but "
		"reacts more slowly when the scene changes; higher values adapt "
		"faster but leave more visible noise."),
		0.01, 1.0, 0.01, 2, m_liveSvgfTemporalAlpha);
	m_liveSvgfMaxHistoryLengthSpin = addSvgfDoubleSpin(0, 1, tr("Max History Length:"),
		tr("The most frames of history a pixel is allowed to build up once "
		"it has settled down. Puts a ceiling on how \"sticky\" - i.e. slow "
		"to update - a settled pixel can become."),
		1.0, 256.0, 1.0, 0, m_liveSvgfMaxHistoryLength);
	m_liveSvgfVarianceBootstrapFramesSpin = addSvgfDoubleSpin(1, 0, tr("Variance Bootstrap Frames:"),
		tr("Until a pixel has built up at least this many frames of "
		"history, its noise estimate is smoothed using its neighboring "
		"pixels instead of trusted on its own. This helps a brand-new "
		"pixel - for example, one just uncovered by a moving object - get "
		"reasonable edge-detection behavior before it has enough history "
		"of its own to judge from."),
		0.0, 32.0, 1.0, 0, m_liveSvgfVarianceBootstrapFrames);
	m_liveSvgfVarianceBootstrapRadiusSpin = addSvgfIntSpin(1, 1, tr("Variance Bootstrap Radius:"),
		tr("How far out, in pixels, the neighbor-smoothing described above "
		"reaches. A radius of 3 means it looks at a 7x7 block of pixels."),
		0, 8, m_liveSvgfVarianceBootstrapRadius);
	m_liveSvgfSigmaNormalSpin = addSvgfDoubleSpin(2, 0, tr("Sigma Normal:"),
		tr("How sensitive the filter is to two neighboring pixels facing "
		"different directions. Higher values treat a smaller difference "
		"in surface angle as a different surface, which keeps the filter "
		"from blurring across curved surfaces or object edges."),
		1.0, 1024.0, 1.0, 0, m_liveSvgfSigmaNormal);
	m_liveSvgfSigmaDepthSpin = addSvgfDoubleSpin(2, 1, tr("Sigma Depth:"),
		tr("How sensitive the filter is to two neighboring pixels sitting "
		"at different distances from the camera. Higher values tolerate "
		"more depth difference before treating a neighbor as a separate, "
		"unrelated surface."),
		0.01, 16.0, 0.1, 2, m_liveSvgfSigmaDepth);
	m_liveSvgfSigmaLuminanceSpin = addSvgfDoubleSpin(3, 0, tr("Sigma Luminance:"),
		tr("How sensitive the filter is to two neighboring pixels having "
		"different brightness. Higher values let it blend across bigger "
		"brightness differences, which smooths more but risks blurring "
		"away real detail."),
		0.1, 32.0, 0.1, 1, m_liveSvgfSigmaLuminance);
	m_liveSvgfAtrousRadiusSpin = addSvgfIntSpin(3, 1, tr("A-trous Radius:"),
		tr("How wide an area, in pixels, each smoothing pass covers. "
		"Limited to 0-2, where 2 covers a 5x5 block - the filter's "
		"internal weighting table only supports that range."),
		0, 2, m_liveSvgfAtrousRadius);
	m_liveSvgfMinAlbedoSpin = addSvgfDoubleSpin(4, 0, tr("Min Albedo:"),
		tr("A minimum surface-color value the filter substitutes in when "
		"it temporarily factors out surface color to smooth the lighting "
		"on its own. Prevents a very dark or black surface from causing "
		"math errors that would show up as flickering noise or a solid "
		"black patch."),
		0.001, 1.0, 0.001, 3, m_liveSvgfMinAlbedo);
	m_liveSvgfAtrousPassesSpin = addSvgfIntSpin(4, 1, tr("A-trous Passes:"),
		tr("How many smoothing passes the filter runs, each one covering "
		"a wider area than the last (the step size doubles every pass: "
		"1, 2, 4, 8, ...). More passes smooth a larger area but cost "
		"proportionally more GPU time."),
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
}

void MainWindow::buildLivePreviewSettingsSection(QWidget *optionsTab, QVBoxLayout *layout) {
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
		tr("Covers ReSTIR DI/GI, the Radiance Cache, Path Guiding, "
		"Exposure, Samples/Max Bounces per frame, and the Firefly Clamp - "
		"all separate from the Advanced Parameters group below (which only "
		"affects Image/Video renders) and from the Denoiser section above. "
		"These settings only actually take effect when Output Mode above "
		"is set to \"Live Preview (interactive)\", but you can still edit "
		"them in any mode.")));
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

	// Own row, spanning every column - a checkbox has nothing to pair with
	// the way the label+field values below do, so it doesn't belong forced
	// into the same 2-per-row grouping.
	m_liveRestirGiCheck = createLiveToggleCheckbox(tr("ReSTIR GI"), m_liveRestirGiEnabled, [this](bool checked) {
		m_liveRestirGiEnabled = checked;
		saveLiveRestirGiEnabled(checked);
		pushLiveRestirGiToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveRestirGiCheck,
		tr("Improves indirect lighting - light that's bounced off at "
		"least one other surface before reaching what you're looking at - "
		"by reusing good light samples found at nearby pixels and in "
		"recent frames, instead of only trying once per pixel. Works "
		"independently of whichever denoiser is active above. Turning it "
		"off falls back to the simpler one-sample-per-pixel method, which "
		"looks noisier but is cheaper to render.")),
		0, 0, 1, 4);

	// Own row too, same reasoning as ReSTIR GI's row above.
	m_liveRestirDiCheck = createLiveToggleCheckbox(tr("ReSTIR DI"), m_liveRestirDiEnabled, [this](bool checked) {
		m_liveRestirDiEnabled = checked;
		saveLiveRestirDiEnabled(checked);
		pushLiveRestirDiToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveRestirDiCheck,
		tr("Improves direct lighting - light that reaches a surface "
		"straight from a light source, with no bounces - the same way "
		"ReSTIR GI above improves indirect lighting: by reusing good "
		"light samples found at nearby pixels and in recent frames "
		"instead of only trying once per pixel. Independent of ReSTIR GI "
		"above (that one handles light that's already bounced at least "
		"once; this one handles light hitting a surface directly). "
		"Turning it off falls back to picking one light sample per pixel "
		"the plain way, which is noisier in scenes with many lights but "
		"cheaper to render.")),
		1, 0, 1, 4);

	// Own row too, same reasoning as ReSTIR GI/DI's rows above.
	m_liveProbeCacheCheck = createLiveToggleCheckbox(tr("Radiance Cache"), m_liveProbeCacheEnabled, [this](bool checked) {
		m_liveProbeCacheEnabled = checked;
		saveLiveProbeCacheEnabled(checked);
		pushLiveProbeCacheToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveProbeCacheCheck,
		tr("Caches and reuses estimates of indirect lighting - light "
		"that's bounced two or more times - across frames and nearby "
		"points in space, instead of recalculating it completely from "
		"scratch every frame. Independent of ReSTIR GI above (that one "
		"only improves the very first bounce; this one covers every "
		"bounce after that). It needs a few seconds to catch up, so "
		"expect the lighting to look patchy or noisy right after you turn "
		"it on or fly the camera into a new area, then smooth out as it "
		"builds up data. Turning it off falls back to computing every "
		"bounce the plain way, which looks noisier in scenes with a lot "
		"of deep indirect light, but shows the correct result immediately "
		"with no warm-up delay.")),
		2, 0, 1, 4);

	// Own row too, same reasoning as the checkboxes above.
	m_livePathGuidingCheck = createLiveToggleCheckbox(tr("Path Guiding"), m_livePathGuidingEnabled, [this](bool checked) {
		m_livePathGuidingEnabled = checked;
		saveLivePathGuidingEnabled(checked);
		pushLivePathGuidingToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_livePathGuidingCheck,
		tr("For shiny/metal surfaces, this learns roughly where the "
		"brightest light is coming from at each point in the scene, so "
		"bounce rays get aimed more toward useful directions instead of "
		"just guessing based on the surface's own reflective properties. "
		"Requires the Radiance Cache above to also be turned on - this "
		"feature reuses that cache's own data and does nothing without "
		"it. Like the Radiance Cache, it needs a few seconds to learn and "
		"improve; turning it off falls back to the material's own plain "
		"reflection-based guessing.")),
		3, 0, 1, 4);

	// Own row, grouped with the plain-checkbox rows above (independent of
	// every one of them - no shared state with the Radiance Cache/Path
	// Guiding) rather than after the numeric/combo rows below, so every
	// checkbox on this panel reads as one contiguous block.
	m_liveNrcCheck = createLiveToggleCheckbox(tr("Neural Radiance Cache"), m_liveNrcEnabled, [this](bool checked) {
		m_liveNrcEnabled = checked;
		saveLiveNrcEnabled(checked);
		pushLiveNrcToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveNrcCheck,
		tr("A small AI model, trained live while you render, that learns "
		"to predict indirect lighting for both plain matte surfaces and "
		"shiny/metal ones - unlike the Radiance Cache above, which only "
		"handles matte surfaces and doesn't account for the angle you're "
		"viewing from. Like the Radiance Cache, it takes a while to catch "
		"up, so expect it to need several frames to settle in after you "
		"turn it on or move the camera into a new area. Turning it off "
		"falls back to tracing every bounce the plain way (or to the "
		"Radiance Cache, if that's also turned on).")),
		4, 0, 1, 4);

	// Depth-of-field override checkbox - see RenderOptions::aperture_override's
	// own comment (render_options.h). No hard dependency on any other Live
	// Preview control, so it groups here with the other independent
	// checkboxes; its own Aperture/Focus Distance value fields live further
	// down with the other numeric rows, not here (a checkbox row has
	// nothing to pair with the way those do).
	m_liveDofCheck = createLiveToggleCheckbox(tr("Depth of Field"), m_liveDofEnabled, [this](bool checked) {
		m_liveDofEnabled = checked;
		saveLiveDofEnabled(checked);
		pushLiveDofToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveDofCheck,
		tr("Overrides the current scene's own camera lens size and focus "
		"distance with the Aperture and Focus Distance values below, "
		"without changing the scene file itself. Only works for scenes "
		"loaded from a scene file - it has no effect on the built-in demo "
		"gallery, which always uses its own fixed camera.")),
		5, 0, 1, 4);

	// Temporal upscale (see this project's own plan) - a 3-way factor
	// choice (Off/2x/4x), not a checkbox, so it gets its own label+combo row
	// (like the Denoiser mode combo above, m_liveDenoiserModeCombo) rather
	// than living in the plain-checkbox rows above. Plain int persistence
	// (m_liveTemporalUpscaleFactor, settings_keys.h's own
	// kLivePreviewTemporalUpscaleFactorKey), not a bool pair - there's no
	// pre-existing bool this decomposes from, unlike the Denoiser combo's
	// own m_liveDenoiseEnabled/m_liveSvgfEnabled pair.
	QWidget *liveTemporalUpscaleRow = new QWidget();
	QHBoxLayout *liveTemporalUpscaleRowLayout = new QHBoxLayout(liveTemporalUpscaleRow);
	liveTemporalUpscaleRowLayout->setContentsMargins(0, 0, 0, 0);
	liveTemporalUpscaleRowLayout->setSpacing(10);

	// Index 0 = Off (factor 1), 1 = 2x, 2 = 4x.
	m_liveTemporalUpscaleCombo = new QComboBox();
	m_liveTemporalUpscaleCombo->addItem(tr("Off"));
	m_liveTemporalUpscaleCombo->addItem(tr("2x"));
	m_liveTemporalUpscaleCombo->addItem(tr("4x"));
	m_liveTemporalUpscaleCombo->setCurrentIndex(m_liveTemporalUpscaleFactor >= 4 ? 2 : (m_liveTemporalUpscaleFactor >= 2 ? 1 : 0));
	styleComboBox(m_liveTemporalUpscaleCombo);
	connect(m_liveTemporalUpscaleCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
		m_liveTemporalUpscaleFactor = (index == 2) ? 4 : (index == 1 ? 2 : 1);
		saveLiveTemporalUpscaleFactor(m_liveTemporalUpscaleFactor);
		pushLiveTemporalUpscaleToSession();
		// Neural Reconstruction is meaningless at "Off" - grey it out (and
		// force it off) rather than silently no-op, same "document the
		// dependency by disabling the control" precedent Path Guiding's own
		// checkbox already established for its probe-cache dependency.
		if (m_liveNeuralUpscaleCheck) {
			m_liveNeuralUpscaleCheck->setEnabled(m_liveTemporalUpscaleFactor > 1);
			if (m_liveTemporalUpscaleFactor <= 1 && m_liveNeuralUpscaleCheck->isChecked()) {
				m_liveNeuralUpscaleCheck->setChecked(false);
			}
		}
	});

	liveTemporalUpscaleRowLayout->addWidget(labelWithInfo(tr("Temporal Upscale:"),
		tr("Builds up a sharper-looking image over several frames using a "
		"repeating pixel-shift pattern, instead of just stretching Live "
		"Preview's native low-resolution image to fit the window. The "
		"cost of rendering each individual frame doesn't change - 2x vs. "
		"4x only changes how many frames it takes to reach a sharp image "
		"(4 frames for 2x, 16 frames for 4x, counting from when the "
		"camera stops moving). 4x uses noticeably more memory (roughly "
		"150-200 MB) than 2x (roughly 40-50 MB). This doesn't combine "
		"with the Denoiser dropdown's SVGF mode or the 'Show latest "
		"frame' option above - if either of those is on, it takes "
		"priority instead.")));
	liveTemporalUpscaleRowLayout->addWidget(m_liveTemporalUpscaleCombo, 1);

	liveRenderSettingsGrid->addWidget(liveTemporalUpscaleRow, 6, 0, 1, 4);

	// Neural temporal upscale (see this project's own plan) - HARD-DEPENDS
	// on Temporal Upscale above being 2x/4x, same "grey out the dependent
	// checkbox" precedent Path Guiding's own checkbox already established
	// for its probe-cache dependency. Kept on its own row directly below
	// Temporal Upscale's own row rather than grouped with the other
	// checkboxes above it, so the dependency reads top-to-bottom.
	m_liveNeuralUpscaleCheck = createLiveToggleCheckbox(tr("Neural Reconstruction"), m_liveNeuralUpscaleEnabled, [this](bool checked) {
		m_liveNeuralUpscaleEnabled = checked;
		saveLiveNeuralUpscaleEnabled(checked);
		pushLiveNeuralUpscaleToSession();
	});
	m_liveNeuralUpscaleCheck->setEnabled(m_liveTemporalUpscaleFactor > 1);
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveNeuralUpscaleCheck,
		tr("Replaces Temporal Upscale's own basic image-building method "
		"with a small AI model, trained live while you render, that "
		"blends nearby samples together more smartly instead of just "
		"copying pixel blocks into place - this cuts down on the blocky, "
		"ghost-like artifacts the plain method can show around moving "
		"object edges. Requires Temporal Upscale above to be set to 2x or "
		"4x (does nothing at Off). Like the Radiance Cache, it takes a "
		"few seconds after you turn it on to start looking good. Works "
		"best with Samples/Frame set to 1 - higher values get averaged "
		"together before this feature sees them, which blurs the data "
		"it's learning from.")),
		7, 0, 1, 4);

	// The numeric values grouped into their own clean 2-per-row grid (rows
	// 8-10), separate from every checkbox/combo above.
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
		tr("A flat brightness multiplier applied to the image before "
		"final color adjustments. Same idea as this tab's own "
		"Output-group Exposure control above, but set separately just "
		"for Live Preview.")),
		8, 0);
	liveRenderSettingsGrid->addWidget(m_liveExposureSpin, 8, 1);

	m_liveSamplesSpinBox = new QSpinBox();
	// Raised from 16 - Live Preview intentionally uses a small per-call
	// sample count so it stays responsive to camera moves/setting changes
	// between frames (see this spinbox's own tooltip and
	// RealtimePreviewWorker::renderLoop()'s own comment on why), but 16 was
	// too low a ceiling for anyone wanting a heavier per-call cost in
	// exchange for faster convergence on a static frame - the caller still
	// chooses how high to actually go.
	m_liveSamplesSpinBox->setRange(1, 64);
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
		tr("How many light rays are traced per pixel each time Live "
		"Preview renders a frame - more samples means a cleaner image but "
		"a slower frame. Separate from the Advanced Parameters group "
		"below, which only affects Image/Video renders.")),
		8, 2);
	liveRenderSettingsGrid->addWidget(m_liveSamplesSpinBox, 8, 3);

	m_liveMaxDepthSpinBox = new QSpinBox();
	m_liveMaxDepthSpinBox->setRange(1, 32);
	m_liveMaxDepthSpinBox->setValue(m_liveMaxDepth);
	styleSpinBox(m_liveMaxDepthSpinBox);
	connect(m_liveMaxDepthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, pushSppMaxDepth);
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Max Bounces:"),
		tr("The most times a light ray is allowed to bounce off surfaces "
		"before Live Preview stops tracing it - higher lets light reach "
		"further into a scene (useful for mirrors, glass, or rooms lit "
		"indirectly) at a higher cost per frame. Separate from the "
		"Advanced Parameters group below, which only affects Image/Video "
		"renders.")),
		9, 0);
	liveRenderSettingsGrid->addWidget(m_liveMaxDepthSpinBox, 9, 1);

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
		tr("Puts a ceiling on how bright any single sample is allowed to "
		"be, to suppress fireflies - an isolated ray that happens to "
		"catch a very bright, small light at just the right angle, "
		"showing up as a stray bright speckle in the image. The tradeoff "
		"is that genuinely bright highlights can get dimmed too. Lower "
		"values clamp more aggressively.")),
		9, 2);
	liveRenderSettingsGrid->addWidget(m_liveFireflyClampSpin, 9, 3);

	m_liveApertureSpin = new QDoubleSpinBox();
	m_liveApertureSpin->setRange(0.0, 100.0);
	m_liveApertureSpin->setSingleStep(0.1);
	m_liveApertureSpin->setValue(m_liveAperture);
	styleSpinBox(m_liveApertureSpin);
	connect(m_liveApertureSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveAperture = value;
		saveLiveAperture(value);
		pushLiveDofToSession();
	});
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Aperture:"),
		tr("How wide the camera's lens opening is, in scene units - "
		"bigger values create more blur outside the focus distance. 0 "
		"means a pinhole-sharp image with no blur at all.")),
		10, 0);
	liveRenderSettingsGrid->addWidget(m_liveApertureSpin, 10, 1);

	m_liveFocusDistanceSpin = new QDoubleSpinBox();
	m_liveFocusDistanceSpin->setRange(0.01, 100000.0);
	m_liveFocusDistanceSpin->setSingleStep(1.0);
	m_liveFocusDistanceSpin->setValue(m_liveFocusDistance);
	styleSpinBox(m_liveFocusDistanceSpin);
	connect(m_liveFocusDistanceSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveFocusDistance = value;
		saveLiveFocusDistance(value);
		pushLiveDofToSession();
	});
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Focus Distance:"),
		tr("How far from the camera things are in perfectly sharp focus, "
		"in scene units.")),
		10, 2);
	liveRenderSettingsGrid->addWidget(m_liveFocusDistanceSpin, 10, 3);

	// Adaptive sampling (Stage 1 of this project's own plan) - own row,
	// grouped with the plain-checkbox rows above in spirit, but appended
	// here at the end rather than renumbering rows 0-10 above to insert it
	// earlier. Currently only drives the noise-heatmap debug view below,
	// not real GPU sampling - see RealtimePreviewWorker::
	// setAdaptiveSampling()'s own comment for the staging.
	m_liveAdaptiveSamplingCheck = createLiveToggleCheckbox(tr("Adaptive Sampling"), m_liveAdaptiveSamplingEnabled, [this](bool checked) {
		m_liveAdaptiveSamplingEnabled = checked;
		saveLiveAdaptiveSamplingEnabled(checked);
		pushLiveAdaptiveSamplingToSession();
	});
	liveRenderSettingsGrid->addWidget(checkboxWithInfo(m_liveAdaptiveSamplingCheck,
		tr("Tracks how noisy each pixel still is and, once enabled, shows a "
		"black-and-white heatmap instead of the normal preview: white where "
		"a pixel is still noisy enough to need more samples (per the "
		"Convergence Threshold below), black where it's already converged. "
		"This is a diagnostic view for now - it doesn't yet change which "
		"pixels actually get sampled.")),
		11, 0, 1, 4);

	m_liveAdaptiveSamplingThresholdSpin = new QDoubleSpinBox();
	m_liveAdaptiveSamplingThresholdSpin->setRange(0.001, 0.5);
	m_liveAdaptiveSamplingThresholdSpin->setSingleStep(0.005);
	m_liveAdaptiveSamplingThresholdSpin->setDecimals(3);
	m_liveAdaptiveSamplingThresholdSpin->setValue(m_liveAdaptiveSamplingThreshold);
	styleSpinBox(m_liveAdaptiveSamplingThresholdSpin);
	connect(m_liveAdaptiveSamplingThresholdSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double value) {
		m_liveAdaptiveSamplingThreshold = value;
		saveLiveAdaptiveSamplingThreshold(value);
		pushLiveAdaptiveSamplingToSession();
	});
	liveRenderSettingsGrid->addWidget(labelWithInfo(tr("Convergence Threshold:"),
		tr("How settled a pixel's brightness needs to be, relative to its "
		"own noise level, before Adaptive Sampling above considers it "
		"converged - lower values demand more certainty (more of the image "
		"reads as still-noisy for longer) before treating a pixel as done. "
		"0.01 matches this project's own CPU/offline --adaptive-threshold "
		"default and Blender Cycles' own default.")),
		12, 0);
	liveRenderSettingsGrid->addWidget(m_liveAdaptiveSamplingThresholdSpin, 12, 1);

	liveRenderSettingsLayout->addRow(liveRenderSettingsRow);

	layout->addWidget(m_liveRenderSettingsGroupBox);
}

#endif // RT_GUI_HAVE_GPU
