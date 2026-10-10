// scene_builder_widget.cpp - see scene_builder_widget.h. The layout view is scene_layout_view.cpp, the property panel scene_builder_inspector.cpp.
#include "scene_builder_widget.h"

#include "scene_builder_common.h"
#include "scene_builder_array_dialog.h"
#include "flow_layout.h"
#include "app_log.h"
#include "window_geometry.h"
#include "atomic_file.h"
#include "../src/shared/scene_doc_diff.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

using namespace scene_builder_ui;

// =====================================================================================================================================
// SceneBuilderWidget
// =====================================================================================================================================

SceneBuilderWidget::SceneBuilderWidget(QWidget *parent) : QWidget(parent) {
	m_autosaveEnabled = !qEnvironmentVariableIsSet("RT_GUI_SELFTEST");
	m_autosaveTimer = new QTimer(this);
	m_autosaveTimer->setSingleShot(true);
	m_autosaveTimer->setInterval(1500);
	connect(m_autosaveTimer, &QTimer::timeout, this, &SceneBuilderWidget::writeAutosave);

	connect(qApp, &QCoreApplication::aboutToQuit, this, &SceneBuilderWidget::flushEditLog);   // also when the program is quit without this widget being destroyed

	buildUi();
	if (!(m_autosaveEnabled && loadAutosave())) newScene();
}

void SceneBuilderWidget::showEvent(QShowEvent *e) {
	QWidget::showEvent(e);
	// The panel can be dragged as narrow as the three buttons above the list allow, whatever the language and theme padding: that width comes from
	// their (now styled) size hints, not from a fixed number.
	if (!m_leftPanel) return;
	m_leftPanel->setMinimumWidth(std::max({m_addButton->sizeHint().width(), m_duplicateButton->sizeHint().width(), m_deleteButton->sizeHint().width()}) + 6);
}

SceneBuilderWidget::~SceneBuilderWidget() {
	flushEditLog();   // the last edit is still waiting for its pause
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
	if (m_dirty && m_autosaveEnabled) writeAutosave();
	if (!qEnvironmentVariableIsSet("RT_GUI_SELFTEST") && m_mainSplit && m_centreSplit) {
		window_geometry::saveSplitter(m_mainSplit, "builder/mainSplit");
		window_geometry::saveSplitter(m_centreSplit, "builder/centreSplit");
	}
}

QString SceneBuilderWidget::workFolder() const {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/ray_tracer_scene_builder";
	QDir().mkpath(dir);
	return dir;
}

QString SceneBuilderWidget::launcherPath() const {
#ifdef Q_OS_WIN
	return QCoreApplication::applicationDirPath() + "/ray_tracer.exe";
#else
	return QCoreApplication::applicationDirPath() + "/ray_tracer";
#endif
}

void SceneBuilderWidget::buildUi() {
	auto *root = new QVBoxLayout(this);

	// ---- top bar: one row of buttons and the scene's name, wrapping to more rows when the window is narrow
	auto button = [this](const QString &text, const QString &tip) {
		auto *b = new QPushButton(text, this);
		b->setToolTip(tip);
		b->setAutoDefault(false);
		return b;
	};
	auto *newB = button(tr("New"), tr("Start again from the example scene"));
	auto *openB = button(tr("Open..."), tr("Open a .pbrt file saved by the Scene Builder"));
	auto *saveB = button(tr("Save"), tr("Save the scene as a .pbrt file"));
	auto *saveAsB = button(tr("Save As..."), tr("Save the scene under a new name"));
	auto *listB = button(tr("Add to scene list"), tr("Save the scene into the scenes folder so it shows up in the Settings tab"));
	m_liveButton = button(tr("Preview live"), tr("Open this scene in Live Preview: walk around it and move things. It is saved as a separate \"(live preview)\" scene in the scene list; your scene and its own listing are not changed."));
	m_liveButton->setObjectName("builderPreviewLiveButton");
	m_liveButton->setVisible(false);
	m_undoButton = button(tr("Undo"), tr("Undo the last change (%1)").arg(shortcutText(QKeySequence::Undo)));
	m_redoButton = button(tr("Redo"), tr("Redo (%1)").arg(shortcutText(QKeySequence::Redo)));
	m_titleLabel = new QLabel(this);
	m_titleLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
	// The scene's name: what the scene list shows, and what a copy added to the list is named after. Edited right here (undoable like any change).
	m_titleEdit = new QLineEdit(this);
	m_titleEdit->setPlaceholderText(tr("Scene name"));
	m_titleEdit->setToolTip(tr("The name of this scene, shown in the scene list"));
	m_titleEdit->setMinimumWidth(150);
	m_titleEdit->setMaximumWidth(320);
	// A flow layout (flow_layout.h): everything on one line when the window is wide, wrapping to a second or third line as it narrows, so nothing is clipped
	// however long a translated label is. The status text keeps its tooltip with all of it.
	auto *toolbar = new QWidget(this);
	auto *flow = new FlowLayout(toolbar, 0, 6, 6);
	for (QPushButton *b : {newB, openB, saveB, saveAsB, listB, m_liveButton}) {
		scene_builder_ui::compactStyle(b);
		flow->addWidget(b);
	}
	flow->addGap(14);
	for (QPushButton *b : {m_undoButton, m_redoButton}) {
		scene_builder_ui::compactStyle(b);
		flow->addWidget(b);
	}
	flow->addGap(14);
	auto *nameBox = new QWidget(toolbar);   // the label and its box stay together when the row wraps
	auto *nameLayout = new QHBoxLayout(nameBox);
	nameLayout->setContentsMargins(0, 0, 0, 0);
	nameLayout->addWidget(new QLabel(tr("Name:"), nameBox));
	nameLayout->addWidget(m_titleEdit);
	flow->addWidget(nameBox);
	m_titleLabel->setParent(toolbar);
	flow->addWidget(m_titleLabel);
	root->addWidget(toolbar);
	connect(m_titleEdit, &QLineEdit::textEdited, this, &SceneBuilderWidget::setSceneName);
	connect(newB, &QPushButton::clicked, this, [this]() { if (confirmDiscard()) newScene(); });
	connect(openB, &QPushButton::clicked, this, &SceneBuilderWidget::onOpenClicked);
	connect(saveB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveClicked);
	connect(saveAsB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveAsClicked);
	connect(listB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveToSceneListClicked);
	connect(m_liveButton, &QPushButton::clicked, this, &SceneBuilderWidget::onPreviewLiveClicked);
	connect(m_undoButton, &QPushButton::clicked, this, [this]() { undo(); });
	connect(m_redoButton, &QPushButton::clicked, this, [this]() { redo(); });

	auto *split = new QSplitter(Qt::Horizontal, this);
	root->addWidget(split, 1);

	// ---- left: scene contents
	auto *left = new QWidget(split);
	auto *leftLayout = new QVBoxLayout(left);
	leftLayout->setContentsMargins(0, 0, 0, 0);
	auto *addB = new QPushButton(tr("Add"), left);
	addB->setAutoDefault(false);
	auto *addMenu = new QMenu(addB);
	addMenu->addSection(tr("Objects"));
	for (ShapeKind k : scene_doc::allShapeKinds())
		addMenu->addAction(shapeLabel(k), this, [this, k]() { addObject(k); });
	addMenu->addSection(tr("Props (several objects at once)"));
	for (scene_doc::PropKind k : scene_doc::allPropKinds()) addMenu->addAction(propLabel(k), this, [this, k]() { addProp(k); });
	// Blocky objects (scene_blocks.h): coloured blocks, creatures and things, each a few dozen boxes added together, in three submenus.
	addMenu->addSection(tr("Blocky (several objects at once)"));
	QMenu *blockMenus[3] = {addMenu->addMenu(tr("Blocks")), addMenu->addMenu(tr("Creatures")), addMenu->addMenu(tr("Things"))};
	for (scene_doc::BlockyKind k : scene_doc::allBlockyKinds())
		blockMenus[static_cast<int>(scene_doc::blockyGroup(k))]->addAction(blockyLabel(k), this, [this, k]() { addBlocky(k); });
	addMenu->addSection(tr("More"));
	addMenu->addAction(tr("Model library..."), this, [this]() { showModelLibrary(); });
	addMenu->addAction(tr("Object from a photo..."), this, [this]() { addObjectFromPhoto(); });
	addMenu->addAction(tr("Light panel (emitting quad)"), this, [this]() {
		edit(QString(), [this]() {
			scene_doc::Object o = scene_doc::makeAreaLightPanel(QString("Light panel %1").arg(m_doc.objects.size() + 1).toStdString());
			m_doc.objects.push_back(o);
			m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
		});
		rebuildList();
		setSelection(m_sel);
	});
	addMenu->addSection(tr("Lights"));
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite})
		addMenu->addAction(lightLabel(k), this, [this, k]() { addLight(k); });
	addB->setMenu(addMenu);
	m_duplicateButton = new QPushButton(tr("Duplicate"), left);
	m_duplicateButton->setToolTip(tr("Make a copy of the selected item beside it (%1)").arg(QKeySequence(Qt::CTRL | Qt::Key_D).toString(QKeySequence::NativeText)));
	m_arrayButton = new QPushButton(tr("Array..."), left);
	m_arrayButton->setToolTip(tr("Make many copies of the selected object: a grid, a ring round a point, or a random scatter"));
	m_deleteButton = new QPushButton(tr("Delete"), left);
	m_duplicateButton->setAutoDefault(false);
	m_arrayButton->setAutoDefault(false);
	m_deleteButton->setAutoDefault(false);
	for (QPushButton *b : {addB, m_duplicateButton, m_arrayButton, m_deleteButton}) scene_builder_ui::compactStyle(b);   // compact, so the list column can be narrow
	auto *listBar = new QWidget(left);   // wraps (Add / Duplicate / Delete on two lines) when the column is dragged narrow
	auto *row = new FlowLayout(listBar, 0, 6, 6);
	row->addWidget(addB);
	row->addWidget(m_duplicateButton);
	row->addWidget(m_arrayButton);
	row->addWidget(m_deleteButton);
	leftLayout->addWidget(listBar);
	m_list = new QListWidget(left);
	m_list->setObjectName("sceneBuilderList");  // sized by the application stylesheet (mainwindow_style.cpp)
	m_list->setSelectionMode(QAbstractItemView::ExtendedSelection);   // Ctrl- or Shift-click picks several: they are moved, copied and deleted together
	leftLayout->addWidget(m_list, 1);
	left->setMinimumWidth(0);  // showEvent() sets the real minimum from the widest button
	m_leftPanel = left;
	m_addButton = addB;
	connect(m_list, &QListWidget::itemSelectionChanged, this, [this]() { onListSelectionChanged(); });
	connect(m_deleteButton, &QPushButton::clicked, this, [this]() { deleteSelected(); });
	connect(m_duplicateButton, &QPushButton::clicked, this, [this]() { duplicateSelected(); });
	connect(m_arrayButton, &QPushButton::clicked, this, [this]() { showArrayDialog(); });
	// Group, Ungroup and Select all, from anywhere in the tab except a text box.
	const auto addKey = [this](const QKeySequence &key, const std::function<void()> &action) {
		auto *shortcut = new QShortcut(key, this);
		shortcut->setContext(Qt::WidgetWithChildrenShortcut);
		connect(shortcut, &QShortcut::activated, this, [action]() { if (!qobject_cast<QLineEdit *>(QApplication::focusWidget())) action(); });
	};
	addKey(QKeySequence(Qt::CTRL | Qt::Key_G), [this]() { groupSelected(); });
	addKey(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_G), [this]() { ungroupSelected(); });
	addKey(QKeySequence(Qt::CTRL | Qt::Key_A), [this]() { selectAll(); });
	// Ctrl+D (Command+D on a Mac) duplicates from anywhere in the tab except a text box.
	auto *duplicateKey = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_D), this);
	duplicateKey->setContext(Qt::WidgetWithChildrenShortcut);
	connect(duplicateKey, &QShortcut::activated, this, [this]() { if (!qobject_cast<QLineEdit *>(QApplication::focusWidget())) duplicateSelected(); });
	// Delete and undo work from the list and the layout view (not inside a text box, where Delete edits text).
	// (Backspace too: a Mac's "delete" key is Backspace, which the standard Delete shortcut does not include there.)
	for (const QKeySequence &key : {QKeySequence(QKeySequence::Delete), QKeySequence(Qt::Key_Backspace)}) {
		auto *delList = new QShortcut(key, m_list);
		delList->setContext(Qt::WidgetShortcut);
		connect(delList, &QShortcut::activated, this, [this]() { deleteSelected(); });
	}

	// ---- centre: layout view above the preview
	auto *centre = new QSplitter(Qt::Vertical, split);
	auto *layoutBox = new QWidget(centre);
	auto *layoutLayout = new QVBoxLayout(layoutBox);
	layoutLayout->setContentsMargins(0, 0, 0, 0);
	createViews(layoutBox, layoutLayout);  // scene_builder_views.cpp: the Top / Front / Side / 3D views and what they share

	auto *previewBox = new QWidget;
	auto *previewLayout = new QVBoxLayout(previewBox);
	previewLayout->setContentsMargins(0, 0, 0, 0);
	auto *renderBar = new QWidget(previewBox);
	auto *renderRow = new FlowLayout(renderBar, 0, 6, 6);
	m_qualityCombo = new QComboBox(previewBox);
	m_qualityCombo->addItem(tr("Draft"), 0);
	m_qualityCombo->addItem(tr("Good"), 1);
	m_qualityCombo->addItem(tr("Best"), 2);
	m_qualityCombo->setCurrentIndex(0);
	m_qualityCombo->setToolTip(tr("Draft: 480 pixels wide, 16 samples. Good: 720 wide, 64 samples. Best: 960 wide, 256 samples."));
	m_gpuCheck = new QCheckBox(tr("Use the GPU"), previewBox);
	m_gpuCheck->setToolTip(tr("Render on the graphics card (NVIDIA OptiX on Windows, Metal on a Mac). Much faster for large pictures; needs a supported GPU."));
	m_previewButton = new QPushButton(tr("Preview"), previewBox);
	m_previewButton->setAutoDefault(false);
	m_finalButton = new QPushButton(tr("Render picture..."), previewBox);
	m_finalButton->setAutoDefault(false);
	m_finalButton->setToolTip(tr("Render at the image size and sample count set under Camera, and save the picture as a PNG"));
	m_turntableButton = new QPushButton(tr("Turntable..."), previewBox);
	m_turntableButton->setAutoDefault(false);
	m_turntableButton->setToolTip(tr("Make a video of the scene from a camera that goes once round it"));
	m_flythroughButton = new QPushButton(tr("Flythrough..."), previewBox);
	m_flythroughButton->setAutoDefault(false);
	m_flythroughButton->setToolTip(tr("Make a video of the scene from a camera that flies through your saved camera views (needs two or more)"));
	m_previewStatus = new QLabel(previewBox);
	renderRow->addWidget(new QLabel(tr("Quality:"), renderBar));
	renderRow->addWidget(m_qualityCombo);
	renderRow->addWidget(m_gpuCheck);
	scene_builder_ui::compactStyle(m_previewButton);
	scene_builder_ui::compactStyle(m_finalButton);
	renderRow->addWidget(m_previewButton);
	renderRow->addWidget(m_finalButton);
	scene_builder_ui::compactStyle(m_turntableButton);
	renderRow->addWidget(m_turntableButton);
	scene_builder_ui::compactStyle(m_flythroughButton);
	renderRow->addWidget(m_flythroughButton);
	m_previewStatus->setWordWrap(true);
	previewLayout->addWidget(renderBar);
	m_previewLabel = new QLabel(previewBox);
	m_previewLabel->setAlignment(Qt::AlignCenter);
	m_previewLabel->setMinimumHeight(120);
	m_previewLabel->installEventFilter(this);  // rescale the picture when the pane is resized
	m_previewLabel->setText(tr("Press Preview to see the scene."));
	m_previewLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	previewLayout->addWidget(m_previewLabel, 1);
	previewLayout->addWidget(m_previewStatus);
	connect(m_previewButton, &QPushButton::clicked, this, [this]() {
		if (m_process) {
			m_cancelRequested = true;
			m_process->kill();  // the button reads Cancel while a render runs
			return;
		}
		startPreview();
	});
	connect(m_finalButton, &QPushButton::clicked, this, &SceneBuilderWidget::onRenderFinalClicked);
	connect(m_turntableButton, &QPushButton::clicked, this, &SceneBuilderWidget::showTurntableDialog);
	connect(m_flythroughButton, &QPushButton::clicked, this, &SceneBuilderWidget::showFlythroughDialog);

	// ---- right: inspector
	auto *right = new QWidget(split);
	auto *inspectorPane = right;
	auto *rightLayout = new QVBoxLayout(inspectorPane);
	rightLayout->setContentsMargins(0, 0, 0, 0);
	m_inspectorScroll = new QScrollArea(inspectorPane);
	m_inspectorScroll->setWidgetResizable(true);
	m_inspectorScroll->setFrameShape(QFrame::NoFrame);
	rightLayout->addWidget(m_inspectorScroll, 1);
	m_problemsLabel = new QLabel(inspectorPane);
	m_problemsLabel->setWordWrap(true);
	m_problemsLabel->setTextFormat(Qt::RichText);
	rightLayout->addWidget(m_problemsLabel);
	m_inspectorPanel = right;  // rebuildInspector() sets its minimum width from the page it shows
	centre->addWidget(previewBox);
	centre->setStretchFactor(0, 3);
	centre->setStretchFactor(1, 2);
	centre->setSizes({820, 740});
	// The column is tall (view + preview) and scrolls when the window is shorter, instead of squeezing them. The splitter's own height is what scrolls, so the
	// divider can still be dragged to give one of the two more room (each keeps a small minimum; the pair cannot be squeezed below the total).
	layoutBox->setMinimumHeight(320);
	previewBox->setMinimumHeight(260);
	centre->setMinimumHeight(1560);

	split->addWidget(left);
	auto *centreScroll = new QScrollArea(split);
	centreScroll->setWidget(centre);
	centreScroll->setWidgetResizable(true);
	centreScroll->setFrameShape(QFrame::NoFrame);
	centreScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);  // always there, so it is clear the column scrolls when the window is short
	split->addWidget(centreScroll);
	split->addWidget(right);
	split->setStretchFactor(0, 0);
	split->setStretchFactor(1, 1);
	split->setStretchFactor(2, 0);
	split->setSizes({262, 560, 320});   // wide enough for Add / Duplicate / Delete on one line
	centreScroll->setMinimumWidth(340);   // wide enough for the Top / Front / Side / 3D row of buttons
	m_mainSplit = split;
	m_centreSplit = centre;
	// The panes come back as the user left them (not under the self-test, which compares screenshots).
	if (!qEnvironmentVariableIsSet("RT_GUI_SELFTEST")) {
		window_geometry::restoreSplitter(split, "builder/mainSplit");
		window_geometry::restoreSplitter(centre, "builder/centreSplit");
	}
}

void SceneBuilderWidget::resetPaneSizes() {
	if (m_mainSplit) m_mainSplit->setSizes({230, 560, 320});
	if (m_centreSplit) m_centreSplit->setSizes({820, 740});
	window_geometry::saveSplitter(m_mainSplit, "builder/mainSplit");
	window_geometry::saveSplitter(m_centreSplit, "builder/centreSplit");
}

// ---- undo, autosave -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::pushUndo() { m_history.record(scene_doc::toJson(m_doc)); }

void SceneBuilderWidget::edit(const QString &key, const std::function<void()> &mutate) {
	const bool merge = !key.isEmpty() && key == m_lastEditKey && (key.startsWith("drag#") || (m_editClock.isValid() && m_editClock.elapsed() < 1200));
	if (!merge) {
		flushEditLog();
		m_logBase = m_doc;
		m_logPending = true;
		pushUndo();
	}
	mutate();
	if (skySignature() != m_skySignature) syncSunAndSky();
	m_lastEditKey = key;
	m_editClock.restart();
	if (!m_logTimer) {
		m_logTimer = new QTimer(this);
		m_logTimer->setSingleShot(true);
		connect(m_logTimer, &QTimer::timeout, this, &SceneBuilderWidget::flushEditLog);
	}
	m_logTimer->start(800);   // a drag or typing is one entry, written when it pauses
	documentChanged();
}

void SceneBuilderWidget::flushEditLog() {
	if (!m_logPending) return;
	m_logPending = false;
	const std::string what = scene_doc::describeChange(m_logBase, m_doc);
	if (!what.empty()) AppLog::info(QStringLiteral("builder"), QStringLiteral("edit: %1").arg(QString::fromStdString(what)));
}

bool SceneBuilderWidget::restore(const std::string &json) {
	Document d;
	std::string err;
	if (!scene_doc::fromJson(json, d, err)) return false;
	m_doc = std::move(d);
	// What was picked stays picked, except items the restored document does not have.
	const auto gone = [this](const BuilderSelection &s) {
		return (s.kind == SelKind::Object && s.index >= static_cast<int>(m_doc.objects.size())) || (s.kind == SelKind::Light && s.index >= static_cast<int>(m_doc.lights.size()));
	};
	m_extra.erase(std::remove_if(m_extra.begin(), m_extra.end(), gone), m_extra.end());
	if (gone(m_sel)) {
		m_sel = {SelKind::None, 0};
		m_extra.clear();
	}
	m_lastEditKey.clear();
	rebuildList();
	applySelection(m_sel, false);
	documentChanged();
	return true;
}

bool SceneBuilderWidget::undo() {
	if (!m_history.canUndo()) return false;
	flushEditLog();
	const scene_doc::Document before = m_doc;
	std::string now = scene_doc::toJson(m_doc);
	if (!restore(m_history.undoTop())) {   // the step stays in the list if it cannot be read back
		AppLog::error(QStringLiteral("builder"), QStringLiteral("undo: the saved step could not be read back"));
		return false;
	}
	m_history.undone(std::move(now));
	AppLog::info(QStringLiteral("builder"), QStringLiteral("undo: %1").arg(QString::fromStdString(scene_doc::describeChange(before, m_doc))));
	return true;
}

bool SceneBuilderWidget::redo() {
	if (!m_history.canRedo()) return false;
	flushEditLog();
	const scene_doc::Document before = m_doc;
	std::string now = scene_doc::toJson(m_doc);
	if (!restore(m_history.redoTop())) {
		AppLog::error(QStringLiteral("builder"), QStringLiteral("redo: the saved step could not be read back"));
		return false;
	}
	m_history.redone(std::move(now));
	AppLog::info(QStringLiteral("builder"), QStringLiteral("redo: %1").arg(QString::fromStdString(scene_doc::describeChange(before, m_doc))));
	return true;
}

std::string SceneBuilderWidget::skySignature() const {
	for (const Light &l : m_doc.lights)
		if (l.kind == LightKind::Infinite && l.physicalSky) return scene_doc::sky::skyFileName(l.sky) + "|" + std::to_string(l.intensity);
	return std::string();
}

void SceneBuilderWidget::syncSunAndSky() {
	for (size_t i = 0; i < m_doc.lights.size(); ++i) {
		if (m_doc.lights[i].kind != LightKind::Infinite || !m_doc.lights[i].physicalSky) continue;
		const size_t before = m_doc.lights.size();
		std::string error;
		if (!scene_doc::sky::applySunAndSky(m_doc.lights, i, skyImageFolder().toStdString(), error)) {
			AppLog::warn(QStringLiteral("builder"), QStringLiteral("Sun & sky: %1").arg(QString::fromStdString(error)));
			m_doc.lights[i].physicalSky = false;   // nothing was generated: do not claim a sky that is not there
		} else if (m_doc.lights.size() != before) {
			rebuildList();   // a new "Sun" light was added
		}
		return;
	}
}

void SceneBuilderWidget::documentChanged() {
	m_skySignature = skySignature();
	m_dirty = true;
	refreshViewCombo();
	refreshListLabels();
	refreshProblems();
	updateTitle();
	updateActions();
	updateViews();
	scheduleAutosave();
}

// ---- list, selection ------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::rebuildList() {
	QSignalBlocker block(m_list);
	m_list->clear();
	auto add = [this](const QString &text, SelKind kind, int index) {
		auto *item = new QListWidgetItem(text, m_list);
		item->setData(Qt::UserRole, static_cast<int>(kind));
		item->setData(Qt::UserRole + 1, index);
	};
	add(tr("Camera and image"), SelKind::Camera, 0);
	for (int i = 0; i < static_cast<int>(m_doc.objects.size()); ++i) add(QString(), SelKind::Object, i);
	for (int i = 0; i < static_cast<int>(m_doc.lights.size()); ++i) add(QString(), SelKind::Light, i);
	refreshListLabels();
	syncListSelection();   // reselect
}

// Highlights the rows of the main item and the others picked with it; the main one is the list's current row.
void SceneBuilderWidget::syncListSelection() {
	QSignalBlocker block(m_list);
	m_list->clearSelection();
	m_list->setCurrentRow(-1);
	for (int r = 0; r < m_list->count(); ++r) {
		QListWidgetItem *it = m_list->item(r);
		const BuilderSelection row{static_cast<SelKind>(it->data(Qt::UserRole).toInt()), it->data(Qt::UserRole + 1).toInt()};
		if (row.kind == SelKind::None) continue;
		if (row == m_sel) m_list->setCurrentRow(r, QItemSelectionModel::ClearAndSelect);
		else if (std::find(m_extra.begin(), m_extra.end(), row) != m_extra.end()) it->setSelected(true);
	}
}

void SceneBuilderWidget::refreshListLabels() {
	for (int r = 0; r < m_list->count(); ++r) {
		QListWidgetItem *it = m_list->item(r);
		const auto kind = static_cast<SelKind>(it->data(Qt::UserRole).toInt());
		const int i = it->data(Qt::UserRole + 1).toInt();
		if (kind == SelKind::Object && i < static_cast<int>(m_doc.objects.size())) {
			const Object &o = m_doc.objects[i];
			it->setText(QString("%1  -  %2%3").arg(QString::fromStdString(o.name), shapeLabel(o.shape).section(' ', 0, 0), o.emissive ? tr(", light") : QString()));
		} else if (kind == SelKind::Light && i < static_cast<int>(m_doc.lights.size())) {
			const Light &l = m_doc.lights[i];
			const QString type = lightLabel(l.kind).section(" (", 0, 0);
			const QString name = QString::fromStdString(l.name);
			it->setText(name.compare(type, Qt::CaseInsensitive) == 0 || name.startsWith(type.section(' ', 0, 0)) ? name : QString("%1  -  %2").arg(name, type));
		}
	}
}

void SceneBuilderWidget::onListSelectionChanged() {
	const QList<QListWidgetItem *> rows = m_list->selectedItems();
	QListWidgetItem *it = m_list->currentItem();
	if (it && !it->isSelected()) it = nullptr;   // the current row can be one that a Ctrl-click just took out of the selection
	if (!it && !rows.isEmpty()) it = rows.first();
	if (!it) {
		setSelection({SelKind::None, 0}, true);
		return;
	}
	const auto toSelection = [](const QListWidgetItem *row) { return BuilderSelection{static_cast<SelKind>(row->data(Qt::UserRole).toInt()), row->data(Qt::UserRole + 1).toInt()}; };
	const BuilderSelection s = toSelection(it);
	std::vector<BuilderSelection> others;
	if (rows.size() > 1)
		for (const QListWidgetItem *row : rows) {
			const BuilderSelection other = toSelection(row);
			if (!(other == s) && other.kind != SelKind::Camera) others.push_back(other);   // the camera is edited on its own
		}
	logSelection(s, "list");
	if (others.empty()) {
		setSelection(s, true);
		return;
	}
	if (s.kind == SelKind::Camera) {   // Ctrl-clicked onto the camera's row: the camera alone
		setSelection(s, true);
		return;
	}
	m_extra = others;
	applySelection(s, true);
	AppLog::info(QStringLiteral("builder"), QStringLiteral("select: %1 items (list)").arg(others.size() + 1));
}

void SceneBuilderWidget::logSelection(const BuilderSelection &s, const char *where) {
	if (s == m_sel) return;   // clicking what is already selected is not news
	QString what;
	switch (s.kind) {
		case SelKind::Camera: what = QStringLiteral("camera and image"); break;
		case SelKind::Object:
			if (s.index >= 0 && s.index < static_cast<int>(m_doc.objects.size())) what = QStringLiteral("object '%1'").arg(QString::fromStdString(m_doc.objects[s.index].name));
			break;
		case SelKind::Light:
			if (s.index >= 0 && s.index < static_cast<int>(m_doc.lights.size())) what = QStringLiteral("light '%1'").arg(QString::fromStdString(m_doc.lights[s.index].name));
			break;
		case SelKind::None: what = QStringLiteral("nothing"); break;
	}
	if (!what.isEmpty()) AppLog::info(QStringLiteral("builder"), QStringLiteral("select: %1 (%2)").arg(what, QString::fromLatin1(where)));
}

void SceneBuilderWidget::setSelection(const BuilderSelection &s, bool fromList) {
	m_extra.clear();
	applySelection(s, fromList);
}

void SceneBuilderWidget::applySelection(const BuilderSelection &s, bool fromList) {
	m_sel = s;
	if (!fromList) syncListSelection();
	selectInViews(s);
	rebuildInspector();
	updateActions();
}

// Presses on object `index` in the layout view, drags by `deltaPx` pixels and releases, with real mouse events; returns whether the object moved.
bool SceneBuilderWidget::dragObjectForTest(int index, const QPointF &deltaPx) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	const Float3 before = m_doc.objects[index].position;
	const QPointF start = m_view->itemScreenPos({SelKind::Object, index});
	auto send = [this](QEvent::Type type, const QPointF &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
		QMouseEvent e(type, pos, m_view->mapToGlobal(pos), button, buttons, Qt::NoModifier);
		QApplication::sendEvent(m_view, &e);
	};
	send(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) send(QEvent::MouseMove, start + deltaPx * (step / 4.0), Qt::NoButton, Qt::LeftButton);
	send(QEvent::MouseButtonRelease, start + deltaPx, Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].position;
	return before.x != after.x || before.y != after.y || before.z != after.z;
}

void SceneBuilderWidget::selectObject(int index) {
	setSelection({SelKind::Object, index});
}

void SceneBuilderWidget::selectLight(int index) {
	setSelection({SelKind::Light, index});
}

// The prop's objects go in together at the drop point, named alike ("Table top", "Table leg 1", ...; a second table gets "Table top 2"...), and the first
// is selected. They are ordinary objects from then on.
void SceneBuilderWidget::addProp(scene_doc::PropKind kind) { addParts(scene_doc::makeProp(kind), propLabel(kind)); }
void SceneBuilderWidget::addBlocky(scene_doc::BlockyKind kind) { addParts(scene_doc::makeBlocky(kind), blockyLabel(kind)); }

void SceneBuilderWidget::addParts(std::vector<Object> parts, const QString &groupName) {
	scene_doc::ItemSet addedItems;
	edit(QString(), [&]() {
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		int copy = 1;   // the first number whose suffix leaves every part's name free
		for (;; ++copy) {
			bool free = true;
			for (const Object &p : parts) free = free && !names.contains(QString::fromStdString(p.name) + (copy > 1 ? QString(" %1").arg(copy) : QString()));
			if (free) break;
		}
		const Float3 c = dropPoint();
		const int first = static_cast<int>(m_doc.objects.size());
		for (Object &p : parts) {
			if (copy > 1) p.name += " " + std::to_string(copy);
			p.position.x += c.x;
			p.position.y += c.y;
			p.position.z += c.z;
			m_doc.objects.push_back(p);
			addedItems.objects.push_back(static_cast<int>(m_doc.objects.size()) - 1);
		}
		if (parts.size() >= 2 && !groupName.isEmpty()) scene_doc::groupObjects(m_doc, addedItems.objects, groupName.toStdString());   // a table is one thing to pick and move
		m_sel = {SelKind::Object, first};
	});
	rebuildList();
	pickItems(addedItems, m_sel);
}

void SceneBuilderWidget::addObject(ShapeKind shape) {
	std::string fileName;
	if (shape == ShapeKind::Mesh) {
		const QString f = QFileDialog::getOpenFileName(this, tr("Choose a mesh"), QString(), tr("Meshes (*.ply *.obj)"));
		if (f.isEmpty()) return;
		fileName = f.toStdString();
	}
	edit(QString(), [&]() {
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		const QString name = uniqueName(shapeLabel(shape).section(' ', 0, 0), names);
		Object o = scene_doc::makeObject(shape, name.toStdString());
		o.meshFile = fileName;
		// Drop it at the middle of the layout view, keeping the shape's own height above the floor where the view does not show it.
		const Float3 c = dropPoint();
		o.position.x = c.x;
		o.position.z = c.z;
		if (c.y != 0.0) o.position.y = c.y;
		m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::addLight(LightKind kind) {
	edit(QString(), [&]() {
		Light l;
		l.kind = kind;
		QStringList names;
		for (const Light &existing : m_doc.lights) names << QString::fromStdString(existing.name);
		l.name = uniqueName(lightLabel(kind).section(' ', 0, 0), names).toStdString();
		switch (kind) {
			case LightKind::Point: l.position = {1.0, 4.0, 2.0}; l.intensity = 40.0; break;
			case LightKind::Spot: l.position = {0.0, 5.0, 2.0}; l.target = {0.0, 0.0, 0.0}; l.intensity = 120.0; break;
			case LightKind::Distant: l.position = {4.0, 6.0, 3.0}; l.target = {0.0, 0.0, 0.0}; l.color = {1.0, 0.95, 0.85}; l.intensity = 3.0; break;
			case LightKind::Infinite: l.color = {0.55, 0.70, 1.0}; l.intensity = 0.5; break;
		}
		const Float3 c = dropPoint();
		if (kind == LightKind::Point || kind == LightKind::Spot) { l.position.x = c.x; l.position.z = c.z; }
		m_doc.lights.push_back(l);
		m_sel = {SelKind::Light, static_cast<int>(m_doc.lights.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

// A numbered copy of what is picked ("Chair" gives "Chair 2"), beside the original, with its place relative to the others kept and a group of its own if it was one;
// the copies are picked afterwards. One undo step.
void SceneBuilderWidget::duplicateSelected() {
	const scene_doc::ItemSet items = pickedItems();
	if (items.empty()) return;
	scene_doc::ItemSet added;
	edit(QString(), [&]() { added = scene_doc::duplicateItems(m_doc, items); });
	rebuildList();
	pickItems(added, added.objects.empty() ? BuilderSelection{SelKind::Light, added.lights.front()} : BuilderSelection{SelKind::Object, added.objects.front()});
}

// Objects made from another one (scene_array.h: a grid, a ring, a scatter) go in together, as one undo step; the first is selected.
void SceneBuilderWidget::addCopies(const std::vector<Object> &copies) {
	if (copies.empty()) return;
	edit(QString(), [&]() {
		const int first = static_cast<int>(m_doc.objects.size());
		for (const Object &o : copies) m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, first};
	});
	AppLog::info(QStringLiteral("builder"), QStringLiteral("array: added %1 copies").arg(copies.size()));
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::showArrayDialog() {
	if (m_sel.kind != SelKind::Object || m_sel.index >= static_cast<int>(m_doc.objects.size())) return;
	std::vector<int> together;   // several picked objects are the unit that is copied; with one, the window suggests the parts that belong with it
	for (const BuilderSelection &other : m_extra)
		if (other.kind == SelKind::Object && other.index < static_cast<int>(m_doc.objects.size())) together.push_back(other.index);
	ArrayDialog dialog(m_sel.index, m_doc.objects, this, together);
	if (dialog.exec() != QDialog::Accepted) return;
	std::vector<Object> made = dialog.copies();
	scene_doc::groupCopies(m_doc, made, dialog.unit());   // copies of a group are groups
	addCopies(made);
}

void SceneBuilderWidget::deleteSelected() {
	const scene_doc::ItemSet items = pickedItems();
	if (items.empty()) return;
	edit(QString(), [&]() { scene_doc::eraseItems(m_doc, items); });
	m_sel = {SelKind::None, 0};
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::updateActions() {
	const bool item = (m_sel.kind == SelKind::Object || m_sel.kind == SelKind::Light);
	m_deleteButton->setEnabled(item);
	m_duplicateButton->setEnabled(item);
	m_arrayButton->setEnabled(m_sel.kind == SelKind::Object);   // (with several picked, the others go into the copies)
	m_undoButton->setEnabled(m_history.canUndo());
	m_redoButton->setEnabled(m_history.canRedo());
	const bool ok = !scene_doc::hasErrors(m_problems);   // from the last refreshProblems(): the document has not changed since
	m_previewButton->setEnabled(ok || m_process);
	m_finalButton->setEnabled(ok && !m_process);
	m_turntableButton->setEnabled(ok && !m_process);
	m_flythroughButton->setEnabled(ok && !m_process && m_doc.cameraViews.size() >= 2);
}

void SceneBuilderWidget::updateTitle() {
	const QString file = m_path.isEmpty() ? tr("not saved yet") : QFileInfo(m_path).fileName();
	const QString name = QString::fromStdString(m_doc.title);
	if (m_titleEdit && m_titleEdit->text() != name) {   // an undo, a new scene, an open: not the user typing
		const QSignalBlocker blocker(m_titleEdit);
		m_titleEdit->setText(name);
	}
	m_titleLabel->setText(tr("(%1)%2  |  %3 objects, %4 lights")
	                          .arg(file, m_dirty ? " *" : "")
	                          .arg(m_doc.objects.size())
	                          .arg(m_doc.lights.size()));
	m_titleLabel->setToolTip(m_titleLabel->text());   // the label is cut short when the page is narrow
}

QString SceneBuilderWidget::problemsText() const {
	QString s;
	for (const auto &p : scene_doc::validate(m_doc))
		s += (p.severity == scene_doc::Problem::Severity::Error ? "Error: " : "Note: ") + QString::fromStdString(p.message) + "\n";
	return s;
}

void SceneBuilderWidget::refreshProblems() {
	m_problems = scene_doc::validate(m_doc);   // once per change - it looks at the picture, mesh and sky files too
	const auto &problems = m_problems;
	{   // each problem is logged once, when it first appears (and again if it goes away and comes back)
		QStringList now;
		for (const auto &p : problems) {
			const QString line = QStringLiteral("%1: %2").arg(p.severity == scene_doc::Problem::Severity::Error ? QStringLiteral("error") : QStringLiteral("note"), QString::fromStdString(p.message));
			now << line;
			if (!m_loggedProblems.contains(line))
				AppLog::write(p.severity == scene_doc::Problem::Severity::Error ? log_format::Level::Warn : log_format::Level::Info, QStringLiteral("builder"), QStringLiteral("scene problem - %1").arg(line));
		}
		m_loggedProblems = now;
	}
	if (problems.empty()) {
		m_problemsLabel->setText(tr("No problems found."));
		return;
	}
	QString html;
	for (const auto &p : problems) {
		const bool error = p.severity == scene_doc::Problem::Severity::Error;
		html += QString("<p style='margin:2px 0'><b>%1</b> %2</p>").arg(error ? tr("Fix this:") : tr("Note:"), QString::fromStdString(p.message).toHtmlEscaped());
	}
	m_problemsLabel->setText(html);
}

bool SceneBuilderWidget::eventFilter(QObject *watched, QEvent *event) {
	if (watched == m_previewLabel && event->type() == QEvent::Resize) updatePreviewPixmap();
	return QWidget::eventFilter(watched, event);
}

void SceneBuilderWidget::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	updatePreviewPixmap();
}
