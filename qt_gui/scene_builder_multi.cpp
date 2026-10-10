// The Scene Builder with several items picked: Ctrl- or Shift-click and a box in the views, groups, and the panel shown while more than one is picked. The
// operations themselves (move, copy, delete, group, copy a look) are src/shared/scene_selection.h; this file is how the editing surface reaches them.
#include "scene_builder_widget.h"

#include "app_log.h"
#include "scene_builder_common.h"

#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>

#include <algorithm>
#include <array>

using namespace scene_builder_ui;

using scene_doc::Float3;
using scene_doc::ItemSet;

namespace {

BuilderSelection objectSel(int i) { return {BuilderSelection::Kind::Object, i}; }
BuilderSelection lightSel(int i) { return {BuilderSelection::Kind::Light, i}; }

}  // namespace

// The main item and the others picked along with it, as indices. The camera is never part of it (it is edited on its own).
ItemSet SceneBuilderWidget::pickedItems() const {
	std::vector<int> objects, lights;
	const auto add = [&](const BuilderSelection &s) {
		if (s.kind == SelKind::Object) objects.push_back(s.index);
		else if (s.kind == SelKind::Light) lights.push_back(s.index);
	};
	add(m_sel);
	for (const BuilderSelection &s : m_extra) add(s);
	return scene_doc::makeItemSet(m_doc, objects, lights);
}

// Picks exactly these items; `mainItem` (if it is one of them) is the main one, else the first object, else the first light.
void SceneBuilderWidget::pickItems(const ItemSet &items, const BuilderSelection &mainItem) {
	std::vector<BuilderSelection> all;
	for (int i : items.objects) all.push_back(objectSel(i));
	for (int i : items.lights) all.push_back(lightSel(i));
	if (all.empty()) {
		setSelection(BuilderSelection{SelKind::None, 0});
		return;
	}
	BuilderSelection main = all.front();
	if (std::find(all.begin(), all.end(), mainItem) != all.end()) main = mainItem;
	m_extra.clear();
	for (const BuilderSelection &s : all)
		if (!(s == main)) m_extra.push_back(s);
	applySelection(main, false);
}

void SceneBuilderWidget::pickObjects(const std::vector<int> &objects, const std::vector<int> &lights) {
	const BuilderSelection first = !objects.empty() ? objectSel(objects.front()) : !lights.empty() ? lightSel(lights.front()) : BuilderSelection{};
	pickItems(scene_doc::makeItemSet(m_doc, objects, lights), first);
}

void SceneBuilderWidget::selectAll() {
	std::vector<int> objects, lights;
	for (int i = 0; i < static_cast<int>(m_doc.objects.size()); ++i) objects.push_back(i);
	for (int i = 0; i < static_cast<int>(m_doc.lights.size()); ++i)
		if (m_doc.lights[i].kind != scene_doc::LightKind::Infinite) lights.push_back(i);   // the sky has nowhere to be moved to
	pickObjects(objects, lights);
	AppLog::info(QStringLiteral("builder"), QStringLiteral("select: all (%1 items)").arg(objects.size() + lights.size()));
}

// A click on an object in a view: the whole group it is in, or just it.
void SceneBuilderWidget::selectWithGroup(const BuilderSelection &s) {
	if (s.kind == SelKind::Object && s.index >= 0 && s.index < static_cast<int>(m_doc.objects.size()) && !m_doc.objects[s.index].group.empty()) {
		const ItemSet mates = scene_doc::withGroupMates(m_doc, scene_doc::makeItemSet(m_doc, {s.index}));
		if (mates.objects.size() > 1) {
			pickItems(mates, s);
			return;
		}
	}
	setSelection(s);
}

// Ctrl- or Shift-click: the item joins what is picked, or leaves it (a group goes in or out as a whole).
void SceneBuilderWidget::togglePicked(const BuilderSelection &s) {
	if (s.kind != SelKind::Object && s.kind != SelKind::Light) return;
	ItemSet items = pickedItems();   // empty when the camera (or nothing) was selected: the click then starts a new pick
	ItemSet touched = s.kind == SelKind::Object ? scene_doc::withGroupMates(m_doc, scene_doc::makeItemSet(m_doc, {s.index})) : scene_doc::makeItemSet(m_doc, {}, {s.index});
	const std::vector<int> &mine = s.kind == SelKind::Object ? items.objects : items.lights;
	const bool picked = std::find(mine.begin(), mine.end(), s.index) != mine.end();
	if (picked) {
		const auto remove = [](std::vector<int> &from, const std::vector<int> &what) {
			from.erase(std::remove_if(from.begin(), from.end(), [&what](int i) { return std::find(what.begin(), what.end(), i) != what.end(); }), from.end());
		};
		remove(items.objects, touched.objects);
		remove(items.lights, touched.lights);
	} else {
		items.objects.insert(items.objects.end(), touched.objects.begin(), touched.objects.end());
		items.lights.insert(items.lights.end(), touched.lights.begin(), touched.lights.end());
		items = scene_doc::makeItemSet(m_doc, items.objects, items.lights);
	}
	AppLog::info(QStringLiteral("builder"), QStringLiteral("select: %1 -> %2 items (Ctrl/Shift-click)").arg(picked ? QStringLiteral("removed") : QStringLiteral("added")).arg(items.size()));
	pickItems(items, picked ? m_sel : s);
}

// A box dragged round items: they are added to what is picked (a group the box touches comes whole).
void SceneBuilderWidget::boxPicked(const QList<BuilderSelection> &boxed, bool additive) {
	if (boxed.isEmpty()) return;
	ItemSet items = additive ? pickedItems() : ItemSet{};
	for (const BuilderSelection &s : boxed) {
		if (s.kind == SelKind::Object) items.objects.push_back(s.index);
		else if (s.kind == SelKind::Light) items.lights.push_back(s.index);
	}
	items = scene_doc::withGroupMates(m_doc, scene_doc::makeItemSet(m_doc, items.objects, items.lights));
	AppLog::info(QStringLiteral("builder"), QStringLiteral("select: box -> %1 items").arg(items.size()));
	pickItems(items, m_sel);
}

void SceneBuilderWidget::groupSelected() {
	const ItemSet items = pickedItems();
	if (items.objects.size() < 2) {
		emit statusMessage(tr("Pick two or more objects to group them."));
		return;
	}
	std::string name;
	edit(QString(), [&]() { name = scene_doc::groupObjects(m_doc, items.objects, "Group"); });
	AppLog::info(QStringLiteral("builder"), QStringLiteral("group: %1 objects -> '%2'").arg(items.objects.size()).arg(QString::fromStdString(name)));
	rebuildList();
	pickItems(items, m_sel);
}

void SceneBuilderWidget::ungroupSelected() {
	const ItemSet items = pickedItems();
	int freed = 0;
	bool any = false;
	for (int i : items.objects) any = any || !m_doc.objects[i].group.empty();
	if (!any) return;
	edit(QString(), [&]() { freed = scene_doc::ungroupObjects(m_doc, items.objects); });
	AppLog::info(QStringLiteral("builder"), QStringLiteral("ungroup: %1 objects freed").arg(freed));
	rebuildList();
	pickItems(items, m_sel);
}

void SceneBuilderWidget::moveSelectedBy(const Float3 &delta) {
	const ItemSet items = pickedItems();
	if (items.empty() || (delta.x == 0.0 && delta.y == 0.0 && delta.z == 0.0)) return;
	edit(QString(), [&]() { scene_doc::translateItems(m_doc, items, delta); });
	refreshInspectorValues();
}

void SceneBuilderWidget::useLookOfMainObject() {
	if (m_sel.kind != SelKind::Object) return;
	const ItemSet items = pickedItems();
	if (items.objects.size() < 2) return;
	edit(QString(), [&]() { scene_doc::copyLook(m_doc, m_sel.index, items.objects); });
}

// The panel for several picked items: what they are, the group they form, a move by an amount, and a look copied from the main object.
void SceneBuilderWidget::inspectMultiple(QFormLayout *f) {
	const ItemSet items = pickedItems();
	auto *heading = new QLabel(tr("<b>%1 items picked</b>").arg(items.size()));
	f->addRow(heading);
	QStringList names;
	for (int i : items.objects) names << QString::fromStdString(m_doc.objects[i].name);
	for (int i : items.lights) names << QString::fromStdString(m_doc.lights[i].name);
	QString shown = names.mid(0, 8).join(QStringLiteral(", "));
	if (names.size() > 8) shown += tr(", and %1 more").arg(names.size() - 8);
	auto *list = new QLabel(shown);
	list->setWordWrap(true);
	f->addRow(list);
	auto *how = new QLabel(tr("Drag any of them to move them all. Delete and Duplicate act on all of them."));
	how->setWordWrap(true);
	f->addRow(how);

	// Group: the name they share, if they are one group; else a button to make them one.
	std::string common;
	bool oneGroup = !items.objects.empty();
	for (int i : items.objects) {
		if (m_doc.objects[i].group.empty() || (!common.empty() && m_doc.objects[i].group != common)) oneGroup = false;
		common = m_doc.objects[i].group;
	}
	bool anyGroup = false;
	for (int i : items.objects) anyGroup = anyGroup || !m_doc.objects[i].group.empty();
	if (oneGroup) f->addRow(tr("Group:"), new QLabel(QString::fromStdString(common).toHtmlEscaped()));
	auto *groupRow = new QWidget;
	auto *groupLayout = new QHBoxLayout(groupRow);
	groupLayout->setContentsMargins(0, 0, 0, 0);
	auto *group = new QPushButton(tr("Group"));
	group->setToolTip(tr("Make the picked objects one group: clicking one in a view picks them all (%1)").arg(QKeySequence(Qt::CTRL | Qt::Key_G).toString(QKeySequence::NativeText)));
	group->setEnabled(items.objects.size() >= 2 && !oneGroup);
	auto *ungroup = new QPushButton(tr("Ungroup"));
	ungroup->setToolTip(tr("Dissolve the group (%1)").arg(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_G).toString(QKeySequence::NativeText)));
	ungroup->setEnabled(anyGroup);
	for (QPushButton *b : {group, ungroup}) {
		b->setAutoDefault(false);
		compactStyle(b);
		groupLayout->addWidget(b);
	}
	groupLayout->addStretch(1);
	f->addRow(groupRow);
	connect(group, &QPushButton::clicked, this, [this]() { groupSelected(); });
	connect(ungroup, &QPushButton::clicked, this, [this]() { ungroupSelected(); });

	// Move by an amount, for when dragging is not exact enough.
	auto *moveRow = new QWidget;
	auto *moveLayout = new QHBoxLayout(moveRow);
	moveLayout->setContentsMargins(0, 0, 0, 0);
	std::array<QDoubleSpinBox *, 3> amounts{};
	const char *axes[3] = {"X", "Y", "Z"};
	for (int a = 0; a < 3; ++a) {
		amounts[a] = new QDoubleSpinBox;
		amounts[a]->setRange(-1000.0, 1000.0);
		amounts[a]->setDecimals(2);
		amounts[a]->setSingleStep(0.25);
		amounts[a]->setPrefix(QString("%1 ").arg(axes[a]));
		amounts[a]->setKeyboardTracking(false);
		moveLayout->addWidget(amounts[a]);
	}
	auto *move = new QPushButton(tr("Move"));
	move->setAutoDefault(false);
	compactStyle(move);
	f->addRow(tr("Move all by:"), moveRow);
	f->addRow(move);
	connect(move, &QPushButton::clicked, this, [this, amounts]() {
		moveSelectedBy(Float3{amounts[0]->value(), amounts[1]->value(), amounts[2]->value()});
		for (int a = 0; a < 3; ++a) amounts[a]->setValue(0.0);
	});

	if (m_sel.kind == SelKind::Object && items.objects.size() >= 2) {
		auto *look = new QPushButton(tr("Use %1's look for all").arg(QString::fromStdString(m_doc.objects[m_sel.index].name)));
		look->setToolTip(tr("Give the other picked objects the same material and light settings as this one (the main one: the first you picked)"));
		look->setAutoDefault(false);
		f->addRow(look);
		connect(look, &QPushButton::clicked, this, [this]() { useLookOfMainObject(); });
	}
}
