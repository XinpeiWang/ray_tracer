#include "scene_builder_array_dialog.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QListWidget>
#include <QGuiApplication>
#include <QScreen>
#include <QScrollArea>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QTabWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

using scene_doc::Object;

namespace {

QDoubleSpinBox *doubleBox(QWidget *parent, double lo, double hi, double step, int decimals, double value, const QString &suffix = QString()) {
	auto *b = new QDoubleSpinBox(parent);
	b->setRange(lo, hi);
	b->setSingleStep(step);
	b->setDecimals(decimals);
	b->setValue(value);
	if (!suffix.isEmpty()) b->setSuffix(suffix);
	return b;
}

QSpinBox *intBox(QWidget *parent, int lo, int hi, int value) {
	auto *b = new QSpinBox(parent);
	b->setRange(lo, hi);
	b->setValue(value);
	return b;
}

double snapQuarter(double v) { return std::round(v * 4.0) / 4.0; }

}  // namespace

ArrayDialog::ArrayDialog(int sourceIndex, const std::vector<Object> &existing, QWidget *parent, const std::vector<int> &together)
    : QDialog(parent), m_source(existing.at(static_cast<size_t>(sourceIndex))), m_sourceIndex(sourceIndex), m_existing(existing) {
	const Object &source = m_source;
	setWindowTitle(tr("Array - copies of %1").arg(QString::fromStdString(source.name)));
	auto *root = new QVBoxLayout(this);
	auto *intro = new QLabel(tr("Makes copies of <b>%1</b>. The copies are ordinary objects: move, recolour or delete them one by one. The whole thing is one undo step.")
	                             .arg(QString::fromStdString(source.name).toHtmlEscaped()), this);
	intro->setWordWrap(true);
	root->addWidget(intro);

	// The parts that go with it: a tree's crown, a table's legs. Ticked ones are copied together with the selected object and keep their places relative to it.
	if (existing.size() > 1) {
		m_partsBox = new QGroupBox(tr("Copy other objects together with it"), this);
		m_partsBox->setCheckable(true);
		auto *partsLayout = new QVBoxLayout(m_partsBox);
		auto *hint = new QLabel(tr("Tick the parts that belong with it (a tree's crown, a table's legs). They are copied, turned and scaled as one."), m_partsBox);
		hint->setWordWrap(true);
		hint->setTextFormat(Qt::PlainText);
		partsLayout->addWidget(hint);
		m_parts = new QListWidget(m_partsBox);
		m_parts->setMaximumHeight(110);
		std::vector<int> likely = together;
		if (likely.empty()) {
			likely = scene_doc::likelyCompanions(source, existing, sourceIndex);
			if (!source.group.empty())   // the other members of its group belong with it
				for (int i = 0; i < static_cast<int>(existing.size()); ++i)
					if (i != sourceIndex && existing[static_cast<size_t>(i)].group == source.group && std::find(likely.begin(), likely.end(), i) == likely.end()) likely.push_back(i);
		}
		for (int i = 0; i < static_cast<int>(existing.size()); ++i) {
			if (i == sourceIndex) continue;
			auto *item = new QListWidgetItem(QString::fromStdString(existing[static_cast<size_t>(i)].name), m_parts);
			item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
			item->setCheckState(std::find(likely.begin(), likely.end(), i) != likely.end() ? Qt::Checked : Qt::Unchecked);
			item->setData(Qt::UserRole, i);
		}
		partsLayout->addWidget(m_parts);
		m_partsBox->setChecked(!likely.empty());
		root->addWidget(m_partsBox);
		// Collapsed while unticked, so the common case (one object) leaves the room to the settings below.
		const auto showParts = [hint, this](bool on) { hint->setVisible(on); m_parts->setVisible(on); };
		showParts(m_partsBox->isChecked());
		connect(m_partsBox, &QGroupBox::toggled, this, [this, showParts](bool on) { showParts(on); refresh(); });
		connect(m_parts, &QListWidget::itemChanged, this, [this]() { refresh(); });
	}

	m_tabs = new QTabWidget(this);
	// Each page scrolls, so the window can stay a comfortable size on a small screen (the scatter page has a dozen settings).
	const auto scrolling = [this](QWidget *page) {
		auto *area = new QScrollArea(m_tabs);
		area->setWidgetResizable(true);
		area->setFrameShape(QFrame::NoFrame);
		area->setWidget(page);
		return area;
	};
	auto *gridPage = new QWidget();
	auto *ringPage = new QWidget();
	auto *scatterPage = new QWidget();
	buildGridPage(gridPage);
	buildRingPage(ringPage);
	buildScatterPage(scatterPage);
	m_tabs->addTab(scrolling(gridPage), tr("Grid"));
	m_tabs->addTab(scrolling(ringPage), tr("Ring"));
	m_tabs->addTab(scrolling(scatterPage), tr("Scatter"));
	root->addWidget(m_tabs, 1);

	m_summary = new QLabel(this);
	m_summary->setWordWrap(true);
	root->addWidget(m_summary);

	auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	m_ok = buttons->button(QDialogButtonBox::Ok);
	m_ok->setText(tr("Add copies"));
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
	root->addWidget(buttons);

	connect(m_tabs, &QTabWidget::currentChanged, this, [this]() { refresh(); });
	refresh();
	// Tall enough for the grid and ring pages whole, never more than about three quarters of the screen (the scatter page then scrolls).
	const int screenHeight = QGuiApplication::primaryScreen() ? QGuiApplication::primaryScreen()->availableGeometry().height() : 900;
	resize(std::max(sizeHint().width(), 520), std::min(std::max(sizeHint().height(), 560), screenHeight * 3 / 4));
}

void ArrayDialog::buildGridPage(QWidget *page) {
	auto *form = new QFormLayout(page);
	auto *note = new QLabel(tr("A line, a rectangle or a block of copies, starting from the original. Counts include the original; spacing is the distance between neighbours."), page);
	note->setWordWrap(true);
	form->addRow(note);
	// A comfortable first spacing: the object's own width and a little more, on the 0.25 grid.
	const double across = std::max(1.0, snapQuarter(2.0 * scene_doc::footprintRadius(m_source) * 1.2));
	const double up = std::max(1.0, snapQuarter(2.0 * scene_doc::boundingRadius(m_source) * 1.1));
	const char *axes[3] = {"X", "Y (up)", "Z"};
	const double spacing[3] = {across, up, across};
	const int counts[3] = {3, 1, 1};
	for (int a = 0; a < 3; ++a) {
		m_gridCount[a] = intBox(page, 1, 100, counts[a]);
		m_gridSpacing[a] = doubleBox(page, -10000.0, 10000.0, 0.25, 2, spacing[a]);
		auto *row = new QHBoxLayout();
		row->addWidget(m_gridCount[a]);
		row->addWidget(new QLabel(tr("apart by"), page));
		row->addWidget(m_gridSpacing[a]);
		form->addRow(tr("Along %1:").arg(QString::fromLatin1(axes[a])), row);
		connect(m_gridCount[a], QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { refresh(); });
		connect(m_gridSpacing[a], QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this]() { refresh(); });
	}
}

void ArrayDialog::buildRingPage(QWidget *page) {
	auto *form = new QFormLayout(page);
	auto *note = new QLabel(tr("Copies spaced evenly round a point on the floor, as chairs round a table. The original counts as one of them and stays where it is; the "
	                           "circle's spacing starts from the original's own angle."), page);
	note->setTextFormat(Qt::PlainText);
	note->setWordWrap(true);
	form->addRow(note);
	const double dist = std::hypot(m_source.position.x, m_source.position.z);
	m_ringCount = intBox(page, 2, scene_doc::kMaxArrayCopies + 1, 8);
	m_ringRadius = doubleBox(page, 0.01, 10000.0, 0.25, 2, dist > 0.1 ? snapQuarter(dist) : 3.0);
	m_ringCentreX = doubleBox(page, -10000.0, 10000.0, 0.25, 2, 0.0);
	m_ringCentreZ = doubleBox(page, -10000.0, 10000.0, 0.25, 2, 0.0);
	m_ringStart = doubleBox(page, -360.0, 360.0, 5.0, 1, 0.0, QStringLiteral(" °"));
	m_ringFace = new QCheckBox(tr("Turn each copy round with the ring (it keeps facing the centre the way the original does)"), page);
	m_ringFace->setChecked(true);
	form->addRow(tr("Objects in the ring:"), m_ringCount);
	form->addRow(tr("Radius:"), m_ringRadius);
	form->addRow(tr("Centre X:"), m_ringCentreX);
	form->addRow(tr("Centre Z:"), m_ringCentreZ);
	form->addRow(tr("Turn the ring by:"), m_ringStart);
	form->addRow(m_ringFace);
	connect(m_ringCount, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { refresh(); });
	for (QDoubleSpinBox *b : {m_ringRadius, m_ringCentreX, m_ringCentreZ, m_ringStart}) connect(b, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this]() { refresh(); });
	connect(m_ringFace, &QCheckBox::toggled, this, [this]() { refresh(); });
}

void ArrayDialog::buildScatterPage(QWidget *page) {
	auto *form = new QFormLayout(page);
	auto *note = new QLabel(tr("Copies placed at random in an area, each a little different in size and turn: trees, rocks, stars. The same Arrangement number "
	                           "always gives the same result."), page);
	note->setTextFormat(Qt::PlainText);
	note->setWordWrap(true);
	form->addRow(note);
	m_scatterCount = intBox(page, 1, scene_doc::kMaxArrayCopies, 20);
	m_scatterShape = new QComboBox(page);
	m_scatterShape->addItem(tr("Disk"));
	m_scatterShape->addItem(tr("Rectangle"));
	m_scatterCentreX = doubleBox(page, -10000.0, 10000.0, 0.5, 2, m_source.position.x);
	m_scatterCentreZ = doubleBox(page, -10000.0, 10000.0, 0.5, 2, m_source.position.z);
	m_scatterWidth = doubleBox(page, 0.1, 10000.0, 0.5, 2, 10.0);
	m_scatterDepth = doubleBox(page, 0.1, 10000.0, 0.5, 2, 10.0);
	m_scatterGround = doubleBox(page, -10000.0, 10000.0, 0.25, 2, 0.0);
	m_scatterScaleMin = doubleBox(page, 1.0, 1000.0, 5.0, 0, 80.0, QStringLiteral(" %"));
	m_scatterScaleMax = doubleBox(page, 1.0, 1000.0, 5.0, 0, 120.0, QStringLiteral(" %"));
	m_scatterSpin = new QCheckBox(tr("Turn each copy at random (about the vertical)"), page);
	m_scatterSpin->setChecked(true);
	m_scatterTilt = doubleBox(page, 0.0, 45.0, 1.0, 0, 0.0, QStringLiteral(" °"));
	m_scatterApart = new QCheckBox(tr("Keep them from overlapping on the floor"), page);
	m_scatterApart->setChecked(true);
	m_scatterGap = doubleBox(page, 0.0, 1000.0, 0.25, 2, 0.0);
	m_scatterSeed = intBox(page, 1, 999999, 1);
	auto *another = new QPushButton(tr("Another arrangement"), page);
	another->setAutoDefault(false);
	form->addRow(tr("How many:"), m_scatterCount);
	form->addRow(tr("Area:"), m_scatterShape);
	form->addRow(tr("Centre X:"), m_scatterCentreX);
	form->addRow(tr("Centre Z:"), m_scatterCentreZ);
	form->addRow(tr("Width (diameter):"), m_scatterWidth);
	form->addRow(tr("Depth:"), m_scatterDepth);
	form->addRow(tr("Floor height:"), m_scatterGround);
	form->addRow(tr("Smallest size:"), m_scatterScaleMin);
	form->addRow(tr("Largest size:"), m_scatterScaleMax);
	form->addRow(m_scatterSpin);
	form->addRow(tr("Lean up to:"), m_scatterTilt);
	form->addRow(m_scatterApart);
	form->addRow(tr("Extra space between:"), m_scatterGap);
	auto *seedRow = new QHBoxLayout();
	seedRow->addWidget(m_scatterSeed);
	seedRow->addWidget(another);
	form->addRow(tr("Arrangement:"), seedRow);
	m_scatterDepth->setEnabled(false);   // a disk has no depth
	connect(m_scatterShape, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int i) {
		m_scatterDepth->setEnabled(i == 1);
		refresh();
	});
	connect(another, &QPushButton::clicked, this, [this]() { m_scatterSeed->setValue(m_scatterSeed->value() % 999999 + 1); });
	for (QSpinBox *b : {m_scatterCount, m_scatterSeed}) connect(b, QOverload<int>::of(&QSpinBox::valueChanged), this, [this]() { refresh(); });
	for (QDoubleSpinBox *b : {m_scatterCentreX, m_scatterCentreZ, m_scatterWidth, m_scatterDepth, m_scatterGround, m_scatterScaleMin, m_scatterScaleMax, m_scatterTilt, m_scatterGap})
		connect(b, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this]() { refresh(); });
	for (QCheckBox *c : {m_scatterSpin, m_scatterApart}) connect(c, &QCheckBox::toggled, this, [this]() { refresh(); });
}

ArrayDialog::Mode ArrayDialog::mode() const {
	switch (m_tabs->currentIndex()) {
		case 1: return Mode::Ring;
		case 2: return Mode::Scatter;
		default: return Mode::Grid;
	}
}

scene_doc::GridParams ArrayDialog::grid() const {
	scene_doc::GridParams p;
	p.countX = m_gridCount[0]->value();
	p.countY = m_gridCount[1]->value();
	p.countZ = m_gridCount[2]->value();
	p.spacing = {m_gridSpacing[0]->value(), m_gridSpacing[1]->value(), m_gridSpacing[2]->value()};
	return p;
}

scene_doc::RingParams ArrayDialog::ring() const {
	scene_doc::RingParams p;
	p.count = m_ringCount->value();
	p.radius = m_ringRadius->value();
	p.centreX = m_ringCentreX->value();
	p.centreZ = m_ringCentreZ->value();
	p.startAngle = m_ringStart->value();
	p.turnWithRing = m_ringFace->isChecked();
	return p;
}

scene_doc::ScatterParams ArrayDialog::scatterParams() const {
	scene_doc::ScatterParams p;
	p.count = m_scatterCount->value();
	p.disk = m_scatterShape->currentIndex() == 0;
	p.centreX = m_scatterCentreX->value();
	p.centreZ = m_scatterCentreZ->value();
	p.width = m_scatterWidth->value();
	p.depth = m_scatterDepth->value();
	p.groundY = m_scatterGround->value();
	p.scaleMin = m_scatterScaleMin->value() / 100.0;
	p.scaleMax = m_scatterScaleMax->value() / 100.0;
	p.randomSpin = m_scatterSpin->isChecked();
	p.tiltDegrees = m_scatterTilt->value();
	p.keepApart = m_scatterApart->isChecked();
	p.extraGap = m_scatterGap->value();
	p.seed = static_cast<std::uint32_t>(m_scatterSeed->value());
	return p;
}

std::vector<Object> ArrayDialog::unit() const {
	std::vector<Object> parts = {m_source};
	if (m_partsBox && m_partsBox->isChecked())
		for (int row = 0; row < m_parts->count(); ++row) {
			const QListWidgetItem *item = m_parts->item(row);
			if (item->checkState() == Qt::Checked) parts.push_back(m_existing[static_cast<size_t>(item->data(Qt::UserRole).toInt())]);
		}
	return parts;
}

std::vector<Object> ArrayDialog::copies(int *asked) const {
	const std::vector<Object> parts = unit();
	const int per = static_cast<int>(parts.size());
	switch (mode()) {
		case Mode::Ring: {
			const scene_doc::RingParams p = ring();
			if (asked) *asked = std::max(p.count - 1, 0) * per;
			return scene_doc::makeRing(parts, p, m_existing);
		}
		case Mode::Scatter: {
			const scene_doc::ScatterParams p = scatterParams();
			scene_doc::ScatterResult r = scene_doc::scatter(parts, p, m_existing);
			if (asked) *asked = r.requested * per;
			return std::move(r.objects);
		}
		case Mode::Grid:
		default: {
			const scene_doc::GridParams p = grid();
			if (asked) *asked = static_cast<int>(std::min<long long>(scene_doc::gridRequested(p) * per, 1000000));
			return scene_doc::makeGrid(parts, p, m_existing);
		}
	}
}

void ArrayDialog::refresh() {
	if (!m_summary || !m_ok) return;
	int asked = 0;
	const int made = static_cast<int>(copies(&asked).size());
	QString text;
	if (made == 0) text = tr("Nothing to add yet: set a count of two or more.");
	else if (mode() != Mode::Scatter && asked > made) text = tr("Adds %1 objects (you asked for %2; one array adds at most %3 copies and %4 objects).").arg(made).arg(asked).arg(scene_doc::kMaxArrayCopies).arg(scene_doc::kMaxArrayObjects);
	else if (made < asked) text = tr("Adds %1 objects (%2 were asked for, but there is no room for more without overlapping: make the area bigger or turn the overlap check off).").arg(made).arg(asked);
	else text = tr("Adds %1 objects.").arg(made);
	m_summary->setText(text);
	m_ok->setEnabled(made > 0);
}

void ArrayDialog::setMode(Mode m) { m_tabs->setCurrentIndex(m == Mode::Ring ? 1 : (m == Mode::Scatter ? 2 : 0)); }
void ArrayDialog::setGridCounts(int x, int y, int z) {
	m_gridCount[0]->setValue(x);
	m_gridCount[1]->setValue(y);
	m_gridCount[2]->setValue(z);
}
void ArrayDialog::setRingCount(int count) { m_ringCount->setValue(count); }
void ArrayDialog::setScatterCount(int count) { m_scatterCount->setValue(count); }

void ArrayDialog::tickPart(const QString &name, bool on) {
	if (!m_parts) return;
	for (int row = 0; row < m_parts->count(); ++row)
		if (m_parts->item(row)->text() == name) m_parts->item(row)->setCheckState(on ? Qt::Checked : Qt::Unchecked);
	if (m_partsBox) m_partsBox->setChecked(true);
}
