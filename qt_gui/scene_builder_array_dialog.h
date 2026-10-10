#pragma once
// The Scene Builder's "Array..." window: make many copies of the selected object as a grid (a line, a rectangle or a block), a ring around a point, or a random
// scatter. The numbers come from src/shared/scene_array.h; this window only gathers the settings and says how many objects they would add.

#include <QDialog>
#include <vector>

#include "../src/shared/scene_array.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QGroupBox;
class QListWidget;
class QLabel;
class QPushButton;
class QSpinBox;
class QTabWidget;

class ArrayDialog : public QDialog {
	Q_OBJECT
public:
	enum class Mode { Grid, Ring, Scatter };

	// `sourceIndex` is the selected object in `existing`, all the document's objects (the defaults come from its size and place; the copies get names nobody uses).
	// `together`: the other objects picked along with it, which are then the ones ticked (instead of the ones the names and the group suggest).
	ArrayDialog(int sourceIndex, const std::vector<scene_doc::Object> &existing, QWidget *parent = nullptr, const std::vector<int> &together = {});

	Mode mode() const;
	scene_doc::GridParams grid() const;
	scene_doc::RingParams ring() const;
	scene_doc::ScatterParams scatterParams() const;

	// The selected object followed by the other objects ticked to go with it: what each copy is made of.
	std::vector<scene_doc::Object> unit() const;
	// The objects the current settings would add (what OK adds), and how many objects the settings asked for before the limit and before a crowded scatter left some out.
	std::vector<scene_doc::Object> copies(int *asked = nullptr) const;

	// For the self-test: pick a page and set its numbers without a mouse.
	void setMode(Mode m);
	void setGridCounts(int x, int y, int z);
	void setRingCount(int count);
	void setScatterCount(int count);
	void tickPart(const QString &name, bool on);   // tick or untick one of the "together with" objects by name

private:
	void buildGridPage(QWidget *page);
	void buildRingPage(QWidget *page);
	void buildScatterPage(QWidget *page);
	void refresh();   // the "this adds N objects" line and the OK button

	scene_doc::Object m_source;
	int m_sourceIndex = 0;
	std::vector<scene_doc::Object> m_existing;
	QGroupBox *m_partsBox = nullptr;
	QListWidget *m_parts = nullptr;
	QTabWidget *m_tabs = nullptr;
	QLabel *m_summary = nullptr;
	QPushButton *m_ok = nullptr;
	// grid
	QSpinBox *m_gridCount[3] = {nullptr, nullptr, nullptr};
	QDoubleSpinBox *m_gridSpacing[3] = {nullptr, nullptr, nullptr};
	// ring
	QSpinBox *m_ringCount = nullptr;
	QDoubleSpinBox *m_ringRadius = nullptr, *m_ringCentreX = nullptr, *m_ringCentreZ = nullptr, *m_ringStart = nullptr;
	QCheckBox *m_ringFace = nullptr;
	// scatter
	QSpinBox *m_scatterCount = nullptr, *m_scatterSeed = nullptr;
	QComboBox *m_scatterShape = nullptr;
	QDoubleSpinBox *m_scatterCentreX = nullptr, *m_scatterCentreZ = nullptr, *m_scatterWidth = nullptr, *m_scatterDepth = nullptr, *m_scatterGround = nullptr;
	QDoubleSpinBox *m_scatterScaleMin = nullptr, *m_scatterScaleMax = nullptr, *m_scatterTilt = nullptr, *m_scatterGap = nullptr;
	QCheckBox *m_scatterSpin = nullptr, *m_scatterApart = nullptr;
};
