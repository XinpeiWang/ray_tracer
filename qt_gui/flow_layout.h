#ifndef FLOW_LAYOUT_H
#define FLOW_LAYOUT_H

// A layout that puts its widgets in a row and starts a new row when the next one would not fit, like text wrapping. A toolbar built with it is one line
// in a wide window and two or three in a narrow one, instead of clipping buttons or forcing the window wide. Its height depends on its width
// (heightForWidth), which the box layouts around it understand.

#include <QLayout>
#include <QList>
#include <QRect>
#include <QSize>

class FlowLayout : public QLayout {
public:
	explicit FlowLayout(QWidget *parent = nullptr, int margin = 0, int horizontalSpacing = 6, int verticalSpacing = 6);
	~FlowLayout() override;

	void addItem(QLayoutItem *item) override;
	int count() const override;
	QLayoutItem *itemAt(int index) const override;
	QLayoutItem *takeAt(int index) override;
	Qt::Orientations expandingDirections() const override { return {}; }
	bool hasHeightForWidth() const override { return true; }
	int heightForWidth(int width) const override;
	void setGeometry(const QRect &rect) override;
	QSize sizeHint() const override;
	QSize minimumSize() const override;

	// Extra room before the next widget (a gap between groups of buttons): added after the widget that was added last.
	void addGap(int pixels);

private:
	int doLayout(const QRect &rect, bool apply) const;

	QList<QLayoutItem *> m_items;
	QList<int> m_gaps;   // the extra gap after each item
	int m_horizontal, m_vertical;
};

#endif  // FLOW_LAYOUT_H
