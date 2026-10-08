#include "flow_layout.h"

#include <QWidget>

#include <algorithm>

FlowLayout::FlowLayout(QWidget *parent, int margin, int horizontalSpacing, int verticalSpacing)
    : QLayout(parent), m_horizontal(horizontalSpacing), m_vertical(verticalSpacing) {
	setContentsMargins(margin, margin, margin, margin);
}

FlowLayout::~FlowLayout() {
	qDeleteAll(m_items);
}

void FlowLayout::addItem(QLayoutItem *item) {
	m_items.append(item);
	m_gaps.append(0);
}

void FlowLayout::addGap(int pixels) {
	if (!m_gaps.isEmpty()) m_gaps.last() = pixels;
}

int FlowLayout::count() const { return m_items.size(); }

QLayoutItem *FlowLayout::itemAt(int index) const { return index >= 0 && index < m_items.size() ? m_items.at(index) : nullptr; }

QLayoutItem *FlowLayout::takeAt(int index) {
	if (index < 0 || index >= m_items.size()) return nullptr;
	m_gaps.removeAt(index);
	return m_items.takeAt(index);
}

int FlowLayout::heightForWidth(int width) const { return doLayout(QRect(0, 0, width, 0), false); }

void FlowLayout::setGeometry(const QRect &rect) {
	QLayout::setGeometry(rect);
	doLayout(rect, true);
}

QSize FlowLayout::sizeHint() const { return minimumSize(); }

// As narrow as the widest single item (everything else wraps), as tall as that makes it.
QSize FlowLayout::minimumSize() const {
	int widest = 0;
	for (const QLayoutItem *item : m_items) widest = std::max(widest, item->minimumSize().width());
	const QMargins m = contentsMargins();
	const int width = widest + m.left() + m.right();
	return QSize(width, heightForWidth(width));
}

// Lays the items out in rows inside `rect`; returns the height used. With `apply` false it only measures.
int FlowLayout::doLayout(const QRect &rect, bool apply) const {
	const QMargins m = contentsMargins();
	const QRect area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom());
	int x = area.x(), y = area.y();
	int rowHeight = 0;
	struct Placed { QLayoutItem *item; int x; QSize size; };
	QList<Placed> row;
	auto flush = [&]() {
		if (apply)
			for (const Placed &p : row) p.item->setGeometry(QRect(QPoint(p.x, y + (rowHeight - p.size.height()) / 2), p.size));   // centred in the row
		row.clear();
	};
	for (int i = 0; i < m_items.size(); ++i) {
		QLayoutItem *item = m_items.at(i);
		if (item->isEmpty()) continue;
		QSize size = item->sizeHint().expandedTo(item->minimumSize());
		if (!row.isEmpty() && x + size.width() > area.right() + 1) {   // does not fit: a new row
			flush();
			y += rowHeight + m_vertical;
			x = area.x();
			rowHeight = 0;
		}
		size.setWidth(std::min(size.width(), std::max(area.width(), 1)));   // never wider than the row
		row.append({item, x, size});
		x += size.width() + m_horizontal + m_gaps.at(i);
		rowHeight = std::max(rowHeight, size.height());
	}
	flush();
	return y + rowHeight - rect.y() + m.bottom();
}
