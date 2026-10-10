#ifndef EXPANDING_TABS_H
#define EXPANDING_TABS_H
// expanding_tabs.h -- the main tab strip's widgets. Split out of mainwindow.h, which includes it.

#include <QTabBar>
#include <QTabWidget>

// ============================================================================
// ExpandingTabBar / ExpandingTabWidget
// ============================================================================
// Makes the main tab strip's 8 tabs fill the window's full width instead of
// sitting left-aligned with blank space to the right of "Diagnostics" once
// the window is wider than the tabs' own natural size.
//
// QTabBar::setExpanding(true) - the documented-sounding way to do this - does
// NOT do this: confirmed empirically (a standalone repro, logging every
// tabSizeHint() call) that it only equalizes tab widths when tabs must
// SHRINK to fit an overflowing bar, never grows them to fill idle space.
// QTabWidget also never stretches its tab bar to the widget's own width in
// the first place - the bar only ever claims its own sizeHint (the sum of
// its tabs' natural widths), which is the real reason the strip left-aligns.
//
// The fix is a tabSizeHint() override that hands out width()/count() per
// tab whenever that's wider than the tab's natural hint - but width() itself
// is circular (the bar's width is DERIVED from summing these same hints), so
// naively reading width() here just converges to some in-between value, not
// the full window width (also confirmed empirically). Reading
// parentWidget()->width() instead - the QTabWidget itself, stable and set
// independently of the tab bar's own size - breaks that circularity.
class ExpandingTabBar : public QTabBar {
public:
	explicit ExpandingTabBar(QWidget *parent = nullptr) : QTabBar(parent) {}

protected:
	// tabSizeHint() below depends on parentWidget()->width() - state entirely
	// external to QTabBar, which it has no way to know changed on its own.
	// Qt only re-invokes tabSizeHint() for a tab when its internal per-tab
	// size cache is marked dirty (tab insert/remove, setShape(), setElideMode(),
	// a style/font change, ...) - a plain resize does NOT dirty that cache by
	// itself, so resizeEvent() alone just re-lays-out the tabs using their
	// STALE cached widths from whenever the cache was last valid (typically
	// once, right after the tabs were first added). That's exactly the
	// "expands right after startup, then stops keeping up with later manual
	// window resizes" symptom this override fixes: re-asserting the tab
	// bar's own (unchanged) shape is a harmless way to force the same
	// internal refresh Qt's own setShape() triggers, so every tab's width
	// gets recomputed from tabSizeHint() on every resize instead of reusing
	// whatever was cached before.
	void resizeEvent(QResizeEvent *event) override {
		setShape(shape());
		QTabBar::resizeEvent(event);
	}

	// Without this the bar lags exactly ONE window resize behind (found by
	// rendering the real app offscreen at 1400px then 2000px wide: only 4 of
	// the 6 tabs showed plus a scroll arrow, the bar ~1365px wide inside a
	// 2000px window - i.e. its width from the PREVIOUS size). QTabWidget's own
	// layout (setUpLayout(), run from ITS resizeEvent) asks this bar for its
	// sizeHint() to decide the bar's width, and QTabBar::sizeHint() is derived
	// from the per-tab size cache - which is only refreshed later, in this
	// bar's own resizeEvent() above. So the layout always reads a hint computed
	// for the old width, hands the bar that stale width, and nothing re-runs
	// the layout once the cache catches up. The tabs then sum to ~the new
	// full width inside a bar still sized for the old one, overflow, and tip
	// into scroll-arrow mode (the "cliff" tabSizeHint() below already warns
	// about). Answering from parentWidget()->width() - the QTabWidget's
	// CURRENT width, already updated by the time its layout runs - skips the
	// stale cache. Same source tabSizeHint() reads, for the same reason.
	QSize sizeHint() const override {
		QSize hint = QTabBar::sizeHint();
		if (parentWidget() && parentWidget()->width() > hint.width())
			hint.setWidth(parentWidget()->width());
		return hint;
	}

	QSize tabSizeHint(int index) const override {
		QSize hint = QTabBar::tabSizeHint(index);
		const int n = count();
		if (n > 0 && parentWidget()) {
			// The "- n * 4" isn't cosmetic: each tab's own QSS margin/border
			// (mainwindow_style.cpp's QTabBar::tab rule) adds a few pixels
			// this per-tab hint doesn't otherwise account for. Without this
			// slack, the summed hints land a handful of pixels OVER the
			// parent's actual width, tipping the whole bar into scroll-arrow
			// mode - which then reserves its own space for the arrows and
			// never revisits this hint, so only the first few (oversized)
			// tabs end up visible at all. A few pixels of unused margin at
			// the right edge is a far smaller cost than that cliff.
			const int evenWidth = (parentWidget()->width() - n * 4) / n;
			if (evenWidth > hint.width()) hint.setWidth(evenWidth);
		}
		return hint;
	}
};

class ExpandingTabWidget : public QTabWidget {
public:
	explicit ExpandingTabWidget(QWidget *parent = nullptr) : QTabWidget(parent) {
		setTabBar(new ExpandingTabBar(this));
	}
};

#endif // EXPANDING_TABS_H
