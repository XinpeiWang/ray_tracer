#pragma once
// window_placement.h -- is a saved window position still usable? Qt-free (plain rectangles), so it is unit-tested; qt_gui/window_geometry.cpp feeds it the real
// screens. A window saved on a monitor that has since been unplugged, or on a resolution that has since changed, must not come back where it cannot be reached.

#include <algorithm>
#include <vector>

namespace window_placement {

struct Rect {
	int x = 0, y = 0, w = 0, h = 0;
	int right() const { return x + w; }
	int bottom() const { return y + h; }
};

// The part of `a` that lies inside `b` (zero width and height when they do not touch).
inline Rect intersection(const Rect& a, const Rect& b) {
	const int x0 = std::max(a.x, b.x), y0 = std::max(a.y, b.y);
	const int x1 = std::min(a.right(), b.right()), y1 = std::min(a.bottom(), b.bottom());
	return x1 > x0 && y1 > y0 ? Rect{x0, y0, x1 - x0, y1 - y0} : Rect{x0, y0, 0, 0};
}

// True when enough of the window's title bar is on some screen to grab and drag it back: its top strip (`strip` pixels tall) shows at least `minWidth` pixels
// on one screen. A window that is mostly off-screen but whose title bar is reachable is fine; one entirely below or beside every screen is not.
inline bool reachable(const Rect& frame, const std::vector<Rect>& screens, int minWidth = 120, int strip = 32) {
	if (frame.w <= 0 || frame.h <= 0) return false;
	const Rect titleBar{frame.x, frame.y, frame.w, std::min(strip, frame.h)};
	for (const Rect& s : screens) {
		const Rect visible = intersection(titleBar, s);
		if (visible.w >= std::min(minWidth, frame.w) && visible.h >= std::min(strip, frame.h) / 2) return true;
	}
	return false;
}

// The size to use on `screen`: the saved one, but never larger than the screen.
inline Rect clampSize(Rect size, const Rect& screen) {
	size.w = std::min(size.w, screen.w);
	size.h = std::min(size.h, screen.h);
	return size;
}

}  // namespace window_placement
