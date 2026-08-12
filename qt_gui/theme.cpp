#include "theme.h"

#include "palette_data.h"

// ============================================================================
// Qt adapter over palette_data
// ============================================================================
// This file used to hold the colour tables themselves - twelve functions, some
// seven hundred lines. They now live in palette_data.cpp, which is Qt-free and
// therefore reachable from the MSVC test binary; see that header for why that
// matters. What is left here is the conversion into QColor and the lookup the
// GUI uses, which is all this layer was ever really for.
// ============================================================================
namespace theme {
namespace {

QColor toQColor(const palette_data::Rgb &c) {
	return QColor(c.r, c.g, c.b);
}

Palette adapt(const palette_data::PaletteData &d) {
	Palette p;
	p.id = QString::fromUtf8(d.id);
	p.name = QString::fromUtf8(d.name);
	p.origin = QString::fromUtf8(d.origin);

	p.surface0 = toQColor(d.surface0);
	p.surface1 = toQColor(d.surface1);
	p.surface2 = toQColor(d.surface2);
	p.surface3 = toQColor(d.surface3);

	p.textBody = toQColor(d.textBody);
	p.textMuted = toQColor(d.textMuted);
	p.textDisabled = toQColor(d.textDisabled);

	p.accentPrimary = toQColor(d.accentPrimary);
	p.accentSecondary = toQColor(d.accentSecondary);
	p.accentDim = toQColor(d.accentDim);

	p.primaryTop = toQColor(d.primaryTop);
	p.primaryBottom = toQColor(d.primaryBottom);
	p.primaryTopHover = toQColor(d.primaryTopHover);
	p.primaryBottomHover = toQColor(d.primaryBottomHover);

	p.border = toQColor(d.border);
	p.borderStrong = toQColor(d.borderStrong);
	p.hoverRow = toQColor(d.hoverRow);

	p.success = toQColor(d.success);
	p.error = toQColor(d.error);

	p.logInfo = toQColor(d.logInfo);
	p.logError = toQColor(d.logError);
	p.logWarning = toQColor(d.logWarning);
	p.logSuccess = toQColor(d.logSuccess);
	p.logGpu = toQColor(d.logGpu);
	p.logCpu = toQColor(d.logCpu);
	p.logPerformance = toQColor(d.logPerformance);
	p.logScene = toQColor(d.logScene);
	p.logInit = toQColor(d.logInit);
	p.logTechnique = toQColor(d.logTechnique);
	p.logCommand = toQColor(d.logCommand);
	p.logDebug = toQColor(d.logDebug);
	p.logSeparator = toQColor(d.logSeparator);

	if (d.backgroundImage) p.backgroundImage = QString::fromUtf8(d.backgroundImage);
	p.backgroundTiled = d.backgroundTiled;
	if (d.backgroundPosition) p.backgroundPosition = QString::fromUtf8(d.backgroundPosition);
	return p;
}

const QVector<Palette> &registry() {
	static const QVector<Palette> themes = []() {
		QVector<Palette> v;
		v.reserve(static_cast<int>(palette_data::builtinCount()));
		for (std::size_t i = 0; i < palette_data::builtinCount(); ++i)
			v.push_back(adapt(palette_data::builtins()[i]));
		return v;
	}();
	return themes;
}

} // namespace

// No default: case, deliberately. Adding a LogSeverity without giving it a
// colour is then a compile error rather than a silent fall-through to logInfo -
// which would surface only as one category quietly rendering in the wrong
// colour, the kind of thing nobody notices for months.
QColor Palette::colourFor(render_output::LogSeverity severity) const {
	using S = render_output::LogSeverity;
	switch (severity) {
	case S::Info:        return logInfo;
	case S::Error:       return logError;
	case S::Warning:     return logWarning;
	case S::Success:     return logSuccess;
	case S::Gpu:         return logGpu;
	case S::Cpu:         return logCpu;
	case S::Performance: return logPerformance;
	case S::Scene:       return logScene;
	case S::Init:        return logInit;
	case S::Technique:   return logTechnique;
	case S::Command:     return logCommand;
	case S::Debug:       return logDebug;
	case S::Separator:   return logSeparator;
	}
	return logInfo;
}

const QVector<Palette> &all() {
	return registry();
}

const Palette &defaultPalette() {
	return registry().first();
}

const Palette &byId(const QString &id) {
	for (const Palette &p : registry()) {
		if (p.id == id) return p;
	}
	// Unknown id: settings written by a newer build, or a theme that has since
	// been removed. Falling back beats refusing to start.
	return defaultPalette();
}

} // namespace theme
