#pragma once

// Shared QSettings location and key names for every persisted UI preference
// (theme, font, language). One definition instead of three independently
// hardcoded copies, so a rename only needs one edit.
//
// Deliberately plain constexpr constants, not a class - theme_switch.cpp's
// loadSavedThemeId()/saveThemeId() are non-static instance methods, while
// language_switch.cpp's and font_switch.cpp's equivalents are static
// (callable from main.cpp before any MainWindow exists), so there's no
// single shared function these could all route through, only the constants.
namespace settings_keys {
constexpr const char *kOrg = "RayTracer";
constexpr const char *kApp = "RayTracerGUI";
constexpr const char *kThemeKey = "ui/theme";
constexpr const char *kFontKey = "ui/font";
constexpr const char *kLanguageKey = "ui/language";
// QSettings array group name (beginWriteArray/beginReadArray) for the
// Recent Renders list - see recent_renders.cpp. The first list-shaped
// value this app persists, hence its own group rather than a scalar key.
constexpr const char *kRecentRendersGroup = "renders/recent";
} // namespace settings_keys
