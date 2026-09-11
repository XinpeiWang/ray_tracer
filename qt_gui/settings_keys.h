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
// Live Preview's own feel preferences - a separate "livePreview/" prefix
// rather than "ui/", since these are specific to one feature rather than
// whole-app chrome the way theme/font/language are.
constexpr const char *kLivePreviewMouseSensitivityKey = "livePreview/mouseSensitivity";
constexpr const char *kLivePreviewKeyboardSensitivityKey = "livePreview/keyboardSensitivity";
constexpr const char *kLivePreviewDenoiseEnabledKey = "livePreview/denoiseEnabled";
constexpr const char *kLivePreviewDenoiseBlendKey = "livePreview/denoiseBlend";
constexpr const char *kLivePreviewDenoiseShowLatestKey = "livePreview/denoiseShowLatest";
constexpr const char *kLivePreviewSvgfEnabledKey = "livePreview/svgfEnabled";
constexpr const char *kLivePreviewRestirGiEnabledKey = "livePreview/restirGiEnabled";
constexpr const char *kLivePreviewExposureKey = "livePreview/exposure";
constexpr const char *kLivePreviewSamplesKey = "livePreview/samples";
constexpr const char *kLivePreviewMaxDepthKey = "livePreview/maxDepth";
constexpr const char *kLivePreviewFireflyClampKey = "livePreview/fireflyClamp";
// SVGF advanced tuning - see gpu/optix/svgf_tuning_params.h's own comment for
// each field's meaning and literature-default value.
constexpr const char *kLivePreviewSvgfTemporalAlphaKey = "livePreview/svgfTemporalAlpha";
constexpr const char *kLivePreviewSvgfMaxHistoryLengthKey = "livePreview/svgfMaxHistoryLength";
constexpr const char *kLivePreviewSvgfVarianceBootstrapFramesKey = "livePreview/svgfVarianceBootstrapFrames";
constexpr const char *kLivePreviewSvgfVarianceBootstrapRadiusKey = "livePreview/svgfVarianceBootstrapRadius";
constexpr const char *kLivePreviewSvgfSigmaNormalKey = "livePreview/svgfSigmaNormal";
constexpr const char *kLivePreviewSvgfSigmaDepthKey = "livePreview/svgfSigmaDepth";
constexpr const char *kLivePreviewSvgfSigmaLuminanceKey = "livePreview/svgfSigmaLuminance";
constexpr const char *kLivePreviewSvgfAtrousRadiusKey = "livePreview/svgfAtrousRadius";
constexpr const char *kLivePreviewSvgfMinAlbedoKey = "livePreview/svgfMinAlbedo";
constexpr const char *kLivePreviewSvgfAtrousPassesKey = "livePreview/svgfAtrousPasses";
} // namespace settings_keys
