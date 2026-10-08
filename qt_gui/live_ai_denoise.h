#ifndef LIVE_AI_DENOISE_H
#define LIVE_AI_DENOISE_H

// The Mac Live Preview's "AI denoise" switch, remembered between runs (QSettings, settings_keys.h's location). The denoising itself is
// RealtimePreviewWorker::aiDenoiseAccum() (Intel Open Image Denoise, loaded at run time: src/shared/oidn_runtime.h).

#include "settings_keys.h"

#include <QSettings>

namespace live_ai_denoise {

inline bool savedEnabled() {
	return QSettings(settings_keys::kOrg, settings_keys::kApp).value(settings_keys::kLivePreviewAiDenoiseKey, false).toBool();
}
inline void saveEnabled(bool on) {
	QSettings(settings_keys::kOrg, settings_keys::kApp).setValue(settings_keys::kLivePreviewAiDenoiseKey, on);
}

}  // namespace live_ai_denoise

#endif  // LIVE_AI_DENOISE_H
