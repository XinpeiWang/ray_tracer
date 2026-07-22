#include "renderer_config.h"

namespace optix_renderer {

// Global configuration instance (default: RECURSIVE with TO_RECURSIVE fallback)
RendererConfig g_rendererConfig = RendererConfig::defaultConfig();

} // namespace optix_renderer
