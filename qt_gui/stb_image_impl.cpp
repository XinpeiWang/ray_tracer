// stb_image_impl.cpp - the one place the GUI compiles stb_image's implementation. The shared mesh reader (src/shared/ply_mesh.h, used for the 3D view's
// mesh preview) decompresses .gz meshes through stb_image's zlib routines and only declares them; the renderers have their own copy of the implementation.
#define STB_IMAGE_IMPLEMENTATION
#include "../src/external/stb_image.h"
