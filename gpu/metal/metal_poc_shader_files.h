// metal_poc_shader_files.h
//
// The single source of truth for how the Metal shader source is split
// across files on disk, and the ORDER they must be concatenated in to
// reproduce the original single-file `metal_poc.metal` exactly (structs/
// functions defined in an earlier file are used by later ones - MSL, like
// C++, needs a prior declaration in the same translation unit).
//
// Originally one 6,495-line file (the shader-side analogue of the
// `metal_poc.mm`/`MetalPocApp` "one giant translation unit" problem
// PR #156 already fixed on the host side - see that PR's own doc section)
// mixing shared types/light functions, sampling/BSDF math utilities, 18
// per-material `shadeXxx()` functions (over half the file on its own),
// the main kernel, and the device-side test kernels. Split the same way:
// by logical section, at exact function boundaries, nothing rewritten.
//
// Unlike the `.mm` split, Metal shader source is compiled at RUNTIME from
// a single string (`newLibraryWithSource:` - see metal_poc.mm's/
// metal_poc_shader_tests.mm's own compile-time-vs-runtime comment), not
// as separate translation units the linker joins later. So this split
// does NOT give each piece its own real compilation unit the way the
// `.mm` split did - every caller must still read all 8 files and
// concatenate them into ONE string, in this exact order, before handing
// it to Metal. This header exists so that "read this list of files, in
// this order" is written down ONCE, not duplicated (and potentially
// drifted) between metal_poc.mm's own loader and
// metal_poc_shader_tests.mm's own separate, simpler one.
//
// Plain C++, no Objective-C/Foundation dependency, so it can be included
// from either .mm file's loader without pulling in more than needed.

inline const char* const* metalShaderFileNames(int* outCount) {
    static const char* const kFiles[] = {
        "metal_poc_types.metal",
        "metal_poc_sampling.metal",
        "metal_poc_materials_specular.metal",
        "metal_poc_materials_diffuse.metal",
        "metal_poc_materials_layered.metal",
        "metal_poc_materials_extra.metal",
        "metal_poc_kernel.metal",
        "metal_poc_test_kernels.metal",
    };
    *outCount = (int)(sizeof(kFiles) / sizeof(kFiles[0]));
    return kFiles;
}
