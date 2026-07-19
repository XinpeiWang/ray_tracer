# Build System Notes

## CUDA Build Improvements

The build system has been improved to handle common issues with CUDA compilation in parallel builds.

### Changes Made:

1. **Semaphore-Based Locking**: Added `.build_lock` file to serialize CUDA compilation across projects
   - Prevents both `ray_tracer` and `raytracing_book` projects from compiling CUDA sources simultaneously
   - Uses PowerShell-based wait loop with 30-second timeout

2. **File System Synchronization**: Added small delays after compilation to ensure files are written before verification

3. **Better Error Messages**: Each project now logs which project is actively building CUDA sources

4. **Absolute Paths**: Use fully-qualified paths for nvcc.exe and cl.exe to avoid working directory issues

5. **Console Output Capture**: `ConsoleToMSBuild="true"` ensures compile errors are visible in Visual Studio

### Build Options:

**Recommended: Single-Threaded Build** (most reliable)
```powershell
msbuild ray_tracer.sln /m:1 /p:Configuration=Release /p:Platform=x64
```

**Alternative: Parallel Build** (faster but may occasionally fail)
```powershell
msbuild ray_tracer.sln /p:Configuration=Release /p:Platform=x64
```

If parallel builds fail with "Permission denied" or "file not found" errors:
1. Clean the solution: `msbuild ray_tracer.sln /t:Clean`
2. Delete lock file: `Remove-Item gpu\cuda\.build_lock`
3. Rebuild with `/m:1` flag

### Visual Studio Settings:

To avoid parallel-build issues in Visual Studio IDE:
1. Go to **Tools → Options**
2. Navigate to **Projects and Solutions → Build and Run**
3. Set "maximum number of parallel project builds" to **1**

### Known Issues:

- CUDA object files are shared between projects, which can cause race conditions in parallel builds
- The lock mechanism may not always work if MSBuild spawns processes too quickly
- Windows antivirus or file indexing can occasionally lock `.obj` files

### Files Added to .gitignore:

```
# CUDA object files
gpu/cuda/obj/
*.obj
```
