# Qt GUI Troubleshooting - Output File Issue

## Problem
User clicks "Start Render" but no output file appears.

## What We Know

1. **Manual test of ray_tracer.exe works perfectly:**
   ```
   ray_tracer.exe --gpu 800 100 50
   ```
   Creates: `RayTracer_Package\output\image.png` ✅

2. **Qt GUI command format is now correct:**
   - Fixed from: `--gpu --width 800 --height 800 --samples 100 --depth 50`
   - To: `--gpu 800 100 50` ✅

3. **Output path is hardcoded in ray_tracer.exe:**
   - Always writes to: `{exe_directory}\output\image.png`
   - Cannot be changed via command line
   - Qt GUI's "Output Path" field is currently ignored

## Possible Issues

### Issue #1: Qt GUI isn't actually launching the process
**Test**: Click "Start Render" and check if you see any error message in the messagebox

**If you see "Failed to start renderer: [error]":**
- The process isn't starting
- Possible causes:
  - ray_tracer.exe not found in same directory as RayTracerGUI.exe
  - Permission issues
  - Missing DLL dependencies

**Fix**: Check that both exes are in `RayTracer_Package\`:
```powershell
Get-ChildItem RayTracer_Package -Filter *.exe
```

### Issue #2: Process starts but fails silently
**Test**: After clicking render, check for output:
```powershell
Test-Path "RayTracer_Package\output\image.png"
```

**If file exists:**
- Render is working!
- Just need to tell user where to find it

**If file doesn't exist:**
- Process is crashing or failing
- Check exit code in error message

### Issue #3: You're looking in the wrong location
**Current behavior:**
- Qt GUI "Output Path" field says: `C:\Users\xinpe\Desktop\render_output.png`
- But ray_tracer.exe ignores this and writes to: `RayTracer_Package\output\image.png`

**Fix applied:**
- Updated success message to show: "Check RayTracer_Package/output/image.png"

## Next Steps For User

**Try this:**
1. Click "Start Render" in the Qt GUI
2. Wait for the render to complete
3. Check if you see an error message or success message
4. If success, check: `C:\Users\xinpe\source\repos\ray_tracer\RayTracer_Package\output\image.png`

**Report back:**
- What message do you see after clicking render?
- Does `RayTracer_Package\output\image.png` exist?
- If error, what does the error say?

## Long-term Fix Needed

To make the Qt GUI output path work properly, need to modify `main.cpp` to:
1. Accept an `--output` or `-o` command-line argument
2. Use that path instead of the hardcoded `output\image.ppm`

Example:
```cpp
// Add to argument parsing:
else if (arg == "--output" || arg == "-o") {
	if (i + 1 < argc) {
		out_path = argv[i + 1];
		i++;
	}
}
```

Then Qt GUI can send:
```
ray_tracer.exe --gpu 800 100 50 --output "C:\Users\xinpe\Desktop\render_output.png"
```
