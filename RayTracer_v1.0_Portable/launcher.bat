@echo off
REM Ray Tracer Launcher
REM Double-click this file to run the ray tracer with default settings

title Ray Tracer - Cornell Box Renderer

echo.
echo ========================================
echo     RAY TRACER - Cornell Box
echo ========================================
echo.
echo Starting interactive mode...
echo Output will be saved to: output\image.ppm
echo.

REM Launch the ray tracer in interactive mode
RayTracer.exe

REM Check if render was successful
if %ERRORLEVEL% EQU 0 (
	echo.
	echo ========================================
	echo Render completed successfully!
	echo ========================================
	echo.
	echo Output saved to: output\image.ppm
	echo.

	REM Ask if user wants to open the output folder
	choice /C YN /M "Open output folder now"
	if %ERRORLEVEL% EQU 1 (
		explorer.exe output
	)
) else (
	echo.
	echo ========================================
	echo Render failed with error code: %ERRORLEVEL%
	echo ========================================
	echo.
	echo Please check:
	echo   - GPU drivers are up to date
	echo   - Sufficient disk space
	echo   - No antivirus interference
	echo.
)

pause
