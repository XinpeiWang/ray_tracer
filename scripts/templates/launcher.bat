@echo off
REM Ray Tracer launcher - starts the GUI if this package has one, otherwise
REM shows the CLI's own help text. Kept as its own file (not generated
REM inline by package.ps1) so its wording can be edited without touching
REM the packaging script.
if exist "%~dp0RayTracerGUI.exe" (
    start "" "%~dp0RayTracerGUI.exe"
) else (
    echo This is the command-line-only (Lite) package - there is no GUI to launch.
    echo.
    "%~dp0RayTracer.exe" --help
    pause
)
