@echo off
REM Run the unified launcher (defaults to GPU)
cd /d "%~dp0"
x64\Release\ray_tracer.exe %*
