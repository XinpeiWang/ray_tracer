@echo off
echo.
echo ==================================================
echo Testing which executable is the startup project
echo ==================================================
echo.
echo Test 1: Running launcher directly (should show GPU by default)
echo.
x64\Release\ray_tracer.exe --help
echo.
echo ==================================================
echo Test 2: Running CPU renderer directly
echo.
x64\Release\raytracing_book.exe --help
echo.
echo ==================================================
echo.
echo If Visual Studio shows "RAY TRACER LAUNCHER" banner,
echo it's using the correct startup project.
echo.
echo If it shows "Attempting to write image to..." immediately,
echo it's still using raytracing_book as startup project.
echo.
pause
