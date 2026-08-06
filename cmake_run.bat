if exist "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test\CMakeCache.txt" del /f "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test\CMakeCache.txt"
if exist "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test\CMakeFiles" rmdir /s /q "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test\CMakeFiles"
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
cmake -G "Visual Studio 18 2026" "C:\Users\xinpe\source\repos\ray_tracer\tests" -B "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test" > "C:\Users\xinpe\source\repos\ray_tracer\cmake_out.txt" 2>&1
cmake --build "C:\Users\xinpe\source\repos\ray_tracer\tests\build_test" --config Debug >> "C:\Users\xinpe\source\repos\ray_tracer\cmake_out.txt" 2>&1
