@echo off
echo Cleaning Visual Studio cache...
echo Please close Visual Studio first!
pause

cd /d "%~dp0"
if exist ".vs" (
	echo Deleting .vs folder...
	rmdir /s /q ".vs"
	echo Done! Now reopen Visual Studio.
) else (
	echo .vs folder not found.
)
pause
