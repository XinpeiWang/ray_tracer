# Qt6 Installation Verification Script

Write-Host "`n🔍 Checking for Qt6 Installation...`n" -ForegroundColor Cyan

# Common installation paths
$possiblePaths = @(
	"C:\Qt",
	"C:\Qt6",
	"$env:USERPROFILE\Qt",
	"C:\Program Files\Qt",
	"C:\Program Files (x86)\Qt"
)

$found = $false
$qtPath = $null

foreach ($basePath in $possiblePaths) {
	if (Test-Path $basePath) {
		Write-Host "✓ Found Qt directory: $basePath" -ForegroundColor Green

		# Look for Qt 6.x versions
		$versions = Get-ChildItem -Path $basePath -Directory -ErrorAction SilentlyContinue | 
				   Where-Object { $_.Name -match '^6\.\d+\.\d+$' } |
				   Sort-Object Name -Descending

		if ($versions) {
			foreach ($version in $versions) {
				$msvcPath = Join-Path $version.FullName "msvc2022_64\bin\qmake.exe"
				if (Test-Path $msvcPath) {
					Write-Host "  ✓ Qt $($version.Name) with MSVC 2022 64-bit" -ForegroundColor Green
					$qtPath = Split-Path (Split-Path $msvcPath -Parent) -Parent
					$found = $true
					break
				}

				# Check for MSVC 2019 as fallback
				$msvc2019Path = Join-Path $version.FullName "msvc2019_64\bin\qmake.exe"
				if (Test-Path $msvc2019Path) {
					Write-Host "  ⚠ Qt $($version.Name) with MSVC 2019 64-bit (will work)" -ForegroundColor Yellow
					$qtPath = Split-Path (Split-Path $msvc2019Path -Parent) -Parent
					$found = $true
					break
				}
			}
			if ($found) { break }
		}
	}
}

if ($found) {
	Write-Host "`n✅ Qt6 IS INSTALLED!" -ForegroundColor Green
	Write-Host "   Path: $qtPath`n" -ForegroundColor Green

	# Check qmake version
	$qmakePath = Join-Path $qtPath "bin\qmake.exe"
	if (Test-Path $qmakePath) {
		Write-Host "📦 Qt Version Information:" -ForegroundColor Cyan
		& $qmakePath -version
	}

	# Check important tools
	Write-Host "`n🔧 Checking Qt Tools:" -ForegroundColor Cyan
	$tools = @("qmake.exe", "moc.exe", "uic.exe", "rcc.exe", "windeployqt.exe")
	foreach ($tool in $tools) {
		$toolPath = Join-Path $qtPath "bin\$tool"
		if (Test-Path $toolPath) {
			Write-Host "  ✓ $tool" -ForegroundColor Green
		} else {
			Write-Host "  ✗ $tool (missing)" -ForegroundColor Red
		}
	}

	# Check environment variable
	Write-Host "`n🌍 Environment Variables:" -ForegroundColor Cyan
	if ($env:Qt6_DIR) {
		Write-Host "  Qt6_DIR = $env:Qt6_DIR" -ForegroundColor Green
	} else {
		Write-Host "  ⚠ Qt6_DIR not set (optional but recommended)" -ForegroundColor Yellow
	}

	# Check if Qt bin is in PATH
	$qtBinPath = Join-Path $qtPath "bin"
	if ($env:Path -like "*$qtBinPath*") {
		Write-Host "  ✓ Qt bin directory is in PATH" -ForegroundColor Green
	} else {
		Write-Host "  ⚠ Qt bin not in PATH (optional but recommended)" -ForegroundColor Yellow
		Write-Host "`n  To add Qt to PATH, run:" -ForegroundColor Cyan
		Write-Host "  [Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$qtBinPath', 'User')" -ForegroundColor Gray
	}

	Write-Host "`n✅ Ready to proceed with Qt GUI development!" -ForegroundColor Green
	Write-Host "   Let GitHub Copilot know Qt is installed to continue.`n" -ForegroundColor Cyan

} else {
	Write-Host "`n❌ Qt6 NOT FOUND" -ForegroundColor Red
	Write-Host "`n📥 Please install Qt6 following these steps:`n" -ForegroundColor Yellow
	Write-Host "1. Download: https://www.qt.io/download-qt-installer-oss" -ForegroundColor White
	Write-Host "2. Run the installer" -ForegroundColor White
	Write-Host "3. Select: Qt 6.x → MSVC 2022 64-bit" -ForegroundColor White
	Write-Host "4. Complete installation (~2-3GB download)" -ForegroundColor White
	Write-Host "`nSee QT6_INSTALLATION_GUIDE.md for detailed instructions.`n" -ForegroundColor Cyan
}
