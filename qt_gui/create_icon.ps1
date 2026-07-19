# PowerShell script to create a simple game-like icon for the Ray Tracer
# This creates a colorful icon with a ray/beam theme

Add-Type -AssemblyName System.Drawing

# Create bitmap (256x256 for high quality)
$size = 256
$bitmap = New-Object System.Drawing.Bitmap($size, $size)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

# Background - Dark gradient
$rect = New-Object System.Drawing.Rectangle(0, 0, $size, $size)
$brush1 = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 20, 20, 40))
$graphics.FillRectangle($brush1, $rect)

# Draw multiple colored rays emanating from center
$centerX = $size / 2
$centerY = $size / 2

# Ray colors (gaming theme: cyan, magenta, yellow)
$colors = @(
	[System.Drawing.Color]::FromArgb(200, 0, 255, 255),    # Cyan
	[System.Drawing.Color]::FromArgb(200, 255, 0, 255),    # Magenta
	[System.Drawing.Color]::FromArgb(200, 255, 255, 0),    # Yellow
	[System.Drawing.Color]::FromArgb(200, 0, 255, 128),    # Green-Cyan
	[System.Drawing.Color]::FromArgb(200, 255, 128, 0)     # Orange
)

# Draw rays
for ($i = 0; $i -lt 8; $i++) {
	$angle = ($i * 45) * [Math]::PI / 180
	$endX = $centerX + [Math]::Cos($angle) * ($size * 0.4)
	$endY = $centerY + [Math]::Sin($angle) * ($size * 0.4)

	$color = $colors[$i % $colors.Length]
	$pen = New-Object System.Drawing.Pen($color, 12)
	$pen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
	$pen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round

	$graphics.DrawLine($pen, $centerX, $centerY, $endX, $endY)
	$pen.Dispose()
}

# Draw central sphere/orb (glossy look)
$orbSize = $size * 0.25
$orbX = $centerX - $orbSize / 2
$orbY = $centerY - $orbSize / 2
$orbRect = New-Object System.Drawing.RectangleF($orbX, $orbY, $orbSize, $orbSize)

# Gradient for 3D effect
$gradientBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
	$orbRect,
	[System.Drawing.Color]::FromArgb(255, 100, 150, 255),
	[System.Drawing.Color]::FromArgb(255, 30, 60, 150),
	45
)
$graphics.FillEllipse($gradientBrush, $orbRect)

# Highlight for glossy effect
$highlightSize = $orbSize * 0.3
$highlightX = $centerX - $orbSize * 0.15
$highlightY = $centerY - $orbSize * 0.15
$highlightRect = New-Object System.Drawing.RectangleF($highlightX, $highlightY, $highlightSize, $highlightSize)
$highlightBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(150, 255, 255, 255))
$graphics.FillEllipse($highlightBrush, $highlightRect)

# Clean up
$graphics.Dispose()

# Save as PNG first
$pngPath = Join-Path $PSScriptRoot "app_icon.png"
$bitmap.Save($pngPath, [System.Drawing.Imaging.ImageFormat]::Png)
Write-Host "Created PNG icon at: $pngPath"

# Convert PNG to ICO using .NET
# Create multi-size ICO (256, 128, 64, 48, 32, 16)
$iconSizes = @(256, 128, 64, 48, 32, 16)
$icoPath = Join-Path $PSScriptRoot "app_icon.ico"

# For a proper ICO file, we'll use a simpler approach - save the 256x256 and let Windows handle it
# Note: For production, you'd want to use a proper ICO creation tool
$icon = [System.Drawing.Icon]::FromHandle($bitmap.GetHicon())
$fileStream = [System.IO.File]::Create($icoPath)
$icon.Save($fileStream)
$fileStream.Close()

Write-Host "Created ICO icon at: $icoPath"

$bitmap.Dispose()
$icon.Dispose()

Write-Host "Icon creation complete! The icon features a glowing orb with colorful rays."
