<#
.SYNOPSIS
    Sets up the optional "Object from a photo" helper for the Scene Builder.

.DESCRIPTION
    Makes a private Python environment (PyTorch, the open TripoSR model, a background remover) in the program's per-user
    data folder, so the Scene Builder's Add > "Object from a photo..." can turn one photo into a 3D object on this
    computer. Nothing is uploaded. It needs about 5 GB of disk and a download of about the same, once; the model
    weights (about 1.7 GB) are fetched from Hugging Face the first time the helper runs.

    Needs Python 3.10, 3.11 or 3.12 and git on PATH. An NVIDIA graphics card makes it take seconds instead of minutes.

.PARAMETER Folder
    Where to put the environment. The default (%LOCALAPPDATA%\RayTracerPhoto) is the folder the Scene Builder looks
    in; change it only together with the RAY_TRACER_PHOTO3D_PYTHON environment variable. Keep it SHORT: PyTorch and
    transformers have deep folders and Windows stops at 260 characters.

.PARAMETER Cpu
    Install the processor-only PyTorch (smaller, but a photo then takes several minutes).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_photo_to_mesh.ps1
#>
param(
	[string]$Folder = (Join-Path $env:LOCALAPPDATA "RayTracerPhoto"),
	[switch]$Cpu
)

$ErrorActionPreference = "Stop"
# The TripoSR commit this was built and tested against.
$triposrCommit = "107cefdc244c39106fa830359024f6a2f1c78871"

function Find-Python {
	foreach ($candidate in @(@("py", "-3.12"), @("py", "-3.11"), @("py", "-3.10"), @("python"))) {
		try {
			$exe = $candidate[0]
			$extra = @($candidate | Select-Object -Skip 1)
			$ver = & $exe @extra -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
			if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.(10|11|12)$') { return ,@($exe) + $extra }
		} catch { }
	}
	return $null
}

$py = Find-Python
if (-not $py) { throw "Python 3.10, 3.11 or 3.12 was not found on PATH. Install one from python.org and run this again." }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git was not found on PATH (it fetches the TripoSR code)." }

Write-Host "Setting up the photo helper in: $Folder"
New-Item -ItemType Directory -Force -Path $Folder | Out-Null
$venv = Join-Path $Folder "venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
	Write-Host "Creating the Python environment..."
	& $py[0] @($py | Select-Object -Skip 1) -m venv $venv
	if ($LASTEXITCODE -ne 0) { throw "Could not create the Python environment." }
}
& $python -m pip install --quiet --upgrade pip

$useGpu = (-not $Cpu) -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($useGpu) {
	Write-Host "Installing PyTorch for NVIDIA graphics cards (about 3 GB)..."
	& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
} else {
	Write-Host "Installing PyTorch for the processor..."
	& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch failed." }

Write-Host "Installing the other packages..."
& $python -m pip install omegaconf einops "transformers>=4.40,<5" trimesh "rembg[cpu]" huggingface-hub xatlas scikit-image scipy pillow numpy onnxruntime
if ($LASTEXITCODE -ne 0) { throw "Installing the packages failed." }

$triposr = Join-Path $Folder "TripoSR"
if (-not (Test-Path (Join-Path $triposr "tsr"))) {
	Write-Host "Fetching the TripoSR code..."
	& git clone --quiet https://github.com/VAST-AI-Research/TripoSR $triposr
	if ($LASTEXITCODE -ne 0) { throw "Could not fetch the TripoSR code." }
}
& git -C $triposr checkout --quiet $triposrCommit
if ($LASTEXITCODE -ne 0) { throw "Could not check out the tested TripoSR version." }

Write-Host "Downloading the model weights (about 1.7 GB)..."
& $python -c "from huggingface_hub import hf_hub_download as d; [d('stabilityai/TripoSR', f) for f in ('config.yaml', 'model.ckpt')]"
if ($LASTEXITCODE -ne 0) { throw "Could not download the model weights." }

Write-Host ""
Write-Host "Done. In the Scene Builder choose Add > Object from a photo..."
if (-not $useGpu) { Write-Host "(No NVIDIA card was used: expect several minutes per photo.)" }
