#!/usr/bin/env bash
# Sets up the optional "Object from a photo" helper for the Scene Builder on macOS (and Linux): the counterpart of setup_photo_to_mesh.ps1.
#
# Makes a private Python environment (PyTorch, the open TripoSR model, a background remover) in the program's per-user data folder, so the Scene
# Builder's Add > "Object from a photo..." can turn one photo into a 3D object on this computer. Nothing is uploaded. About 5 GB of disk and a
# download of about the same, once; the model weights (about 1.7 GB) come from Hugging Face.
#
# Needs Python 3.10, 3.11 or 3.12 and git. On a Mac: `brew install python@3.12`, or the installer from python.org (git comes with the Xcode command
# line tools: `xcode-select --install`). Apple-silicon Macs use the Apple GPU through PyTorch's MPS backend (several minutes per photo become tens of
# seconds), falling back to the processor if MPS cannot run something. Intel Macs get PyTorch 2.2.2, the last version built for them, and run on the processor.
#
#   scripts/setup_photo_to_mesh.sh [FOLDER]
#
# FOLDER defaults to ~/Library/Application Support/RayTracerPhoto (the folder the Scene Builder looks in; change it only together with the
# RAY_TRACER_PHOTO3D_PYTHON environment variable).
set -euo pipefail

# The TripoSR commit this was built and tested against.
TRIPOSR_COMMIT="107cefdc244c39106fa830359024f6a2f1c78871"

if [[ "$(uname -s)" == "Darwin" ]]; then
	DEFAULT_FOLDER="$HOME/Library/Application Support/RayTracerPhoto"
else
	DEFAULT_FOLDER="${XDG_DATA_HOME:-$HOME/.local/share}/RayTracerPhoto"
fi
FOLDER="${1:-$DEFAULT_FOLDER}"

# A program started from Finder has a minimal PATH, without Homebrew or python.org's folders: add the usual places.
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/Library/Frameworks/Python.framework/Versions/3.11/bin:/Library/Frameworks/Python.framework/Versions/3.10/bin:$PATH"

find_python() {
	local candidate version
	for candidate in python3.12 python3.11 python3.10 python3; do
		if command -v "$candidate" >/dev/null 2>&1; then
			version="$("$candidate" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true)"
			case "$version" in
				3.10|3.11|3.12) command -v "$candidate"; return 0 ;;
			esac
		fi
	done
	return 1
}

PY="$(find_python)" || { echo "Python 3.10, 3.11 or 3.12 was not found. Install one (macOS: brew install python@3.12, or the installer from python.org) and run this again." >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git was not found (it fetches the TripoSR code). On a Mac: xcode-select --install" >&2; exit 1; }

echo "Setting up the photo helper in: $FOLDER"
echo "Using Python: $PY ($("$PY" --version 2>&1))"
mkdir -p "$FOLDER"
VENV="$FOLDER/venv"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
	echo "Creating the Python environment..."
	"$PY" -m venv "$VENV" || { echo "Could not create the Python environment." >&2; exit 1; }
fi
"$PYTHON" -m pip install --quiet --upgrade pip

ARCH="$("$PYTHON" -c "import platform; print(platform.machine())")"
EXTRA_PINS=()
if [[ "$(uname -s)" == "Darwin" && "$ARCH" == "x86_64" ]]; then
	# PyTorch stopped publishing Intel-Mac wheels after 2.2.2, which needs NumPy 1.x.
	echo "Installing PyTorch 2.2.2 (the last build for Intel Macs; it runs on the processor)..."
	"$PYTHON" -m pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2" || { echo "Installing PyTorch failed." >&2; exit 1; }
	EXTRA_PINS=("numpy<2")
else
	echo "Installing PyTorch (about 1 GB)..."
	"$PYTHON" -m pip install torch torchvision || { echo "Installing PyTorch failed." >&2; exit 1; }
fi

echo "Installing the other packages..."
"$PYTHON" -m pip install omegaconf einops "transformers>=4.40,<5" trimesh "rembg[cpu]" huggingface-hub xatlas scikit-image scipy pillow numpy onnxruntime "${EXTRA_PINS[@]+"${EXTRA_PINS[@]}"}" \
	|| { echo "Installing the packages failed." >&2; exit 1; }

TRIPOSR="$FOLDER/TripoSR"
if [[ ! -d "$TRIPOSR/tsr" ]]; then
	echo "Fetching the TripoSR code..."
	git clone --quiet https://github.com/VAST-AI-Research/TripoSR "$TRIPOSR" || { echo "Could not fetch the TripoSR code." >&2; exit 1; }
fi
git -C "$TRIPOSR" checkout --quiet "$TRIPOSR_COMMIT" || { echo "Could not check out the tested TripoSR version." >&2; exit 1; }

echo "Downloading the model weights (about 1.7 GB)..."
"$PYTHON" -c "from huggingface_hub import hf_hub_download as d; [d('stabilityai/TripoSR', f) for f in ('config.yaml', 'model.ckpt')]" \
	|| { echo "Could not download the model weights." >&2; exit 1; }

echo "Downloading the background-removal model (U2-Net, about 176 MB)..."
"$PYTHON" -c "import rembg; rembg.new_session('u2net')" || { echo "Could not download the background-removal model." >&2; exit 1; }

echo
echo 'Done. In the Scene Builder choose Add > Object from a photo...'
echo "Diagnostics (the Diagnostics tab) lists what this helper has and what is missing."
if [[ "$(uname -s)" == "Darwin" && "$ARCH" == "x86_64" ]]; then echo "(An Intel Mac has no GPU path here: expect several minutes per photo.)"; fi
