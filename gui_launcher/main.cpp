#include <windows.h>
#include <commctrl.h>
#include <string>
#include <thread>
#include <filesystem>
#include "resource.h"

// Image writer utilities
#include "../src/external/image_writer.h"

// Forward declarations of the render APIs
extern "C" {
	int gpu_is_available();
	int gpu_render_main(int width, int height, int samples_per_pixel, int max_depth, const char* output_path);
	int cpu_render_main(int width, int height, int samples_per_pixel, int max_depth, const char* output_path);
}

#pragma comment(lib, "comctl32.lib")
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

// Global variables
HWND g_hDlg = NULL;
bool g_rendering = false;
int g_currentTab = 0; // 0 = Basic, 1 = Advanced

struct RenderSettings {
	bool useGPU;
	int width;
	int height;
	int samples;
	int maxDepth;
};

struct QualityPreset {
	const wchar_t* name;
	const wchar_t* description;
	int width;
	int samples;
	int depth;
};

// Quality presets for basic mode
QualityPreset g_presets[] = {
	{ L"Low (Fast Preview)", L"Quick renders for testing (400x400, 50 samples)", 400, 50, 10 },
	{ L"Medium (Balanced)", L"Good quality with reasonable time (600x600, 200 samples)", 600, 200, 20 },
	{ L"High (Recommended)", L"High quality for most uses (800x800, 500 samples)", 800, 500, 30 },
	{ L"Very High (Slow)", L"Excellent quality, takes time (1080x1080, 1000 samples)", 1080, 1000, 40 },
	{ L"Extreme (Production)", L"Best quality, very slow (2048x2048, 2000 samples)", 2048, 2000, 50 }
};

void UpdateStatusText(const char* text) {
	SetDlgItemTextA(g_hDlg, IDC_STATIC_STATUS, text);
}

void SetProgressBar(int percent) {
	SendDlgItemMessage(g_hDlg, IDC_PROGRESS, PBM_SETPOS, percent, 0);
}

void ShowBasicControls(bool show) {
	ShowWindow(GetDlgItem(g_hDlg, IDC_COMBO_QUALITY), show ? SW_SHOW : SW_HIDE);
	ShowWindow(GetDlgItem(g_hDlg, IDC_STATIC_QUALITY_DESC), show ? SW_SHOW : SW_HIDE);
}

void ShowAdvancedControls(bool show) {
	ShowWindow(GetDlgItem(g_hDlg, IDC_COMBO_RESOLUTION), show ? SW_SHOW : SW_HIDE);
	ShowWindow(GetDlgItem(g_hDlg, IDC_SLIDER_SAMPLES), show ? SW_SHOW : SW_HIDE);
	ShowWindow(GetDlgItem(g_hDlg, IDC_SLIDER_DEPTH), show ? SW_SHOW : SW_HIDE);
	ShowWindow(GetDlgItem(g_hDlg, IDC_LABEL_SAMPLES), show ? SW_SHOW : SW_HIDE);
	ShowWindow(GetDlgItem(g_hDlg, IDC_LABEL_DEPTH), show ? SW_SHOW : SW_HIDE);
}

void SwitchTab(int tabIndex) {
	g_currentTab = tabIndex;
	if (tabIndex == 0) {
		// Basic mode
		ShowBasicControls(true);
		ShowAdvancedControls(false);
	} else {
		// Advanced mode
		ShowBasicControls(false);
		ShowAdvancedControls(true);
	}
}

void RenderThread(RenderSettings settings) {
	// Get executable directory for output path
	char exePath[MAX_PATH];
	GetModuleFileNameA(NULL, exePath, MAX_PATH);
	std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
	std::filesystem::path outputPath = exeDir / "output" / "image.ppm";

	// Create output directory
	std::filesystem::create_directories(outputPath.parent_path());

	std::string outputStr = outputPath.string();

	// Start progress animation
	SetProgressBar(10);

	int result = 0;
	if (settings.useGPU) {
		UpdateStatusText("Rendering with GPU (CUDA)...");
		SetProgressBar(30);
		result = gpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
		SetProgressBar(80);
	} else {
		UpdateStatusText("Rendering with CPU (multi-threaded)...");
		SetProgressBar(30);
		result = cpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
		SetProgressBar(80);
	}

	g_rendering = false;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), TRUE);

	if (result == 0) {
		// Convert PPM to PNG
		SetProgressBar(90);
		UpdateStatusText("Converting to PNG...");
		std::filesystem::path pngPath = outputPath.parent_path() / "image.png";

		bool pngOk = convert_ppm_to_png(outputStr.c_str(), pngPath.string().c_str());

		SetProgressBar(100);

		std::string statusMsg = "Render complete! Saved:\n";
		if (pngOk) statusMsg += "✓ image.png\n";
		statusMsg += "✓ image.ppm";

		UpdateStatusText(statusMsg.c_str());

		std::string msgText = "Render completed successfully!\n\nSaved formats:\n";
		if (pngOk) msgText += "• PNG (lossless)\n";
		msgText += "• PPM (raw)\n\nOpening output folder...";

		MessageBoxA(g_hDlg, 
			msgText.c_str(),
			"Render Complete", 
			MB_OK | MB_ICONINFORMATION);

		// Optionally open the output folder
		std::string outputDir = outputPath.parent_path().string();
		ShellExecuteA(NULL, "explore", outputDir.c_str(), NULL, NULL, SW_SHOWNORMAL);

		// Reset progress bar
		SetProgressBar(0);
	} else {
		SetProgressBar(0);
		UpdateStatusText("Render failed!");
		MessageBoxA(g_hDlg, "Rendering failed. Please check if you have sufficient GPU memory or try CPU mode.", "Error", MB_OK | MB_ICONERROR);
	}
}

void StartRender() {
	if (g_rendering) return;

	RenderSettings settings;

	// Get renderer selection
	settings.useGPU = (IsDlgButtonChecked(g_hDlg, IDC_RADIO_GPU) == BST_CHECKED);

	if (g_currentTab == 0) {
		// Basic mode: use quality preset
		int qualityIdx = SendDlgItemMessage(g_hDlg, IDC_COMBO_QUALITY, CB_GETCURSEL, 0, 0);
		if (qualityIdx >= 0 && qualityIdx < 5) {
			settings.width = g_presets[qualityIdx].width;
			settings.height = g_presets[qualityIdx].width;
			settings.samples = g_presets[qualityIdx].samples;
			settings.maxDepth = g_presets[qualityIdx].depth;
		} else {
			// Default to Medium
			settings.width = 600;
			settings.height = 600;
			settings.samples = 200;
			settings.maxDepth = 20;
		}
	} else {
		// Advanced mode: use manual settings
		int resIdx = SendDlgItemMessage(g_hDlg, IDC_COMBO_RESOLUTION, CB_GETCURSEL, 0, 0);
		switch (resIdx) {
			case 0: settings.width = 400; settings.height = 400; break;
			case 1: settings.width = 600; settings.height = 600; break;
			case 2: settings.width = 800; settings.height = 800; break;
			case 3: settings.width = 1080; settings.height = 1080; break;
			case 4: settings.width = 2048; settings.height = 2048; break;
			default: settings.width = 600; settings.height = 600; break;
		}

		// Get samples and depth from sliders
		settings.samples = SendDlgItemMessage(g_hDlg, IDC_SLIDER_SAMPLES, TBM_GETPOS, 0, 0);
		settings.maxDepth = SendDlgItemMessage(g_hDlg, IDC_SLIDER_DEPTH, TBM_GETPOS, 0, 0);
	}

	// Disable render button
	g_rendering = true;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), FALSE);

	// Reset progress bar
	SetProgressBar(0);

	// Start rendering thread
	std::thread renderThread(RenderThread, settings);
	renderThread.detach();
}

void UpdateSliderLabels() {
	int samples = SendDlgItemMessage(g_hDlg, IDC_SLIDER_SAMPLES, TBM_GETPOS, 0, 0);
	int depth = SendDlgItemMessage(g_hDlg, IDC_SLIDER_DEPTH, TBM_GETPOS, 0, 0);

	char text[256];
	sprintf_s(text, "Samples per pixel: %d", samples);
	SetDlgItemTextA(g_hDlg, IDC_LABEL_SAMPLES, text);

	sprintf_s(text, "Max ray depth: %d", depth);
	SetDlgItemTextA(g_hDlg, IDC_LABEL_DEPTH, text);
}

INT_PTR CALLBACK DialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
	switch (message) {
		case WM_INITDIALOG: {
			g_hDlg = hDlg;

			// Set icon
			HICON hIcon = LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_APP_ICON));
			SendMessage(hDlg, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
			SendMessage(hDlg, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);

			// Check GPU availability and set default
			if (gpu_is_available()) {
				CheckDlgButton(hDlg, IDC_RADIO_GPU, BST_CHECKED);
				SetDlgItemTextA(hDlg, IDC_STATIC_STATUS, "GPU detected! Ready to render.");
			} else {
				CheckDlgButton(hDlg, IDC_RADIO_CPU, BST_CHECKED);
				EnableWindow(GetDlgItem(hDlg, IDC_RADIO_GPU), FALSE);
				SetDlgItemTextA(hDlg, IDC_STATIC_STATUS, "No GPU detected. CPU mode selected.");
			}

			// Setup tab control
			HWND hTab = GetDlgItem(hDlg, IDC_TAB_CONTROL);
			TCITEM tie;
			tie.mask = TCIF_TEXT;
			tie.pszText = (LPWSTR)L"Basic";
			TabCtrl_InsertItem(hTab, 0, &tie);
			tie.pszText = (LPWSTR)L"Advanced";
			TabCtrl_InsertItem(hTab, 1, &tie);

			// Populate quality presets (Basic tab)
			for (int i = 0; i < 5; i++) {
				SendDlgItemMessage(hDlg, IDC_COMBO_QUALITY, CB_ADDSTRING, 0, (LPARAM)g_presets[i].name);
			}
			SendDlgItemMessage(hDlg, IDC_COMBO_QUALITY, CB_SETCURSEL, 2, 0); // Default: High

			// Populate resolution combo (Advanced tab)
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"400 x 400 (Quick Preview)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"600 x 600 (Balanced)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"800 x 800 (High Quality)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"1080 x 1080 (Very High)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"2048 x 2048 (2K Ultra)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_SETCURSEL, 1, 0); // Default: 600x600

			// Setup samples slider (10-2000)
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETRANGE, TRUE, MAKELONG(10, 2000));
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETPOS, TRUE, 500);
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETTICFREQ, 100, 0);

			// Setup depth slider (5-50)
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETRANGE, TRUE, MAKELONG(5, 50));
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETPOS, TRUE, 20);
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETTICFREQ, 5, 0);

			// Initialize progress bar
			SendDlgItemMessage(hDlg, IDC_PROGRESS, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
			SendDlgItemMessage(hDlg, IDC_PROGRESS, PBM_SETPOS, 0, 0);

			UpdateSliderLabels();

			// Show Basic tab by default
			SwitchTab(0);

			return TRUE;
		}

		case WM_NOTIFY: {
			LPNMHDR pnmhdr = (LPNMHDR)lParam;
			if (pnmhdr->idFrom == IDC_TAB_CONTROL && pnmhdr->code == TCN_SELCHANGE) {
				int tabIndex = TabCtrl_GetCurSel(GetDlgItem(hDlg, IDC_TAB_CONTROL));
				SwitchTab(tabIndex);
			}
			return TRUE;
		}

		case WM_HSCROLL: {
			// Slider moved
			UpdateSliderLabels();
			return TRUE;
		}

		case WM_COMMAND: {
			if (LOWORD(wParam) == IDC_BUTTON_RENDER) {
				StartRender();
				return TRUE;
			}
			if (LOWORD(wParam) == IDCANCEL) {
				if (g_rendering) {
					MessageBoxA(hDlg, "Please wait for rendering to complete.", "Busy", MB_OK | MB_ICONWARNING);
					return TRUE;
				}
				EndDialog(hDlg, 0);
				return TRUE;
			}
			break;
		}

		case WM_CLOSE: {
			if (g_rendering) {
				MessageBoxA(hDlg, "Please wait for rendering to complete.", "Busy", MB_OK | MB_ICONWARNING);
				return TRUE;
			}
			EndDialog(hDlg, 0);
			return TRUE;
		}
	}
	return FALSE;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	InitCommonControls();

	DialogBox(hInstance, MAKEINTRESOURCE(IDD_MAIN_DIALOG), NULL, DialogProc);

	return 0;
}
