#include <windows.h>
#include <commctrl.h>
#include <string>
#include <thread>
#include <filesystem>
#include "resource.h"

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

struct RenderSettings {
	bool useGPU;
	int width;
	int height;
	int samples;
	int maxDepth;
};

void UpdateStatusText(const char* text) {
	SetDlgItemTextA(g_hDlg, IDC_STATIC_STATUS, text);
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

	int result = 0;
	if (settings.useGPU) {
		UpdateStatusText("Rendering with GPU (CUDA)...");
		result = gpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
	} else {
		UpdateStatusText("Rendering with CPU (multi-threaded)...");
		result = cpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
	}

	g_rendering = false;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), TRUE);

	if (result == 0) {
		UpdateStatusText("Render complete! Image saved to output/image.ppm");
		MessageBoxA(g_hDlg, 
			"Render completed successfully!\n\nOutput: output\\image.ppm\n\nYou can open this file with image viewers like IrfanView or convert to PNG.",
			"Render Complete", 
			MB_OK | MB_ICONINFORMATION);

		// Optionally open the output folder
		std::string outputDir = outputPath.parent_path().string();
		ShellExecuteA(NULL, "explore", outputDir.c_str(), NULL, NULL, SW_SHOWNORMAL);
	} else {
		UpdateStatusText("Render failed!");
		MessageBoxA(g_hDlg, "Rendering failed. Please check if you have sufficient GPU memory or try CPU mode.", "Error", MB_OK | MB_ICONERROR);
	}
}

void StartRender() {
	if (g_rendering) return;

	RenderSettings settings;

	// Get renderer selection
	settings.useGPU = (IsDlgButtonChecked(g_hDlg, IDC_RADIO_GPU) == BST_CHECKED);

	// Get resolution
	int resIdx = SendDlgItemMessage(g_hDlg, IDC_COMBO_RESOLUTION, CB_GETCURSEL, 0, 0);
	switch (resIdx) {
		case 0: settings.width = 400; settings.height = 400; break;
		case 1: settings.width = 600; settings.height = 600; break;
		case 2: settings.width = 800; settings.height = 800; break;
		case 3: settings.width = 1920; settings.height = 1080; break;
		default: settings.width = 600; settings.height = 600; break;
	}

	// Get samples and depth from sliders
	settings.samples = SendDlgItemMessage(g_hDlg, IDC_SLIDER_SAMPLES, TBM_GETPOS, 0, 0);
	settings.maxDepth = SendDlgItemMessage(g_hDlg, IDC_SLIDER_DEPTH, TBM_GETPOS, 0, 0);

	// Disable render button
	g_rendering = true;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), FALSE);

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

			// Populate resolution combo
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)"400 x 400 (Quick Preview)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)"600 x 600 (Balanced)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)"800 x 800 (High Quality)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)"1920 x 1080 (Full HD)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_SETCURSEL, 1, 0); // Default: 600x600

			// Setup samples slider (10-2000)
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETRANGE, TRUE, MAKELONG(10, 2000));
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETPOS, TRUE, 500);
			SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETTICFREQ, 100, 0);

			// Setup depth slider (5-50)
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETRANGE, TRUE, MAKELONG(5, 50));
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETPOS, TRUE, 20);
			SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETTICFREQ, 5, 0);

			UpdateSliderLabels();

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
