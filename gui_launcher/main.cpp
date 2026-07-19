#include <windows.h>
#include <commctrl.h>
#include <richedit.h>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>
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

// Modern UI colors (inspired by VS Code / Discord)
HBRUSH g_hBackgroundBrush = NULL;
HBRUSH g_hLightBackgroundBrush = NULL;
HBRUSH g_hAccentBrush = NULL;
COLORREF g_colorBackground = RGB(32, 32, 36);          // Deep dark grey
COLORREF g_colorLightBackground = RGB(47, 49, 54);    // Lighter surface
COLORREF g_colorText = RGB(236, 236, 236);            // Almost white
COLORREF g_colorAccent = RGB(88, 101, 242);           // Discord blurple
COLORREF g_colorAccentHover = RGB(109, 120, 255);    // Lighter blurple
COLORREF g_colorSuccess = RGB(87, 242, 135);          // Green accent

HFONT g_hTitleFont = NULL;
HFONT g_hNormalFont = NULL;

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
	{ L"⚡ Low (Fast Preview)", L"Quick renders for testing (400x400, 50 samples)", 400, 50, 10 },
	{ L"💎 Medium (Balanced)", L"Good quality with reasonable time (600x600, 200 samples)", 600, 200, 20 },
	{ L"🌟 High (Recommended)", L"High quality for most uses (800x800, 500 samples)", 800, 500, 30 },
	{ L"🔥 Very High (Impressive)", L"Excellent quality, takes time (1080x1080, 1000 samples)", 1080, 1000, 40 },
	{ L"💫 Extreme (Ultra Quality)", L"Best quality, very slow (2048x2048, 2000 samples)", 2048, 2000, 50 }
};

void UpdateStatusText(const wchar_t* text) {
	SetDlgItemTextW(g_hDlg, IDC_STATIC_STATUS, text);
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

	// Start timing
	auto start_time = std::chrono::high_resolution_clock::now();

	// Start progress animation
	SetProgressBar(10);

	int result = 0;
	if (settings.useGPU) {
		UpdateStatusText(L"⚡ Rendering with GPU (CUDA)...");
		SetProgressBar(30);
		result = gpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
		SetProgressBar(80);
	} else {
		UpdateStatusText(L"🔧 Rendering with CPU (multi-threaded)...");
		SetProgressBar(30);
		result = cpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputStr.c_str());
		SetProgressBar(80);
	}

	// Calculate elapsed time
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	double seconds = duration.count() / 1000.0;

	g_rendering = false;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), TRUE);

	if (result == 0) {
		// Convert PPM to PNG
		SetProgressBar(90);
		UpdateStatusText(L"🎨 Converting to PNG...");
		std::filesystem::path pngPath = outputPath.parent_path() / "image.png";

		bool pngOk = convert_ppm_to_png(outputStr.c_str(), pngPath.string().c_str());

		SetProgressBar(100);

		// Format time string
		char timeStr[128];
		wchar_t wTimeStr[128];
		if (seconds < 1.0) {
			sprintf_s(timeStr, "%.0f ms", duration.count());
			swprintf_s(wTimeStr, L"%.0f ms", duration.count());
		} else if (seconds < 60.0) {
			sprintf_s(timeStr, "%.2f seconds", seconds);
			swprintf_s(wTimeStr, L"%.2f seconds", seconds);
		} else {
			int minutes = (int)(seconds / 60);
			double remainingSeconds = seconds - (minutes * 60);
			sprintf_s(timeStr, "%d min %.1f sec", minutes, remainingSeconds);
			swprintf_s(wTimeStr, L"%d min %.1f sec", minutes, remainingSeconds);
		}

		std::wstring statusMsg = L"✅ Render complete! Time: ";
		statusMsg += wTimeStr;
		statusMsg += L"\nSaved:\n";
		if (pngOk) statusMsg += L"✓ image.png\n";
		statusMsg += L"✓ image.ppm";

		UpdateStatusText(statusMsg.c_str());

		std::wstring msgText = L"🎉 Render completed successfully!\n\n";
		msgText += L"⏱️ Render time: ";
		msgText += wTimeStr;
		msgText += L"\n\n📦 Saved formats:\n";
		if (pngOk) msgText += L"✓ PNG (lossless)\n";
		msgText += L"✓ PPM (raw)\n\n📂 Opening output folder...";

		MessageBoxW(g_hDlg, 
			msgText.c_str(),
			L"🎨 Render Complete", 
			MB_OK | MB_ICONINFORMATION);

		// Optionally open the output folder
		std::string outputDir = outputPath.parent_path().string();
		ShellExecuteA(NULL, "explore", outputDir.c_str(), NULL, NULL, SW_SHOWNORMAL);

		// Reset progress bar
		SetProgressBar(0);
	} else {
		SetProgressBar(0);
		UpdateStatusText(L"❌ Render failed!");
		MessageBoxW(g_hDlg, L"❌ Rendering failed. Please check if you have sufficient GPU memory or try CPU mode.", L"Error", MB_OK | MB_ICONERROR);
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

			// Create dark theme brushes
			g_hBackgroundBrush = CreateSolidBrush(g_colorBackground);
			g_hLightBackgroundBrush = CreateSolidBrush(g_colorLightBackground);
			g_hAccentBrush = CreateSolidBrush(g_colorAccent);

			// Create modern fonts
			g_hTitleFont = CreateFont(22, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
				DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
				CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
			g_hNormalFont = CreateFont(10, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
				DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
				CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

			// Set emoji font to all controls FIRST (before setting text)
			SendDlgItemMessage(hDlg, IDC_STATIC_TITLE, WM_SETFONT, (WPARAM)g_hTitleFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_GROUPBOX_RENDERER, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_RADIO_GPU, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_RADIO_CPU, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_BUTTON_RENDER, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_STATIC_QUALITY_DESC, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_STATIC_ADVANCED_DESC, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			SendDlgItemMessage(hDlg, IDC_STATIC_STATUS, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);

			// NOW set emoji text using escaped Unicode sequences
			SetDlgItemTextW(hDlg, IDC_STATIC_TITLE, L"\U0001F3A8 RAY TRACER");
			SetDlgItemTextW(hDlg, IDC_GROUPBOX_RENDERER, L"\U0001F5A5\uFE0F Rendering Engine");
			SetDlgItemTextW(hDlg, IDC_RADIO_GPU, L"\u26A1 GPU (CUDA) - Recommended");
			SetDlgItemTextW(hDlg, IDC_RADIO_CPU, L"\U0001F527 CPU (Multi-threaded)");
			SetDlgItemTextW(hDlg, IDC_BUTTON_RENDER, L"\u25B6\uFE0F RENDER");
			SetDlgItemTextW(hDlg, IDC_STATIC_QUALITY_DESC, L"Select a quality preset based on your needs.\nHigher quality = better image quality but longer render time.\n\n\U0001F4A1 Tip: Start with Medium or High for best results.");
			SetDlgItemTextW(hDlg, IDC_STATIC_ADVANCED_DESC, L"\U0001F4A1 Higher samples = better quality but slower render.\n\u26A1 GPU recommended for high sample counts.");

			// Set icon
			HICON hIcon = LoadIcon(GetModuleHandle(NULL), MAKEINTRESOURCE(IDI_APP_ICON));
			SendMessage(hDlg, WM_SETICON, ICON_BIG, (LPARAM)hIcon);
			SendMessage(hDlg, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);

			// Configure RichEdit controls for colored emoji support
			HWND richEditControls[] = {
				GetDlgItem(hDlg, IDC_STATIC_TITLE),
				GetDlgItem(hDlg, IDC_STATIC_QUALITY_DESC),
				GetDlgItem(hDlg, IDC_STATIC_ADVANCED_DESC),
				GetDlgItem(hDlg, IDC_STATIC_STATUS)
			};

			for (int i = 0; i < 4; i++) {
				HWND hRichEdit = richEditControls[i];
				if (hRichEdit) {
					// Enable transparent background
					LONG style = GetWindowLong(hRichEdit, GWL_EXSTYLE);
					SetWindowLong(hRichEdit, GWL_EXSTYLE, style | WS_EX_TRANSPARENT);

					// Set background color for dark theme
					SendMessage(hRichEdit, EM_SETBKGNDCOLOR, 0, (LPARAM)g_colorBackground);

					// Disable selection hiding and enable auto URL detection
					SendMessage(hRichEdit, EM_SETOPTIONS, ECOOP_OR, ECO_NOHIDESEL);

					// Set text color to white for dark theme
					CHARFORMAT2W cf = { 0 };
					cf.cbSize = sizeof(CHARFORMAT2W);
					cf.dwMask = CFM_COLOR;
					cf.crTextColor = g_colorText;
					SendMessage(hRichEdit, EM_SETCHARFORMAT, SCF_ALL, (LPARAM)&cf);
				}
			}

			// Check GPU availability and set default
			if (gpu_is_available()) {
				CheckDlgButton(hDlg, IDC_RADIO_GPU, BST_CHECKED);
				SetDlgItemTextW(hDlg, IDC_STATIC_STATUS, L"\U0001F680 GPU detected! Ready to render.");
			} else {
				CheckDlgButton(hDlg, IDC_RADIO_CPU, BST_CHECKED);
				EnableWindow(GetDlgItem(hDlg, IDC_RADIO_GPU), FALSE);
				SetDlgItemTextW(hDlg, IDC_STATIC_STATUS, L"\u2699\uFE0F No GPU detected. CPU mode selected.");
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
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"⚡ 400 x 400 (Quick Preview)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"💎 600 x 600 (Balanced)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"🌟 800 x 800 (High Quality)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"🔥 1080 x 1080 (Very High)");
			SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)L"💫 2048 x 2048 (2K Ultra)");
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
					MessageBoxW(hDlg, L"⏳ Please wait for rendering to complete.", L"Busy", MB_OK | MB_ICONWARNING);
					return TRUE;
				}
				EndDialog(hDlg, 0);
				return TRUE;
			}
			break;
		}

		case WM_CLOSE: {
			if (g_rendering) {
				MessageBoxW(hDlg, L"⏳ Please wait for rendering to complete.", L"Busy", MB_OK | MB_ICONWARNING);
				return TRUE;
			}
			// Clean up resources
			if (g_hBackgroundBrush) DeleteObject(g_hBackgroundBrush);
			if (g_hLightBackgroundBrush) DeleteObject(g_hLightBackgroundBrush);
			if (g_hTitleFont) DeleteObject(g_hTitleFont);
			if (g_hNormalFont) DeleteObject(g_hNormalFont);
			EndDialog(hDlg, 0);
			return TRUE;
		}

		case WM_CTLCOLORDLG:
		case WM_CTLCOLORSTATIC:
		case WM_CTLCOLOREDIT: {  // Handle RichEdit controls for dark theme
			HDC hdcStatic = (HDC)wParam;
			SetTextColor(hdcStatic, g_colorText);
			SetBkColor(hdcStatic, g_colorBackground);
			return (INT_PTR)g_hBackgroundBrush;
		}

		case WM_CTLCOLORBTN: {
			HDC hdcButton = (HDC)wParam;
			SetTextColor(hdcButton, g_colorText);
			SetBkColor(hdcButton, g_colorLightBackground);
			return (INT_PTR)g_hLightBackgroundBrush;
		}
	}
	return FALSE;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	InitCommonControls();

	// Load RichEdit library for colored emoji support
	HMODULE hRichEdit = LoadLibraryW(L"msftedit.dll");
	if (!hRichEdit) {
		MessageBoxW(NULL, L"Failed to load RichEdit library (msftedit.dll).\nColored emoji may not display correctly.", 
			L"Warning", MB_OK | MB_ICONWARNING);
	}

	DialogBox(hInstance, MAKEINTRESOURCE(IDD_MAIN_DIALOG), NULL, DialogProc);

	// Cleanup
	if (hRichEdit) {
		FreeLibrary(hRichEdit);
	}

	return 0;
}
