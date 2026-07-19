#include <windows.h>
#include <commctrl.h>
#include <gdiplus.h>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>
#include "resource.h"

// Link GDI+ for modern graphics
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(linker,"\"\/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

using namespace Gdiplus;

// Image writer utilities
#include "../src/external/image_writer.h"

// Forward declarations of the render APIs
extern "C" {
	int gpu_is_available();
	int gpu_render_main(int width, int height, int samples_per_pixel, int max_depth, const char* output_path);
	int cpu_render_main(int width, int height, int samples_per_pixel, int max_depth, const char* output_path);
}

// Modern UI colors (Material Design inspired)
const COLORREF COLOR_PRIMARY = RGB(98, 0, 238);           // Deep purple
const COLORREF COLOR_PRIMARY_DARK = RGB(55, 0, 179);     // Darker purple
const COLORREF COLOR_PRIMARY_LIGHT = RGB(179, 136, 255); // Light purple
const COLORREF COLOR_ACCENT = RGB(0, 229, 255);          // Cyan accent
const COLORREF COLOR_BACKGROUND = RGB(18, 18, 18);       // Almost black
const COLORREF COLOR_SURFACE = RGB(30, 30, 30);          // Dark grey
const COLORREF COLOR_CARD = RGB(40, 40, 40);             // Card background
const COLORREF COLOR_TEXT_PRIMARY = RGB(255, 255, 255);  // White
const COLORREF COLOR_TEXT_SECONDARY = RGB(180, 180, 180);// Grey text
const COLORREF COLOR_SUCCESS = RGB(76, 175, 80);         // Green
const COLORREF COLOR_ERROR = RGB(244, 67, 54);           // Red

// GDI+ globals
ULONG_PTR g_gdiplusToken = 0;

// Global variables
HWND g_hDlg = NULL;
bool g_rendering = false;
int g_currentTab = 0; // 0 = Basic, 1 = Advanced
HWND g_hoveredButton = NULL;

// Fonts
HFONT g_hTitleFont = NULL;
HFONT g_hLargeFont = NULL;
HFONT g_hNormalFont = NULL;
HFONT g_hSmallFont = NULL;

// Custom button data
struct ModernButton {
	HWND hwnd;
	bool isHovered;
	bool isPressed;
	COLORREF normalColor;
	COLORREF hoverColor;
	COLORREF pressedColor;
};

std::vector<ModernButton*> g_buttons;

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
	{ L"⚡ Fast Preview", L"Quick test renders (400x400, 50 samples)", 400, 50, 10 },
	{ L"💎 Balanced", L"Good quality, reasonable time (600x600, 200 samples)", 600, 200, 20 },
	{ L"⭐ High Quality", L"Recommended for most scenes (800x800, 500 samples)", 800, 500, 30 },
	{ L"🔥 Very High", L"Impressive results (1080x1080, 1000 samples)", 1080, 1000, 40 },
	{ L"💫 Ultra", L"Maximum quality (2048x2048, 2000 samples)", 2048, 2000, 50 }
};

// Helper: Create rounded rectangle path
GraphicsPath* CreateRoundRectPath(Rect rect, int radius) {
	GraphicsPath* path = new GraphicsPath();
	path->AddArc(rect.X, rect.Y, radius, radius, 180, 90);
	path->AddArc(rect.X + rect.Width - radius, rect.Y, radius, radius, 270, 90);
	path->AddArc(rect.X + rect.Width - radius, rect.Y + rect.Height - radius, radius, radius, 0, 90);
	path->AddArc(rect.X, rect.Y + rect.Height - radius, radius, radius, 90, 90);
	path->CloseFigure();
	return path;
}

// Custom draw a modern button
void DrawModernButton(HDC hdc, RECT rect, const wchar_t* text, bool isHovered, bool isPressed, bool isPrimary = false) {
	Graphics graphics(hdc);
	graphics.SetSmoothingMode(SmoothingModeAntiAlias);
	graphics.SetTextRenderingHint(TextRenderingHintAntiAlias);

	// Button colors
	COLORREF bgColor = isPrimary ? COLOR_PRIMARY : COLOR_SURFACE;
	if (isPressed) bgColor = isPrimary ? COLOR_PRIMARY_DARK : RGB(50, 50, 50);
	else if (isHovered) bgColor = isPrimary ? RGB(120, 30, 255) : RGB(60, 60, 60);

	// Create gradient brush
	Rect gdiRect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
	Color color1(GetRValue(bgColor), GetGValue(bgColor), GetBValue(bgColor));
	Color color2(
		min(GetRValue(bgColor) + 20, 255),
		min(GetGValue(bgColor) + 20, 255),
		min(GetBValue(bgColor) + 20, 255)
	);

	LinearGradientBrush gradientBrush(gdiRect, color1, color2, LinearGradientModeVertical);

	// Draw shadow if not pressed
	if (!isPressed) {
		Rect shadowRect = gdiRect;
		shadowRect.Offset(0, 2);
		shadowRect.Inflate(2, 2);
		GraphicsPath* shadowPath = CreateRoundRectPath(shadowRect, 8);
		PathGradientBrush shadowBrush(shadowPath);
		Color centerColor(60, 0, 0, 0);
		Color edgeColor(0, 0, 0, 0);
		shadowBrush.SetCenterColor(centerColor);
		int count = 1;
		shadowBrush.SetSurroundColors(&edgeColor, &count);
		graphics.FillPath(&shadowBrush, shadowPath);
		delete shadowPath;
	}

	// Draw button
	GraphicsPath* buttonPath = CreateRoundRectPath(gdiRect, 6);
	graphics.FillPath(&gradientBrush, buttonPath);

	// Draw border
	Color borderColor(isHovered ? 80 : 60, 255, 255, 255);
	Pen borderPen(borderColor, 1.0f);
	graphics.DrawPath(&borderPen, buttonPath);
	delete buttonPath;

	// Draw text
	FontFamily fontFamily(L"Segoe UI");
	Font font(&fontFamily, 11, FontStyleBold, UnitPoint);
	Color textColor(255, 255, 255, 255);
	SolidBrush textBrush(textColor);
	StringFormat stringFormat;
	stringFormat.SetAlignment(StringAlignmentCenter);
	stringFormat.SetLineAlignment(StringAlignmentCenter);
	RectF textRect((REAL)rect.left, (REAL)rect.top, (REAL)(rect.right - rect.left), (REAL)(rect.bottom - rect.top));
	graphics.DrawString(text, -1, &font, textRect, &stringFormat, &textBrush);
}

// Modern card background with gradient
void DrawCard(HDC hdc, RECT rect) {
	Graphics graphics(hdc);
	graphics.SetSmoothingMode(SmoothingModeAntiAlias);

	Rect gdiRect(rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);

	// Create gradient
	Color color1(GetRValue(COLOR_CARD), GetGValue(COLOR_CARD), GetBValue(COLOR_CARD));
	Color color2(
		min(GetRValue(COLOR_CARD) + 10, 255),
		min(GetGValue(COLOR_CARD) + 10, 255),
		min(GetBValue(COLOR_CARD) + 10, 255)
	);
	LinearGradientBrush gradientBrush(gdiRect, color1, color2, LinearGradientModeVertical);

	// Draw rounded rectangle
	GraphicsPath* cardPath = CreateRoundRectPath(gdiRect, 12);
	graphics.FillPath(&gradientBrush, cardPath);

	// Subtle border
	Color borderColor(30, 255, 255, 255);
	Pen borderPen(borderColor, 1.0f);
	graphics.DrawPath(&borderPen, cardPath);
	delete cardPath;
}

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
		ShowBasicControls(true);
		ShowAdvancedControls(false);
	} else {
		ShowBasicControls(false);
		ShowAdvancedControls(true);
	}
}

void RenderThread(RenderSettings settings) {
	// Get executable directory for output path
	char exePath[MAX_PATH];
	GetModuleFileNameA(NULL, exePath, MAX_PATH);
	std::filesystem::path exeDir = std::filesystem::path(exePath).parent_path();
	std::filesystem::path outputDir = exeDir / "output";
	std::filesystem::create_directories(outputDir);

	auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
	std::string filename = "render_" + std::to_string(timestamp) + ".png";
	std::filesystem::path outputPath = outputDir / filename;

	UpdateStatusText(settings.useGPU ? L"🚀 Rendering on GPU..." : L"💻 Rendering on CPU...");
	SetProgressBar(0);

	auto startTime = std::chrono::high_resolution_clock::now();

	int result = settings.useGPU ?
		gpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputPath.string().c_str()) :
		cpu_render_main(settings.width, settings.height, settings.samples, settings.maxDepth, outputPath.string().c_str());

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

	if (result == 0) {
		std::wstring statusMsg = L"✅ Render completed in " + std::to_wstring(duration / 1000.0) + L"s\nSaved to: output\\" + std::filesystem::path(filename).wstring();
		UpdateStatusText(statusMsg.c_str());
		SetProgressBar(100);
		ShellExecuteW(NULL, L"explore", outputDir.wstring().c_str(), NULL, NULL, SW_SHOWNORMAL);
	} else {
		UpdateStatusText(L"❌ Render failed. Check GPU memory if using GPU mode.");
		SetProgressBar(0);
	}

	g_rendering = false;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), TRUE);
	SetDlgItemTextW(g_hDlg, IDC_BUTTON_RENDER, L"🎨 Start Render");
}

RenderSettings GetCurrentSettings() {
	RenderSettings settings;
	settings.useGPU = (SendDlgItemMessage(g_hDlg, IDC_RADIO_GPU, BM_GETCHECK, 0, 0) == BST_CHECKED);

	if (g_currentTab == 0) {
		// Basic mode
		int presetIndex = (int)SendDlgItemMessage(g_hDlg, IDC_COMBO_QUALITY, CB_GETCURSEL, 0, 0);
		if (presetIndex >= 0 && presetIndex < sizeof(g_presets) / sizeof(g_presets[0])) {
			settings.width = g_presets[presetIndex].width;
			settings.height = g_presets[presetIndex].width;
			settings.samples = g_presets[presetIndex].samples;
			settings.maxDepth = g_presets[presetIndex].depth;
		}
	} else {
		// Advanced mode
		int resolutionIndex = (int)SendDlgItemMessage(g_hDlg, IDC_COMBO_RESOLUTION, CB_GETCURSEL, 0, 0);
		int resolutions[] = { 400, 600, 800, 1080, 1920, 2048 };
		settings.width = (resolutionIndex >= 0 && resolutionIndex < 6) ? resolutions[resolutionIndex] : 800;
		settings.height = settings.width;
		settings.samples = (int)SendDlgItemMessage(g_hDlg, IDC_SLIDER_SAMPLES, TBM_GETPOS, 0, 0);
		settings.maxDepth = (int)SendDlgItemMessage(g_hDlg, IDC_SLIDER_DEPTH, TBM_GETPOS, 0, 0);
	}

	return settings;
}

void OnRenderClick() {
	if (g_rendering) return;

	g_rendering = true;
	EnableWindow(GetDlgItem(g_hDlg, IDC_BUTTON_RENDER), FALSE);
	SetDlgItemTextW(g_hDlg, IDC_BUTTON_RENDER, L"⏳ Rendering...");

	RenderSettings settings = GetCurrentSettings();
	std::thread(RenderThread, settings).detach();
}

INT_PTR CALLBACK DialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
	switch (message) {
	case WM_INITDIALOG: {
		g_hDlg = hDlg;

		// Create modern fonts
		g_hTitleFont = CreateFontW(24, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
			OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
		g_hLargeFont = CreateFontW(13, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
			OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
		g_hNormalFont = CreateFontW(10, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
			OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
		g_hSmallFont = CreateFontW(9, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
			OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

		// Set fonts
		SendDlgItemMessage(hDlg, IDC_STATIC_TITLE, WM_SETFONT, (WPARAM)g_hTitleFont, TRUE);
		SendDlgItemMessage(hDlg, IDC_TAB, WM_SETFONT, (WPARAM)g_hLargeFont, TRUE);

		// Apply normal font to all other controls
		EnumChildWindows(hDlg, [](HWND hwnd, LPARAM lParam) -> BOOL {
			int id = GetDlgCtrlID(hwnd);
			if (id != IDC_STATIC_TITLE && id != IDC_TAB) {
				SendMessage(hwnd, WM_SETFONT, (WPARAM)g_hNormalFont, TRUE);
			}
			return TRUE;
		}, 0);

		// Initialize tab control
		TCITEMW tie;
		tie.mask = TCIF_TEXT;
		tie.pszText = (LPWSTR)L"⚙️ Basic";
		TabCtrl_InsertItem(GetDlgItem(hDlg, IDC_TAB), 0, &tie);
		tie.pszText = (LPWSTR)L"🔧 Advanced";
		TabCtrl_InsertItem(GetDlgItem(hDlg, IDC_TAB), 1, &tie);

		// Initialize quality presets
		for (const auto& preset : g_presets) {
			SendDlgItemMessageW(hDlg, IDC_COMBO_QUALITY, CB_ADDSTRING, 0, (LPARAM)preset.name);
		}
		SendDlgItemMessage(hDlg, IDC_COMBO_QUALITY, CB_SETCURSEL, 2, 0); // Default to High Quality

		// Initialize resolution combo
	const wchar_t* resolutions[] = { L"400x400", L"600x600", L"800x800 (HD)", L"1080x1080 (Full HD)", L"1920x1920 (2K)", L"2048x2048 (2K+)" };
		for (const auto* res : resolutions) {
			SendDlgItemMessageW(hDlg, IDC_COMBO_RESOLUTION, CB_ADDSTRING, 0, (LPARAM)res);
		}
		SendDlgItemMessage(hDlg, IDC_COMBO_RESOLUTION, CB_SETCURSEL, 2, 0);

		// Initialize sliders
		SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETRANGE, TRUE, MAKELONG(10, 2000));
		SendDlgItemMessage(hDlg, IDC_SLIDER_SAMPLES, TBM_SETPOS, TRUE, 500);
		SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETRANGE, TRUE, MAKELONG(5, 100));
		SendDlgItemMessage(hDlg, IDC_SLIDER_DEPTH, TBM_SETPOS, TRUE, 30);

		// Set GPU as default if available
		if (gpu_is_available()) {
			CheckDlgButton(hDlg, IDC_RADIO_GPU, BST_CHECKED);
			CheckDlgButton(hDlg, IDC_RADIO_CPU, BST_UNCHECKED);
		} else {
			CheckDlgButton(hDlg, IDC_RADIO_CPU, BST_CHECKED);
			CheckDlgButton(hDlg, IDC_RADIO_GPU, BST_UNCHECKED);
			EnableWindow(GetDlgItem(hDlg, IDC_RADIO_GPU), FALSE);
		}

		// Initialize progress bar
		SendDlgItemMessage(hDlg, IDC_PROGRESS, PBM_SETRANGE, 0, MAKELPARAM(0, 100));
		SendDlgItemMessage(hDlg, IDC_PROGRESS, PBM_SETSTEP, 1, 0);

		// Start in basic mode
		SwitchTab(0);

		// Center window
		RECT rc, rcDlg, rcOwner;
		GetWindowRect(GetDesktopWindow(), &rcOwner);
		GetWindowRect(hDlg, &rcDlg);
		CopyRect(&rc, &rcOwner);
		OffsetRect(&rcDlg, -rcDlg.left, -rcDlg.top);
		OffsetRect(&rc, -rc.left, -rc.top);
		OffsetRect(&rc, -rcDlg.right, -rcDlg.bottom);
		SetWindowPos(hDlg, HWND_TOP, rcOwner.left + (rc.right / 2), rcOwner.top + (rc.bottom / 2), 0, 0, SWP_NOSIZE);

		UpdateStatusText(L"💎 Ready to render your scene");
		return TRUE;
	}

	case WM_CTLCOLORDLG:
	case WM_CTLCOLORSTATIC: {
		HDC hdcStatic = (HDC)wParam;
		SetTextColor(hdcStatic, COLOR_TEXT_PRIMARY);
		SetBkColor(hdcStatic, COLOR_BACKGROUND);
		return (INT_PTR)CreateSolidBrush(COLOR_BACKGROUND);
	}

	case WM_DRAWITEM: {
		DRAWITEMSTRUCT* dis = (DRAWITEMSTRUCT*)lParam;
		if (dis->CtlType == ODT_BUTTON) {
			bool isHovered = (dis->itemState & ODS_SELECTED) || (g_hoveredButton == dis->hwndItem);
			bool isPressed = (dis->itemState & ODS_SELECTED);
			bool isPrimary = (dis->CtlID == IDC_BUTTON_RENDER);

			wchar_t text[256];
			GetWindowTextW(dis->hwndItem, text, 256);
			DrawModernButton(dis->hDC, dis->rcItem, text, isHovered, isPressed, isPrimary);
			return TRUE;
		}
		break;
	}

	case WM_MOUSEMOVE: {
		POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
		ClientToScreen(hDlg, &pt);

		HWND hButton = ChildWindowFromPoint(hDlg, pt);
		if (hButton != g_hoveredButton) {
			if (g_hoveredButton) InvalidateRect(g_hoveredButton, NULL, FALSE);
			g_hoveredButton = hButton;
			if (g_hoveredButton) {
				InvalidateRect(g_hoveredButton, NULL, FALSE);
				TRACKMOUSEEVENT tme = { sizeof(tme), TME_LEAVE, hDlg, 0 };
				TrackMouseEvent(&tme);
			}
		}
		break;
	}

	case WM_MOUSELEAVE:
		if (g_hoveredButton) {
			InvalidateRect(g_hoveredButton, NULL, FALSE);
			g_hoveredButton = NULL;
		}
		break;

	case WM_NOTIFY: {
		NMHDR* nmhdr = (NMHDR*)lParam;
		if (nmhdr->idFrom == IDC_TAB && nmhdr->code == TCN_SELCHANGE) {
			int tabIndex = TabCtrl_GetCurSel(nmhdr->hwndFrom);
			SwitchTab(tabIndex);
		}
		break;
	}

	case WM_COMMAND:
		switch (LOWORD(wParam)) {
		case IDC_BUTTON_RENDER:
			OnRenderClick();
			return TRUE;
		case IDC_BUTTON_EXIT:
		case IDCANCEL:
			EndDialog(hDlg, 0);
			return TRUE;
		case IDC_COMBO_QUALITY:
			if (HIWORD(wParam) == CBN_SELCHANGE) {
				int index = (int)SendDlgItemMessage(hDlg, IDC_COMBO_QUALITY, CB_GETCURSEL, 0, 0);
				if (index >= 0 && index < sizeof(g_presets) / sizeof(g_presets[0])) {
					SetDlgItemTextW(hDlg, IDC_STATIC_QUALITY_DESC, g_presets[index].description);
				}
			}
			break;
		}
		break;

	case WM_CLOSE:
		EndDialog(hDlg, 0);
		return TRUE;

	case WM_DESTROY:
		if (g_hTitleFont) DeleteObject(g_hTitleFont);
		if (g_hLargeFont) DeleteObject(g_hLargeFont);
		if (g_hNormalFont) DeleteObject(g_hNormalFont);
		if (g_hSmallFont) DeleteObject(g_hSmallFont);
		PostQuitMessage(0);
		return TRUE;
	}

	return FALSE;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
	// Initialize GDI+
	GdiplusStartupInput gdiplusStartupInput;
	GdiplusStartup(&g_gdiplusToken, &gdiplusStartupInput, NULL);

	// Initialize common controls
	INITCOMMONCONTROLSEX icc;
	icc.dwSize = sizeof(icc);
	icc.dwICC = ICC_WIN95_CLASSES | ICC_BAR_CLASSES | ICC_TAB_CLASSES | ICC_PROGRESS_CLASS;
	InitCommonControlsEx(&icc);

	// Create and show dialog
	DialogBox(hInstance, MAKEINTRESOURCE(IDD_MAIN), NULL, DialogProc);

	// Shutdown GDI+
	GdiplusShutdown(g_gdiplusToken);

	return 0;
}
