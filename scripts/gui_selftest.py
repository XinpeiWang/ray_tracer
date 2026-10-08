#!/usr/bin/env python3
"""Automated smoke test of the real Qt GUI, on macOS and Windows (the project has no other GUI test).

It runs the built RayTracerGUI with RT_GUI_SELFTEST (qt_gui/mainwindow_selftest.cpp) in each requested mode, headless, and checks the
result. Screenshots are of the app's own window (never the screen) and are written, with a text log per mode, to the output directory.

  ui           the main window comes up and the Output Mode list is filled (with --live-preview: it contains an ENABLED
               "Live Preview (interactive)" item)
  options      the Options controls respond
  builder      the Scene Builder tab: edits, drag, undo/redo, save and re-open, a CPU preview (and with --live-preview a GPU preview too)
  queue        two real tiny renders through the Render button: the second queues behind the first, both rows end Done with a time and stay in
               the queue table, Clear Finished empties it
  diagnostics  the Diagnostics report is produced
  tour         (not in the default set) a screenshot of every tab, for looking over the UI: RT_GUI_SELFTEST_WIDTH/_HEIGHT size the window, RT_GUI_SELFTEST_THEME
               picks a theme id (e.g. solarized-light)
  installphoto the Diagnostics tab's "Install Photo Helper" flow with stand-in installer scripts (not in the default set; no big download)
  livepreview  (--live-preview) selects Live Preview, starts it, lets it render, orbits the camera like a mouse drag, and requires
               frames to flow AND the picture to change

Usage:
  python3 scripts/gui_selftest.py [APP] [--out DIR] [--modes ui,options,...] [--live-preview] [--plugins DIR]
    APP   RayTracerGUI.app (macOS) or RayTracerGUI.exe / the folder holding it (Windows).
          Default: RayTracer_Package_macOS/RayTracerGUI.app, or RayTracer_Package/RayTracerGUI.exe.

It runs with a throwaway home folder, so the app never reads or writes your Pictures or your real preferences (macOS asks permission for
Pictures, and an unanswered prompt blocks startup invisibly), and with Qt's `offscreen` platform plugin, taken from the Qt install next to
`qmake` (macdeployqt/windeployqt bundle only the real window-system plugin) unless --plugins or QT_QPA_PLATFORM_PLUGIN_PATH says otherwise.
The app is started from the root folder, where a Finder/Dock/Explorer launch can start, so a scene file that only resolves from the
working directory fails here instead of on a user's machine.

Exit status 0 when every requested mode passed.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IS_WINDOWS = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def find_exe(arg):
    if arg is None:
        arg = os.path.join(REPO, "RayTracer_Package_macOS", "RayTracerGUI.app") if IS_MAC else os.path.join(REPO, "RayTracer_Package", "RayTracerGUI.exe")
    arg = os.path.abspath(arg)
    if os.path.isdir(arg):
        mac = os.path.join(arg, "Contents", "MacOS", "RayTracerGUI")
        win = os.path.join(arg, "RayTracerGUI.exe")
        arg = mac if os.path.exists(mac) else win
    return arg


def find_plugins(explicit):
    """The directory holding Qt's offscreen platform plugin."""
    name = "qoffscreen.dll" if IS_WINDOWS else "libqoffscreen.dylib" if IS_MAC else "libqoffscreen.so"
    candidates = [explicit, os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")]
    qmake = shutil.which("qmake")
    if qmake:
        try:
            plugins = subprocess.run([qmake, "-query", "QT_INSTALL_PLUGINS"], capture_output=True, text=True, timeout=30).stdout.strip()
            if plugins:
                candidates.append(os.path.join(plugins, "platforms"))
        except (OSError, subprocess.SubprocessError):
            pass
    for c in candidates:
        if c and os.path.exists(os.path.join(c, name)):
            return c
    sys.exit("ERROR: Qt's offscreen platform plugin (%s) not found. Put Qt's bin folder (qmake) on PATH, or pass --plugins DIR." % name)


def needs_rosetta(exe):
    """An x86_64-only build on an Apple-silicon Mac must be started under Rosetta; an arm64 or universal build runs natively."""
    if not IS_MAC:
        return False
    archs = subprocess.run(["lipo", "-archs", exe], capture_output=True, text=True).stdout.split()
    apple_silicon = subprocess.run(["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True).stdout.strip() == "1"
    return apple_silicon and "arm64" not in archs


def installphoto_env(fake_home):
    """Stand-in installers for the installphoto mode (the real one downloads about 5 GB): one that prints some lines, including a lone-CR progress update,
    and succeeds, and one that fails with a CRLF error line. The Python the helper would use points at a file that does not exist, so the Diagnostics report
    lists the helper as missing and the install button has something to do."""
    if IS_WINDOWS:
        ok_body = 'Write-Host "Setting up the photo helper (stand-in)"\r\nWrite-Host "Installing the packages..."\r\nexit 0\r\n'
        fail_body = '[Console]::Out.Write("ERROR: no network (CRLF)`r`n")\r\nexit 1\r\n'
        ext = ".ps1"
    else:
        ok_body = '#!/bin/bash\necho "Setting up the photo helper (stand-in)"\nprintf "progress 10%%\\r"\necho "Installing the packages..."\nexit 0\n'
        fail_body = '#!/bin/bash\nprintf "ERROR: no network (CRLF)\\r\\n"\nexit 1\n'
        ext = ".sh"
    extra = {}
    if not IS_WINDOWS:
        pid_file = os.path.join(fake_home, "hang_child.pid")
        hang_path = os.path.join(fake_home, "stand_in_setup_hang.sh")
        with open(hang_path, "w", newline="") as f:
            f.write('#!/bin/bash\nsleep 120 &\necho $! > "%s"\nwait\n' % pid_file)
        os.chmod(hang_path, 0o755)
        extra = {"RT_GUI_SELFTEST_SETUP_HANG": hang_path, "RT_GUI_SELFTEST_HANG_PID": pid_file}
    ok_path = os.path.join(fake_home, "stand_in_setup_ok" + ext)
    fail_path = os.path.join(fake_home, "stand_in_setup_fail" + ext)
    for path, body in ((ok_path, ok_body), (fail_path, fail_body)):
        with open(path, "w", newline="") as f:
            f.write(body)
        os.chmod(path, 0o755)
    return {"RAY_TRACER_PHOTO3D_PYTHON": os.path.join(fake_home, "no-such-python"), "RAY_TRACER_PHOTO3D_SETUP": ok_path,
            "RT_GUI_SELFTEST_SETUP_FAIL": fail_path, **extra}


def run_mode(exe, mode, wait_s, out, plugins, fake_home, gpu=False):
    prefix = os.path.join(out, mode)
    for ext in (".txt", ".stdout"):
        if os.path.exists(prefix + ext):
            os.remove(prefix + ext)
    env = dict(os.environ)
    env.update({
        "QT_QPA_PLATFORM": "offscreen", "QT_QPA_PLATFORM_PLUGIN_PATH": plugins,
        "RT_GUI_SELFTEST": mode, "RT_GUI_SELFTEST_OUT": prefix,
        "HOME": fake_home, "CFFIXED_USER_HOME": fake_home, "USERPROFILE": fake_home,
        "APPDATA": os.path.join(fake_home, "AppData", "Roaming"), "LOCALAPPDATA": os.path.join(fake_home, "AppData", "Local"),
        # Qt's per-user folders ignore HOME on a Mac (they come from the system), so say where the program's own per-user folders are: otherwise the
        # builder mode's "Add to scene list" put test scenes into the real user's My Scenes.
        "RAY_TRACER_USER_ASSETS": os.path.join(fake_home, "user_assets"),
    })
    if gpu:
        env["RT_GUI_SELFTEST_GPU"] = "1"   # the builder mode also previews through "Use the GPU"
    if mode == "installphoto":
        env.update(installphoto_env(fake_home))
    cmd = (["arch", "-x86_64"] if needs_rosetta(exe) else []) + [exe]
    with open(prefix + ".stdout", "wb") as log:
        # cwd is the root folder on purpose (see the module docstring).
        proc = subprocess.Popen(cmd, env=env, cwd=os.path.abspath(os.sep), stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print("FAIL [%s]: still running after %ds (hung)" % (mode, wait_s))
            return False
    if rc != 0:
        print("FAIL [%s]: exit code %d" % (mode, rc))
        if os.path.exists(prefix + ".txt"):
            print(open(prefix + ".txt", errors="replace").read())
        return False
    print("PASS [%s]" % mode)
    return True


def forget_settings():
    """The app's self-test settings domain is separate from the real one; remove what a run left (the app also clears it at start)."""
    try:
        if IS_MAC:
            subprocess.run(["defaults", "delete", "com.raytracer.RayTracerGUI-selftest"], capture_output=True)
            plist = os.path.expanduser("~/Library/Preferences/com.raytracer.RayTracerGUI-selftest.plist")
            if os.path.exists(plist):
                os.remove(plist)
        elif IS_WINDOWS:
            subprocess.run(["reg", "delete", r"HKCU\Software\RayTracer\RayTracerGUI-selftest", "/f"], capture_output=True)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("app", nargs="?", help="RayTracerGUI.app, RayTracerGUI.exe, or the folder holding it")
    ap.add_argument("--out", help="directory for logs and screenshots (default: a new temp directory)")
    ap.add_argument("--modes", default="ui,options,builder,queue,diagnostics", help="comma-separated self-test modes (default: %(default)s)")
    ap.add_argument("--live-preview", action="store_true", help="also require an enabled Live Preview item and run the livepreview mode (needs a GPU)")
    ap.add_argument("--plugins", help="directory holding Qt's offscreen platform plugin")
    args = ap.parse_args()

    exe = find_exe(args.app)
    if not os.path.exists(exe):
        sys.exit("ERROR: %s not found - build the app first (scripts/build_and_deploy_macos.sh, or the Windows build)" % exe)
    plugins = find_plugins(args.plugins)
    out = os.path.abspath(args.out or tempfile.mkdtemp(prefix="gui_selftest."))
    os.makedirs(out, exist_ok=True)
    fake_home = tempfile.mkdtemp(prefix="gui_selftest_home.")
    os.makedirs(os.path.join(fake_home, "Pictures"), exist_ok=True)

    modes = [m for m in args.modes.split(",") if m]
    if args.live_preview and "livepreview" not in modes:
        modes.append("livepreview")
    waits = {"livepreview": 60}
    ok = True
    try:
        for mode in modes:
            passed = run_mode(exe, mode, waits.get(mode, 60), out, plugins, fake_home, gpu=args.live_preview)
            ok = ok and passed
            log = os.path.join(out, mode + ".txt")
            text = open(log, errors="replace").read() if os.path.exists(log) else ""
            if mode == "ui" and passed and args.live_preview and 'Live Preview (interactive)" enabled=1' not in text:
                print("FAIL [ui]: no enabled Live Preview item")
                ok = False
            if mode == "livepreview":
                for line in text.splitlines():
                    if any(k in line for k in ("frames=", "picture change", "RESULT")):
                        print("  " + line.strip())
    finally:
        shutil.rmtree(fake_home, ignore_errors=True)
        forget_settings()
    print("logs and screenshots: " + out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
