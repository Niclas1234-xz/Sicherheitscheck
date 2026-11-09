#!/usr/bin/env python3
"""
Richtet den Sicherheitscheck als Infobereich-App ein:

- Windows: Eintrag im Startmenü, Autostart beim Anmelden (still im Infobereich), kein Desktop-Symbol
- macOS:   App «Sicherheitscheck» im Ordner ~/Programme (Launchpad, Spotlight), Autostart, kein Dock-Symbol
- Linux:   Eintrag im Anwendungsmenü, Autostart, kein Desktop-Symbol

Wird von den Startskripten einmalig automatisch ausgeführt. Kann jederzeit erneut gestartet werden,
z. B. nachdem du den Programmordner verschoben hast:
    python verknuepfung.py               (einrichten, Autostart ein)
    python verknuepfung.py --ohne-autostart
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "sicherheitscheck.py"
ICON_DIR = HERE / "icon"
APP_NAME = "Sicherheitscheck"
DESCRIPTION = "Schutz vor fremden Blicken auf deinen Bildschirm"
TRAY_ARG = "--tray"   # startet still im Infobereich, ohne Fenster


def venv_python(windowed: bool = False) -> Path:
    if sys.platform.startswith("win"):
        exe = HERE / ".venv" / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
        if exe.is_file():
            return exe
        current = Path(sys.executable)
        if windowed and current.name.lower() == "python.exe":
            alt = current.with_name("pythonw.exe")
            if alt.is_file():
                return alt
        return current
    exe = HERE / ".venv" / "bin" / "python"
    return exe if exe.exists() else Path(sys.executable)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

def _ps(s) -> str:
    """Text für einen PowerShell-String in einfachen Anführungszeichen."""
    return str(s).replace("'", "''")


def _run_powershell(script: str) -> None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    res = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, creationflags=flags)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="replace").strip() or "PowerShell-Fehler")


def _ps_shortcut(folder_expr: str, args: str) -> str:
    """PowerShell-Befehl, der eine Verknüpfung im angegebenen Ordner anlegt."""
    target = venv_python(windowed=True)
    icon = ICON_DIR / "sicherheitscheck.ico"
    return f"""
$f = {folder_expr}
if ($f) {{
    $s = $ws.CreateShortcut((Join-Path $f '{_ps(APP_NAME)}.lnk'))
    $s.TargetPath = '{_ps(target)}'
    $s.Arguments = '"{_ps(SCRIPT)}"{(" " + args) if args else ""}'
    $s.WorkingDirectory = '{_ps(HERE)}'
    $s.IconLocation = '{_ps(icon)},0'
    $s.Description = '{_ps(DESCRIPTION)}'
    $s.Save()
}}
"""


def _win_startup_link() -> Path:
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / f"{APP_NAME}.lnk"


def setup_windows(autostart: bool) -> list[str]:
    script = "$ErrorActionPreference = 'Stop'\n$ws = New-Object -ComObject WScript.Shell\n"
    script += _ps_shortcut("[Environment]::GetFolderPath('Programs')", "")
    # altes Desktop-Symbol entfernen
    script += (f"$d = Join-Path ([Environment]::GetFolderPath('Desktop')) '{_ps(APP_NAME)}.lnk'\n"
               "if (Test-Path -LiteralPath $d) { Remove-Item -LiteralPath $d -Force }\n")
    _run_powershell(script)
    done = ["Startmenü"]
    set_autostart(autostart)
    if autostart:
        done.append("Autostart beim Anmelden")
    return done


# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #

def _mac_app() -> Path:
    return Path.home() / "Applications" / f"{APP_NAME}.app"


def _mac_agent() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "local.sicherheitscheck.plist"


def setup_macos(autostart: bool) -> list[str]:
    app = _mac_app()
    app.parent.mkdir(exist_ok=True)
    if app.exists() or app.is_symlink():
        shutil.rmtree(app) if app.is_dir() and not app.is_symlink() else app.unlink()
    macos = app / "Contents" / "MacOS"
    res = app / "Contents" / "Resources"
    macos.mkdir(parents=True)
    res.mkdir(parents=True)

    launcher = macos / APP_NAME
    launcher.write_text(
        "#!/bin/bash\n"
        f"cd {shlex.quote(str(HERE))}\n"
        f'exec {shlex.quote(str(venv_python()))} {shlex.quote(str(SCRIPT))} "$@"\n',
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    shutil.copy2(ICON_DIR / "sicherheitscheck.icns", res / f"{APP_NAME}.icns")
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": "local.sicherheitscheck.app",
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.1",
        "CFBundleVersion": "2",
        "LSMinimumSystemVersion": "11.0",
        "LSUIElement": True,          # kein Symbol im Dock, nur in der Menüleiste
        "NSHighResolutionCapable": True,
        "NSCameraUsageDescription": "Der Sicherheitscheck wertet die Webcam ausschliesslich lokal aus, "
                                    "um fremde Blicke auf deinen Bildschirm zu erkennen.",
    }
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump(info, fh)

    # altes Schreibtisch-Symbol entfernen
    link = Path.home() / "Desktop" / f"{APP_NAME}.app"
    if link.is_symlink():
        link.unlink()

    lsregister = ("/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                  "LaunchServices.framework/Support/lsregister")
    if os.path.exists(lsregister):
        subprocess.run([lsregister, "-f", str(app)], capture_output=True)
    done = ["Programme (Launchpad, Spotlight)"]
    set_autostart(autostart)
    if autostart:
        done.append("Autostart beim Anmelden")
    return done


# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #

def _desktop_dir() -> Path:
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip()
        if out and Path(out).is_dir() and Path(out) != Path.home():
            return Path(out)
    except Exception:
        pass
    for name in ("Desktop", "Schreibtisch", "Arbeitsfläche"):
        if (Path.home() / name).is_dir():
            return Path.home() / name
    return Path.home() / "Desktop"


def _linux_entry(args: str = "") -> str:
    def q(p) -> str:
        return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`") + '"'
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        f"Comment={DESCRIPTION}\n"
        f"Exec={q(venv_python())} {q(SCRIPT)}{(' ' + args) if args else ''}\n"
        f"Path={HERE}\n"
        f"Icon={ICON_DIR / 'sicherheitscheck.png'}\n"
        "Terminal=false\n"
        "Categories=Utility;Security;\n"
    )


def _linux_autostart_file() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "autostart" / "sicherheitscheck.desktop"


def setup_linux(autostart: bool) -> list[str]:
    apps = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    menu = apps / "sicherheitscheck.desktop"
    menu.write_text(_linux_entry(), encoding="utf-8")
    menu.chmod(0o755)
    old = _desktop_dir() / "sicherheitscheck.desktop"  # altes Schreibtisch-Symbol entfernen
    if old.is_file():
        old.unlink()
    done = ["Anwendungsmenü"]
    set_autostart(autostart)
    if autostart:
        done.append("Autostart beim Anmelden")
    return done


# --------------------------------------------------------------------------- #
# Autostart ein/aus (wird auch vom Programm selbst über das Menü im Infobereich genutzt)
# --------------------------------------------------------------------------- #

def autostart_supported() -> bool:
    return sys.platform.startswith("win") or sys.platform == "darwin" or sys.platform.startswith("linux")


def is_autostart_enabled() -> bool:
    if sys.platform.startswith("win"):
        return _win_startup_link().is_file()
    if sys.platform == "darwin":
        return _mac_agent().is_file()
    return _linux_autostart_file().is_file()


def set_autostart(enabled: bool) -> None:
    if sys.platform.startswith("win"):
        link = _win_startup_link()
        if enabled:
            link.parent.mkdir(parents=True, exist_ok=True)
            script = "$ErrorActionPreference = 'Stop'\n$ws = New-Object -ComObject WScript.Shell\n"
            script += _ps_shortcut(f"'{_ps(link.parent)}'", TRAY_ARG)
            _run_powershell(script)
        elif link.is_file():
            link.unlink()
        return

    if sys.platform == "darwin":
        agent = _mac_agent()
        if enabled:
            agent.parent.mkdir(parents=True, exist_ok=True)
            app = _mac_app()
            if app.is_dir():  # über die App starten, damit macOS den Kamerazugriff der App zuordnet
                args = ["/usr/bin/open", "-g", "-a", str(app), "--args", TRAY_ARG]
            else:
                args = [str(venv_python()), str(SCRIPT), TRAY_ARG]
            with open(agent, "wb") as fh:
                plistlib.dump({"Label": "local.sicherheitscheck", "ProgramArguments": args,
                               "RunAtLoad": True, "ProcessType": "Interactive"}, fh)
        elif agent.is_file():
            agent.unlink()
        return

    path = _linux_autostart_file()
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_linux_entry(TRAY_ARG) + "X-GNOME-Autostart-enabled=true\n", encoding="utf-8")
    elif path.is_file():
        path.unlink()


# --------------------------------------------------------------------------- #

def main() -> int:
    autostart = "--ohne-autostart" not in sys.argv
    try:
        if sys.platform.startswith("win"):
            where = setup_windows(autostart)
        elif sys.platform == "darwin":
            where = setup_macos(autostart)
        else:
            where = setup_linux(autostart)
    except Exception as exc:
        print(f"Die Einrichtung ist fehlgeschlagen: {exc}")
        return 1
    print(f"«{APP_NAME}» eingerichtet: " + ", ".join(where))
    return 0


if __name__ == "__main__":
    sys.exit(main())
