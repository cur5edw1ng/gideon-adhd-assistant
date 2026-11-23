"""
Gideon - Native Windows Desktop Launcher (Level 3.5)

Opens Gideon in its own desktop window (pywebview + Microsoft WebView2), so you
never need a browser tab.

How it works:
  1. A background thread starts the Streamlit server on a free 127.0.0.1 port.
     It runs as a separate hidden process, because Streamlit installs signal
     handlers that only work in a program's main thread, and pywebview needs
     the main thread for the window.
  2. The window opens instantly with a dark "Starting" screen, then loads Gideon
     as soon as the server answers its health check.
  3. When you close the window (or this launcher exits for any reason), the
     whole Streamlit process tree is killed, so no stray Python processes stay
     running.

Run with:   python desktop.py
No console: pythonw.exe desktop.py
Options:    --frameless   no title bar (drag from anywhere, close with Alt+F4)
            --debug       open the WebView2 developer tools

Server output is written to gideon_desktop.log next to this file.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import html
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
APP_SCRIPT = BASE_DIR / "app.py"
LOG_PATH = BASE_DIR / "gideon_desktop.log"
STORAGE_DIR = BASE_DIR / ".webview"

WINDOW_TITLE = "GIDEON // Time Vault"
WINDOW_SIZE = (1360, 840)
MIN_SIZE = (1100, 700)
BACKGROUND = "#05080d"
HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 90

SPLASH_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body { height: 100%; margin: 0; background: #05080d; color: #7f93a6;
    font-family: "Bahnschrift", "Segoe UI", system-ui, sans-serif; }
  .wrap { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .orb { width: 96px; height: 96px; border-radius: 50%;
    background: radial-gradient(circle at 38% 32%, #e9fdff 0%, #3de9ff 28%, #2f7bff 62%, rgba(5,8,13,0) 100%);
    box-shadow: 0 0 34px rgba(61,233,255,.55), 0 0 80px rgba(47,123,255,.35);
    animation: breathe 2.4s ease-in-out infinite; }
  .title { margin-top: 28px; color: #3de9ff; letter-spacing: .18em; font-size: 20px;
    text-shadow: 0 0 14px rgba(61,233,255,.55); }
  .sub { margin-top: 8px; font-size: 13px; letter-spacing: .06em; }
  @keyframes breathe { 0%,100% { transform: scale(.94); opacity: .85; } 50% { transform: scale(1.06); opacity: 1; } }
  @media (prefers-reduced-motion: reduce) { .orb { animation: none; } }
</style></head><body><div class="wrap">
  <div class="orb"></div>
  <div class="title">GIDEON // TIME VAULT</div>
  <div class="sub">Starting local systems...</div>
</div></body></html>"""


def log_line(message: str) -> None:
    """Append a timestamped line to the launcher log (works even without a console)."""
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n")
    except OSError:
        pass


def error_page(message: str) -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8"></head>'
        '<body style="margin:0;background:#05080d;color:#d7e6f2;font-family:Segoe UI,system-ui,sans-serif;">'
        '<div style="max-width:640px;margin:14vh auto;padding:0 24px;">'
        '<h2 style="color:#ff5d73;letter-spacing:.06em;">Gideon could not start</h2>'
        f"<p>{html.escape(message)}</p>"
        f'<p style="color:#7f93a6;">Details are in {html.escape(str(LOG_PATH))}</p>'
        "</div></body></html>"
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return int(probe.getsockname()[1])


def python_executable() -> str:
    """Use python.exe (not pythonw.exe) for the server so its output can be redirected cleanly."""
    executable = Path(sys.executable)
    if executable.name.lower() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.exists():
            return str(candidate)
    return str(executable)


class StreamlitServer:
    """Runs `streamlit run app.py` as a hidden child process and can stop its whole process tree."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.stopped = False
        self._log_handle = None
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.port}"

    def start(self) -> None:
        with self._lock:
            if self.process is not None or self.stopped:
                return
            command = [
                python_executable(),
                "-m",
                "streamlit",
                "run",
                str(APP_SCRIPT),
                "--server.address",
                HOST,
                "--server.port",
                str(self.port),
                "--server.headless",
                "true",
                "--server.fileWatcherType",
                "none",
                "--browser.gatherUsageStats",
                "false",
                "--client.toolbarMode",
                "minimal",
            ]
            self._log_handle = open(LOG_PATH, "ab")
            self._log_handle.write(f"\n--- Streamlit starting on port {self.port} ---\n".encode("utf-8"))
            self._log_handle.flush()
            flags = 0
            if os.name == "nt":
                flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        log_line(f"Streamlit process {self.process.pid} started on {self.url}")

    def wait_until_ready(self, timeout: float) -> None:
        """Block until the server answers its health check. Raises RuntimeError on failure."""
        health_url = f"{self.url}/_stcore/health"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.stopped:
                raise RuntimeError("Gideon was closed while it was starting.")
            process = self.process
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"The Streamlit server stopped unexpectedly (exit code {process.returncode}). "
                    "Check that you ran the install commands inside the C:\\GIDEON virtual environment."
                )
            try:
                with self._opener.open(health_url, timeout=1.5) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.25)
        raise RuntimeError(f"Gideon did not start within {int(timeout)} seconds.")

    def stop(self) -> None:
        """Kill the server and everything it spawned. Safe to call more than once."""
        self.stopped = True
        process = self.process
        if process is not None and process.poll() is None:
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        capture_output=True,
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    process.terminate()
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            try:
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            log_line(f"Streamlit process {process.pid} stopped")
        handle = self._log_handle
        self._log_handle = None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


def apply_dark_title_bar(window) -> None:
    """Switch the native Windows title bar to dark mode (Windows 10 2004+ and Windows 11)."""
    if os.name != "nt":
        return
    try:
        handle = int(window.native.Handle.ToInt64())
        value = ctypes.c_int(1)
        for attribute in (20, 19):  # 20 = current, 19 = older Windows 10 builds
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(handle), attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
            if result == 0:
                return
    except Exception as exc:
        log_line(f"Dark title bar not applied: {exc}")


def configure_webview2() -> None:
    """Chromium switches for the embedded browser. Must be set before the window is created.

    --autoplay-policy: lets Gideon's spoken replies play without an extra click.
    --use-fake-ui-for-media-stream: grants microphone access automatically instead of asking
      every time. Gideon only ever loads its own page from 127.0.0.1, so this is safe here.
    """
    flags = "--autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream"
    existing = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (existing + " " + flags).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Gideon desktop launcher")
    parser.add_argument("--frameless", action="store_true", help="hide the title bar (drag anywhere, close with Alt+F4)")
    parser.add_argument("--debug", action="store_true", help="open the WebView2 developer tools")
    args = parser.parse_args()

    try:
        import webview
    except ImportError:
        message = "pywebview is not installed. Run: pip install pywebview"
        log_line(message)
        print(message)
        return 1

    configure_webview2()
    STORAGE_DIR.mkdir(exist_ok=True)

    server = StreamlitServer(find_free_port())
    atexit.register(server.stop)

    # Start the server in a background thread right away so it boots while the window opens.
    starter = threading.Thread(target=server.start, name="gideon-server-start", daemon=True)
    starter.start()

    window = webview.create_window(
        WINDOW_TITLE,
        html=SPLASH_HTML,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=MIN_SIZE,
        background_color=BACKGROUND,
        text_select=True,
        frameless=args.frameless,
        easy_drag=args.frameless,
    )
    window.events.closed += server.stop
    window.events.shown += lambda: apply_dark_title_bar(window)

    def boot(win) -> None:
        """Runs once the window exists: wait for Streamlit, then load it into the window."""
        try:
            starter.join()
            server.wait_until_ready(STARTUP_TIMEOUT_SECONDS)
            apply_dark_title_bar(win)
            win.load_url(server.url)
            log_line(f"Gideon loaded at {server.url}")
        except Exception as exc:
            log_line(f"Startup failed: {exc}\n{traceback.format_exc()}")
            if not server.stopped:
                win.load_html(error_page(str(exc)))

    try:
        webview.start(
            boot,
            (window,),
            gui="edgechromium",
            private_mode=False,  # keep the microphone permission and settings between launches
            storage_path=str(STORAGE_DIR),
            debug=args.debug,
        )
    except Exception as exc:
        log_line(f"Window failed: {exc}\n{traceback.format_exc()}")
        print(
            "The desktop window could not open. Make sure the Microsoft Edge WebView2 Runtime is installed "
            "(it ships with Windows 11 and current Windows 10). Details: " + str(exc)
        )
        return 1
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log_line("Launcher crashed:\n" + traceback.format_exc())
        raise
