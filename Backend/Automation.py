import os
import subprocess
import webbrowser
import asyncio
import keyboard
import requests
from AppOpener import close, open as appopen
from dotenv import dotenv_values
from bs4 import BeautifulSoup
from pywhatkit import search, playonyt
import shutil
from pathlib import Path
import sys
import pyautogui
import datetime
import keyboard
import time
import pygetwindow as gw
# ---- Optional: resolve .lnk shortcuts ----
try:
    from win32com.client import Dispatch
    CAN_RESOLVE_LNK = True
except:
    CAN_RESOLVE_LNK = False

# Load env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_vars = dotenv_values(os.path.join(BASE_DIR, ".env"))
GroqAPIKey = env_vars.get("GroqAPIKey")

# ------------------------------
#  Helper Logging
# ------------------------------
def _log(msg):
    print(f"[Automation] {msg}")

# ------------------------------
#  App Opening Logic (NEW + FIXED)
# ------------------------------

COMMON_PATHS = [
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    os.path.expanduser(r"~\AppData\Local\Programs"),
]

def _try_appopener(app):
    try:
        appopen(app, match_closest=True, output=True, throw_error=True)
        _log(f"Opened using AppOpener: {app}")
        return True
    except Exception as e:
        _log(f"AppOpener failed: {e}")
        return False

def _try_shutil_which(app):
    exe = shutil.which(app) or shutil.which(app + ".exe")
    if exe:
        try:
            subprocess.Popen([exe])
            _log(f"Opened using PATH exe: {exe}")
            return True
        except Exception as e:
            _log(f"Failed to launch PATH exe: {e}")
    return False

def _scan_program_files(app):
    search_name = app.lower().replace(" ", "")
    for base in COMMON_PATHS:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".exe"):
                    fn = f.lower().replace(" ", "")
                    if search_name in fn:
                        exe_path = os.path.join(root, f)
                        try:
                            subprocess.Popen([exe_path])
                            _log(f"Opened (ProgramFiles scan): {exe_path}")
                            return True
                        except Exception:
                            pass
    return False

def _resolve_lnk(lnk):
    if not CAN_RESOLVE_LNK:
        return None
    try:
        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(lnk)
        return shortcut.Targetpath
    except:
        return None

def _open_startmenu_shortcut(app):
    search_name = app.lower().replace(" ", "")
    start_paths = [
        os.path.join(os.environ.get("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
        os.path.join(os.environ.get("ProgramData", ""), r"Microsoft\Windows\Start Menu\Programs"),
    ]
    for base in start_paths:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".lnk") and search_name in f.lower():
                    full = os.path.join(root, f)
                    target = _resolve_lnk(full)
                    if target:
                        subprocess.Popen([target])
                        _log(f"Opened via shortcut target: {target}")
                        return True
                    else:
                        os.startfile(full)
                        _log(f"Opened shortcut directly: {full}")
                        return True
    return False

def OpenApp(app):
    _log(f"Trying to open app: {app}")

    # 1. AppOpener
    if _try_appopener(app):
        return True

    # 2. PATH
    if _try_shutil_which(app):
        return True

    # 3. Program Files scan
    if _scan_program_files(app):
        return True

    # 4. Start Menu shortcuts
    if _open_startmenu_shortcut(app):
        return True

    # 5. Final fallback: open Microsoft Store search
    try:
        url = f"https://www.microsoft.com/en-us/search?q={app}"
        webbrowser.open(url)
        _log("Fallback: opening Microsoft Store search")
        return True
    except:
        pass

    _log("FAILED to open app.")
    return False

# ------------------------------
# Close App (Improved)
# ------------------------------

def CloseApp(app):
    _log(f"Trying to close: {app}")

    # special handling for Chrome
    if "chrome" in app.lower():
        try:
            subprocess.run(["taskkill", "/f", "/im", "chrome.exe"], check=True)
            _log("Force closed chrome.exe")
            return True
        except:
            pass

    # generic close
    try:
        close(app, match_closest=True, output=True, throw_error=True)
        _log(f"Closed using AppOpener: {app}")
        return True
    except Exception as e:
        _log(f"AppOpener close failed: {e}")
        return False
    app = app.lower().strip()

    # Special case for Windows Camera
    if "camera" in app:
        try:
            subprocess.run(["taskkill", "/f", "/im", "WindowsCamera.exe"], check=True)
            _log("Closed Windows Camera using taskkill")
            return True
        except Exception as e:
            _log(f"Failed to close Camera: {e}")
            return False
# ------------------------------
# Google, YouTube, System Commands
# ------------------------------

def GoogleSearch(topic):
    search(topic)
    return True

def YouTubeSearch(topic):
    url = f"https://www.youtube.com/results?search_query={topic}"
    webbrowser.open(url)
    return True

def PlayYoutube(query):
    try:
        playonyt(query)
        return True
    except:
        return False
def TakeScreenshot():
    try:
        os.makedirs("Data", exist_ok=True)
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join("Data", filename)

        img = pyautogui.screenshot()
        img.save(filepath)
        _log(f"Screenshot saved: {filepath}")
        return True

    except Exception as e:
        _log(f"Screenshot error: {e}")
        return False
def CaptureWindowsCameraPhoto():
    try:
        # find Camera window
        windows = gw.getWindowsWithTitle("Camera")
        if not windows:
            _log("Camera is not open. Opening...")
            OpenApp("camera")
            time.sleep(3)  # wait for camera to load
            windows = gw.getWindowsWithTitle("Camera")

        cam_window = windows[0]
        cam_window.activate()
        time.sleep(1)

        keyboard.press_and_release("space")   # press shutter
        _log("Clicked photo in Windows Camera")
        return True

    except Exception as e:
        _log(f"Photo capture failed: {e}")
        return False
def System(command):
    try:
        if command == "mute":
            keyboard.press_and_release("volume mute")
        elif command == "unmute":
            keyboard.press_and_release("volume mute")
        elif command == "volume up":
            keyboard.press_and_release("volume up")
        elif command == "volume down":
            keyboard.press_and_release("volume down")
        else:
            _log(f"Unknown system command: {command}")
            return False
        return True
    except Exception as e:
        _log(f"System command error: {e}")
        return False

# ------------------------------
# Translate + Execute
# ------------------------------

async def TranslateAndExecute(commands: list[str]):
    tasks = []

    for command in commands:
        command = command.strip().lower()
        _log(f"Processing command: {command}")

        if command.startswith("open "):
            app_name = command[5:].strip()
            tasks.append(asyncio.to_thread(OpenApp, app_name))

        elif command.startswith("close "):
            app_name = command[6:].strip()
            tasks.append(asyncio.to_thread(CloseApp, app_name))

        elif command.startswith("play "):
            query = command[5:].strip()
            tasks.append(asyncio.to_thread(PlayYoutube, query))

        elif command.startswith("google search "):
            query = command.replace("google search", "").strip()
            tasks.append(asyncio.to_thread(GoogleSearch, query))

        elif command.startswith("youtube search "):
            query = command.replace("youtube search", "").strip()
            tasks.append(asyncio.to_thread(YouTubeSearch, query))

        elif command.startswith("system "):
            syscmd = command.replace("system", "").strip()
            tasks.append(asyncio.to_thread(System, syscmd))
            
        elif command in ["take screenshot", "capture screen", "screenshot"]:
            tasks.append(asyncio.to_thread(TakeScreenshot))
        elif command in ["take photo", "click photo", "capture photo", "capture image"]:
            tasks.append(asyncio.to_thread(CaptureWindowsCameraPhoto))

        elif command.startswith("close camera"):
            tasks.append(asyncio.to_thread(CloseApp, "camera"))

        else:
            _log(f"No matching command: {command}")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            yield r
    else:
        _log("No valid tasks to execute")

# ------------------------------
# Automation Entry
# ------------------------------

async def Automation(commands: list[str]):
    _log(f"Starting Automation → {commands}")
    output = []
    async for res in TranslateAndExecute(commands):
        output.append(res)
    _log(f"Automation Finished → {output}")
    return True
