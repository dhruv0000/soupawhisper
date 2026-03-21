#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
"""

import argparse
import configparser
import errno
import grp
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from select import select

from evdev import InputDevice, ecodes, list_devices
from faster_whisper import WhisperModel

__version__ = "0.1.0"

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"


def load_config():
    config = configparser.ConfigParser()

    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": "f12",
        "auto_type": "true",
        "notifications": "true",
    }

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    return {
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": config.get("whisper", "device", fallback=defaults["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


CONFIG = load_config()
SESSION_TYPE = os.environ.get("XDG_SESSION_TYPE", "unknown").lower() or "unknown"
PYNPUT_KEYBOARD = None


def normalize_key_name(key_name):
    return key_name.strip().lower().replace("-", "_").replace(" ", "_")


def get_pynput_hotkey(key_name):
    """Map key name to a pynput key."""
    keyboard = get_pynput_keyboard()
    key_name = normalize_key_name(key_name)
    if hasattr(keyboard.Key, key_name):
        return getattr(keyboard.Key, key_name)
    if len(key_name) == 1:
        return keyboard.KeyCode.from_char(key_name)

    print(f"Unknown pynput key: {key_name}, defaulting to f12")
    return keyboard.Key.f12


def get_pynput_keyboard():
    global PYNPUT_KEYBOARD

    if PYNPUT_KEYBOARD is not None:
        return PYNPUT_KEYBOARD

    from pynput import keyboard

    PYNPUT_KEYBOARD = keyboard
    return PYNPUT_KEYBOARD


def get_evdev_hotkey(key_name):
    """Map key name to an evdev keycode."""
    key_name = normalize_key_name(key_name)

    if key_name.startswith("key_"):
        candidate = key_name.upper()
    elif len(key_name) == 1 and key_name.isalnum():
        candidate = f"KEY_{key_name.upper()}"
    else:
        candidate = f"KEY_{key_name.upper()}"

    return getattr(ecodes, candidate, None)


def format_hotkey_name(key_name):
    key_name = normalize_key_name(key_name)
    return key_name.upper() if len(key_name) == 1 else key_name


HOTKEY = None if SESSION_TYPE == "wayland" else get_pynput_hotkey(CONFIG["key"])
EVDEV_HOTKEY = get_evdev_hotkey(CONFIG["key"])
HOTKEY_LABEL = format_hotkey_name(CONFIG["key"])
MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
AUTO_TYPE = CONFIG["auto_type"]
NOTIFICATIONS = CONFIG["notifications"]


def user_in_current_group(group_name):
    group_ids = set(os.getgroups())
    return any(grp.getgrgid(gid).gr_name == group_name for gid in group_ids)


def user_is_listed_in_group(group_name):
    try:
        group = grp.getgrnam(group_name)
    except KeyError:
        return False
    return os.environ.get("USER", "") in group.gr_mem


def get_keyboard_devices(hotkey_only=False):
    devices = {}

    for path in sorted(list_devices()):
        try:
            device = InputDevice(path)
        except OSError:
            continue

        capabilities = device.capabilities().get(ecodes.EV_KEY, [])
        if capabilities and (not hotkey_only or EVDEV_HOTKEY in capabilities):
            devices[path] = device
        else:
            device.close()

    return devices


def refresh_keyboard_devices(devices, hotkey_only=False):
    current_paths = set()

    for path in sorted(list_devices()):
        try:
            device = InputDevice(path)
        except OSError:
            continue

        capabilities = device.capabilities().get(ecodes.EV_KEY, [])
        if not capabilities or (hotkey_only and EVDEV_HOTKEY not in capabilities):
            device.close()
            continue

        current_paths.add(path)
        if path in devices:
            device.close()
        else:
            devices[path] = device

    for path in list(devices):
        if path not in current_paths:
            try:
                devices[path].close()
            except OSError:
                pass
            devices.pop(path, None)

    return devices


def close_keyboard_devices(devices):
    for device in devices.values():
        try:
            device.close()
        except OSError:
            pass


def debug_keys():
    devices = get_keyboard_devices()
    if not devices:
        print("No readable keyboard devices found.")
        print("If you just added yourself to the 'input' group, log out and back in first.")
        sys.exit(1)

    print("Watching global key events from evdev. Press Ctrl+C to quit.")
    for device in devices.values():
        print(f"  {device.path}: {device.name}")

    try:
        while True:
            ready, _, _ = select(list(devices.values()), [], [], 2.0)
            if not ready:
                refresh_keyboard_devices(devices)
                continue
            for device in ready:
                try:
                    events = device.read()
                except OSError as e:
                    if e.errno == errno.ENODEV:
                        devices.pop(device.path, None)
                        refresh_keyboard_devices(devices)
                        continue
                    raise

                for event in events:
                    if event.type != ecodes.EV_KEY:
                        continue
                    key_name = ecodes.KEY.get(event.code, f"KEY_{event.code}")
                    action = {0: "up", 1: "down", 2: "hold"}.get(event.value, str(event.value))
                    print(f"{device.name}: {key_name} {action}", flush=True)
    finally:
        close_keyboard_devices(devices)


class Dictation:
    def __init__(self):
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True

        print(f"Loading Whisper model ({MODEL_SIZE})...")
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        try:
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            self.model_loaded.set()
            print("Model loaded. Ready for dictation!")
            print(f"Hold [{HOTKEY_LABEL}] to record, release to transcribe.")
            print("Press Ctrl+C to quit.")
        except Exception as e:
            self.model_error = str(e)
            self.model_loaded.set()
            print(f"Failed to load model: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                print("Hint: Try setting device = cpu in your config, or install cuDNN.")

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification."""
        if not NOTIFICATIONS:
            return
        subprocess.run(
            [
                "notify-send",
                "-a", "SoupaWhisper",
                "-i", icon,
                "-t", str(timeout),
                "-h", "string:x-canonical-private-synchronous:soupawhisper",
                title,
                message,
            ],
            capture_output=True,
        )

    def start_recording(self):
        if self.recording or self.model_error:
            return

        self.recording = True
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()

        self.record_process = subprocess.Popen(
            [
                "arecord",
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-t",
                "wav",
                self.temp_file.name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("Recording...")
        self.notify("Recording...", f"Release {HOTKEY_LABEL.upper()} when done", "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            return

        self.recording = False

        if self.record_process:
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        print("Transcribing...")
        self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 30000)

        self.model_loaded.wait()

        if self.model_error:
            print("Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            return

        try:
            segments, _info = self.model.transcribe(
                self.temp_file.name,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                process = subprocess.Popen(
                    ["xclip", "-selection", "clipboard"],
                    stdin=subprocess.PIPE,
                )
                process.communicate(input=text.encode())

                auto_type_failed = False
                if AUTO_TYPE:
                    result = subprocess.run(
                        ["xdotool", "type", "--clearmodifiers", text],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        auto_type_failed = True
                        error_text = (result.stderr or result.stdout or "xdotool failed").strip()
                        print(f"Auto-type failed: {error_text}")

                print(f"Copied: {text}")
                if auto_type_failed:
                    self.notify("Copied to clipboard", "Auto-type failed; paste manually", "dialog-warning", 3500)
                else:
                    self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                print("No speech detected")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            print(f"Error: {e}")
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)

    def on_press(self, key):
        if key == HOTKEY:
            self.start_recording()

    def on_release(self, key):
        if key == HOTKEY:
            self.stop_recording()

    def stop(self):
        print("\nExiting...")
        self.running = False
        os._exit(0)

    def run_pynput_listener(self):
        keyboard = get_pynput_keyboard()
        print("Using pynput global hotkey listener.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def run_evdev_listener(self):
        if EVDEV_HOTKEY is None:
            print(f"Unsupported evdev hotkey: {CONFIG['key']}")
            print("Use: poetry run python dictate.py --debug-keys")
            sys.exit(1)

        devices = get_keyboard_devices(hotkey_only=True)
        if not devices:
            print("No readable keyboard devices found for evdev.")
            print("Wayland requires access to /dev/input/event* for global hotkeys.")
            print("Run: sudo usermod -aG input $USER")
            if not user_in_current_group("input") and user_is_listed_in_group("input"):
                print("The 'input' group was added, but this login session does not have it yet.")
                print("Log out completely, log back in, then restart the service.")
            else:
                print("After adding the group, log out completely and log back in.")
            sys.exit(1)

        print(f"Wayland detected. Using evdev global hotkey listener for [{HOTKEY_LABEL}].")
        print("Keyboard devices are only being watched, not grabbed.")
        if AUTO_TYPE:
            print("Wayland note: clipboard copy should work, but xdotool auto-typing may fail in native Wayland apps.")

        try:
            while True:
                ready, _, _ = select(list(devices.values()), [], [], 2.0)
                if not ready:
                    refresh_keyboard_devices(devices, hotkey_only=True)
                    if not devices:
                        time.sleep(0.5)
                    continue
                for device in ready:
                    try:
                        events = device.read()
                    except OSError as e:
                        if e.errno == errno.ENODEV:
                            devices.pop(device.path, None)
                            refresh_keyboard_devices(devices, hotkey_only=True)
                            continue
                        raise

                    for event in events:
                        if event.type == ecodes.EV_KEY and event.code == EVDEV_HOTKEY:
                            if event.value == 1:
                                self.start_recording()
                            elif event.value == 0:
                                self.stop_recording()
        finally:
            close_keyboard_devices(devices)

    def run(self):
        if SESSION_TYPE == "wayland":
            self.run_evdev_listener()
            return

        self.run_pynput_listener()


def check_dependencies():
    """Check that required system commands are available."""
    missing = []

    for cmd in ["arecord", "xclip"]:
        if shutil.which(cmd) is None:
            pkg = "alsa-utils" if cmd == "arecord" else cmd
            missing.append((cmd, pkg))

    if AUTO_TYPE and shutil.which("xdotool") is None:
        missing.append(("xdotool", "xdotool"))

    if missing:
        print("Missing dependencies:")
        for cmd, pkg in missing:
            print(f"  {cmd} - install with: sudo apt install {pkg}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="SoupaWhisper - Push-to-talk voice dictation")
    parser.add_argument(
        "--debug-keys",
        action="store_true",
        help="Print global key events detected through evdev and exit.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"SoupaWhisper {__version__}",
    )
    args = parser.parse_args()

    print(f"SoupaWhisper v{__version__}")
    print(f"Config: {CONFIG_PATH}")
    print(f"Session: {SESSION_TYPE}")

    if args.debug_keys:
        debug_keys()
        return

    check_dependencies()

    dictation = Dictation()

    def handle_sigint(_sig, _frame):
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)

    dictation.run()


if __name__ == "__main__":
    main()
