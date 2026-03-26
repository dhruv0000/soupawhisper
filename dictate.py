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

PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"
DEFAULT_CONFIG = {
    "model": "base.en",
    "device": "cpu",
    "compute_type": "int8",
    "key": "f12",
    "auto_type": "true",
    "notifications": "true",
}
DEPENDENCY_PACKAGES = {
    "arecord": "alsa-utils",
    "xclip": "xclip",
    "xdotool": "xdotool",
}


def load_env_file(env_path):
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def load_config():
    config = configparser.ConfigParser()

    if CONFIG_PATH.exists():
        config.read(CONFIG_PATH)

    env_hotkeys = (
        os.environ.get("SOUPAWHISPER_KEYS")
        or os.environ.get("SOUPAWHISPER_HOTKEYS")
    )

    return {
        "model": config.get("whisper", "model", fallback=DEFAULT_CONFIG["model"]),
        "device": config.get("whisper", "device", fallback=DEFAULT_CONFIG["device"]),
        "compute_type": config.get("whisper", "compute_type", fallback=DEFAULT_CONFIG["compute_type"]),
        "key": env_hotkeys or config.get("hotkey", "key", fallback=DEFAULT_CONFIG["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
    }


load_env_file(ENV_PATH)
CONFIG = load_config()
SESSION_TYPE = os.environ.get("XDG_SESSION_TYPE", "unknown").lower() or "unknown"
PYNPUT_KEYBOARD = None
HOTKEY_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "option": "alt",
    "super": "meta",
    "win": "meta",
    "windows": "meta",
    "cmd": "meta",
    "command": "meta",
    "escape": "esc",
    "return": "enter",
    "spacebar": "space",
    "scrolllock": "scroll_lock",
    "capslock": "caps_lock",
    "numlock": "num_lock",
    "pageup": "page_up",
    "pagedown": "page_down",
    "pgup": "page_up",
    "pgdn": "page_down",
    "leftctrl": "left_ctrl",
    "leftcontrol": "left_ctrl",
    "ctrl_l": "left_ctrl",
    "rightctrl": "right_ctrl",
    "rightcontrol": "right_ctrl",
    "ctrl_r": "right_ctrl",
    "leftalt": "left_alt",
    "alt_l": "left_alt",
    "rightalt": "right_alt",
    "alt_r": "right_alt",
    "altgr": "right_alt",
    "alt_gr": "right_alt",
    "leftshift": "left_shift",
    "shift_l": "left_shift",
    "rightshift": "right_shift",
    "shift_r": "right_shift",
    "leftmeta": "left_meta",
    "meta_l": "left_meta",
    "super_l": "left_meta",
    "win_l": "left_meta",
    "cmd_l": "left_meta",
    "rightmeta": "right_meta",
    "meta_r": "right_meta",
    "super_r": "right_meta",
    "win_r": "right_meta",
    "cmd_r": "right_meta",
}
HOTKEY_MATCH_ALIASES = {
    "ctrl": frozenset({"ctrl", "left_ctrl", "right_ctrl"}),
    "alt": frozenset({"alt", "left_alt", "right_alt"}),
    "shift": frozenset({"shift", "left_shift", "right_shift"}),
    "meta": frozenset({"meta", "left_meta", "right_meta"}),
}
EVDEV_KEY_ALIASES = {
    "esc": "KEY_ESC",
    "enter": "KEY_ENTER",
    "page_up": "KEY_PAGEUP",
    "page_down": "KEY_PAGEDOWN",
    "scroll_lock": "KEY_SCROLLLOCK",
    "caps_lock": "KEY_CAPSLOCK",
    "num_lock": "KEY_NUMLOCK",
    "left_ctrl": "KEY_LEFTCTRL",
    "right_ctrl": "KEY_RIGHTCTRL",
    "left_alt": "KEY_LEFTALT",
    "right_alt": "KEY_RIGHTALT",
    "left_shift": "KEY_LEFTSHIFT",
    "right_shift": "KEY_RIGHTSHIFT",
    "left_meta": "KEY_LEFTMETA",
    "right_meta": "KEY_RIGHTMETA",
}
EVDEV_GROUP_CODES = {
    "ctrl": frozenset({ecodes.KEY_LEFTCTRL, ecodes.KEY_RIGHTCTRL}),
    "alt": frozenset({ecodes.KEY_LEFTALT, ecodes.KEY_RIGHTALT}),
    "shift": frozenset({ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}),
    "meta": frozenset({ecodes.KEY_LEFTMETA, ecodes.KEY_RIGHTMETA}),
}


def normalize_key_name(key_name):
    return key_name.strip().lower().replace("-", "_").replace(" ", "_")


def canonicalize_hotkey_part(key_name):
    return HOTKEY_ALIASES.get(normalize_key_name(key_name), normalize_key_name(key_name))


def parse_hotkey(key_name):
    return tuple(
        canonicalize_hotkey_part(part)
        for part in key_name.split("+")
        if normalize_key_name(part)
    )


def parse_hotkeys(key_names):
    return tuple(
        hotkey
        for hotkey in (parse_hotkey(key_name) for key_name in key_names.split(","))
        if hotkey
    )


def get_pynput_keyboard():
    global PYNPUT_KEYBOARD

    if PYNPUT_KEYBOARD is not None:
        return PYNPUT_KEYBOARD

    from pynput import keyboard

    PYNPUT_KEYBOARD = keyboard
    return PYNPUT_KEYBOARD


def get_pynput_key_name(key):
    key_name = getattr(key, "name", None)
    if key_name:
        return canonicalize_hotkey_part(key_name)

    char = getattr(key, "char", None)
    if char == " ":
        return "space"
    if char:
        return canonicalize_hotkey_part(char)

    return None


def get_evdev_hotkey_codes(key_name):
    """Map a hotkey part to one or more evdev keycodes."""
    key_name = canonicalize_hotkey_part(key_name)

    if key_name in EVDEV_GROUP_CODES:
        return EVDEV_GROUP_CODES[key_name]

    candidate = EVDEV_KEY_ALIASES.get(key_name, f"KEY_{key_name.upper()}")
    keycode = getattr(ecodes, candidate, None)
    if keycode is None:
        return None

    return frozenset({keycode})


def build_evdev_hotkeys(hotkeys):
    supported_hotkeys = []
    unsupported_hotkeys = []

    for hotkey in hotkeys:
        hotkey_codes = []
        unsupported_parts = []

        for part in hotkey:
            keycodes = get_evdev_hotkey_codes(part)
            if keycodes is None:
                unsupported_parts.append(part)
                continue
            hotkey_codes.append(keycodes)

        if unsupported_parts:
            unsupported_hotkeys.append((hotkey, tuple(unsupported_parts)))
        else:
            supported_hotkeys.append(tuple(hotkey_codes))

    return tuple(supported_hotkeys), tuple(unsupported_hotkeys)


def get_evdev_key_name(code):
    key_name = ecodes.KEY.get(code)
    if isinstance(key_name, list):
        key_name = key_name[0]
    if not key_name:
        return None
    if key_name.startswith("KEY_"):
        key_name = key_name[4:]
    return canonicalize_hotkey_part(key_name)


def get_hotkey_match_names(key_part):
    return HOTKEY_MATCH_ALIASES.get(key_part, frozenset({key_part}))


def format_hotkey_part(key_part):
    if len(key_part) == 1 and key_part.isalnum():
        return key_part.upper()
    return key_part.replace("_", " ").title()


def format_hotkey_name(key_parts):
    return "+".join(format_hotkey_part(part) for part in key_parts)


def format_hotkey_names(hotkeys):
    return ", ".join(format_hotkey_name(hotkey) for hotkey in hotkeys)


HOTKEYS = parse_hotkeys(CONFIG["key"])
HOTKEY_MATCH_GROUPS = tuple(
    tuple(get_hotkey_match_names(part) for part in hotkey)
    for hotkey in HOTKEYS
)
EVDEV_HOTKEY_GROUPS, EVDEV_UNSUPPORTED_HOTKEYS = build_evdev_hotkeys(HOTKEYS)
HOTKEY_LABEL = format_hotkey_names(HOTKEYS) if HOTKEYS else CONFIG["key"].strip()
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


def supports_evdev_hotkey(capabilities):
    capability_set = {
        capability[0] if isinstance(capability, tuple) else capability
        for capability in capabilities
    }
    return any(
        all(bool(part_codes & capability_set) for part_codes in hotkey_codes)
        for hotkey_codes in EVDEV_HOTKEY_GROUPS
    )


def iter_key_events(devices, hotkey_only=False, on_device_removed=None):
    while True:
        ready, _, _ = select(list(devices.values()), [], [], 2.0)
        if not ready:
            refresh_keyboard_devices(devices, hotkey_only=hotkey_only)
            if hotkey_only and not devices:
                time.sleep(0.5)
            continue

        for device in ready:
            try:
                events = device.read()
            except OSError as e:
                if e.errno == errno.ENODEV:
                    devices.pop(device.path, None)
                    if on_device_removed is not None:
                        on_device_removed()
                    refresh_keyboard_devices(devices, hotkey_only=hotkey_only)
                    continue
                raise

            for event in events:
                if event.type == ecodes.EV_KEY:
                    yield device, event


def get_keyboard_devices(hotkey_only=False):
    devices = {}

    for path in sorted(list_devices()):
        try:
            device = InputDevice(path)
        except OSError:
            continue

        capabilities = device.capabilities().get(ecodes.EV_KEY, [])
        if capabilities and (not hotkey_only or supports_evdev_hotkey(capabilities)):
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
        if not capabilities or (hotkey_only and not supports_evdev_hotkey(capabilities)):
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
        for device, event in iter_key_events(devices):
            key_name = ecodes.KEY.get(event.code, f"KEY_{event.code}")
            action = {0: "up", 1: "down", 2: "hold"}.get(event.value, str(event.value))
            print(f"{device.name}: {key_name} {action}", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        close_keyboard_devices(devices)


class Dictation:
    def __init__(self):
        self.recording = False
        self.processing = False
        self.record_process = None
        self.temp_file = None
        self.model = None
        self.model_loaded = threading.Event()
        self.model_error = None
        self.state_lock = threading.Lock()
        self.pynput_pressed_keys = set()
        self.evdev_pressed_keys = set()
        self.hotkey_active = False

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

    def copy_text_to_clipboard(self, text):
        process = subprocess.Popen(
            ["xclip", "-selection", "clipboard"],
            stdin=subprocess.PIPE,
        )
        process.communicate(input=text.encode())

    def auto_type_text(self, text):
        if not AUTO_TYPE:
            return False

        result = subprocess.run(
            ["xdotool", "type", "--clearmodifiers", text],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return False

        error_text = (result.stderr or result.stdout or "xdotool failed").strip()
        print(f"Auto-type failed: {error_text}")
        return True

    def update_pressed_keys(self, pressed_keys, key_name, is_pressed):
        if is_pressed:
            pressed_keys.add(key_name)
        else:
            pressed_keys.discard(key_name)
        self.update_hotkey_state(pressed_keys)

    def start_recording(self):
        with self.state_lock:
            if self.recording or self.processing or self.model_error:
                return

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()

        try:
            record_process = subprocess.Popen(
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
                    temp_file.name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            print(f"Failed to start recording: {e}")
            self.notify("Error", "Unable to start recording", "dialog-error", 3000)
            return

        with self.state_lock:
            self.recording = True
            self.record_process = record_process
            self.temp_file = temp_file.name

        print("Recording...")
        release_hint = HOTKEY_LABEL if len(HOTKEYS) == 1 else "the hotkey"
        self.notify("Recording...", f"Release {release_hint} when done", "audio-input-microphone", 30000)

    def stop_recording(self):
        with self.state_lock:
            if not self.recording or self.processing:
                return

            self.recording = False
            self.processing = True
            record_process = self.record_process
            temp_file = self.temp_file
            self.record_process = None
            self.temp_file = None

        threading.Thread(
            target=self._finish_recording,
            args=(record_process, temp_file),
            daemon=True,
        ).start()

    def _finish_recording(self, record_process, temp_file):
        try:
            if record_process:
                record_process.terminate()
                try:
                    record_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record_process.kill()
                    record_process.wait()

            if not temp_file or not os.path.exists(temp_file):
                print("Recording file missing")
                self.notify("Error", "Recording file missing", "dialog-error", 3000)
                return

            print("Transcribing...")
            self.notify("Transcribing...", "Processing your speech", "emblem-synchronizing", 30000)

            self.model_loaded.wait()

            if self.model_error:
                print("Cannot transcribe: model failed to load")
                self.notify("Error", "Model failed to load", "dialog-error", 3000)
                return

            segments, _info = self.model.transcribe(
                temp_file,
                beam_size=5,
                vad_filter=True,
            )

            text = " ".join(segment.text.strip() for segment in segments)

            if text:
                self.copy_text_to_clipboard(text)
                auto_type_failed = self.auto_type_text(text)

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
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
            with self.state_lock:
                self.processing = False

    def update_hotkey_state(self, pressed_keys):
        hotkey_pressed = any(
            all(bool(match_names & pressed_keys) for match_names in hotkey_match_parts)
            for hotkey_match_parts in HOTKEY_MATCH_GROUPS
        )

        if hotkey_pressed and not self.hotkey_active:
            self.hotkey_active = True
            self.start_recording()
        elif not hotkey_pressed and self.hotkey_active:
            self.hotkey_active = False
            self.stop_recording()

    def on_press(self, key):
        key_name = get_pynput_key_name(key)
        if key_name is None:
            return
        self.update_pressed_keys(self.pynput_pressed_keys, key_name, True)

    def on_release(self, key):
        key_name = get_pynput_key_name(key)
        if key_name is None:
            return
        self.update_pressed_keys(self.pynput_pressed_keys, key_name, False)

    def stop(self):
        print("\nExiting...")
        os._exit(0)

    def run_pynput_listener(self):
        keyboard = get_pynput_keyboard()
        print("Using pynput global hotkey listener.")
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def run_evdev_listener(self):
        if not HOTKEYS:
            print(f"Invalid hotkey: {CONFIG['key']}")
            print("Set [hotkey] key to a key name such as f12, a combo like ctrl+space, or a list like f12, ctrl+space.")
            sys.exit(1)

        if not EVDEV_HOTKEY_GROUPS:
            print(f"Unsupported evdev hotkey: {CONFIG['key']}")
            for hotkey, unsupported_parts in EVDEV_UNSUPPORTED_HOTKEYS:
                print(f"Unsupported part(s) for {format_hotkey_name(hotkey)}: {', '.join(unsupported_parts)}")
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
        print("Keyboard devices are only watched on Wayland; they are not grabbed.")
        if EVDEV_UNSUPPORTED_HOTKEYS:
            for hotkey, unsupported_parts in EVDEV_UNSUPPORTED_HOTKEYS:
                print(f"Ignoring unsupported hotkey [{format_hotkey_name(hotkey)}]: {', '.join(unsupported_parts)}")
        if any(len(part) == 1 and part.isalnum() for hotkey in HOTKEYS for part in hotkey):
            print("Wayland note: character-key hotkeys can still reach the focused app.")
            print("Dedicated keys like F12, Scroll Lock, or Pause are less likely to conflict.")
        if AUTO_TYPE:
            print("Wayland note: clipboard copy should work, but xdotool auto-typing may fail in native Wayland apps.")

        try:
            for _device, event in iter_key_events(
                devices,
                hotkey_only=True,
                on_device_removed=self.handle_evdev_device_removed,
            ):
                key_name = get_evdev_key_name(event.code)
                if key_name is None:
                    continue

                self.update_pressed_keys(
                    self.evdev_pressed_keys,
                    key_name,
                    event.value in (1, 2),
                )
        finally:
            close_keyboard_devices(devices)

    def handle_evdev_device_removed(self):
        self.evdev_pressed_keys.clear()
        self.update_hotkey_state(self.evdev_pressed_keys)

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
            missing.append((cmd, DEPENDENCY_PACKAGES[cmd]))

    if AUTO_TYPE and shutil.which("xdotool") is None:
        missing.append(("xdotool", DEPENDENCY_PACKAGES["xdotool"]))

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
