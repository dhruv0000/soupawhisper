# SoupaWhisper

This fork adds Wayland support, evdev global hotkeys, and related Linux/service fixes.

A simple push-to-talk voice dictation tool for Linux using faster-whisper. Hold a key to record, release to transcribe, and it automatically copies to clipboard and types into the active input.

## Requirements

- Python 3.10+
- Poetry
- Linux with ALSA audio
- X11, or Wayland with access to the `input` group for global hotkeys

## Supported Distros

- Ubuntu / Pop!_OS / Debian (apt)
- Fedora (dnf)
- Arch Linux (pacman)
- openSUSE (zypper)

## Installation

```bash
git clone https://github.com/ksred/soupawhisper.git
cd soupawhisper
chmod +x install.sh
./install.sh
```

The installer will:
1. Detect your package manager
2. Install system dependencies
3. Install Python dependencies via Poetry
4. Set up the config file
5. Optionally install as a systemd service

### Manual Installation

```bash
# Ubuntu/Debian
sudo apt install alsa-utils xclip xdotool libnotify-bin

# Fedora
sudo dnf install alsa-utils xclip xdotool libnotify

# Arch
sudo pacman -S alsa-utils xclip xdotool libnotify

# Then install Python deps
poetry install
```

### GPU Support (Optional)

For NVIDIA GPU acceleration, install cuDNN 9:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libcudnn9-cuda-12
```

Then edit `~/.config/soupawhisper/config.ini`:
```ini
device = cuda
compute_type = float16
```

For AMD GPU acceleration, install ROCm and a ROCm-enabled build of `ctranslate2`, then edit `~/.config/soupawhisper/config.ini`:
```ini
device = amd
compute_type = float16
```

`device = amd`, `device = rocm`, and `device = hip` are accepted aliases in SoupaWhisper. Internally, these map to the CTranslate2 GPU backend, which requires a ROCm-capable install on AMD hardware.

## Usage

```bash
poetry run python dictate.py
```

- Hold **F12** to record
- Release to transcribe → copies to clipboard and types into active input
- Press **Ctrl+C** to quit (when running manually)
- On media-key-first keyboards, you may need **Fn+F12** unless you switch the top row to function-key mode
- Hotkeys are best configured from `make debug-keys` output using exact `KEY_*` names in `.env`
- To inspect what key your system is actually sending, run `poetry run python dictate.py --debug-keys`

## Run as a systemd Service

The installer can set this up automatically. If you skipped it, run:

```bash
./install.sh  # Select 'y' when prompted for systemd
```

### Service Commands

```bash
systemctl --user start soupawhisper     # Start
systemctl --user stop soupawhisper      # Stop
systemctl --user restart soupawhisper   # Restart
systemctl --user status soupawhisper    # Status
journalctl --user -u soupawhisper -f    # View logs
```

## Configuration

Edit `~/.config/soupawhisper/config.ini`:

```ini
[whisper]
# Model size: tiny.en, base.en, small.en, medium.en, large-v3
model = base.en

# Device: cpu, auto, cuda/nvidia, or amd/rocm
device = cpu

# Compute type: int8 for CPU, float16 for GPU
compute_type = int8

[hotkey]
# Optional fallback when .env is not used. Prefer KEY_* names from --debug-keys.
key = f12

[behavior]
# Type text into active input field
auto_type = true

# Show desktop notification
notifications = true
```

Create the config directory and file if it doesn't exist:
```bash
mkdir -p ~/.config/soupawhisper
# ./ is '/path/to/soupawhisper/'
cp ./config.example.ini ~/.config/soupawhisper/config.ini
```

To override hotkeys from the repo checkout instead, create `.env` from `.env.example`:
```bash
cp .env.example .env
```

Example:
```dotenv
SOUPAWHISPER_KEYS=KEY_F12,KEY_LEFTCTRL+KEY_SPACE
```

When `.env` is present, `SOUPAWHISPER_KEYS` overrides `[hotkey] key` from `~/.config/soupawhisper/config.ini`.

## Troubleshooting

**No audio recording:**
```bash
# Check your input device
arecord -l

# Test recording
arecord -d 3 test.wav && aplay test.wav
```

**Permission issues with keyboard:**
```bash
sudo usermod -aG input $USER
# Then log out completely and back in before restarting the service
```

**Wayland notes:**
```bash
make debug-keys
```
Use this to find the exact key names your keyboard is sending, then paste them into `.env`.

Examples:
```dotenv
SOUPAWHISPER_KEYS=KEY_F12
SOUPAWHISPER_KEYS=KEY_LEFTCTRL+KEY_SPACE
SOUPAWHISPER_KEYS=KEY_F12,KEY_LEFTCTRL+KEY_SPACE
```

On Wayland, SoupaWhisper only watches keyboard events for the configured hotkey. It does not grab or replay your keyboard input, because partial grabs can leave mismatched key press/release state behind.

On Wayland, clipboard copy should still work, but `xdotool` auto-typing may not work in native Wayland apps.

**cuDNN errors with GPU:**
```
Unable to load any of {libcudnn_ops.so.9...}
```
Install cuDNN 9 (see GPU Support section above) or switch to CPU mode.

**AMD ROCm errors with GPU:**
If `device = amd` fails at startup, the most common cause is that `ctranslate2` is still using the default non-ROCm wheel or ROCm is not installed on the system. In that case, switch back to `device = cpu` until the ROCm stack is installed.

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~75MB | Fastest | Basic |
| base.en | ~150MB | Fast | Good |
| small.en | ~500MB | Medium | Better |
| medium.en | ~1.5GB | Slower | Great |
| large-v3 | ~3GB | Slowest | Best |

For dictation, `base.en` or `small.en` is usually the sweet spot.
