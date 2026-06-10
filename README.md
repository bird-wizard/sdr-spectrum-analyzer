# Unraveling the Radio Spectrum

**Real-Time RF Environment Analyzer**
WES 207 Capstone | Master of Advanced Studies in Wireless Embedded Systems | UC San Diego | Spring 2026
Author: Mario Infante

---

## Overview

A spectrum analyzer that uses a USB radio dongle from 25 MHz all the way up to 1.75 GHz. It picks up active signals, identifies what they are (FM stations, aircraft transponders, etc.), and keeps a log of everything it detects.

Built on a Nooelec RTL-SDR V5 dongle running Python on Ubuntu Linux.

---

## What You Need

**Hardware**
- Nooelec RTL-SDR V5 USB dongle
- An antenna (a basic whip antenna works for most bands)
- A computer with a USB port

**Software**
- Ubuntu 22.04 or 24.04 (tested in VirtualBox on Windows 11)
- Python 3.11

---

## Installation

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y cmake build-essential libusb-dev libusb-1.0-0-dev pkg-config python3-pip
```

### 2. Build the RTL-SDR driver

> **Note:** The version available through apt is missing a function this project needs. You have to build it from source.

```bash
git clone https://github.com/librtlsdr/librtlsdr.git
cd librtlsdr
mkdir build && cd build
cmake ../ -DINSTALL_UDEV_RULES=ON
make
sudo make install
sudo ldconfig
```

### 3. Install Python packages

```bash
pip install pyrtlsdr numpy scipy PyQt6 pyqtgraph
```

---

## Project Files

```
spectrum_analyzer/
    Analyzer.py    # Run this to start the app
    Worker.py      # Handles radio tuning and signal processing
    Logger.py      # Saves detections to a log file
    config.py      # Settings: frequency range, sensitivity, band definitions
    spectrum.db    # Activity log (created automatically)
    README.md
```

---

## Running It

```bash
cd spectrum_analyzer
python Analyzer.py
```

The app will open and start scanning. You should see signals appear on the display within a few seconds.

---

## The Interface

| Panel | What it shows |
|---|---|
| **Waterfall** | A scrolling color map showing where signals are active over time |
| **Spectrum** | A live graph of signal strength across the current frequency range |
| **RF Band Reference** | A reference list of common frequency bands (FM, aircraft, etc.) |
| **RF Activity Log** | A running list of everything detected, with frequency and signal strength |
| **Parameter Sliders** | Adjust sensitivity and scan speed on the fly |

---

## Scan Modes

**Single-band mode** locks onto one frequency range and watches it continuously.

**Sweep mode** scans the full range from 25 MHz to 1.75 GHz, stepping through each band and building up a complete picture of everything active. Scan speed and range are adjustable with the sliders.

---

## Frequency Bands Covered

| Band | Range | What you might see |
|---|---|---|
| FM Radio | 88 to 108 MHz | Local FM stations |
| Aircraft | 108 to 137 MHz | Plane navigation and approach signals |
| ADS-B | 1090 MHz | Aircraft position broadcasts |
| ISM (900 MHz) | 902 to 928 MHz | LoRa and similar devices |

---

## Troubleshooting

**`undefined symbol: rtlsdr_set_dithering`**
The wrong version of the driver is installed. Redo step 2 of the installation and run `sudo ldconfig` when done.

---

**Device not found or permission denied**
Your user account needs permission to access USB devices. Run this, then log out and back in:

```bash
sudo usermod -aG plugdev $USER
```

---

**App opens but no signals appear**
- Check that the dongle is recognized: `rtl_test -t`
- Turn up the gain slider (30 to 40 is a good starting point)
- Make sure the antenna is connected

*UC San Diego | Master of Advanced Studies, Wireless Embedded Systems | Spring 2026*
