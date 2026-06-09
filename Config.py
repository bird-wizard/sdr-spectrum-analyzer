import numpy as np
from scipy.signal.windows import blackmanharris
from rtlsdr import RtlSdr

# FFT size controls frequency resolution.
# Larger = finer resolution but more compute per frame
FFT_SIZE = 4096

# how many frames deep the waterfall scrolls
# (more = longer history but more memory and slower updates)
NUM_ROWS = 200

# how many raw IQ samples to show in the time plot
# (more = wider time window but more compute and visual clutter)
TIME_PLOT_N = 500

# Defaults
DEFAULT_CENTER = 99.5e6
DEFAULT_GAIN_DB = 30
SAMPLE_RATES_MHZ = [2.4, 2.048, 1.024]
DEFAULT_SAMPLE_RATE = SAMPLE_RATES_MHZ[0] * 1e6

# Frequency slider
# converted back to Hz before writing to the SDR.
FREQ_MIN_KHZ  = 24_000
FREQ_MAX_KHZ  = 1_700_000
FREQ_TICK_KHZ = 100_000

# Known San Diego FM stations (MHz).
# Any detected peak within STATION_MATCH_TOLERANCE_MHZ of one of these
# gets labeled with the station frequency on the spectrum plot.
KNOWN_STATIONS_MHZ = [
    88.3, 89.5, 90.3, 90.7, 91.1, 91.7, 92.1, 92.5, 93.3, 94.1,
    94.9, 95.7, 96.5, 98.1, 99.3, 99.7, 100.7, 101.5, 102.1,
    102.9, 103.7, 104.5, 104.9, 105.3, 105.7, 106.5, 107.3,
]
STATION_MATCH_TOLERANCE_MHZ = 0.2


# PSD running average: 0.01 (sluggish) -> 1.0 (raw)
# (higher alpha = more weight on the new frame, less on the old average).
DEFAULT_AVG_ALPHA = 0.20

# gaussian smoothing width in FFT bins
# 0 disables smoothing entirely (gaussian_filter1d is a no-op at sigma=0).
DEFAULT_SMOOTH_SIGMA = 4.0

# how far a peak must stand above its surroundings
DEFAULT_PEAK_PROMINENCE_DB = 6.0

# absolute floor: peaks below this dBm are ignored
DEFAULT_PEAK_MIN_HEIGHT_DB = -40.0

# expected minimum peak width (approx FM channel spacing)
# (too narrow = noise spikes trigger false positives,
#  too wide  = nearby stations merge into one)
DEFAULT_PEAK_BW_KHZ = 40.0


# Each step moves the center frequency by (sample_rate * SWEEP_STEP_FRAC).
# Keeping it below 1.0 gives overlap between steps so filter rolloff at the
# edges of each slice doesn't leave blind spots in the composite view.
# NOT ADJUSTABLE.
SWEEP_STEP_FRAC = 0.80

# How many frames to accumulate at each step before running peak detection
# and advancing.
DEFAULT_SWEEP_DWELL = 10

# global SDR
sdr = RtlSdr()
sdr.sample_rate = DEFAULT_SAMPLE_RATE
sdr.center_freq = DEFAULT_CENTER
sdr.freq_correction = 60
sdr.gain = 'auto'

# Discrete gain table from the R820T tuner.
VALID_GAINS = sdr.valid_gains_db


def snap_gain(target_db: float) -> float:
    # Return the nearest gain value the tuner actually supports
    best_gain = VALID_GAINS[0]
    best_diff = abs(best_gain - target_db)

    for gain in VALID_GAINS[1:]:
        diff = abs(gain - target_db)
        if diff < best_diff:
            best_gain = gain
            best_diff = diff

    return best_gain


# Precompute the Blackman-Harris window once.
WINDOW = blackmanharris(FFT_SIZE).astype(np.float32)

BAND_SWEEP_RANGE = [{25.0, 1700.0}]

# Band ranges table
# https://www.fcc.gov/sites/default/files/fcctable.pdf
BAND_RANGES = [
    ("VHF",                                  27.0,   88.0),
    ("FM Radio",                             88.0,  108.0),
    ("Aeronautical/Space/Land Mobile VHF",  108.0,  162.0),
    ("VHF/UHF",                             162.0,  400.0),
    ("UHF",                                 400.0,  894.0),
    ("UHF Extended",                        894.0, 1400.0),
    ("Astronomy (Radio)",                    1400.0, 1626.5),
]

# Return the name of the band while in that band
def classify_band(freq_mhz: float) -> str:
    for name, lo, hi in BAND_RANGES:
        if lo <= freq_mhz <= hi:
            return name
    return "Unknown"
