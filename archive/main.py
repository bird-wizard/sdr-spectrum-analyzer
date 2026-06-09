#!/usr/bin/env python3
import argparse
import numpy as np

from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from scipy.signal import welch, find_peaks
from scipy.ndimage import gaussian_filter1d

from rtlsdr import RtlSdr



# FFT Adjustments
CENTER_FREQ_HZ   = 103e6
SAMPLE_RATE_HZ   = 2.4e6
GAIN             = 10.0
FFT_SIZE         = 4096
NUM_SAMPLES      = FFT_SIZE * 256
OVERLAP          = 0.67

# Peak detection adjustments
# dB above neighbours to count as a peak
PEAK_MIN_PROMINENCE_DB = 6
# absolute minimum height
PEAK_MIN_HEIGHT_DB     = -90
# peak bandwidth
PEAK_TARGET_BW = 0.04
PEAK_WIDTH_BINS = PEAK_TARGET_BW * 1e6 / (SAMPLE_RATE_HZ / FFT_SIZE)

# Full FM band sweep
FM_BAND_START_HZ  = 87.5e6
FM_BAND_END_HZ    = 108.0e6
# keep usable spectrum, ignore roll off
SWEEP_USABLE_FRAC = 0.80   
# step in 70% of sample_rate per chunk for overlap
SWEEP_STEP_FRAC   = 0.70

# Low pass filter smoothing
SMOOTH_SIGMA_HZ = 25e1
SMOOTH_SIGMA_BINS = SMOOTH_SIGMA_HZ / (SAMPLE_RATE_HZ / FFT_SIZE)

# https://allusaradiostation.e-monsite.com/pages/city/san-diego.html
KNOWN_STATIONS_MHZ = [
    88.3,  # KSDS Jazz
    89.5,  # KPBS
    90.3,  # Z90
    90.7,  # XTIM
    91.1,  # 91X
    91.7,  # XGLX
    92.1,  # KSON Country
    92.5,  # Magic 925
    93.3,  # KHTS 933
    94.1,  # Star 941
    94.9,  # KBZT
    95.7,  # KSSX
    96.5,  # KYXY 
    98.1,  # KXSN
    99.3,  # XOCL Spanish
    99.7,  # XHTY
    100.7, # KFBG
    101.5, # KGB
    102.1, # KLVJ K love
    102.9, # KLQV Spanish
    103.7, # KPRI Country
    104.5, # XLTN
    104.9, # XLNC
    105.3, # KIOZ Rock
    105.7, # XPRS
    106.5, # KLNV Mexican
    107.3  # XHFG
]

def capture_iq(center_freq: float, sample_rate: float, num_samples: int, gain) -> np.ndarray:
    sdr = RtlSdr()
    try:
        sdr.sample_rate   = sample_rate
        sdr.center_freq   = center_freq
        sdr.gain          = gain
        print(f"SDR Center: {center_freq/1e6:.1f} MHz | "
              f"Sample Rate: {sample_rate/1e6:.2f} MS/s")
        samples = sdr.read_samples(num_samples)
    finally:
        sdr.close()
    return np.array(samples, dtype=np.complex64)


def compute_psd(samples: np.ndarray, sample_rate: float, fft_size: int, overlap: float, center_freq: float = CENTER_FREQ_HZ):
    nperseg  = fft_size
    noverlap = int(fft_size * overlap)
    freqs_bb, psd = welch(samples,
                          fs=sample_rate,
                          nperseg=nperseg,
                          noverlap=noverlap,
                          window='blackmanharris',
                          return_onesided=False,
                          scaling='density')

    # shift from baseband to FM
    freqs_bb = np.fft.fftshift(freqs_bb)
    psd      = np.fft.fftshift(psd)
    freqs_hz = freqs_bb + center_freq

    # avoid log(0)
    psd_db = 10 * np.log10(np.abs(psd) + 1e-20)

    # low pass filter
    psd_db = gaussian_filter1d(psd_db, sigma=SMOOTH_SIGMA_BINS)

    return freqs_hz, freqs_bb, psd_db


def sweep_fm_band(start_hz: float, end_hz: float, sample_rate: float, fft_size: int,
                  num_samples_per_chunk: int, gain, overlap: float,
                  usable_frac: float = SWEEP_USABLE_FRAC,
                  step_frac: float = SWEEP_STEP_FRAC):

    step_hz   = sample_rate * step_frac
    half_bw   = (sample_rate * usable_frac) / 2
    bin_width = sample_rate / fft_size

    centers = np.arange(start_hz + half_bw, end_hz - half_bw + step_hz, step_hz)

    # start, end, and step size
    grid_freqs = np.arange(start_hz, end_hz + bin_width, bin_width)
    grid_psd   = np.full(len(grid_freqs), -np.inf, dtype=np.float64)

    sdr = RtlSdr()
    try:
        sdr.sample_rate = sample_rate
        sdr.gain        = gain
        
        for i, cf in enumerate(centers):
            sdr.center_freq = cf
            sdr.read_samples(int(fft_size))
            samples = np.array(sdr.read_samples(num_samples_per_chunk),dtype=np.complex64)
            print(f"SDR Center: {cf/1e6:.1f} MHz")

            freqs_hz, freqs_bb, psd_db = compute_psd(samples, sample_rate, fft_size, overlap, center_freq=cf)

            # keep only the usable middle portion of each chunk
            usable = np.abs(freqs_bb) <= half_bw
            
            chunk_freqs = freqs_bb[usable] + cf
            chunk_psd   = psd_db[usable]

            # place onto the common grid
            idx = np.round((chunk_freqs - start_hz) / bin_width).astype(int)
            valid = (idx >= 0) & (idx < len(grid_freqs))
            np.maximum.at(grid_psd, idx[valid], chunk_psd[valid])
    finally:
        sdr.close()

    # drop any bins that never received data
    has_data = np.isfinite(grid_psd)
    return grid_freqs[has_data], grid_psd[has_data]


def detect_peaks(freqs_hz: np.ndarray, psd_db: np.ndarray, min_prominence: float, min_height: float):
    peak_indices, props = find_peaks(psd_db,
                                     width=PEAK_WIDTH_BINS,
                                     prominence=min_prominence,
                                     height=min_height)
    peak_freqs_mhz  = freqs_hz[peak_indices] / 1e6
    peak_powers_db  = psd_db[peak_indices]
    return peak_freqs_mhz, peak_powers_db


def match_known_stations(peak_freqs_mhz: np.ndarray, known_mhz: list, tolerance_mhz: float = 0.2):
    matched = []
    for pf in peak_freqs_mhz:
        for kf in known_mhz:
            if abs(pf - kf) <= tolerance_mhz:
                matched.append((pf, kf))
                break
    return matched


def plot_spectrum(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db, matched,
                  title: str = None, output_path: str = 'fm_spectrum.png',
                  figsize=(15, 5)):
    freqs_mhz = freqs_hz / 1e6

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freqs_mhz, psd_db, color='steelblue', lw=0.8, label='PSD')

    # mark detected peaks
    for pf, pb in zip(peak_freqs_mhz, peak_powers_db):
        ax.plot(pf, pb, 'o', color='orange', ms=4)
        ax.annotate(f'{pf:.1f}',
                    xy=(pf, pb),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center',
                    fontsize=7,
                    color='crimson',
                    fontweight='bold')

    # plot misses
    detected_known = {m[1] for m in matched}
    for kf in KNOWN_STATIONS_MHZ:
        if freqs_mhz[0] <= kf <= freqs_mhz[-1]:
            color = 'green' if kf in detected_known else 'grey'
            ax.axvline(kf, color=color, alpha=0.3, lw=2, ls='--')

    if title is None:
        title = (f'FM Power Spectrum Centre {CENTER_FREQ_HZ/1e6:.0f} MHz, '
                 f'BW {SAMPLE_RATE_HZ/1e6:.1f} MHz')

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(title)
    ax.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)

    legend_items = [
        Line2D([0], [0], color='steelblue',  lw=1,   label='PSD'),
        Line2D([0], [0], marker='o', color='orange', lw=0, ms=6, label='Detected peak'),
        Line2D([0], [0], color='green', lw=1, ls='--', label='Matched Stations'),
        Line2D([0], [0], color='grey',  lw=1, ls='--', label='Known Stations'),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"saved {output_path}")
    plt.show()


def report(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db):
    print(f"\nFrequency range: {freqs_hz[0]/1e6:.2f} - {freqs_hz[-1]/1e6:.2f} MHz")
    print(f"\nPeaks Detected: {len(peak_freqs_mhz)}\n")
    for pf, pb in sorted(zip(peak_freqs_mhz, peak_powers_db)):
        print(f"  {pf:7.2f} MHz  {pb:+.1f} dB")

    matched = match_known_stations(peak_freqs_mhz, KNOWN_STATIONS_MHZ)
    print(f"\n{len(matched)} known San Diego stations confirmed:")

    matched_sorted = sorted(matched, key=lambda x: x[1])
    for detected_mhz, known_mhz in matched_sorted:
        print(f"  {known_mhz:.1f} MHz (detected at {detected_mhz:.2f} MHz)")


    matched_freqs = {m[1] for m in matched}
    freq_min_mhz = freqs_hz[0] / 1e6
    freq_max_mhz = freqs_hz[-1] / 1e6

    missed = []
    for kf in KNOWN_STATIONS_MHZ:
        if freq_min_mhz <= kf <= freq_max_mhz and kf not in matched_freqs:
            missed.append(kf)

    if missed:
        print(f"\n Missing: {missed}")
    return matched


def main():
    parser = argparse.ArgumentParser(description="FM-band spectrum analyzer using RTL-SDR")
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        freqs_hz, psd_db = sweep_fm_band(FM_BAND_START_HZ, 
                                         FM_BAND_END_HZ,
                                         SAMPLE_RATE_HZ, 
                                         FFT_SIZE, 
                                         NUM_SAMPLES,
                                         GAIN, OVERLAP)
        
        peak_freqs_mhz, peak_powers_db = detect_peaks(freqs_hz, psd_db,
                                                     PEAK_MIN_PROMINENCE_DB,
                                                     PEAK_MIN_HEIGHT_DB)
        print(peak_freqs_mhz)
        matched = report(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db)
        
        title = (f'Full FM Band Sweep {FM_BAND_START_HZ/1e6:.1f} - {FM_BAND_END_HZ/1e6:.1f} MHz ')
        
        plot_spectrum(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db, matched,
                      title=title, output_path='fm_spectrum_full.png', figsize=(20, 5))
    else:
        samples = capture_iq(CENTER_FREQ_HZ, SAMPLE_RATE_HZ, NUM_SAMPLES, GAIN)
        print(f"Samples Captured: {len(samples):,}")

        freqs_hz, freqs_bb, psd_db = compute_psd(samples, 
                                          SAMPLE_RATE_HZ, 
                                          FFT_SIZE, 
                                          OVERLAP,
                                          center_freq=CENTER_FREQ_HZ)
        peak_freqs_mhz, peak_powers_db = detect_peaks(freqs_hz, psd_db,
                                                     PEAK_MIN_PROMINENCE_DB,
                                                     PEAK_MIN_HEIGHT_DB)
        print(peak_freqs_mhz)
        matched = report(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db)
        plot_spectrum(freqs_hz, psd_db, peak_freqs_mhz, peak_powers_db, matched)


if __name__ == '__main__':
    main()
