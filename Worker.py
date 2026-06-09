import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from Config import (
    sdr, snap_gain, SAMPLE_RATES_MHZ, DEFAULT_SAMPLE_RATE, DEFAULT_CENTER,
    FFT_SIZE, TIME_PLOT_N, WINDOW, NUM_ROWS,
    DEFAULT_AVG_ALPHA, DEFAULT_SMOOTH_SIGMA,
    DEFAULT_PEAK_PROMINENCE_DB, DEFAULT_PEAK_MIN_HEIGHT_DB, DEFAULT_PEAK_BW_KHZ,
    BAND_SWEEP_RANGE, SWEEP_STEP_FRAC, DEFAULT_SWEEP_DWELL,
    KNOWN_STATIONS_MHZ, STATION_MATCH_TOLERANCE_MHZ,
)

class SDRWorker(QObject):
    # time plot updates
    time_plot_update = pyqtSignal(np.ndarray)
    # (freqs_mhz, psd_db)
    freq_plot_update = pyqtSignal(np.ndarray, np.ndarray)
    # psd_db 
    waterfall_plot_update = pyqtSignal(np.ndarray)
    # [(freq_mhz, power_db, station_or_None)]
    peaks_update = pyqtSignal(list)
    end_of_run = pyqtSignal()

    # update frequency during sweeps
    freq_axis_update = pyqtSignal(float, float)

    # (current_step_idx, total_steps)
    sweep_progress   = pyqtSignal(int, int)
    # sweep peaks
    composite_update = pyqtSignal(list)

    def __init__(self):
        # run QObject
        super().__init__()

        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.center_freq = DEFAULT_CENTER

        self.spectrogram = -80.0 * np.ones((FFT_SIZE, NUM_ROWS), dtype=np.float32)
        self.psd_avg = -80.0 * np.ones(FFT_SIZE, dtype=np.float32)

        # twiddles
        self.avg_alpha = DEFAULT_AVG_ALPHA
        self.smooth_sigma = DEFAULT_SMOOTH_SIGMA
        self.peak_prominence_db = DEFAULT_PEAK_PROMINENCE_DB
        self.peak_min_height_db = DEFAULT_PEAK_MIN_HEIGHT_DB
        self.peak_bw_hz = DEFAULT_PEAK_BW_KHZ * 1e3

        # Sweeping
        self.sweep_mode = False
        # frames per step
        self.sweep_dwell = DEFAULT_SWEEP_DWELL
        # precomputed list of center Hz for each step
        self.sweep_freqs = []
        # current position in sweep_freqs
        self.sweep_idx = 0
        # frames collected at current step so far
        self._sweep_frame_count = 0
        # accumulated psd_db frames (sum)
        self._sweep_frame_acc = None
        # peaks collected across the entire current pass
        self._sweep_all_peaks = []

        self._running = True

    # Update functions for GUI
    def update_freq(self, val_khz: int):
        freq_hz = val_khz * 1e3
        sdr.center_freq  = freq_hz
        self.center_freq = freq_hz
        # reset psd average
        self.psd_avg[:] = -80.0

    def update_gain(self, val_db: int):
        sdr.gain = snap_gain(val_db)

    def update_sample_rate(self, idx: int):
        sr = SAMPLE_RATES_MHZ[idx] * 1e6
        sdr.sample_rate  = sr
        self.sample_rate = sr
        # reset
        self.psd_avg[:]  = -80.0

    def update_avg_alpha(self, val_int: int):
        # alpha 0.01-1.00
        self.avg_alpha = val_int / 100.0

    def update_smooth_sigma(self, val_int: int):
        # range 0-40 -> sigma 0.0-20.0 bins (half-steps)
        # FFT size 4096, sample rate 2.4 MHz -> bin width ~586 Hz, so sigma=20 is ~12 kHz smoothing width.
        self.smooth_sigma = val_int / 2.0

    def update_peak_prominence(self, val_int: int):
        # range 1-30 -> prominence 1-30 dB (direct)
        self.peak_prominence_db = float(val_int)

    def update_peak_min_height(self, val_int: int):
        # range -100 to -10 -> height in dBm (direct, negative)
        self.peak_min_height_db = float(val_int)

    def update_peak_bw_khz(self, val_int: int):
        # range 10-200 -> bandwidth in Hz
        self.peak_bw_hz = val_int * 1e3

    def set_sweep_mode(self, enabled: bool):
        # Toggle sweep mode on or off.
        self.sweep_mode = enabled
        
        if enabled:
            self.sweep_freqs = self._compute_sweep_freqs()
            self.sweep_idx = 0
            self._sweep_frame_count = 0
            self._sweep_frame_acc = None
            self._sweep_all_peaks = []
            sdr.center_freq = self.sweep_freqs[0]
            self.center_freq = self.sweep_freqs[0]
            self.psd_avg[:] = -80.0

    def update_sweep_dwell(self, val_int: int):
        # Slider range 1-20 -> frames per step
        self.sweep_dwell = val_int

    def _compute_sweep_freqs(self) -> list:
        # Build a flat list of center frequencies (in Hz) that covers every band
        # in BAND_SWEEP_RANGES with SWEEP_STEP_FRAC overlap between steps.
        step_hz = self.sample_rate * SWEEP_STEP_FRAC
        freqs = []
        for lo_mhz, hi_mhz in BAND_SWEEP_RANGE:
            lo_hz = lo_mhz * 1e6
            hi_hz = hi_mhz * 1e6
            center = lo_hz + self.sample_rate / 2
            while center <= hi_hz + self.sample_rate / 2:
                freqs.append(center)
                center += step_hz
        return freqs

    def stop(self):
        self._running = False

    # run thread loop
    def run(self):
        if not self._running:
            return

        # Step 1: Pull a block of complex IQ samples from the SDR dongle.
        # FFT_SIZE samples gives us exactly one FFT frame with no zero-padding.
        try:
            samples = sdr.read_samples(FFT_SIZE).astype(np.complex64)
        except Exception as e:
            print("SDR read failed:", e)
            self.end_of_run.emit()
            return

        # Step 2: Send the raw IQ to the time-domain plot
        self.time_plot_update.emit(samples[:TIME_PLOT_N])

        # Step 3: FFT pipeline -> power spectral density in dB.
        fft    = np.fft.fftshift(np.fft.fft(samples * WINDOW))
        psd    = (np.abs(fft) ** 2) / FFT_SIZE
        psd_db = 10.0 * np.log10(psd + 1e-20).astype(np.float32)

        # kill center peak
        # Replace with neighborhood median so it doesn't trigger false peaks.
        c = FFT_SIZE // 2
        psd_db[c] = np.median(psd_db[c - 3 : c + 4])

        # Gaussian smoothing
        if self.smooth_sigma > 0:
            psd_db = gaussian_filter1d(psd_db, sigma=self.smooth_sigma).astype(np.float32)

        # running average for the spectrum curve.
        self.psd_avg = ((1 - self.avg_alpha) * self.psd_avg + self.avg_alpha * psd_db).astype(np.float32)

        # Step 4: Build the frequency axis for this frame.
        freqs_mhz = (np.linspace(-self.sample_rate / 2,
                                   self.sample_rate / 2,
                                   FFT_SIZE,
                                   dtype=np.float32)
                     + self.center_freq) / 1e6

        # Step 5: Waterfall updates
        self.spectrogram = np.roll(self.spectrogram, 1, axis=1)
        self.spectrogram[:, 0] = psd_db
        self.waterfall_plot_update.emit(self.spectrogram)

        # Steps 6-7 check mode
        if not self.sweep_mode:
            self._run_live(freqs_mhz)
        else:
            self._run_sweep(psd_db, freqs_mhz)

        # EXECUTE
        self.end_of_run.emit()

    def _run_live(self, freqs_mhz):
        # Emit the running-average PSD and peak list for the current fixed window.
        self.freq_plot_update.emit(freqs_mhz, self.psd_avg.copy())

        peak_width_bins = self.peak_bw_hz / (self.sample_rate / FFT_SIZE)
        peak_idx, _ = find_peaks(
            self.psd_avg,
            prominence=self.peak_prominence_db,
            height=self.peak_min_height_db,
            width=peak_width_bins,
        )
        peaks = []
        for i in peak_idx:
            fm = float(freqs_mhz[i])
            pm = float(self.psd_avg[i])
            station = next(
                (kf for kf in KNOWN_STATIONS_MHZ
                 if abs(fm - kf) <= STATION_MATCH_TOLERANCE_MHZ),
                None
            )
            peaks.append((fm, pm, station))

        self.peaks_update.emit(peaks)

    def _run_sweep(self, psd_db, freqs_mhz):
        # Accumulate 'sweep_dwell' frames at the current step, then:
        if not self.sweep_freqs:
            return

        #  - Average the accumulated frames and run peak detection
        if self._sweep_frame_acc is None:
            self._sweep_frame_acc = psd_db.copy()
        else:
            self._sweep_frame_acc += psd_db
        self._sweep_frame_count += 1

        # print progress
        if self._sweep_frame_count < self.sweep_dwell:
            self.sweep_progress.emit(self.sweep_idx, len(self.sweep_freqs))
            return

        avg_psd = (self._sweep_frame_acc / self.sweep_dwell).astype(np.float32)

        peak_width_bins = self.peak_bw_hz / (self.sample_rate / FFT_SIZE)
        peak_idx, _ = find_peaks(
            avg_psd,
            prominence=self.peak_prominence_db,
            height=self.peak_min_height_db,
            width=peak_width_bins,
        )
        peaks = []
        for i in peak_idx:
            fm = float(freqs_mhz[i])
            pm = float(avg_psd[i])
            station = next(
                (kf for kf in KNOWN_STATIONS_MHZ
                 if abs(fm - kf) <= STATION_MATCH_TOLERANCE_MHZ),
                None
            )
            peaks.append((fm, pm, station))

        #  - Emit peaks to the logger (peaks_update) and the composite overlay
        self.peaks_update.emit(peaks)
        self._sweep_all_peaks.extend(peaks)
        self.composite_update.emit(list(self._sweep_all_peaks))

        self.sweep_idx = (self.sweep_idx + 1) % len(self.sweep_freqs)

        if self.sweep_idx == 0:
            self._sweep_all_peaks.clear()

        #  - Advance center frequency to the next step
        next_freq = self.sweep_freqs[self.sweep_idx]
        sdr.center_freq  = next_freq
        self.center_freq = next_freq
        self.psd_avg[:]  = -80.0

        # Update frequency axis
        lo_hz = next_freq - self.sample_rate / 2.0
        hi_hz = next_freq + self.sample_rate / 2.0
        self.freq_axis_update.emit(lo_hz / 1e6, hi_hz / 1e6)

        #  - Clear composite and restart
        self._sweep_frame_acc   = None
        self._sweep_frame_count = 0

        self.sweep_progress.emit(self.sweep_idx, len(self.sweep_freqs))
