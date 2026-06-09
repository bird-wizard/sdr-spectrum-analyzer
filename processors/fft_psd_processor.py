import numpy as np
from scipy.signal.windows import blackmanharris

class FFTPSDProcessor:
    def __init__(self, sample_rate: float, center_freq: float, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
        self.fft_size = fft_size

    # return frequencies and power spectrum
    def process(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # split samples into frames
        num_frames = len(samples) // self.fft_size
        total_samples = num_frames * self.fft_size
    
        # reshape into fft_sized chunks with num_frames per row
        frames = samples[:total_samples].reshape(num_frames, self.fft_size)

        # blackman-harris window the frames
        window = blackmanharris(self.fft_size)
        frames = frames * window

        # compute power spectrum
        fft_result = np.fft.fft(frames, axis=1)
        fft_shifted = np.fft.fftshift(fft_result, axes=1)
        powers = np.abs(fft_shifted) ** 2
        psd = np.mean(powers, axis=0)

        psd_dbm = 10 * np.log10(psd)

        # move to FM frequency
        freqs = np.fft.fftfreq(self.fft_size) * self.sample_rate
        freqs = np.fft.fftshift(freqs)
        
        # adjust to MHz values
        freqs_mhz = (freqs + self.center_freq) / 1e6

        return freqs_mhz, psd_dbm