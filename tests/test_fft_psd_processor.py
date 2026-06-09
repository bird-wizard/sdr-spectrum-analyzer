import numpy as np
from sources.file_source import FileSource
from processors.fft_psd_processor import FFTPSDProcessor

def test_fft_psd_processor():
    src = FileSource(
        #filepath="data/fm_rds_250k_1Msamples.iq",
        #filepath="data/bfm.2021-12-16T14_23_16_147.wav",
        #filepath="data/SDRuno_20200907_184033Z_88110kHz.wav"
    )
    samples = src.read()
    print(f"Loaded {len(samples)} samples")
    
    psd = FFTPSDProcessor(
        center_freq=99.5e6,
        fft_size=4096
    )

    freqs_mhz, psd_dbm = psd.process(samples)

    