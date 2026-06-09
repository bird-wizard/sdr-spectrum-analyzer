import numpy as np
from sources.file_source import FileSource

def test_file_source():
    src = FileSource(
        #filepath="data/fm_rds_250k_1Msamples.iq",
        #filepath="data/bfm.2021-12-16T14_23_16_147.wav",
        #filepath="data/SDRuno_20200907_184033Z_88110kHz.wav",
    )
    samples = src.read()
    assert len(samples) > 0
    print(f"Loaded {len(samples)} samples")