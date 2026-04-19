# %%
from pathlib import Path
NOD_ROOT = Path("../../DB/mne/NOD-MEG")
NSD1w_ROOT = Path("../../DB/ephys/GhTest_v0.0.0/sessions/251024_FanFan_nsd1w_MSB")

class TestPath:
    root = None
    session = None
    subject = None
    task = None
    run = None
    desc = None
    probe = None
    suffix = None
    extension = None
    modality = None

class TestEphysPath(TestPath):
    root = NSD1w_ROOT
    session = "251024_FanFan_nsd1w_MSB"
    desc = "TrialRaster"
    probe = 1
    modality = "ephys"

class TestMNEPath(TestPath):
    root = NOD_ROOT / "derivatives" / "preprocessed" / "raw"
    subject = "01"
    session = "ImageNet01"
    task = "ImageNet"
    run = "01"
    extension = ".fif"
    suffix = "clean"
    modality = "mne"

class TestBIDSPath(TestPath):
    root = NOD_ROOT
    subject = "01"
    session = "ImageNet01"
    task = "ImageNet"
    run = "01"
    extension = "fif"
    suffix = "clean"
    modality = "mne"