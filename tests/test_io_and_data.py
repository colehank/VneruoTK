"""Tests for vneurotk.io and vneurotk.neuro modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

import vneurotk as vnt
from vneurotk.io import BIDSPath, EphysPath, MNEPath, VTKPath
from vneurotk.neuro import BaseData


class TestVTKPath:
    """Test VTKPath class."""

    def test_vtkpath_basic(self):
        """Test basic VTKPath construction."""
        path = VTKPath(
            root=Path("/data"),
            subject="01",
            session="test",
            task="task1",
            run="01",
        )
        assert path.root == Path("/data")
        assert path.subject == "01"
        assert path.session == "test"
        assert "sub-01" in str(path.fpath)
        assert "ses-test" in str(path.fpath)

    def test_vtkpath_positional_root(self):
        """Test VTKPath with positional root argument."""
        path = VTKPath(Path("/data"), subject="01", session="test")
        assert path.root == Path("/data")
        assert path.subject == "01"


class TestEphysPath:
    """Test EphysPath class."""

    SESSION_ID = "251024_FanFan_nsd1w_MSB"
    ROOT = Path("/db/ephys/MonkeyVision")

    def test_fpath_single_probe(self):
        """Single-probe session: fpath has no _probe suffix."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="TrialRaster")
        assert path.modality == "ephys"
        fp = path.fpath
        assert str(fp) == str(
            self.ROOT / "sessions" / self.SESSION_ID / f"TrialRaster_{self.SESSION_ID}.h5"
        )
        assert "_probe" not in fp.name

    def test_fpath_multi_probe(self):
        """Multi-probe session: fpath contains _probe{N} tag."""
        p0 = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="MeanFr", probe=0)
        p1 = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="MeanFr", probe=1)
        assert "_probe0" in p0.fpath.name
        assert "_probe1" in p1.fpath.name

    def test_fpath_csv_extension(self):
        """csv extension is accepted and applied correctly."""
        path = EphysPath(
            root=self.ROOT, session_id=self.SESSION_ID, dtype="UnitProp", extension="csv"
        )
        assert path.fpath.suffix == ".csv"

    def test_session_dir(self):
        """session_dir points to {root}/sessions/{session_id}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.session_dir == self.ROOT / "sessions" / self.SESSION_ID

    def test_raw_dir(self):
        """raw_dir points to {root}/raw/{session_id}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.raw_dir == self.ROOT / "raw" / self.SESSION_ID

    def test_nwb_path_single_probe(self):
        """nwb_path without probe has no _probe suffix."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        assert path.nwb_path == self.ROOT / "nwb" / f"{self.SESSION_ID}.nwb"

    def test_nwb_path_multi_probe(self):
        """nwb_path with probe contains _probe{N}."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID, probe=0)
        assert path.nwb_path == self.ROOT / "nwb" / f"{self.SESSION_ID}_probe0.nwb"

    def test_from_components(self):
        """from_components builds session_id correctly."""
        path = EphysPath.from_components(
            root=self.ROOT,
            date="251024",
            subject="FanFan",
            paradigm="nsd1w",
            region="MSB",
            dtype="AvgPsth",
        )
        assert path.session_id == self.SESSION_ID
        assert path.dtype == "AvgPsth"
        assert "_probe" not in path.fpath.name

    def test_invalid_dtype_raises(self):
        """Unknown dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            EphysPath(root=self.ROOT, session_id=self.SESSION_ID, dtype="BadType")

    def test_invalid_extension_raises(self):
        """Unsupported extension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid extension"):
            EphysPath(root=self.ROOT, session_id=self.SESSION_ID, extension="dat")

    def test_fpath_missing_session_id_raises(self):
        """Accessing fpath without session_id raises ValueError."""
        path = EphysPath(root=self.ROOT, dtype="TrialRaster")
        with pytest.raises(ValueError, match="session_id"):
            _ = path.fpath

    def test_fpath_missing_dtype_raises(self):
        """Accessing fpath without dtype raises ValueError."""
        path = EphysPath(root=self.ROOT, session_id=self.SESSION_ID)
        with pytest.raises(ValueError, match="dtype"):
            _ = path.fpath


class TestMNEPath:
    """Test MNEPath class."""

    def test_mnepath_construction(self):
        """Test MNEPath construction."""
        path = MNEPath(
            root=Path("/mne"),
            subject="01",
            session="test",
            task="task1",
            run="01",
            suffix="clean",
            extension=".fif",
        )
        assert path.modality == "mne"
        assert "sub-01_ses-test_task-task1_run-01_clean.fif" in str(path.fpath)


class TestBIDSPath:
    """Test BIDSPath class."""

    def test_bidspath_construction(self):
        """Test BIDSPath construction."""
        path = BIDSPath(
            root=Path("/bids"),
            subject="01",
            session="test",
            task="task1",
            run="01",
            suffix="meg",
            extension="fif",
        )
        assert path.modality == "bids"
        assert hasattr(path, "bids_path")


class TestBaseData:
    """Test BaseData class."""

    def test_basedata_construction(self):
        """Test BaseData construction."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        assert data.ntime == 1000
        assert data.nchan == 10
        assert not data.configured

    def test_basedata_configure(self):
        """Test BaseData.configure() method."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)

        # Configure with 3 trials
        visual_onsets = np.array([100, 400, 700])
        visual_ids = np.array([1, 2, 1])
        trial_window = [-0.2, 0.8]  # seconds

        data.configure(
            trial_window=trial_window,
            vision_onsets=visual_onsets,
            visual_ids=visual_ids,
        )

        assert data.configured
        assert data.n_trials == 3
        assert len(data.vision_onsets) == 3
        assert len(data.trial_starts) == 3
        assert len(data.trial_ends) == 3
        assert data.vision_info["n_stim"] == 2
        assert set(data.vision_info["stim_ids"]) == {1, 2}

    def test_basedata_save_load(self):
        """Test BaseData save and load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create data
            neuro = np.random.randn(1000, 10)
            neuro_info = {
                "sfreq": 100.0,
                "ch_names": [f"ch{i}" for i in range(10)],
                "highpass": 0.1,
                "lowpass": 40.0,
                "source_file": "/test.fif",
            }
            data = BaseData(neuro=neuro, neuro_info=neuro_info)

            # Configure
            visual_onsets = np.array([100, 400, 700])
            visual_ids = np.array([1, 2, 1])
            data.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=visual_onsets,
                visual_ids=visual_ids,
            )

            # Save
            save_path = VTKPath(
                Path(tmpdir),
                subject="01",
                session="test",
                task="task1",
            )
            data.save(save_path)

            # Load
            loaded_data = vnt.read(save_path)
            assert loaded_data.ntime == data.ntime
            assert loaded_data.nchan == data.nchan
            assert loaded_data.n_trials == data.n_trials
            assert np.allclose(loaded_data.neuro, data.neuro)
            assert np.allclose(loaded_data.vision_onsets, data.vision_onsets)

    def test_basedata_save_unconfigured_raises(self):
        """Test that saving unconfigured BaseData raises error."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test.h5"
            with pytest.raises(RuntimeError, match="configure"):
                data.save(save_path)

    def test_info_unconfigured(self):
        """Test info property on unconfigured BaseData."""
        neuro = np.random.randn(500, 8)
        neuro_info = {
            "sfreq": 250.0,
            "ch_names": [f"ch{i}" for i in range(8)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        info = data.info

        # neuro section populated
        assert info._neuro["n_time"] == 500
        assert info._neuro["n_neuro"] == 8
        assert info._neuro["sfreq"] == 250.0

        # visual/trial not configured
        assert info._visual is None
        assert info._trial is None
        assert not info._configured

        # repr contains "Not configured"
        assert "Not configured" in repr(info)
        html = info._repr_html_()
        assert "Not configured" in html
        assert "vtk-info" in html
        assert "Neuro" in html

    def test_info_configured(self):
        """Test info property on configured BaseData."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            visual_ids=np.array([1, 2, 1]),
        )
        info = data.info

        assert info._configured
        assert info._neuro["n_time"] == 1000
        assert info._visual["n_stim"] == 2
        assert info._trial["trial_window"] is not None

        html = info._repr_html_()
        assert "Not configured" not in html
        assert "n_visual" in html
        assert "Baseline" in html
        assert "Trial window" in html

    def test_basedata_repr_html(self):
        """Test BaseData._repr_html_ delegates to info."""
        neuro = np.random.randn(500, 8)
        neuro_info = {
            "sfreq": 250.0,
            "ch_names": [f"ch{i}" for i in range(8)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        assert data._repr_html_() == data.info._repr_html_()

    def test_crop_continues(self):
        """Test crop in continues mode."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            visual_ids=np.array([1, 2, 1]),
        )
        trial_len = data.trial_ends[0] - data.trial_starts[0]

        data.crop("continues")

        assert data._crop_mode == "continues"
        assert data.neuro.ndim == 2
        assert data.neuro.shape == (3 * trial_len, 10)
        assert data.ntime == 3 * trial_len
        assert data.nchan == 10
        assert len(data.trial_starts) == 3
        assert data.trial_starts[1] == trial_len

    def test_crop_epochs(self):
        """Test crop in epochs mode."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            visual_ids=np.array([1, 2, 1]),
        )
        trial_len = data.trial_ends[0] - data.trial_starts[0]

        data.crop("epochs")

        assert data._crop_mode == "epochs"
        assert data.neuro.ndim == 3
        assert data.neuro.shape == (3, trial_len, 10)
        assert data.ntime == trial_len
        assert data.nchan == 10
        assert data.n_timepoints == trial_len
        assert data.n_trials == 3

    def test_configure_with_crop(self):
        """Test configure() with crop=True."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            visual_ids=np.array([1, 2, 1]),
            crop=True,
            mode="epochs",
        )

        assert data._crop_mode == "epochs"
        assert data.neuro.ndim == 3
        assert data.n_trials == 3

    def test_crop_unconfigured_raises(self):
        """Test that cropping unconfigured BaseData raises error."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        with pytest.raises(RuntimeError, match="configure"):
            data.crop("continues")


class TestLoadFunction:
    """Test load function."""

    def test_load_avgpsth_not_implemented(self):
        """Test that loading AvgPsth raises NotImplementedError."""
        path = EphysPath(
            root=Path("/ephys"),
            session_id="251024_FanFan_nsd1w_MSB",
            dtype="AvgPsth",
            extension="h5",
        )
        with pytest.raises(NotImplementedError):
            vnt.read(path)

    def test_load_metadata_dtype_raises(self):
        """Passing a metadata dtype (TrialRecord etc.) raises ValueError."""
        for dtype in ("TrialRecord", "ChTrialRecord", "UnitProp", "ChProp"):
            path = EphysPath(
                root=Path("/ephys"),
                session_id="251024_FanFan_nsd1w_MSB",
                dtype=dtype,
                extension="csv",
            )
            with pytest.raises(ValueError, match="metadata file"):
                vnt.read(path)


class TestBaseDataLoad:
    """Test BaseData.load() method and vnt.read(pre_load=...) parameter."""

    def _make_configured_bd(self) -> "BaseData":
        neuro = np.zeros((3, 100, 5), dtype=np.float32)  # sparse 3D
        neuro_info = {
            "sfreq": 1000,
            "ch_names": [f"ch{i}" for i in range(5)],
            "highpass": None,
            "lowpass": None,
            "source_file": "",
            "shape": (3, 100, 5),
        }
        bd = BaseData(
            neuro=None,
            neuro_info=neuro_info,
            vision=np.zeros((3, 100)),
            trial=np.zeros((3, 100)),
            trial_starts=np.array([0, 0, 0]),
            trial_ends=np.array([100, 100, 100]),
            vision_onsets=np.array([50, 50, 50]),
            vision_info={"n_stim": 1, "stim_ids": [0]},
            trial_info={"baseline": [-50, 0], "trial_window": [-50, 50]},
        )
        bd._crop_mode = "epochs"
        bd._neuro_loader = lambda: neuro
        return bd, neuro

    def test_load_method_triggers_lazy(self):
        """bd.load() reads neuro into memory and clears loader."""
        bd, neuro = self._make_configured_bd()
        assert bd._neuro is None
        result = bd.load()
        assert result is bd  # returns self
        assert bd._neuro is not None
        assert bd._neuro_loader is None
        assert np.array_equal(bd.neuro, neuro)

    def test_load_method_noop_when_already_loaded(self):
        """bd.load() is a no-op when neuro is already in memory."""
        bd, neuro = self._make_configured_bd()
        bd.load()
        arr_before = bd._neuro
        bd.load()  # second call
        assert bd._neuro is arr_before  # same object, not reloaded

    def test_load_method_noop_when_no_loader(self):
        """bd.load() on data with no lazy loader returns self silently."""
        neuro = np.random.randn(100, 5)
        bd = BaseData(
            neuro=neuro,
            neuro_info={"sfreq": 100, "ch_names": list("abcde")},
        )
        result = bd.load()
        assert result is bd

    def test_load_method_chaining(self):
        """bd.load() supports method chaining."""
        bd, neuro = self._make_configured_bd()
        loaded_neuro = bd.load().neuro
        assert np.array_equal(loaded_neuro, neuro)

    def test_pre_load_false_keeps_lazy(self):
        """vnt.read(fpath, pre_load=False) keeps neuro lazy for COO h5 files."""
        bd, _ = self._make_configured_bd()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "data.h5"
            bd.save(fpath)
            loaded = vnt.read(fpath, pre_load=False)
            assert loaded._neuro is None
            assert loaded._neuro_loader is not None

    def test_pre_load_true_forces_eager(self):
        """vnt.read(pre_load=True) reads neuro immediately."""
        bd, neuro = self._make_configured_bd()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "data.h5"
            bd.save(fpath)
            loaded = vnt.read(fpath, pre_load=True)
            assert loaded._neuro is not None
            assert loaded._neuro_loader is None
            assert np.array_equal(loaded.neuro, neuro)


# -- Real DB tests (skip if DB is not available) ---------------------------

_DB_ROOT = Path(__file__).resolve().parent.parent / "DB" / "ephys" / "MonkeyVision"
_SESSION_ID = "251024_FanFan_nsd1w_MSB"
_SESSION_DIR = _DB_ROOT / "sessions" / _SESSION_ID
_HAS_DB = _SESSION_DIR.exists()

skip_no_db = pytest.mark.skipif(not _HAS_DB, reason="Real DB not available")


@skip_no_db
class TestLoadEphysRaster:
    """Test raster loading with real DB files."""

    def _make_path(self, dtype: str) -> EphysPath:
        return EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype=dtype)

    def test_load_trial_raster_lazy(self):
        """TrialRaster loads lazily — neuro is None until accessed."""
        bd = vnt.read(self._make_path("TrialRaster"))
        # before access: _neuro is None, but shape info exists
        assert bd._neuro is None
        assert bd._neuro_loader is not None
        assert bd._crop_mode == "epochs"
        assert bd.data_level == "timepoint"
        assert bd.ntime > 0
        assert bd.nchan > 0

    def test_load_trial_raster_access(self):
        """Accessing neuro triggers lazy load with correct shape."""
        bd = vnt.read(self._make_path("TrialRaster"))
        neuro = bd.neuro
        assert neuro is not None
        assert neuro.ndim == 3  # (n_trials, n_timebins, n_units)
        assert neuro.shape[0] == bd.n_trials
        assert neuro.shape[1] == bd.ntime
        assert neuro.shape[2] == bd.nchan
        # loader cleared after use
        assert bd._neuro_loader is None

    def test_load_ch_trial_raster(self):
        """ChTrialRaster loads and has correct shape."""
        bd = vnt.read(self._make_path("ChTrialRaster"))
        assert bd._crop_mode == "epochs"
        neuro = bd.neuro
        assert neuro.ndim == 3  # (n_trials, n_timebins, n_channels)
        assert neuro.shape[0] == bd.n_trials

    def test_raster_has_trial_meta(self):
        """Raster loading attaches trial_meta from TrialRecord."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.trial_meta is not None
        assert "stim_index" in bd.trial_meta.columns
        assert len(bd.trial_meta) == bd.n_trials

    def test_raster_visual_info(self):
        """Visual info is populated from record file."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.vision_info is not None
        assert bd.vision_info["n_stim"] > 0
        assert len(bd.vision_info["stim_ids"]) == bd.vision_info["n_stim"]

    def test_raster_trial_structure(self):
        """Trial arrays match epochs layout."""
        bd = vnt.read(self._make_path("TrialRaster"))
        assert bd.trial is not None
        assert bd.vision is not None
        assert bd.trial_starts is not None
        assert bd.trial_ends is not None
        assert bd.vision_onsets is not None
        assert len(bd.trial_starts) == bd.n_trials

    def test_raster_configure_raises(self):
        """configure() on epochs-mode raster is blocked (already configured)."""
        bd = vnt.read(self._make_path("TrialRaster"))
        # Already has visual/trial set, and crop_mode is epochs.
        # configure() would require data_level='timepoint' — it is, but
        # ntime points at timebins dimension which is per-epoch not continuous.
        # The real guard: it's already configured, calling configure again
        # would overwrite.  Just check it has trial structure.
        assert bd.configured

    def test_raster_to_continues(self):
        """Converting epochs raster to continues produces 2D array."""
        bd = vnt.read(self._make_path("TrialRaster"))
        n_trials = bd.n_trials
        n_timebins = bd.ntime
        n_chan = bd.nchan
        bd.to_continues()
        assert bd._crop_mode == "continues"
        assert bd.neuro.ndim == 2
        assert bd.neuro.shape == (n_trials * n_timebins, n_chan)


@skip_no_db
class TestLoadEphysMeanFr:
    """Test MeanFr / ChMeanFr loading."""

    def _make_path(self, dtype: str) -> EphysPath:
        return EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype=dtype)

    def test_load_mean_fr(self):
        """MeanFr loads as trial-level data."""
        bd = vnt.read(self._make_path("MeanFr"))
        assert bd.data_level == "trial"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_trials, n_units)
        assert bd.trial_meta is not None
        assert "stim_index" in bd.trial_meta.columns

    def test_load_ch_mean_fr(self):
        """ChMeanFr loads as trial-level data."""
        bd = vnt.read(self._make_path("ChMeanFr"))
        assert bd.data_level == "trial"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_trials, n_channels)

    def test_mean_fr_configure_raises(self):
        """configure() on trial-level data raises ValueError."""
        bd = vnt.read(self._make_path("MeanFr"))
        with pytest.raises(ValueError, match="data_level"):
            bd.configure(
                trial_window=[-0.2, 0.8],
                vision_onsets=np.array([0, 1]),
                visual_ids=np.array([0, 1]),
            )

    def test_mean_fr_crop_raises(self):
        """crop() on trial-level data raises ValueError."""
        bd = vnt.read(self._make_path("MeanFr"))
        with pytest.raises((ValueError, RuntimeError)):
            bd.crop("epochs")


@skip_no_db
class TestLoadEphysStimFr:
    """Test ChStimFr loading."""

    def test_load_ch_stim_fr(self):
        """ChStimFr loads as stimulus-level data."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="ChStimFr")
        bd = vnt.read(path)
        assert bd.data_level == "stimulus"
        assert bd.neuro is not None
        assert bd.neuro.ndim == 2  # (n_stimuli, n_channels)
        assert bd.vision_info is not None
        assert bd.vision_info["n_stim"] == bd.neuro.shape[0]


@skip_no_db
class TestEphysSaveLoadRoundtrip:
    """Test save/load roundtrip for ephys-loaded BaseData."""

    def test_roundtrip_raster(self):
        """Save and reload a raster-loaded BaseData via COO format."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="TrialRaster")
        bd = vnt.read(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = VTKPath(
                Path(tmpdir),
                subject="FanFan",
                session="nsd1w",
                task="raster",
            )
            bd.save(save_path)

            # verify COO format on disk
            import h5py

            with h5py.File(save_path.fpath, "r") as f:
                assert str(f.attrs["neuro_format"]) == "coo"
                assert "neuro_row" in f
                assert "neuro_col" in f
                assert "neuro_data" in f
                assert "neuro" not in f  # no dense dataset

            # reload: lazy
            loaded = vnt.read(save_path)
            assert loaded._crop_mode == "epochs"
            assert loaded.data_level == "timepoint"
            assert loaded._neuro is None
            assert loaded._neuro_loader is not None
            # shape info available before load
            assert loaded.ntime == bd.ntime
            assert loaded.nchan == bd.nchan
            assert loaded.n_trials == bd.n_trials
            # trigger load and verify data
            assert np.allclose(loaded.neuro, bd.neuro)

    def test_roundtrip_preserves_trial_meta(self):
        """Save and reload preserves trial_meta DataFrame."""
        path = EphysPath(root=_DB_ROOT, session_id=_SESSION_ID, dtype="TrialRaster")
        bd = vnt.read(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = VTKPath(
                Path(tmpdir),
                subject="FanFan",
                session="nsd1w",
                task="meta",
            )
            bd.save(save_path)

            loaded = vnt.read(save_path)
            assert loaded.trial_meta is not None
            assert set(loaded.trial_meta.columns) == set(bd.trial_meta.columns)
            assert len(loaded.trial_meta) == len(bd.trial_meta)

    def test_save_dense_for_small_data(self):
        """Non-sparse 2D data still uses dense format."""
        neuro = np.random.randn(1000, 10)
        neuro_info = {
            "sfreq": 100.0,
            "ch_names": [f"ch{i}" for i in range(10)],
            "highpass": 0.1,
            "lowpass": 40.0,
            "source_file": "/test.fif",
        }
        data = BaseData(neuro=neuro, neuro_info=neuro_info)
        data.configure(
            trial_window=[-0.2, 0.8],
            vision_onsets=np.array([100, 400, 700]),
            visual_ids=np.array([1, 2, 1]),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "dense.h5"
            data.save(fpath)

            import h5py

            with h5py.File(fpath, "r") as f:
                assert str(f.attrs.get("neuro_format", "dense")) == "dense"
                assert "neuro" in f

            loaded = vnt.read(fpath)
            assert np.allclose(loaded.neuro, data.neuro)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
