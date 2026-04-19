"""Data reading functions for VneuroTK."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from vneurotk.neuro.base import BaseData
from vneurotk.io.path import BIDSPath, EphysPath, MNEPath, VTKPath

_META_DTYPES = frozenset([
    "TrialRecord", "ChTrialRecord", "UnitProp", "ChProp",
])


def read(
    path: VTKPath | EphysPath | MNEPath | BIDSPath | Path | str,
    pre_load: bool = False,
) -> BaseData:
    """Read data from various sources into BaseData.

    Parameters
    ----------
    path : VTKPath, EphysPath, MNEPath, BIDSPath, Path, or str
        Data source.  Plain ``Path`` / ``str`` is treated as a direct
        file path (e.g. an ``.h5`` file saved by :meth:`BaseData.save`).
    pre_load : bool
        If ``True``, eagerly load neuro data into memory before returning
        (calls :meth:`BaseData.load` internally).
        If ``False`` (default), data is loaded lazily on first access to
        :attr:`BaseData.neuro` — call :meth:`BaseData.load` explicitly to
        trigger loading at a chosen point.  For data types that carry no
        lazy loader (already eager), this flag is a no-op.

    Returns
    -------
    BaseData
        Data as BaseData object.

    Raises
    ------
    NotImplementedError
        If loading AvgPsth (not yet implemented).
    ValueError
        If path type is unknown or file format unsupported.
    FileNotFoundError
        If the specified file does not exist.
    """
    # Resolve to a pathlib.Path
    if isinstance(path, (str, Path)):
        fpath = Path(path)
    elif hasattr(path, "fpath"):
        fpath = path.fpath
    else:
        raise ValueError(f"Unknown path type: {type(path)}")

    # EphysPath dispatch
    if isinstance(path, EphysPath):
        bd = _load_from_ephys(path)
        return bd.load() if pre_load else bd

    # Check if it's an h5 file (saved VTK data)
    if fpath.suffix == ".h5":
        bd = _load_from_h5(fpath)
        return bd.load() if pre_load else bd

    # Load based on modality
    if isinstance(path, MNEPath):
        bd = _load_from_mne(path)
        return bd.load() if pre_load else bd
    elif isinstance(path, BIDSPath):
        bd = _load_from_bids(path)
        return bd.load() if pre_load else bd
    else:
        raise ValueError(f"Unsupported file format: {fpath.suffix}")


# ======================================================================
# Ephys loading
# ======================================================================


def _load_from_ephys(path: EphysPath) -> BaseData:
    """Dispatch ephys loading by dtype."""
    dtype = path.dtype
    if dtype is None:
        raise ValueError("EphysPath.dtype must be set for loading")

    if dtype in _META_DTYPES:
        raise ValueError(
            f"'{dtype}' is a metadata file. Pass a neuro dtype "
            f"(TrialRaster, MeanFr, etc.) instead."
        )

    if dtype == "TrialRaster":
        return _load_ephys_raster(path, level="unit")
    elif dtype == "ChTrialRaster":
        return _load_ephys_raster(path, level="channel")
    elif dtype == "MeanFr":
        return _load_ephys_mean_fr(path, level="unit")
    elif dtype == "ChMeanFr":
        return _load_ephys_mean_fr(path, level="channel")
    elif dtype == "ChStimFr":
        return _load_ephys_stim_fr(path)
    elif dtype == "AvgPsth":
        raise NotImplementedError("AvgPsth loading not yet implemented")
    else:
        raise ValueError(f"Unsupported ephys dtype for loading: {dtype}")


def _load_ephys_raster(path: EphysPath, level: str) -> BaseData:
    """Load TrialRaster / ChTrialRaster into an epochs-mode BaseData.

    The raster h5 is NOT read eagerly — a lazy loader is attached so
    the actual COO → dense conversion happens on first ``bd.neuro`` access.
    """
    # --- companion files ---
    if level == "unit":
        record_dtype, prop_dtype = "TrialRecord", "UnitProp"
    else:
        record_dtype, prop_dtype = "ChTrialRecord", "ChProp"

    record_path = EphysPath(
        root=path.root, session_id=path.session_id,
        dtype=record_dtype, probe=path.probe, extension="csv",
    )
    prop_path = EphysPath(
        root=path.root, session_id=path.session_id,
        dtype=prop_dtype, probe=path.probe, extension="csv",
    )

    # --- read raster metadata only (no data) ---
    raster_fpath = path.fpath
    if not raster_fpath.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_fpath}")

    with h5py.File(raster_fpath, "r") as f:
        original_shape = tuple(int(x) for x in f.attrs["original_shape"])
        stored_dtype = str(f.attrs["dtype"])
        meta = dict(f["metadata"].attrs)

    # --- compute target shape after transpose ---
    if level == "unit":
        # (n_units, n_trials, n_timebins) → (n_trials, n_timebins, n_units)
        target_shape = (original_shape[1], original_shape[2], original_shape[0])
    else:
        # (n_trials, n_channels, n_timebins) → (n_trials, n_timebins, n_channels)
        target_shape = (original_shape[0], original_shape[2], original_shape[1])

    n_trials, n_timebins, n_chan = target_shape

    # --- build lazy loader closure ---
    _level = level

    def _neuro_loader() -> np.ndarray:
        from scipy.sparse import coo_matrix

        logger.info("Loading COO sparse data from {}", raster_fpath)
        with h5py.File(raster_fpath, "r") as f:
            row = f["row"][:]
            col = f["col"][:]
            data = f["data"][:]
        flat_shape = (original_shape[0] * original_shape[1], original_shape[2])
        sparse = coo_matrix(
            (data, (row, col)), shape=flat_shape, dtype=stored_dtype
        )
        dense = sparse.toarray().reshape(original_shape)
        if _level == "unit":
            return dense.transpose(1, 2, 0)
        else:
            return dense.transpose(0, 2, 1)

    # --- read companion files ---
    if not record_path.fpath.exists():
        raise FileNotFoundError(f"Record file not found: {record_path.fpath}")
    if not prop_path.fpath.exists():
        raise FileNotFoundError(f"Prop file not found: {prop_path.fpath}")

    record_df = pd.read_csv(record_path.fpath)
    prop_df = pd.read_csv(prop_path.fpath)

    visual_ids = record_df["stim_index"].values
    ch_names = prop_df["id"].astype(str).tolist()

    # --- timing from metadata ---
    pre_onset = int(meta.get("pre_onset", meta.get("pre_stim_ms", 50)))
    post_onset = n_timebins - pre_onset
    sfreq = int(meta.get("sampling_rate", 1000))

    # --- build visual array (n_trials, n_timebins) ---
    unique_ids = np.unique(visual_ids)
    is_str_ids = unique_ids.dtype.kind in ("U", "S", "O")
    if is_str_ids:
        visual = np.empty((n_trials, n_timebins), dtype=object)
        visual[:] = np.nan
    else:
        visual = np.full((n_trials, n_timebins), np.nan)
    for i, sid in enumerate(visual_ids):
        visual[i, pre_onset] = sid

    # --- build trial array (n_trials, n_timebins) ---
    trial = np.empty((n_trials, n_timebins))
    for i in range(n_trials):
        trial[i, :] = i

    # --- assemble BaseData ---
    bd = BaseData(
        neuro=None,
        neuro_info={
            "sfreq": sfreq,
            "ch_names": ch_names,
            "highpass": None,
            "lowpass": None,
            "source_file": str(raster_fpath),
            "shape": target_shape,
        },
        vision=visual,
        vision_info={
            "n_stim": len(unique_ids),
            "stim_ids": sorted(unique_ids.tolist()),
        },
        trial=trial,
        trial_info={
            "baseline": [-pre_onset, 0],
            "trial_window": [-pre_onset, post_onset],
        },
        trial_starts=np.zeros(n_trials, dtype=int),
        trial_ends=np.full(n_trials, n_timebins, dtype=int),
        vision_onsets=np.full(n_trials, pre_onset, dtype=int),
        trial_meta=record_df,
    )
    bd._crop_mode = "epochs"
    bd._neuro_loader = _neuro_loader

    logger.info(
        "Loaded ephys {} (lazy): {} trials, {} timebins, {} channels",
        path.dtype, n_trials, n_timebins, n_chan,
    )
    return bd


def _load_ephys_mean_fr(path: EphysPath, level: str) -> BaseData:
    """Load MeanFr / ChMeanFr into a trial-level BaseData."""
    if level == "unit":
        record_dtype, prop_dtype = "TrialRecord", "UnitProp"
    else:
        record_dtype, prop_dtype = "ChTrialRecord", "ChProp"

    record_path = EphysPath(
        root=path.root, session_id=path.session_id,
        dtype=record_dtype, probe=path.probe, extension="csv",
    )
    prop_path = EphysPath(
        root=path.root, session_id=path.session_id,
        dtype=prop_dtype, probe=path.probe, extension="csv",
    )

    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"MeanFr file not found: {fpath}")

    with h5py.File(fpath, "r") as f:
        neuro = f["data"][:]

    record_df = pd.read_csv(record_path.fpath)
    prop_df = pd.read_csv(prop_path.fpath)

    visual_ids = record_df["stim_index"].values
    ch_names = prop_df["id"].astype(str).tolist()
    unique_ids = np.unique(visual_ids)

    bd = BaseData(
        neuro=neuro,
        neuro_info={
            "sfreq": None,
            "ch_names": ch_names,
            "highpass": None,
            "lowpass": None,
            "source_file": str(fpath),
        },
        vision_info={
            "n_stim": len(unique_ids),
            "stim_ids": sorted(unique_ids.tolist()),
        },
        trial_meta=record_df,
        data_level="trial",
    )

    logger.info(
        "Loaded ephys {} (trial-level): shape {}",
        path.dtype, neuro.shape,
    )
    return bd


def _load_ephys_stim_fr(path: EphysPath) -> BaseData:
    """Load ChStimFr into a stimulus-level BaseData."""
    prop_path = EphysPath(
        root=path.root, session_id=path.session_id,
        dtype="ChProp", probe=path.probe, extension="csv",
    )

    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"ChStimFr file not found: {fpath}")

    with h5py.File(fpath, "r") as f:
        neuro = f["data"][:]
        allstim = f["allstim"][:] if "allstim" in f else None
        teststim = f["teststim"][:] if "teststim" in f else None

    prop_df = pd.read_csv(prop_path.fpath)
    ch_names = prop_df["id"].astype(str).tolist()

    vision_info: dict = {"n_stim": neuro.shape[0]}
    if allstim is not None:
        vision_info["allstim"] = allstim.tolist()
    if teststim is not None:
        vision_info["teststim"] = teststim.tolist()

    bd = BaseData(
        neuro=neuro,
        neuro_info={
            "sfreq": None,
            "ch_names": ch_names,
            "highpass": None,
            "lowpass": None,
            "source_file": str(fpath),
        },
        vision_info=vision_info,
        data_level="stimulus",
    )

    logger.info(
        "Loaded ephys ChStimFr (stimulus-level): shape {}",
        neuro.shape,
    )
    return bd


# ======================================================================
# MNE / BIDS / H5 loading
# ======================================================================


def _load_from_mne(path: MNEPath) -> BaseData:
    """Load data from MNE raw file (lazy by default)."""
    try:
        import mne
    except ImportError as e:
        raise ImportError("mne is required for loading MNE data") from e

    fpath = path.fpath
    if not fpath.exists():
        raise FileNotFoundError(f"MNE file not found: {fpath}")

    logger.info(f"Loading MNE metadata from {fpath}")
    raw = mne.io.read_raw(fpath, preload=False, verbose=False)

    neuro_info = {
        "sfreq": raw.info["sfreq"],
        "ch_names": raw.info["ch_names"],
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "source_file": str(fpath),
        "shape": (len(raw.times), len(raw.ch_names)),
    }

    def _neuro_loader(_raw=raw) -> np.ndarray:
        logger.info("Reading MNE data into memory from {}", fpath)
        return _raw.get_data().T

    bd = BaseData(neuro=None, neuro_info=neuro_info)
    bd._neuro_loader = _neuro_loader
    logger.info(
        "MNE metadata loaded: {} timepoints, {} channels, sfreq={} Hz",
        len(raw.times), len(raw.ch_names), raw.info["sfreq"],
    )
    return bd


def _load_from_bids(path: BIDSPath) -> BaseData:
    """Load data from BIDS dataset (lazy by default)."""
    try:
        from mne_bids import read_raw_bids
    except ImportError as e:
        raise ImportError("mne_bids is required for loading BIDS data") from e

    if path.bids_path is None:
        raise ValueError("BIDSPath not properly initialized")

    logger.info(f"Loading BIDS metadata from {path.fpath}")
    raw = read_raw_bids(path.bids_path, verbose=False)

    neuro_info = {
        "sfreq": raw.info["sfreq"],
        "ch_names": raw.info["ch_names"],
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "source_file": str(path.fpath),
        "shape": (len(raw.times), len(raw.ch_names)),
    }

    def _neuro_loader(_raw=raw) -> np.ndarray:
        logger.info("Reading BIDS data into memory from {}", path.fpath)
        _raw.load_data()
        return _raw.get_data().T

    bd = BaseData(neuro=None, neuro_info=neuro_info)
    bd._neuro_loader = _neuro_loader
    logger.info(
        "BIDS metadata loaded: {} timepoints, {} channels, sfreq={} Hz",
        len(raw.times), len(raw.ch_names), raw.info["sfreq"],
    )
    return bd


def _load_from_h5(fpath: Path) -> BaseData:
    """Load data from h5 file (saved VTK data)."""
    if not fpath.exists():
        raise FileNotFoundError(f"H5 file not found: {fpath}")

    logger.info(f"Loading VTK data from {fpath}")

    with h5py.File(fpath, "r") as f:
        # --- neuro: detect COO vs dense format ---
        neuro_format = str(f.attrs.get("neuro_format", "dense"))
        _neuro_loader = None

        if neuro_format == "coo":
            neuro = None
            neuro_shape = tuple(int(x) for x in f.attrs["neuro_shape"])
            neuro_dtype = str(f.attrs["neuro_dtype"])

            def _neuro_loader(
                _fpath=fpath, _shape=neuro_shape, _dtype=neuro_dtype,
            ) -> np.ndarray:
                from scipy.sparse import coo_matrix as _coo

                logger.info("Lazy-loading COO data from {}", _fpath)
                with h5py.File(_fpath, "r") as fh:
                    row = fh["neuro_row"][:]
                    col = fh["neuro_col"][:]
                    data = fh["neuro_data"][:]
                flat_shape = (_shape[0] * _shape[1], _shape[2])
                sparse = _coo((data, (row, col)), shape=flat_shape, dtype=_dtype)
                return sparse.toarray().reshape(_shape)
        else:
            neuro = f["neuro"][:]
            neuro_shape = None

        # neuro_info
        neuro_info = {}
        if "neuro_info" in f:
            for key in f["neuro_info"].attrs:
                value = f["neuro_info"].attrs[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                neuro_info[key] = value
        # ensure shape is available for lazy loading fallback
        if neuro_shape is not None and "shape" not in neuro_info:
            neuro_info["shape"] = list(neuro_shape)

        # vision
        vision = None
        if "vision" in f:
            vis_is_str = bool(f.attrs.get("vision_is_str", False))
            vis_shape = f.attrs.get("vision_shape", None)
            raw_vis = f["vision"][:]

            if vis_is_str:
                vision = np.empty(len(raw_vis), dtype=object)
                for i, v in enumerate(raw_vis):
                    s = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    vision[i] = np.nan if s == "" else s
            else:
                vision = raw_vis

            if vis_shape is not None:
                vis_shape = [int(x) for x in vis_shape]
                vision = vision.reshape(vis_shape)

        # vision_info
        vision_info = {}
        if "vision_info" in f:
            vi_grp = f["vision_info"]
            for key in vi_grp.attrs:
                value = vi_grp.attrs[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                vision_info[key] = value
            # stim_ids may be stored as dataset (large) or attr
            if "stim_ids" in vi_grp:
                vision_info["stim_ids"] = vi_grp["stim_ids"][:].tolist()

        # trial
        trial = f["trial"][:] if "trial" in f else None

        # trial_info
        trial_info = {}
        if "trial_info" in f:
            for key in f["trial_info"].attrs:
                value = f["trial_info"].attrs[key]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                trial_info[key] = value

        # trial metadata
        trial_starts = f["trial_starts"][:] if "trial_starts" in f else None
        trial_ends = f["trial_ends"][:] if "trial_ends" in f else None
        vision_onsets = f["vision_onsets"][:] if "vision_onsets" in f else None

        # trial_meta (DataFrame)
        trial_meta = None
        if "trial_meta" in f:
            cols = {}
            for col_name in f["trial_meta"]:
                vals = f["trial_meta"][col_name][:]
                if vals.dtype.kind == "S" or vals.dtype.kind == "O":
                    vals = np.array([
                        v.decode("utf-8") if isinstance(v, bytes) else str(v)
                        for v in vals
                    ])
                cols[col_name] = vals
            trial_meta = pd.DataFrame(cols)

        # crop mode & data_level
        crop_mode_raw = f.attrs.get("crop_mode", "")
        crop_mode = str(crop_mode_raw) if crop_mode_raw else None
        data_level = str(f.attrs.get("data_level", "timepoint"))

    if neuro is not None:
        logger.info(f"Loaded VTK data: neuro shape {neuro.shape}")
    else:
        logger.info(f"Loaded VTK data (lazy): neuro shape {neuro_shape}")

    bd = BaseData(
        neuro=neuro,
        neuro_info=neuro_info,
        vision=vision,
        vision_info=vision_info,
        trial=trial,
        trial_info=trial_info,
        trial_starts=trial_starts,
        trial_ends=trial_ends,
        vision_onsets=vision_onsets,
        trial_meta=trial_meta,
        data_level=data_level,
    )
    bd._crop_mode = crop_mode
    if _neuro_loader is not None:
        bd._neuro_loader = _neuro_loader
    return bd
