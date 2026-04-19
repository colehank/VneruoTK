"""Core data container for VneuroTK.

This module provides :class:`BaseData`, the unified container that holds
neural recordings together with visual-stimulus and trial metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from loguru import logger

from vneurotk.neuro.info import Info


class BaseData:
    """Unified container for neural data, stimulus labels, and trial structure.

    Parameters
    ----------
    neuro : np.ndarray | None
        Neural data array.  ``None`` when using lazy loading (see
        *_neuro_loader*).
    neuro_info : dict
        Metadata dict.  Required key: ``sfreq``.  Optional keys:
        ``ch_names``, ``highpass``, ``lowpass``, ``source_file``, ``shape``.
    vision : np.ndarray | None
        Stimulus-label array of shape ``(ntime,)``.  ``np.nan`` at
        non-stimulus timepoints, stimulus ID at onset timepoints.
    vision_info : dict | None
        Dict with ``n_stim`` (int) and ``stim_ids`` (list).
    trial : np.ndarray | None
        Trial-ID array of shape ``(ntime,)``.  ``np.nan`` outside trials.
    trial_info : dict | None
        Dict with ``baseline`` (list[int]) and ``trial_window`` (list).
    trial_starts : np.ndarray | None
        Start sample indices per trial, shape ``(n_trials,)``.
    trial_ends : np.ndarray | None
        End sample indices per trial, shape ``(n_trials,)``.
    vision_onsets : np.ndarray | None
        Stimulus onset sample indices, shape ``(n_trials,)``.
    trial_meta : pd.DataFrame | None
        Per-trial metadata table (e.g. from TrialRecord).
    data_level : str
        ``"timepoint"`` (default) for time-series data,
        ``"trial"`` for per-trial aggregates,
        ``"stimulus"`` for per-stimulus aggregates.

    Examples
    --------
    >>> import numpy as np
    >>> neuro = np.random.randn(1000, 64)
    >>> info = dict(sfreq=250.0, ch_names=[f"ch{i}" for i in range(64)],
    ...             highpass=0.1, lowpass=40.0, source_file="raw.fif")
    >>> bd = BaseData(neuro, info)
    >>> bd
    BaseData(ntime=1000, nchan=64, n_trials=0, configured=False)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        neuro: np.ndarray | None,
        neuro_info: dict[str, Any],
        vision: np.ndarray | None = None,
        vision_info: dict[str, Any] | None = None,
        trial: np.ndarray | None = None,
        trial_info: dict[str, Any] | None = None,
        trial_starts: np.ndarray | None = None,
        trial_ends: np.ndarray | None = None,
        vision_onsets: np.ndarray | None = None,
        trial_meta: Any = None,
        data_level: str = "timepoint",
    ) -> None:
        # neuro is stored as _neuro; accessed via property for lazy loading
        self._neuro: np.ndarray | None = np.asarray(neuro) if neuro is not None else None
        self._neuro_loader: Callable[[], np.ndarray] | None = None
        self.neuro_info = neuro_info

        self.vision = vision
        self.vision_info = vision_info
        self.trial = trial
        self.trial_info = trial_info
        self.trial_starts = trial_starts
        self.trial_ends = trial_ends
        self.vision_onsets = vision_onsets
        self.trial_meta = trial_meta
        self.data_level = data_level

        self._crop_mode: str | None = None

        logger.debug("BaseData created: ntime={}, nchan={}", self.ntime, self.nchan)

    # ------------------------------------------------------------------
    # neuro property (lazy loading)
    # ------------------------------------------------------------------

    @property
    def neuro(self) -> np.ndarray:
        """Neural data array.  Loaded lazily on first access if a loader was set."""
        if self._neuro is None and self._neuro_loader is not None:
            logger.info("Lazy-loading neuro data...")
            self._neuro = self._neuro_loader()
            self._neuro_loader = None
        return self._neuro

    @neuro.setter
    def neuro(self, value: np.ndarray | None) -> None:
        self._neuro = np.asarray(value) if value is not None else None
        self._neuro_loader = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ntime(self) -> int:
        """Number of time samples (per trial in epochs mode)."""
        if self._neuro is not None:
            if self._crop_mode == "epochs":
                return self._neuro.shape[1]
            return self._neuro.shape[0]
        # Fallback: infer from neuro_info["shape"] without triggering load
        shape = self.neuro_info.get("shape")
        if shape is not None:
            if self._crop_mode == "epochs":
                return shape[1]
            return shape[0]
        return 0

    @property
    def nchan(self) -> int:
        """Number of channels."""
        if self._neuro is not None:
            return self._neuro.shape[-1]
        shape = self.neuro_info.get("shape")
        if shape is not None:
            return shape[-1]
        ch_names = self.neuro_info.get("ch_names")
        if ch_names is not None:
            return len(ch_names)
        return 0

    @property
    def n_timepoints(self) -> int:
        """Time points per trial (same as *ntime* in epochs mode)."""
        if self._crop_mode == "epochs":
            return self.neuro.shape[1]
        if self.trial_starts is not None and self.trial_ends is not None:
            return int(self.trial_ends[0] - self.trial_starts[0])
        return self.ntime

    @property
    def configured(self) -> bool:
        """Whether :meth:`configure` has been called."""
        return self.vision is not None and self.trial is not None

    @property
    def n_trials(self) -> int:
        """Number of trials (0 if not configured)."""
        if self.trial_starts is None:
            return 0
        return len(self.trial_starts)

    @property
    def trial_stim_ids(self) -> np.ndarray:
        """Stimulus ID at the onset of each trial, shape ``(n_trials,)``.

        Returns
        -------
        np.ndarray
            Array of stimulus identifiers aligned to trials.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        if not self.configured:
            raise RuntimeError("BaseData not configured. Call configure() first.")
        return np.array([self.vision[int(self.vision_onsets[i])] for i in range(self.n_trials)])

    @property
    def info(self) -> Info:
        """Summary of neuro, visual, and trial metadata."""
        return Info(
            neuro={
                "n_time": self.ntime,
                "n_neuro": self.nchan,
                "sfreq": self.neuro_info.get("sfreq"),
                "highpass": self.neuro_info.get("highpass"),
                "lowpass": self.neuro_info.get("lowpass"),
            },
            visual=self.vision_info,
            trial=self.trial_info,
            configured=self.configured,
            crop_mode=self._crop_mode,
            data_level=self.data_level,
        )

    # ------------------------------------------------------------------
    # configure()
    # ------------------------------------------------------------------

    def configure(
        self,
        trial_window: list[float | int],
        vision_onsets: np.ndarray,
        visual_ids: np.ndarray,
        crop: bool = False,
        mode: str = "continues",
    ) -> None:
        """Attach stimulus and trial structure to the data.

        Parameters
        ----------
        trial_window : list of float | int
            Two-element list ``[start, end]`` relative to each vision onset.
            If elements are *float*, they are interpreted as **seconds** and
            converted to samples via ``sfreq``.  If *int*, they are treated
            as **samples** directly.
        vision_onsets : np.ndarray
            1-D array of stimulus onset sample indices.
        visual_ids : np.ndarray
            1-D array of stimulus IDs, one per onset.
        crop : bool
            If ``True``, automatically call :meth:`crop` after configuring.
        mode : str
            Crop mode passed to :meth:`crop` when *crop* is ``True``.
        """
        sfreq: float = self.neuro_info["sfreq"]
        if self.data_level != "timepoint":
            raise ValueError(
                f"configure() requires data_level='timepoint', got '{self.data_level}'"
            )
        vision_onsets = np.asarray(vision_onsets, dtype=int)
        visual_ids = np.asarray(visual_ids)

        # --- convert trial_window to samples ---
        tw_samples = self._window_to_samples(trial_window, sfreq)

        # --- trial boundaries ---
        trial_starts = vision_onsets + tw_samples[0]
        trial_ends = vision_onsets + tw_samples[1]

        # --- build vision array ---
        if visual_ids.dtype.kind in ("U", "S", "O"):  # string-like IDs
            vision = np.empty(self.ntime, dtype=object)
            vision[:] = np.nan
        else:
            vision = np.full(self.ntime, np.nan)
        for onset, sid in zip(vision_onsets, visual_ids):
            vision[onset] = sid

        # --- build trial array ---
        trial = np.full(self.ntime, np.nan)
        for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
            trial[ts:te] = i

        # --- info dicts ---
        unique_ids = np.unique(visual_ids).tolist()
        self.vision_info = {"n_stim": len(unique_ids), "stim_ids": unique_ids}
        self.trial_info = {
            "baseline": [tw_samples[0], 0],
            "trial_window": tw_samples,
        }

        # --- store ---
        self.vision = vision
        self.trial = trial
        self.trial_starts = trial_starts
        self.trial_ends = trial_ends
        self.vision_onsets = vision_onsets

        logger.info(
            "Configured: {} trials, {} unique stimuli",
            len(trial_starts),
            len(unique_ids),
        )

        if crop:
            self.crop(mode)

    # ------------------------------------------------------------------
    # crop()
    # ------------------------------------------------------------------

    def crop(self, mode: str = "continues") -> None:
        """Crop data to keep only trial time points.

        Parameters
        ----------
        mode : ``"continues"`` | ``"epochs"``
            ``"continues"`` concatenates trials into a 2-D array
            ``(total_trial_samples, nchan)``.
            ``"epochs"`` stacks trials into a 3-D array
            ``(n_trials, n_timepoints, nchan)``.
        """
        if not self.configured:
            raise RuntimeError("Cannot crop unconfigured BaseData. Call configure() first.")
        if self.data_level != "timepoint":
            raise ValueError(f"crop() requires data_level='timepoint', got '{self.data_level}'")
        if mode not in ("continues", "epochs"):
            raise ValueError(f"mode must be 'continues' or 'epochs', got '{mode}'")

        # --- extract per-trial segments ---
        seg_neuro, seg_vision, seg_trial = [], [], []
        for ts, te in zip(self.trial_starts, self.trial_ends):
            seg_neuro.append(self.neuro[ts:te])
            seg_vision.append(self.vision[ts:te])
            seg_trial.append(self.trial[ts:te])

        trial_len = int(self.trial_ends[0] - self.trial_starts[0])
        onset_offset = int(self.vision_onsets[0] - self.trial_starts[0])
        n = len(self.trial_starts)

        if mode == "continues":
            self.neuro = np.concatenate(seg_neuro, axis=0)
            self.vision = np.concatenate(seg_vision, axis=0)
            self.trial = np.concatenate(seg_trial, axis=0)

            self.trial_starts = np.arange(n, dtype=int) * trial_len
            self.trial_ends = self.trial_starts + trial_len
            self.vision_onsets = self.trial_starts + onset_offset
        else:  # epochs
            self.neuro = np.stack(seg_neuro, axis=0)
            self.vision = np.stack(seg_vision, axis=0)
            self.trial = np.stack(seg_trial, axis=0)

            self.trial_starts = np.zeros(n, dtype=int)
            self.trial_ends = np.full(n, trial_len, dtype=int)
            self.vision_onsets = np.full(n, onset_offset, dtype=int)

        self._crop_mode = mode

        logger.info(
            "Cropped to {} mode: neuro shape {}",
            mode,
            self.neuro.shape,
        )

    # ------------------------------------------------------------------
    # Explicit load
    # ------------------------------------------------------------------

    def load(self) -> "BaseData":
        """Explicitly load neuro data into memory and return self.

        For ephys lazy data this triggers the COO → dense conversion.
        For MNE / BIDS data this reads the raw file into memory (equivalent
        to ``raw.load_data()``).
        If neuro is already loaded or there is no lazy loader, this is a
        no-op.  Supports method chaining: ``bd = vnt.read(path).load()``.

        Returns
        -------
        BaseData
            self, for method chaining.
        """
        if self._neuro is None and self._neuro_loader is not None:
            _ = self.neuro  # triggers the property which calls the loader
        elif self._neuro is not None:
            logger.debug("neuro already loaded, skipping .load()")
        return self

    # ------------------------------------------------------------------
    # Mode conversion
    # ------------------------------------------------------------------

    def to_continues(self) -> None:
        """Convert cropped data to continues (2-D) layout.

        Raises
        ------
        RuntimeError
            If data has not been cropped yet.
        """
        if self._crop_mode is None:
            raise RuntimeError("Cannot convert uncropped data. Call crop() first.")
        if self._crop_mode == "continues":
            logger.warning("Data is already in continues mode.")
            return

        # epochs (n_trials, n_timepoints, nchan) -> continues
        n = self.neuro.shape[0]
        trial_len = self.neuro.shape[1]
        onset_offset = int(self.vision_onsets[0])

        self.neuro = self.neuro.reshape(-1, self.neuro.shape[-1])
        self.vision = self.vision.reshape(-1)
        self.trial = self.trial.reshape(-1)

        self.trial_starts = np.arange(n, dtype=int) * trial_len
        self.trial_ends = self.trial_starts + trial_len
        self.vision_onsets = self.trial_starts + onset_offset

        self._crop_mode = "continues"
        logger.info("Converted to continues mode: neuro shape {}", self.neuro.shape)

    def to_epochs(self) -> None:
        """Convert cropped data to epochs (3-D) layout.

        Raises
        ------
        RuntimeError
            If data has not been cropped yet.
        """
        if self._crop_mode is None:
            raise RuntimeError("Cannot convert uncropped data. Call crop() first.")
        if self._crop_mode == "epochs":
            logger.warning("Data is already in epochs mode.")
            return

        # continues (total_samples, nchan) -> epochs
        trial_len = int(self.trial_ends[0] - self.trial_starts[0])
        onset_offset = int(self.vision_onsets[0] - self.trial_starts[0])
        n = len(self.trial_starts)

        self.neuro = self.neuro.reshape(n, trial_len, -1)
        self.vision = self.vision.reshape(n, trial_len)
        self.trial = self.trial.reshape(n, trial_len)

        self.trial_starts = np.zeros(n, dtype=int)
        self.trial_ends = np.full(n, trial_len, dtype=int)
        self.vision_onsets = np.full(n, onset_offset, dtype=int)

        self._crop_mode = "epochs"
        logger.info("Converted to epochs mode: neuro shape {}", self.neuro.shape)

    # ------------------------------------------------------------------
    # plot()
    # ------------------------------------------------------------------

    def plot(
        self,
        window: tuple[float | int, float | int] = (0.0, 5.0),
        figsize: tuple[float, float] = (6, 3),
        cmap_neuro: str = "Greys",
        cmap_ontime: str = "summer",
        color_offtime: str = "black",
        marker_size: float = 40,
    ):
        """Plot neural activity alongside stimulus labels.

        Parameters
        ----------
        window : tuple of float | int
            Display window.  *float* values are **seconds**,
            *int* values are **samples**.
        figsize : tuple of float
            Figure size ``(width, height)``.
        cmap_neuro : str
            Colormap for neural heatmap.
        cmap_ontime : str
            Colormap for in-trial time.
        color_offtime : str
            Color for off-trial points.
        marker_size : float
            Scatter marker size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        from vneurotk.viz.data import plot_data

        tw = self.trial_info["trial_window"] if self.trial_info is not None else None

        # epochs mode: flatten (n_trials, n_timebins, n_chan) → (n_total, n_chan)
        neuro = self.neuro
        vision = self.vision
        trial = self.trial
        if self._crop_mode == "epochs":
            neuro = neuro.reshape(-1, neuro.shape[-1])
            if vision is not None:
                vision = vision.ravel()
            if trial is not None:
                trial = trial.ravel()

        return plot_data(
            neuro=neuro,
            visual=vision,
            sfreq=self.neuro_info["sfreq"],
            trial=trial,
            trial_window=tw,
            figsize=figsize,
            window=window,
            cmap_neuro=cmap_neuro,
            cmap_ontime=cmap_ontime,
            color_offtime=color_offtime,
            marker_size=marker_size,
        )

    # ------------------------------------------------------------------
    # save()
    # ------------------------------------------------------------------

    def save(self, path: Any) -> None:
        """Persist the configured data to an HDF5 file.

        Parameters
        ----------
        path : VTKPath | pathlib.Path | str
            Destination file path.  If a ``VTKPath`` instance is given its
            ``.fpath`` attribute is used.

        Raises
        ------
        RuntimeError
            If :meth:`configure` has not been called yet.
        """
        if not self.configured:
            raise RuntimeError("Cannot save unconfigured BaseData. Call configure() first.")

        fpath = self._resolve_path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(fpath, "w") as f:
            # --- neuro: COO sparse for large 3D data, dense otherwise ---
            neuro_arr = self.neuro
            use_coo = neuro_arr.ndim == 3 and (neuro_arr == 0).sum() / neuro_arr.size > 0.5
            if use_coo:
                from scipy.sparse import coo_matrix

                flat = neuro_arr.reshape(-1, neuro_arr.shape[-1])
                sparse = coo_matrix(flat)
                f.create_dataset("neuro_row", data=sparse.row)
                f.create_dataset("neuro_col", data=sparse.col)
                f.create_dataset("neuro_data", data=sparse.data)
                f.attrs["neuro_format"] = "coo"
                f.attrs["neuro_shape"] = list(neuro_arr.shape)
                f.attrs["neuro_dtype"] = str(neuro_arr.dtype)
            else:
                f.create_dataset("neuro", data=neuro_arr)
                f.attrs["neuro_format"] = "dense"

            # vision: flatten for storage, save shape/dtype metadata
            vis_flat = self.vision.ravel()
            if vis_flat.dtype == object:
                vis_arr = np.array(
                    [v if isinstance(v, str) else "" for v in vis_flat],
                    dtype=h5py.string_dtype(),
                )
                f.create_dataset("vision", data=vis_arr)
                f.attrs["vision_is_str"] = True
            else:
                f.create_dataset("vision", data=vis_flat)
                f.attrs["vision_is_str"] = False
            f.attrs["vision_shape"] = list(self.vision.shape)

            f.create_dataset("trial", data=self.trial)
            f.create_dataset("trial_starts", data=self.trial_starts)
            f.create_dataset("trial_ends", data=self.trial_ends)
            f.create_dataset("vision_onsets", data=self.vision_onsets)

            # --- crop mode ---
            f.attrs["crop_mode"] = self._crop_mode if self._crop_mode else ""
            f.attrs["data_level"] = self.data_level

            # --- neuro_info (as attributes on a group) ---
            ni = f.create_group("neuro_info")
            for k, v in self.neuro_info.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    ni.attrs[k] = (
                        np.array(v, dtype=h5py.string_dtype())
                        if all(isinstance(x, str) for x in v)
                        else np.array(v)
                    )
                else:
                    ni.attrs[k] = v

            # --- vision_info ---
            vi = f.create_group("vision_info")
            vi.attrs["n_stim"] = self.vision_info["n_stim"]
            stim_ids = self.vision_info["stim_ids"]
            if stim_ids and isinstance(stim_ids[0], str):
                vi.create_dataset(
                    "stim_ids",
                    data=np.array(stim_ids, dtype=h5py.string_dtype()),
                )
            else:
                vi.create_dataset("stim_ids", data=np.array(stim_ids))

            # --- trial_info ---
            ti = f.create_group("trial_info")
            ti.attrs["baseline"] = np.array(self.trial_info["baseline"])
            ti.attrs["trial_window"] = np.array(self.trial_info["trial_window"])

            # --- trial_meta (DataFrame → per-column datasets) ---
            if self.trial_meta is not None:
                tm = f.create_group("trial_meta")
                for col in self.trial_meta.columns:
                    vals = self.trial_meta[col].values
                    if vals.dtype == object:
                        vals = np.array([str(v) for v in vals], dtype=h5py.string_dtype())
                    tm.create_dataset(col, data=vals)

        logger.info("Saved BaseData to {}", fpath)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _window_to_samples(trial_window: list[float | int], sfreq: float) -> list[int]:
        """Convert a trial window to sample offsets.

        Float values are treated as seconds; int values as raw samples.
        """
        result: list[int] = []
        for val in trial_window:
            if isinstance(val, float):
                result.append(int(round(val * sfreq)))
            else:
                result.append(int(val))
        return result

    @staticmethod
    def _resolve_path(path: Any) -> Path:
        """Return a :class:`pathlib.Path` from *path*.

        Supports ``VTKPath`` (via ``.fpath``), ``pathlib.Path``, and ``str``.
        """
        if hasattr(path, "fpath"):
            return Path(path.fpath)
        return Path(path)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [
            f"BaseData(ntime={self.ntime}, nchan={self.nchan}",
            f"n_trials={self.n_trials}, configured={self.configured}",
        ]
        if self.data_level != "timepoint":
            parts.append(f"data_level='{self.data_level}'")
        if self._neuro is None and self._neuro_loader is not None:
            parts.append("neuro=<lazy>")
        return ", ".join(parts) + ")"

    def _repr_html_(self) -> str:
        return self.info._repr_html_()
