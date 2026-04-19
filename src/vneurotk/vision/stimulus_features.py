"""Experiment-level stimulus feature container.

:class:`StimulusFeatures` aggregates per-image activations across all
stimuli into numpy arrays indexed by stimulus ID.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vneurotk.vision.representation import ModelMeta

__all__ = ["StimulusFeatures"]


class StimulusFeatures:
    """Multi-stimulus feature container indexed by stimulus ID.

    Parameters
    ----------
    stim_ids : list
        Ordered stimulus identifiers corresponding to the first axis of
        all arrays.
    features : dict[str, np.ndarray]
        Layer-name → activation array of shape ``(n_stim, ...)``.
    final_embedding : np.ndarray
        Shape ``(n_stim, D)``.  The designated embedding per stimulus.
    model_meta : ModelMeta
        Model-level metadata shared across all stimuli.

    Examples
    --------
    >>> sf = extractor.extract_stimuli(images)
    >>> sf.final_embedding.shape
    (100, 768)
    >>> sf["blocks_11"].shape
    (100, 197, 768)
    >>> sub = sf.select([0, 1, 2])
    >>> sub.final_embedding.shape
    (3, 768)
    """

    def __init__(
        self,
        stim_ids: list,
        features: dict[str, np.ndarray],
        final_embedding: np.ndarray,
        model_meta: ModelMeta,
    ) -> None:
        self.stim_ids = list(stim_ids)
        self.features = features
        self.final_embedding = np.asarray(final_embedding)
        self.model_meta = model_meta
        self._id_to_idx: dict[Any, int] = {
            sid: i for i, sid in enumerate(self.stim_ids)
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_stim(self) -> int:
        """Number of stimuli."""
        return len(self.stim_ids)

    @property
    def layer_names(self) -> list[str]:
        """Names of all retained layers."""
        return list(self.features.keys())

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, layer: str) -> np.ndarray:
        """Return feature array for *layer*, shape ``(n_stim, ...)``."""
        return self.features[layer]

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, ids: np.ndarray | list) -> StimulusFeatures:
        """Return a subset of stimuli by their IDs.

        Parameters
        ----------
        ids : np.ndarray or list
            Stimulus IDs to select.  Order is preserved.

        Returns
        -------
        StimulusFeatures

        Raises
        ------
        KeyError
            If any ID in *ids* is not in :attr:`stim_ids`.
        """
        ids_list = list(ids)
        indices = np.array([self._id_to_idx[sid] for sid in ids_list])
        return StimulusFeatures(
            stim_ids=ids_list,
            features={layer: arr[indices] for layer, arr in self.features.items()},
            final_embedding=self.final_embedding[indices],
            model_meta=self.model_meta,
        )

    def select_by_index(self, indices: np.ndarray | list) -> StimulusFeatures:
        """Return a subset of stimuli by positional index.

        Parameters
        ----------
        indices : np.ndarray or list
            Integer indices into :attr:`stim_ids`.

        Returns
        -------
        StimulusFeatures
        """
        idx = np.asarray(indices)
        return StimulusFeatures(
            stim_ids=[self.stim_ids[i] for i in idx],
            features={layer: arr[idx] for layer, arr in self.features.items()},
            final_embedding=self.final_embedding[idx],
            model_meta=self.model_meta,
        )

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def to_tensor(self, layer: str | None = None) -> Any:
        """Convert features to a PyTorch tensor.

        Parameters
        ----------
        layer : str or None
            If given, return that layer's array as a tensor
            ``(n_stim, ...)``.  If ``None``, return
            :attr:`final_embedding`.

        Returns
        -------
        torch.Tensor

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for to_tensor()") from exc
        arr = self.features[layer] if layer is not None else self.final_embedding
        return torch.from_numpy(np.asarray(arr))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"StimulusFeatures("
            f"n_stim={self.n_stim}, "
            f"n_layers={len(self.features)}, "
            f"embedding_shape={self.final_embedding.shape}, "
            f"model={self.model_meta.model_name!r})"
        )
