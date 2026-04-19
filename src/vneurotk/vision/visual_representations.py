"""Unified DNN activation container.

:class:`VisualRepresentations` is the single output type for all
:meth:`~vneurotk.vision.VisionExtractor.extract` calls.  Single-image
extraction stores activations as ``n_stim=1``; batch extraction stacks
along the first axis.  Both numpy and PyTorch tensor views are available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vneurotk.vision.representation import ModelMeta

__all__ = ["VisualRepresentations"]


class VisualRepresentations:
    """Unified container for DNN activations from one or more images.

    Parameters
    ----------
    stim_ids : list
        Ordered stimulus identifiers.  ``[0]`` for single-image extraction.
    features : dict[str, np.ndarray]
        Layer name → activation array of shape ``(n_stim, ...)``.
    final_embedding : np.ndarray
        Shape ``(n_stim, ...)``.  Derived by the active
        :class:`~vneurotk.vision.extractor.policy.EmbeddingPolicy`.
    model_meta : ModelMeta
        Model-level metadata shared across all stimuli.

    Examples
    --------
    Single image:

    >>> vr = extractor.extract(img)
    >>> vr.n_stim
    1
    >>> vr.final_embedding.shape
    (1, 197, 768)
    >>> vr["blocks.0"].shape
    (1, 197, 768)
    >>> vr.numpy()                   # final_embedding as ndarray
    array(...)
    >>> vr.to_tensor("blocks.0")     # torch.Tensor
    tensor(...)

    Multiple images:

    >>> vrs = extractor.extract(images)   # images: dict[id, PIL.Image]
    >>> vrs.n_stim
    100
    >>> vrs.final_embedding.shape
    (100, 197, 768)
    >>> sub = vrs.select([0, 1, 2])
    >>> sub.n_stim
    3
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
        self._id_to_idx: dict[Any, int] = {sid: i for i, sid in enumerate(self.stim_ids)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_stim(self) -> int:
        """Number of stimuli (1 for single-image extraction)."""
        return len(self.stim_ids)

    @property
    def layer_names(self) -> list[str]:
        """Names of all hooked layers, in hook-registration order."""
        return list(self.features.keys())

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------

    def __getitem__(self, layer: str) -> np.ndarray:
        """Return feature array for *layer*, shape ``(n_stim, ...)``."""
        return self.features[layer]

    # ------------------------------------------------------------------
    # Numpy / Tensor access
    # ------------------------------------------------------------------

    def numpy(self, layer: str | None = None) -> np.ndarray:
        """Return activations as a numpy array.

        Parameters
        ----------
        layer : str or None
            Layer name to fetch.  ``None`` returns
            :attr:`final_embedding`.

        Returns
        -------
        np.ndarray
            Shape ``(n_stim, ...)``.
        """
        if layer is None:
            return self.final_embedding
        return self.features[layer]

    def to_tensor(self, layer: str | None = None) -> Any:
        """Return activations as a PyTorch tensor.

        Parameters
        ----------
        layer : str or None
            Layer name.  ``None`` converts :attr:`final_embedding`.

        Returns
        -------
        torch.Tensor
            Shape ``(n_stim, ...)``.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required for to_tensor()") from exc
        arr = self.features[layer] if layer is not None else self.final_embedding
        return torch.from_numpy(np.asarray(arr))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self, ids: np.ndarray | list) -> VisualRepresentations:
        """Return a subset of stimuli by their IDs.

        Parameters
        ----------
        ids : np.ndarray or list
            Stimulus IDs to select.  Order is preserved.

        Returns
        -------
        VisualRepresentations

        Raises
        ------
        KeyError
            If any ID in *ids* is not in :attr:`stim_ids`.
        """
        ids_list = list(ids)
        indices = np.array([self._id_to_idx[sid] for sid in ids_list])
        return VisualRepresentations(
            stim_ids=ids_list,
            features={layer: arr[indices] for layer, arr in self.features.items()},
            final_embedding=self.final_embedding[indices],
            model_meta=self.model_meta,
        )

    def select_by_index(self, indices: np.ndarray | list) -> VisualRepresentations:
        """Return a subset of stimuli by positional index.

        Parameters
        ----------
        indices : np.ndarray or list
            Integer indices into :attr:`stim_ids`.

        Returns
        -------
        VisualRepresentations
        """
        idx = np.asarray(indices)
        return VisualRepresentations(
            stim_ids=[self.stim_ids[i] for i in idx],
            features={layer: arr[idx] for layer, arr in self.features.items()},
            final_embedding=self.final_embedding[idx],
            model_meta=self.model_meta,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VisualRepresentations("
            f"n_stim={self.n_stim}, "
            f"n_layers={len(self.features)}, "
            f"embedding_shape={self.final_embedding.shape}, "
            f"model={self.model_meta.model_name!r})"
        )
