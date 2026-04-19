"""VisionExtractor — unified entry point for DNN feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from vneurotk.vision.extractor.backend.base import BaseBackend
from vneurotk.vision.extractor.policy import EmbeddingPolicy
from vneurotk.vision.extractor.selector import BlockLevelSelector, LayerSelector
from vneurotk.vision.registry import REGISTRY, ModelConfig
from vneurotk.vision.visual_representations import VisualRepresentations

__all__ = ["VisionExtractor"]


class VisionExtractor:
    """Unified interface for extracting DNN representations from images.

    Composes a :class:`~vneurotk.vision.extractor.backend.base.BaseBackend`,
    a :class:`~vneurotk.vision.extractor.selector.LayerSelector`, and an
    :class:`~vneurotk.vision.extractor.policy.EmbeddingPolicy`.

    Parameters
    ----------
    model_name : str
        Short name looked up in :data:`~vneurotk.vision.registry.REGISTRY`,
        or a library-specific identifier when combined with an explicit
        *backend*.
    selector : LayerSelector or None
        Layer selection strategy.  Defaults to
        :class:`~vneurotk.vision.extractor.selector.BlockLevelSelector`.
    embedding_policy : EmbeddingPolicy or str or None
        Strategy for computing the final embedding.  When ``None`` the
        policy is inferred from the registry entry.
    device : str
        Inference device (default ``"cpu"``).
    pretrained : bool
        Load pretrained weights (default ``True``).
    custom_fn : callable or None
        Required when *embedding_policy* is ``"custom"``.
    """

    def __init__(
        self,
        model_name: str,
        selector: LayerSelector | None = None,
        embedding_policy: EmbeddingPolicy | str | None = None,
        device: str = "cpu",
        pretrained: bool = True,
        custom_fn: Any = None,
    ) -> None:
        self._custom_fn = custom_fn
        self._selector = selector or BlockLevelSelector()

        cfg = REGISTRY.get(model_name) if model_name in REGISTRY else None
        self._backend = self._build_backend(model_name, cfg, device)
        self._backend.load(
            cfg.model_id if cfg else model_name,
            pretrained=pretrained,
        )

        policy_str = embedding_policy or (cfg.policy if cfg else "mean_pool")
        self._policy = EmbeddingPolicy(policy_str)

        hook_model = getattr(self._backend, "_hook_model", self._backend.model)
        layer_names = self._selector.select(hook_model)
        self._backend.register_hooks(layer_names)
        self._layer_names = layer_names

        logger.info(
            "VisionExtractor ready | model={} | layers={} | policy={}",
            model_name,
            len(layer_names),
            self._policy,
        )

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model: Any,
        backend: BaseBackend,
        embedding_policy: EmbeddingPolicy | str,
        model_name: str = "custom",
        selector: LayerSelector | None = None,
        custom_fn: Any = None,
    ) -> VisionExtractor:
        """Build a VisionExtractor from an already-loaded model.

        Parameters
        ----------
        model : nn.Module
            Pre-loaded PyTorch model.
        backend : BaseBackend
            Backend instance with *model* already assigned.
        embedding_policy : EmbeddingPolicy or str
            Embedding strategy.
        model_name : str
            Label used in metadata.
        selector : LayerSelector or None
            Defaults to :class:`BlockLevelSelector`.
        custom_fn : callable or None
            Required for ``EmbeddingPolicy.CUSTOM``.

        Returns
        -------
        VisionExtractor
        """
        inst = object.__new__(cls)
        inst._custom_fn = custom_fn
        inst._selector = selector or BlockLevelSelector()
        inst._backend = backend
        inst._backend.model = model
        inst._policy = EmbeddingPolicy(embedding_policy)

        layer_names = inst._selector.select(model)
        backend.register_hooks(layer_names)
        inst._layer_names = layer_names
        return inst

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, image: Any) -> VisualRepresentations:
        """Extract DNN activations for one image or a collection of stimuli.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or Path or dict
            Single image → returns :class:`VisualRepresentations` with
            ``n_stim=1``.  A ``dict`` mapping stimulus IDs to images →
            returns :class:`VisualRepresentations` with ``n_stim=len(dict)``.
            ``pathlib.Path`` values are opened automatically.

        Returns
        -------
        VisualRepresentations
            Always.  ``n_stim=1`` for single images, ``n_stim=N`` for dicts.
        """
        if isinstance(image, dict):
            return self._extract_batch(image)
        return self._extract_single(image)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def layer_names(self) -> list[str]:
        """Names of all hooked layers."""
        return list(self._layer_names)

    @property
    def policy(self) -> EmbeddingPolicy:
        """Active embedding policy."""
        return self._policy

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_single(self, image: Any) -> VisualRepresentations:
        from pathlib import Path

        from PIL import Image as PILImage

        if isinstance(image, Path):
            image = PILImage.open(image).convert("RGB")

        inputs = self._backend.preprocess(image)
        raw_output = self._backend.forward(inputs)
        activations = self._backend.collect_activations()

        final_embedding = (
            self._policy.apply(raw_output, activations, self._custom_fn).detach().cpu().numpy()
        )

        features = {name: act.numpy()[np.newaxis] for name, act in activations.items()}

        model_meta = self._backend.get_model_meta()
        model_meta.embedding_policy = self._policy.value

        return VisualRepresentations(
            stim_ids=[0],
            features=features,
            final_embedding=final_embedding[np.newaxis],
            model_meta=model_meta,
        )

    def _extract_batch(self, images: dict) -> VisualRepresentations:
        stim_ids = list(images.keys())
        all_features: dict[str, list[np.ndarray]] = {}
        all_embeddings: list[np.ndarray] = []

        for sid in stim_ids:
            rep = self._extract_single(images[sid])
            for layer, arr in rep.features.items():
                all_features.setdefault(layer, []).append(arr[0])
            all_embeddings.append(rep.final_embedding[0])

        features = {layer: np.stack(arrs, axis=0) for layer, arrs in all_features.items()}
        final_embedding = np.stack(all_embeddings, axis=0)

        model_meta = self._backend.get_model_meta()
        model_meta.embedding_policy = self._policy.value

        logger.info(
            "Extracted batch | n={} | layers={} | embedding_shape={}",
            len(stim_ids),
            len(features),
            final_embedding.shape,
        )
        return VisualRepresentations(
            stim_ids=stim_ids,
            features=features,
            final_embedding=final_embedding,
            model_meta=model_meta,
        )

    @staticmethod
    def _build_backend(
        model_name: str,
        cfg: ModelConfig | None,
        device: str,
    ) -> BaseBackend:
        from vneurotk.vision.extractor.backend.timm_backend import TimmBackend
        from vneurotk.vision.extractor.backend.transformers_backend import TransformersBackend

        source = cfg.source if cfg else _guess_source(model_name)
        paradigm = cfg.paradigm if cfg else "supervised"

        if source == "timm":
            return TimmBackend(device=device, learning_paradigm=paradigm)
        if source == "transformers":
            return TransformersBackend(device=device, learning_paradigm=paradigm)
        if source == "thingsvision":
            from vneurotk.vision.extractor.backend.thingsvision_backend import ThingsVisionBackend

            return ThingsVisionBackend(device=device, learning_paradigm=paradigm)
        raise ValueError(
            f"Unknown backend source {source!r}.  "
            "Expected 'timm', 'transformers', or 'thingsvision'."
        )


def _guess_source(model_name: str) -> str:
    """HuggingFace model IDs contain a slash."""
    return "transformers" if "/" in model_name else "timm"
