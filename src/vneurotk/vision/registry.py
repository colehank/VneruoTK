"""Model registry for VneuroTK vision module.

Provides :class:`ModelRegistry` and :data:`REGISTRY` (the built-in
registry with five pre-configured models).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ModelConfig", "ModelRegistry", "REGISTRY"]


@dataclass
class ModelConfig:
    """Configuration for a registered vision model.

    Parameters
    ----------
    source : str
        Backend identifier: ``"timm"``, ``"transformers"``, or
        ``"thingsvision"``.
    model_id : str
        Library-specific model identifier.
    policy : str
        :class:`~vneurotk.vision.extractor.policy.EmbeddingPolicy` value
        string, e.g. ``"cls_token"``.
    paradigm : str
        Learning paradigm, e.g. ``"supervised"`` or ``"selfsupervised"``.
    extra : dict
        Any additional keyword arguments forwarded to the backend's
        ``load()`` call.
    """

    source: str
    model_id: str
    policy: str
    paradigm: str
    extra: dict[str, Any] | None = None


class ModelRegistry:
    """Registry mapping short names to :class:`ModelConfig`.

    Parameters
    ----------
    configs : dict[str, ModelConfig]
        Initial registry entries.

    Examples
    --------
    >>> cfg = REGISTRY.get("vit-b-16-imagenet")
    >>> cfg.source
    'timm'
    """

    def __init__(self, configs: dict[str, ModelConfig] | None = None) -> None:
        self._configs: dict[str, ModelConfig] = dict(configs or {})

    def get(self, name: str) -> ModelConfig:
        """Retrieve a registered model configuration.

        Parameters
        ----------
        name : str
            Short model name.

        Returns
        -------
        ModelConfig

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        if name not in self._configs:
            available = ", ".join(sorted(self._configs))
            raise KeyError(f"Model {name!r} not in registry.  Available: {available}")
        return self._configs[name]

    def register(self, name: str, config: ModelConfig) -> None:
        """Add or replace a registry entry.

        Parameters
        ----------
        name : str
            Short model name.
        config : ModelConfig
            Model configuration.
        """
        self._configs[name] = config

    def list(self) -> list[str]:
        """Return a sorted list of registered model names.

        Returns
        -------
        list[str]
        """
        return sorted(self._configs)

    def __contains__(self, name: str) -> bool:
        return name in self._configs

    def __repr__(self) -> str:
        return f"ModelRegistry(n={len(self._configs)}, models={self.list()})"


# ------------------------------------------------------------------
# Built-in registry
# ------------------------------------------------------------------

REGISTRY = ModelRegistry(
    {
        # ── timm supervised ───────────────────────────────────────────────
        "vit-b-16-imagenet": ModelConfig(
            source="timm",
            model_id="vit_base_patch16_224",
            policy="all_tokens",
            paradigm="supervised",
        ),
        "vit-b-16-in21k": ModelConfig(
            source="timm",
            model_id="vit_base_patch16_224.augreg2_in21k_ft_in1k",
            policy="all_tokens",
            paradigm="supervised",
        ),
        "resnet50": ModelConfig(
            source="timm",
            model_id="resnet50",
            policy="pre_head",
            paradigm="supervised",
        ),
        "resnet50-a1": ModelConfig(
            source="timm",
            model_id="resnet50.a1_in1k",
            policy="pre_head",
            paradigm="supervised",
        ),
        "resnetv2-50": ModelConfig(
            source="timm",
            model_id="resnetv2_50.a1h_in1k",
            policy="pre_head",
            paradigm="supervised",
        ),
        # ── timm self-supervised ──────────────────────────────────────────
        "vit-b-16-dino": ModelConfig(
            source="timm",
            model_id="vit_base_patch16_224.dino",
            policy="all_tokens",
            paradigm="selfsupervised",
        ),
        # ── transformers self-supervised ──────────────────────────────────
        "dinov2-vit-b": ModelConfig(
            source="transformers",
            model_id="facebook/dinov2-base",
            policy="all_tokens",
            paradigm="selfsupervised",
        ),
        "dinov3-vit-b": ModelConfig(
            source="transformers",
            model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
            policy="all_tokens",
            paradigm="selfsupervised",
        ),
        "dinov3-vit-s": ModelConfig(
            source="transformers",
            model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
            policy="all_tokens",
            paradigm="selfsupervised",
        ),
        # ── transformers contrastive ──────────────────────────────────────
        "clip-vit-b-32": ModelConfig(
            source="transformers",
            model_id="openai/clip-vit-base-patch32",
            policy="projection_out",
            paradigm="contrastive",
        ),
        "siglip-b-16": ModelConfig(
            source="transformers",
            model_id="google/siglip-base-patch16-224",
            policy="mean_pool",
            paradigm="contrastive",
        ),
        "siglip2-b-16": ModelConfig(
            source="transformers",
            model_id="google/siglip2-base-patch16-224",
            policy="mean_pool",
            paradigm="contrastive",
        ),
    }
)
