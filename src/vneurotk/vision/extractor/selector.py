"""Layer selection strategies.

Three concrete selectors are provided:

- :class:`BlockLevelSelector` — major blocks (ViT blocks, ResNet layers)
- :class:`AllLeafSelector`    — all leaf modules (no children)
- :class:`CustomSelector`     — explicit user-supplied list
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

import torch.nn as nn

__all__ = [
    "LayerSelector",
    "BlockLevelSelector",
    "AllLeafSelector",
    "CustomSelector",
]


class LayerSelector(ABC):
    """Abstract base class for layer selection strategies.

    Subclasses implement :meth:`select`, which receives a model and
    returns an ordered list of module name strings to hook.
    """

    @abstractmethod
    def select(self, model: nn.Module) -> list[str]:
        """Return layer names to hook.

        Parameters
        ----------
        model : nn.Module
            The PyTorch model.

        Returns
        -------
        list[str]
            Ordered module names as produced by ``model.named_modules()``.
        """


class BlockLevelSelector(LayerSelector):
    """Select major block-level modules appropriate for the architecture.

    Uses regex patterns matched against module names.  Architecture
    patterns are tried in order; the first match wins.  Falls back to
    top-level children if no pattern matches.

    Parameters
    ----------
    max_depth : int
        Maximum nesting depth to include (default 2).  Controls how
        deeply nested sub-blocks are included.
    include_patterns : list[str] or None
        Additional regex patterns to include alongside defaults.
    """

    _ARCH_PATTERNS: list[tuple[str, int]] = [
        (r"^blocks\.\d+$", 2),  # timm ViT
        (r"^encoder\.layers\.\d+$", 3),  # HF ViT (plural)
        (r"^encoder\.layer\.\d+$", 3),  # HF DINOv2 (singular)
        (r"^model\.layer\.\d+$", 3),  # HF DINOv3
        (r"^layer\d+\.\d+$", 3),  # ResNet
        (r"^features\.\d+$", 2),  # VGG / EfficientNet
        (r"^stages\.\d+$", 2),  # ConvNeXt
        (r"^layers\.\d+$", 2),  # Swin / generic
        (r"^vision_model\.encoder\.layers\.\d+$", 4),  # SigLIP / SigLIP2
    ]

    def __init__(
        self,
        max_depth: int = 2,
        include_patterns: list[str] | None = None,
    ) -> None:
        self.max_depth = max_depth
        self._extra = [re.compile(p) for p in (include_patterns or [])]

    def select(self, model: nn.Module) -> list[str]:
        """Select block-level layers from *model*.

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        list[str]
        """
        compiled = [(re.compile(p), d) for p, d in self._ARCH_PATTERNS]
        selected: list[str] = []

        for name, _ in model.named_modules():
            if not name:
                continue
            depth = name.count(".") + 1

            matched = any(pat.match(name) and depth <= max_d for pat, max_d in compiled)
            if not matched:
                matched = any(pat.search(name) for pat in self._extra)

            if matched:
                selected.append(name)

        if not selected:
            selected = [n for n, _ in model.named_children() if n]

        return selected


class AllLeafSelector(LayerSelector):
    """Select all leaf modules (modules with no children).

    Parameters
    ----------
    exclude_types : tuple[type, ...] or None
        Module types to skip.  Defaults to activation and regularization
        layers that carry no representational content.
    """

    _DEFAULT_EXCLUDE = (
        nn.Dropout,
        nn.Identity,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Sigmoid,
        nn.Softmax,
    )

    def __init__(self, exclude_types: tuple | None = None) -> None:
        self.exclude_types = exclude_types if exclude_types is not None else self._DEFAULT_EXCLUDE

    def select(self, model: nn.Module) -> list[str]:
        """Return names of all non-excluded leaf modules.

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        list[str]
        """
        return [
            name
            for name, module in model.named_modules()
            if name and not list(module.children()) and not isinstance(module, self.exclude_types)
        ]


class CustomSelector(LayerSelector):
    """Use an explicit user-supplied list of layer names.

    Parameters
    ----------
    layer_names : list[str]
        Exact module names as they appear in ``model.named_modules()``.

    Raises
    ------
    ValueError
        During :meth:`select` if any name is not found in the model.
    """

    def __init__(self, layer_names: list[str]) -> None:
        self.layer_names = list(layer_names)

    def select(self, model: nn.Module) -> list[str]:
        """Validate and return the configured layer names.

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        list[str]

        Raises
        ------
        ValueError
            If any layer name is absent from the model.
        """
        available = {name for name, _ in model.named_modules()}
        missing = [n for n in self.layer_names if n not in available]
        if missing:
            raise ValueError(
                f"Layer(s) not found in model: {missing}. "
                f"Inspect available names with model.named_modules()."
            )
        return list(self.layer_names)
