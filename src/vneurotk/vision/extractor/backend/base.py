"""Abstract base class for vision model backends.

A backend encapsulates:
1. Loading a model from its native library.
2. Preprocessing images into the required input format.
3. Running the forward pass.
4. Enumerating available layers.
5. Managing forward hooks to capture intermediate activations.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from vneurotk.vision.representation import ModelMeta

__all__ = ["BaseBackend", "LayerInfo"]


@dataclass
class LayerInfo:
    """Minimal metadata for an enumerated layer.

    Parameters
    ----------
    name : str
        Module name as from ``named_modules()``.
    module_type : str
        Class name of the module.
    depth : int
        Nesting depth in the module tree.
    n_params : int
        Approximate number of parameters in this module.
    """

    name: str
    module_type: str
    depth: int
    n_params: int = 0


class BaseBackend(ABC):
    """Abstract base for all feature-extraction backends.

    Subclasses implement :meth:`load`, :meth:`preprocess`,
    :meth:`forward`, :meth:`enumerate_layers`, and
    :meth:`get_model_meta`.  Hook management is provided here and shared.

    Parameters
    ----------
    device : str or torch.device
        Device for inference (default ``"cpu"``).
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model: nn.Module | None = None
        self._hooks: list[Any] = []
        self._activations: OrderedDict[str, Tensor] = OrderedDict()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load the model into memory.

        Parameters
        ----------
        model_name : str
            Library-specific model identifier.
        pretrained : bool
            Whether to load pretrained weights.
        """

    @abstractmethod
    def preprocess(self, image: Any) -> dict[str, Any]:
        """Convert a PIL Image or Tensor to the model's input format.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray or Tensor
            Raw input image.

        Returns
        -------
        dict[str, Any]
            Keyword arguments ready for ``forward()``.
        """

    @abstractmethod
    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run a forward pass and return the raw model output.

        Parameters
        ----------
        inputs : dict[str, Any]
            Preprocessed inputs from :meth:`preprocess`.

        Returns
        -------
        Any
            Raw model output (Tensor or HuggingFace ModelOutput).
        """

    @abstractmethod
    def enumerate_layers(self) -> list[LayerInfo]:
        """Return metadata for all available layers.

        Returns
        -------
        list[LayerInfo]
        """

    @abstractmethod
    def get_model_meta(self) -> ModelMeta:
        """Return model-level metadata.

        Returns
        -------
        ModelMeta
        """

    # ------------------------------------------------------------------
    # Hook management (shared implementation)
    # ------------------------------------------------------------------

    def register_hooks(self, layer_names: list[str]) -> None:
        """Attach forward hooks to the specified layers.

        Hooks store a detached CPU copy of each layer output in
        :attr:`_activations`.  Call :meth:`remove_hooks` when done.

        Parameters
        ----------
        layer_names : list[str]
            Module names to hook.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        ValueError
            If any name is not found in the model.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before registering hooks.")

        self.remove_hooks()
        self._activations.clear()

        named = dict(self.model.named_modules())
        missing = [n for n in layer_names if n not in named]
        if missing:
            raise ValueError(f"Layer(s) not found in model: {missing}")

        for name in layer_names:
            module = named[name]

            def _hook(mod: nn.Module, inp: Any, output: Any, _n: str = name) -> None:  # noqa: ARG001
                act = output[0] if isinstance(output, tuple) else output
                if act.ndim > 1 and act.shape[0] == 1:
                    act = act.squeeze(0)
                self._activations[_n] = act.detach().cpu()

            handle = module.register_forward_hook(_hook)
            self._hooks.append(handle)

        logger.debug("Registered {} hooks", len(layer_names))

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Removed all hooks")

    def collect_activations(self) -> OrderedDict[str, Tensor]:
        """Return captured activations and clear the buffer.

        Returns
        -------
        OrderedDict[str, Tensor]
        """
        result = OrderedDict(self._activations)
        self._activations.clear()
        return result

    # ------------------------------------------------------------------
    # Layer name normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_layer_name(raw_name: str) -> str:
        """Convert a raw module name to a normalized identifier.

        Parameters
        ----------
        raw_name : str
            E.g. ``"blocks.11.attn.proj"``.

        Returns
        -------
        str
            E.g. ``"blocks_11_attn_proj"``.
        """
        name = raw_name.replace(".", "_")
        return re.sub(r"[^a-zA-Z0-9_]", "_", name)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _move_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move all Tensor values in *inputs* to :attr:`device`."""
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in inputs.items()
        }
