"""HuggingFace Transformers vision model backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from vneurotk.vision.extractor.backend.base import BaseBackend, LayerInfo
from vneurotk.vision.representation import ModelMeta

__all__ = ["TransformersBackend"]


class TransformersBackend(BaseBackend):
    """Backend powered by HuggingFace ``transformers``.

    Supports any vision model loadable via ``AutoModel.from_pretrained()``.
    CLIP models are detected automatically and handled via
    ``CLIPModel`` + ``get_image_features()``.

    Parameters
    ----------
    device : str or torch.device
        Inference device (default ``"cpu"``).
    learning_paradigm : str
        Training paradigm label, e.g. ``"supervised"`` or ``"contrastive"``.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        learning_paradigm: str = "supervised",
    ) -> None:
        super().__init__(device)
        self._model_name: str = ""
        self._learning_paradigm = learning_paradigm
        self._processor = None
        self._architecture: str = ""
        self._is_clip: bool = False

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a HuggingFace vision model.

        Parameters
        ----------
        model_name : str
            HuggingFace model ID, e.g. ``"openai/clip-vit-base-patch32"``
            or ``"facebook/dinov2-base"``.
        pretrained : bool
            If ``False``, load with randomized weights (for testing).
        """
        try:
            from transformers import (
                AutoModel,
                AutoProcessor,
                CLIPProcessor,
                CLIPVisionModelWithProjection,
                SiglipVisionModel,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TransformersBackend.  "
                "Install with: uv add transformers"
            ) from exc

        logger.info("Loading transformers model: {} (pretrained={})", model_name, pretrained)

        n = model_name.lower()
        self._is_clip = "clip" in n
        self._is_siglip = "siglip" in n

        if self._is_clip:
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                model_name, use_safetensors=True
            )
        elif self._is_siglip:
            self._processor = AutoProcessor.from_pretrained(model_name)
            self.model = SiglipVisionModel.from_pretrained(model_name, use_safetensors=True)
        else:
            self._processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        self._hook_model = self.model

        self.model.eval()
        self.model.to(self.device)

        self._model_name = model_name
        self._architecture = self._infer_arch(model_name)
        logger.info(
            "Loaded transformers model: {} | arch={} | clip={}",
            model_name,
            self._architecture,
            self._is_clip,
        )

    def preprocess(self, image: Any) -> dict[str, Any]:
        """Preprocess a PIL Image using the model's registered processor.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray
            Input image.

        Returns
        -------
        dict[str, Any]
            Processor output dict with ``pixel_values`` and any other
            required inputs.
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        if self._is_clip:
            inputs = self._processor(images=image, return_tensors="pt")
        else:
            inputs = self._processor(images=image, return_tensors="pt")
        return dict(inputs)

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run the model forward pass.

        For CLIP models the ``image_embeds`` attribute is attached to the
        vision encoder output so that ``EmbeddingPolicy.PROJECTION_OUT``
        can retrieve it.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Any
            HuggingFace ``ModelOutput`` or ``Tensor``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        moved = self._move_to_device(inputs)
        with torch.no_grad():
            if self._is_clip:
                # CLIPVisionModelWithProjection returns CLIPVisionModelOutput with image_embeds
                return self.model(**moved)
            return self.model(**moved)

    def enumerate_layers(self) -> list[LayerInfo]:
        """Return metadata for all named modules.

        For CLIP, enumerates the vision encoder sub-modules only.

        Returns
        -------
        list[LayerInfo]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        source = self._hook_model if self._is_clip else self.model
        result = []
        for name, module in source.named_modules():
            if not name:
                continue
            result.append(
                LayerInfo(
                    name=name,
                    module_type=type(module).__name__,
                    depth=name.count(".") + 1,
                    n_params=sum(p.numel() for p in module.parameters()),
                )
            )
        return result

    def get_model_meta(self) -> ModelMeta:
        """Return ModelMeta for the loaded transformers model."""
        return ModelMeta(
            model_name=self._model_name,
            source="transformers",
            architecture=self._architecture,
            learning_paradigm=self._learning_paradigm,
            encoder_type="vision",
            embedding_policy="",
        )

    # ------------------------------------------------------------------
    # Hook management override for CLIP
    # ------------------------------------------------------------------

    def register_hooks(self, layer_names: list[str]) -> None:
        """Attach hooks on the vision encoder (CLIP) or full model.

        Parameters
        ----------
        layer_names : list[str]
            Module names as returned by :meth:`enumerate_layers`.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded before registering hooks.")

        self.remove_hooks()
        self._activations.clear()

        source = self._hook_model if self._is_clip else self.model

        import torch.nn as nn

        named = dict(source.named_modules())
        missing = [n for n in layer_names if n not in named]
        if missing:
            raise ValueError(f"Layer(s) not found in model: {missing}")

        for name in layer_names:
            module = named[name]

            def _hook(mod: nn.Module, inp: Any, output: Any, _n: str = name) -> None:  # noqa: ARG001
                act = output[0] if isinstance(output, tuple) else output
                if hasattr(act, "last_hidden_state"):
                    act = act.last_hidden_state
                if isinstance(act, Tensor) and act.ndim > 1 and act.shape[0] == 1:
                    act = act.squeeze(0)
                if isinstance(act, Tensor):
                    self._activations[_n] = act.detach().cpu()

            handle = module.register_forward_hook(_hook)
            self._hooks.append(handle)

        logger.debug(
            "Registered {} hooks on {}",
            len(layer_names),
            "CLIP vision_model" if self._is_clip else "model",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_arch(model_name: str) -> str:
        n = model_name.lower()
        for arch in ("clip", "vit", "dino", "resnet", "convnext", "swin", "deit", "beit"):
            if arch in n:
                return arch
        return "unknown"
