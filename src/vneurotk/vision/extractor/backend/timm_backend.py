"""timm-based vision model backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

from vneurotk.vision.extractor.backend.base import BaseBackend, LayerInfo
from vneurotk.vision.representation import ModelMeta

__all__ = ["TimmBackend"]


class TimmBackend(BaseBackend):
    """Backend powered by the ``timm`` library.

    Any model available via ``timm.create_model()`` is supported.
    Preprocessing uses the model's registered data config so no
    ImageNet mean/std are hard-coded.

    Parameters
    ----------
    device : str or torch.device
        Inference device (default ``"cpu"``).
    learning_paradigm : str
        Training paradigm label, e.g. ``"supervised"``.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        learning_paradigm: str = "supervised",
    ) -> None:
        super().__init__(device)
        self._model_name: str = ""
        self._learning_paradigm = learning_paradigm
        self._transform = None
        self._architecture: str = ""

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a timm model.

        Parameters
        ----------
        model_name : str
            E.g. ``"vit_base_patch16_224"`` or ``"resnet50"``.
        pretrained : bool
            Load pretrained weights.
        """
        try:
            import timm
            from timm.data import create_transform, resolve_data_config
        except ImportError as exc:
            raise ImportError(
                "timm is required for TimmBackend.  Install with: uv add timm"
            ) from exc

        logger.info("Loading timm model: {} (pretrained={})", model_name, pretrained)
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        self.model.to(self.device)

        self._model_name = model_name
        cfg = self.model.pretrained_cfg if hasattr(self.model, "pretrained_cfg") else {}
        data_config = resolve_data_config(cfg or {})
        self._transform = create_transform(**data_config)
        self._architecture = self._infer_arch(model_name)
        logger.info("Loaded timm model: {} | arch={}", model_name, self._architecture)

    def preprocess(self, image: Any) -> dict[str, Any]:
        """Preprocess a PIL Image using the model's registered transforms.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray
            Input image.

        Returns
        -------
        dict[str, Any]
            ``{"pixel_values": Tensor}`` with shape ``(1, C, H, W)``.
        """
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        tensor = self._transform(image)
        return {"pixel_values": tensor.unsqueeze(0)}

    def forward(self, inputs: dict[str, Any]) -> Any:
        """Run the timm model forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Tensor
            Model output, shape ``(1, n_classes)`` or ``(1, D)``.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        px = self._move_to_device(inputs)["pixel_values"]
        with torch.no_grad():
            if hasattr(self.model, "forward_features"):
                return self.model.forward_features(px)
            return self.model(px)

    def enumerate_layers(self) -> list[LayerInfo]:
        """Return metadata for all named modules.

        Returns
        -------
        list[LayerInfo]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        result = []
        for name, module in self.model.named_modules():
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
        """Return ModelMeta for the loaded timm model."""
        return ModelMeta(
            model_name=self._model_name,
            source="timm",
            architecture=self._architecture,
            learning_paradigm=self._learning_paradigm,
            encoder_type="vision",
            embedding_policy="",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_arch(model_name: str) -> str:
        n = model_name.lower()
        for arch in ("vit", "resnet", "resnext", "efficientnet", "convnext", "swin", "deit"):
            if arch in n:
                return arch
        return "unknown"
