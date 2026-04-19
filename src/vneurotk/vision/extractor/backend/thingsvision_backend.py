"""ThingsVision backend for DNN feature extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from loguru import logger

from vneurotk.vision.extractor.backend.base import BaseBackend, LayerInfo
from vneurotk.vision.representation import ModelMeta

__all__ = ["ThingsVisionBackend"]


class ThingsVisionBackend(BaseBackend):
    """Backend powered by the ``thingsvision`` library.

    ``thingsvision`` must be installed before instantiating this class;
    a missing installation raises ``ImportError`` immediately (fail-fast).

    Parameters
    ----------
    source : str
        Model source string for thingsvision, e.g. ``"timm"`` or
        ``"torchvision"``.
    device : str or torch.device
        Inference device (default ``"cpu"``).
    learning_paradigm : str
        Training paradigm label.
    """

    def __init__(
        self,
        source: str = "torchvision",
        device: str | torch.device = "cpu",
        learning_paradigm: str = "supervised",
    ) -> None:
        try:
            from thingsvision.model_class import Model  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "thingsvision is required for ThingsVisionBackend.  "
                "Install with: uv add thingsvision"
            ) from exc

        super().__init__(device)
        self._tv_source = source
        self._learning_paradigm = learning_paradigm
        self._model_name: str = ""
        self._architecture: str = ""
        self._extractor = None
        self._transform = None

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def load(self, model_name: str, pretrained: bool = True) -> None:
        """Load a model via thingsvision.

        Parameters
        ----------
        model_name : str
            Model name as accepted by ``thingsvision.get_extractor()``.
        pretrained : bool
            Load pretrained weights.
        """
        from thingsvision.model_class import Model

        logger.info(
            "Loading thingsvision model: {} (source={}, pretrained={})",
            model_name,
            self._tv_source,
            pretrained,
        )

        device_str = str(self.device)
        self._tv_model = Model(
            model_name=model_name,
            pretrained=pretrained,
            device=device_str,
            backend="pt",
        )
        self.model = self._tv_model.model
        self.model.eval()
        self._transform = self._tv_model.get_transformations()
        self._model_name = model_name
        self._architecture = self._infer_arch(model_name)
        logger.info("Loaded thingsvision model: {} | arch={}", model_name, self._architecture)

    def preprocess(self, image: Any) -> dict[str, Any]:
        """Preprocess a PIL Image using thingsvision's transforms.

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
        """Run the model forward pass.

        Parameters
        ----------
        inputs : dict[str, Any]
            Output of :meth:`preprocess`.

        Returns
        -------
        Tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        px = self._move_to_device(inputs)["pixel_values"]
        with torch.no_grad():
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
        """Return ModelMeta for the loaded thingsvision model."""
        return ModelMeta(
            model_name=self._model_name,
            source="thingsvision",
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
        for arch in (
            "vit",
            "resnet",
            "resnext",
            "efficientnet",
            "convnext",
            "swin",
            "deit",
            "alexnet",
            "vgg",
        ):
            if arch in n:
                return arch
        return "unknown"
