"""Tests for vneurotk.vision module.

Covers:
  - VisualRepresentations (no real model needed)
  - EmbeddingPolicy (pure Tensor)
  - LayerSelector (tiny nn.Sequential)
  - ModelRegistry (no model download)
  - BaseData.trial_stim_ids
  - VisionExtractor with MockBackend
  - Smoke tests for timm and transformers (skipped if not installed)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vneurotk.vision.extractor.backend.base import BaseBackend, LayerInfo
from vneurotk.vision.extractor.policy import EmbeddingPolicy
from vneurotk.vision.extractor.selector import (
    AllLeafSelector,
    BlockLevelSelector,
    CustomSelector,
)
from vneurotk.vision.registry import REGISTRY, ModelConfig, ModelRegistry
from vneurotk.vision.representation import ModelMeta
from vneurotk.vision.visual_representations import VisualRepresentations

# ===========================================================================
# Helpers / Fixtures
# ===========================================================================


def _make_model_meta(**kw) -> ModelMeta:
    defaults = dict(
        model_name="test",
        source="timm",
        architecture="vit",
        learning_paradigm="supervised",
        encoder_type="vision",
        embedding_policy="cls_token",
    )
    defaults.update(kw)
    return ModelMeta(**defaults)


def _make_vr(n_stim: int = 5, d: int = 8) -> VisualRepresentations:
    return VisualRepresentations(
        stim_ids=list(range(n_stim)),
        features={
            "layer_a": np.random.rand(n_stim, d).astype(np.float32),
            "layer_b": np.random.rand(n_stim, 4, d).astype(np.float32),
        },
        final_embedding=np.random.rand(n_stim, d).astype(np.float32),
        model_meta=_make_model_meta(),
    )


# ===========================================================================
# TestVisualRepresentations
# ===========================================================================


class TestVisualRepresentations:
    def test_basic_properties(self):
        vr = _make_vr(n_stim=10, d=16)
        assert vr.n_stim == 10
        assert set(vr.layer_names) == {"layer_a", "layer_b"}

    def test_getitem(self):
        vr = _make_vr(n_stim=5, d=8)
        arr = vr["layer_a"]
        assert arr.shape == (5, 8)

    def test_numpy_layer(self):
        vr = _make_vr(n_stim=5, d=8)
        arr = vr.numpy("layer_a")
        assert arr.shape == (5, 8)

    def test_numpy_final_embedding(self):
        vr = _make_vr(n_stim=5, d=8)
        arr = vr.numpy()
        assert arr.shape == (5, 8)

    def test_to_tensor_embedding(self):
        vr = _make_vr(n_stim=4, d=8)
        t = vr.to_tensor()
        assert isinstance(t, torch.Tensor)
        assert t.shape == (4, 8)

    def test_to_tensor_layer(self):
        vr = _make_vr(n_stim=4, d=8)
        t = vr.to_tensor("layer_a")
        assert t.shape == (4, 8)

    def test_select_by_id(self):
        vr = _make_vr(n_stim=5, d=8)
        sub = vr.select([1, 3])
        assert sub.n_stim == 2
        assert sub.stim_ids == [1, 3]
        assert sub.final_embedding.shape == (2, 8)
        assert sub["layer_a"].shape == (2, 8)

    def test_select_by_index(self):
        vr = _make_vr(n_stim=5, d=8)
        sub = vr.select_by_index([0, 4])
        assert sub.stim_ids == [0, 4]

    def test_repr(self):
        vr = _make_vr()
        r = repr(vr)
        assert "VisualRepresentations" in r
        assert "n_stim=5" in r

    def test_select_missing_id_raises(self):
        vr = _make_vr(n_stim=3)
        with pytest.raises(KeyError):
            vr.select([99])


# ===========================================================================
# TestEmbeddingPolicy
# ===========================================================================


class TestEmbeddingPolicy:
    def _flat(self, d=16):
        return torch.randn(d)

    def _seq(self, t=10, d=16):
        return torch.randn(1, t, d)

    def test_cls_token(self):
        out = self._seq()
        emb = EmbeddingPolicy.CLS_TOKEN.apply(out, {})
        assert emb.shape == (16,)

    def test_mean_pool(self):
        out = self._seq()
        emb = EmbeddingPolicy.MEAN_POOL.apply(out, {})
        assert emb.shape == (16,)

    def test_all_tokens(self):
        out = self._seq(t=10, d=16)
        emb = EmbeddingPolicy.ALL_TOKENS.apply(out, {})
        assert emb.shape == (10, 16)

    def test_pre_head_with_activations(self):
        acts = {"layer_a": torch.randn(10, 8), "layer_b": torch.randn(10, 8)}
        emb = EmbeddingPolicy.PRE_HEAD.apply(torch.randn(8), acts)
        assert emb.shape == (8,)

    def test_pre_head_fallback(self):
        out = self._seq()
        emb = EmbeddingPolicy.PRE_HEAD.apply(out, {})
        assert emb.shape == (16,)

    def test_custom_fn(self):
        def fn(o, a):
            return o.squeeze()

        out = torch.randn(1, 8)
        emb = EmbeddingPolicy.CUSTOM.apply(out, {}, custom_fn=fn)
        assert emb.shape == (8,)

    def test_custom_no_fn_raises(self):
        with pytest.raises(ValueError, match="custom_fn"):
            EmbeddingPolicy.CUSTOM.apply(torch.randn(8), {})

    def test_backbone_out(self):
        out = torch.randn(1, 16)
        emb = EmbeddingPolicy.BACKBONE_OUT.apply(out, {})
        assert emb.shape == (16,)


# ===========================================================================
# TestLayerSelector
# ===========================================================================


def _tiny_model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


class TestLayerSelector:
    def test_all_leaf(self):
        model = _tiny_model()
        sel = AllLeafSelector()
        names = sel.select(model)
        assert all(isinstance(model._modules[n], nn.Linear) for n in names)
        assert len(names) == 2

    def test_all_leaf_custom_exclude(self):
        model = _tiny_model()
        sel = AllLeafSelector(exclude_types=(nn.ReLU,))
        names = sel.select(model)
        assert len(names) == 2

    def test_custom_selector(self):
        model = _tiny_model()
        sel = CustomSelector(["0", "2"])
        names = sel.select(model)
        assert names == ["0", "2"]

    def test_custom_selector_missing_raises(self):
        model = _tiny_model()
        sel = CustomSelector(["nonexistent"])
        with pytest.raises(ValueError, match="not found"):
            sel.select(model)

    def test_block_level_fallback(self):
        model = _tiny_model()
        sel = BlockLevelSelector()
        names = sel.select(model)
        assert len(names) >= 1


# ===========================================================================
# TestModelRegistry
# ===========================================================================


class TestModelRegistry:
    def test_builtin_entries(self):
        names = REGISTRY.list()
        assert "vit-b-16-imagenet" in names
        assert "clip-vit-b-32" in names
        assert "resnet50" in names

    def test_get_config(self):
        cfg = REGISTRY.get("resnet50")
        assert cfg.source == "timm"
        assert cfg.model_id == "resnet50"
        assert cfg.policy == "pre_head"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            REGISTRY.get("nonexistent-model-xyz")

    def test_register_custom(self):
        reg = ModelRegistry()
        reg.register(
            "my-model",
            ModelConfig(
                source="timm", model_id="resnet18", policy="mean_pool", paradigm="supervised"
            ),
        )
        assert "my-model" in reg
        assert reg.get("my-model").model_id == "resnet18"

    def test_list_sorted(self):
        names = REGISTRY.list()
        assert names == sorted(names)

    def test_repr(self):
        assert "ModelRegistry" in repr(REGISTRY)


# ===========================================================================
# TestTrialStimIds
# ===========================================================================


class TestTrialStimIds:
    def _make_bd(self, stim_ids=None):
        from vneurotk.neuro.base import BaseData

        neuro = np.random.randn(500, 4)
        neuro_info = dict(sfreq=100.0, ch_names=["c0", "c1", "c2", "c3"])
        vision = np.full(500, np.nan)
        if stim_ids is None:
            stim_ids = [10, 20, 30]
        onsets = np.array([50, 150, 250])
        for i, onset in enumerate(onsets):
            vision[onset] = stim_ids[i]

        bd = BaseData(
            neuro=neuro,
            neuro_info=neuro_info,
            vision=vision,
            vision_info={"n_stim": 3, "stim_ids": stim_ids},
        )
        bd.configure(
            trial_window=[-10, 40],
            vision_onsets=onsets,
            visual_ids=np.array(stim_ids),
        )
        return bd, stim_ids, onsets

    def test_trial_stim_ids_values(self):
        bd, stim_ids, onsets = self._make_bd()
        ids = bd.trial_stim_ids
        assert len(ids) == bd.n_trials
        for i in range(bd.n_trials):
            assert ids[i] == stim_ids[i]

    def test_unconfigured_raises(self):
        from vneurotk.neuro.base import BaseData

        bd = BaseData(
            neuro=np.zeros((10, 2)),
            neuro_info=dict(sfreq=1.0),
        )
        with pytest.raises(RuntimeError, match="configure"):
            _ = bd.trial_stim_ids


# ===========================================================================
# MockBackend
# ===========================================================================


class _MockBackend(BaseBackend):
    """Minimal backend wrapping a tiny nn.Sequential for hook tests."""

    def load(self, model_name: str, pretrained: bool = True) -> None:
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.model.eval()
        self._model_name = model_name

    def preprocess(self, image: Any) -> dict[str, Any]:
        t = torch.from_numpy(np.asarray(image, dtype=np.float32)).flatten()[:4]
        if t.shape[0] < 4:
            t = torch.nn.functional.pad(t, (0, 4 - t.shape[0]))
        return {"pixel_values": t.unsqueeze(0)}

    def forward(self, inputs: dict[str, Any]) -> Any:
        px = inputs["pixel_values"]
        with torch.no_grad():
            return self.model(px)

    def enumerate_layers(self) -> list[LayerInfo]:
        return [
            LayerInfo(name=n, module_type=type(m).__name__, depth=1)
            for n, m in self.model.named_modules()
            if n
        ]

    def get_model_meta(self) -> ModelMeta:
        return _make_model_meta(model_name=getattr(self, "_model_name", "mock"))


# ===========================================================================
# TestVisionExtractorMock
# ===========================================================================


class TestVisionExtractorMock:
    def _make_extractor(self):
        from vneurotk.vision.extractor.extractor import VisionExtractor
        from vneurotk.vision.extractor.selector import CustomSelector

        backend = _MockBackend(device="cpu")
        backend.load("mock")
        selector = CustomSelector(["0", "2"])
        return VisionExtractor.from_model(
            model=backend.model,
            backend=backend,
            embedding_policy="mean_pool",
            model_name="mock",
            selector=selector,
        )

    def test_extract_single_returns_visual_representations(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vr = ext.extract(image)
        assert isinstance(vr, VisualRepresentations)
        assert vr.n_stim == 1

    def test_extract_single_embedding_shape(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vr = ext.extract(image)
        assert vr.final_embedding.ndim == 2  # (1, D)
        assert vr.final_embedding.shape[0] == 1

    def test_extract_single_layer_exists(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vr = ext.extract(image)
        assert "0" in vr.features or "2" in vr.features

    def test_extract_batch_shape(self):
        ext = self._make_extractor()
        images = {i: np.random.rand(4).astype(np.float32) for i in range(6)}
        vr = ext.extract(images)
        assert isinstance(vr, VisualRepresentations)
        assert vr.n_stim == 6
        assert vr.final_embedding.shape[0] == 6
        for arr in vr.features.values():
            assert arr.shape[0] == 6

    def test_numpy_and_tensor(self):
        ext = self._make_extractor()
        image = np.random.rand(4).astype(np.float32)
        vr = ext.extract(image)
        arr = vr.numpy()
        assert isinstance(arr, np.ndarray)
        t = vr.to_tensor()
        assert isinstance(t, torch.Tensor)

    def test_layer_names_property(self):
        ext = self._make_extractor()
        assert set(ext.layer_names) == {"0", "2"}


# ===========================================================================
# Smoke tests — require real packages
# ===========================================================================


def _timm_installed():
    try:
        import timm  # noqa: F401

        return True
    except ImportError:
        return False


def _network_available(host: str = "huggingface.co", port: int = 443) -> bool:
    import socket

    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def _transformers_installed():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _thingsvision_installed():
    try:
        import thingsvision  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _timm_installed(), reason="timm not installed")
class TestTimmSmokeTest:
    def test_resnet18_end_to_end(self):
        from PIL import Image

        from vneurotk.vision.extractor.backend.timm_backend import TimmBackend
        from vneurotk.vision.extractor.selector import BlockLevelSelector

        backend = TimmBackend(device="cpu")
        backend.load("resnet18", pretrained=False)

        sel = BlockLevelSelector()
        layer_names = sel.select(backend.model)
        assert len(layer_names) > 0

        backend.register_hooks(layer_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        out = backend.forward(inputs)
        acts = backend.collect_activations()

        assert isinstance(out, torch.Tensor)
        assert len(acts) == 2
        backend.remove_hooks()


@pytest.mark.skipif(
    not _transformers_installed() or not _network_available(),
    reason="transformers not installed or network unavailable",
)
class TestTransformersSmokeTest:
    def test_dinov2_all_tokens(self):
        from PIL import Image

        from vneurotk.vision.extractor.backend.transformers_backend import TransformersBackend
        from vneurotk.vision.extractor.selector import BlockLevelSelector

        backend = TransformersBackend(device="cpu", learning_paradigm="selfsupervised")
        backend.load("facebook/dinov2-base")

        sel = BlockLevelSelector()
        layer_names = sel.select(backend.model)
        assert len(layer_names) > 0

        backend.register_hooks(layer_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        out = backend.forward(inputs)
        acts = backend.collect_activations()

        assert len(acts) == 2
        backend.remove_hooks()

        emb = EmbeddingPolicy.ALL_TOKENS.apply(out, acts)
        assert emb.ndim == 2  # (T, D)


@pytest.mark.skipif(not _thingsvision_installed(), reason="thingsvision not installed")
class TestThingsVisionSmokeTest:
    def test_resnet18_end_to_end(self):
        from PIL import Image

        from vneurotk.vision.extractor.backend.thingsvision_backend import ThingsVisionBackend
        from vneurotk.vision.extractor.selector import BlockLevelSelector

        backend = ThingsVisionBackend(source="torchvision", device="cpu")
        backend.load("resnet18", pretrained=False)

        sel = BlockLevelSelector()
        layer_names = sel.select(backend.model)
        assert len(layer_names) > 0

        backend.register_hooks(layer_names[:2])

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        inputs = backend.preprocess(img)
        out = backend.forward(inputs)
        acts = backend.collect_activations()

        assert isinstance(out, torch.Tensor)
        assert len(acts) == 2
        backend.remove_hooks()
