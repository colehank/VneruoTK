"""Model and layer metadata dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LayerMeta", "ModelMeta"]


@dataclass
class LayerMeta:
    """Metadata describing a single extracted layer.

    Parameters
    ----------
    name : str
        Raw module name, e.g. ``"blocks.11.attn"``.
    normalized_name : str
        Dot-replaced identifier, e.g. ``"blocks_11_attn"``.
    module_type : str
        PyTorch class name, e.g. ``"Attention"``.
    shape : tuple
        Output tensor shape excluding batch dim, e.g. ``(197, 768)``.
    shape_type : str
        One of ``"token_seq"``, ``"spatial"``, or ``"flat"``.
    depth : int
        Nesting depth in the module tree.
    is_final : bool
        Whether this layer is the designated final embedding source.
    """

    name: str
    normalized_name: str
    module_type: str
    shape: tuple
    shape_type: str
    depth: int
    is_final: bool = False


@dataclass
class ModelMeta:
    """Model-level metadata for a feature extraction run.

    Parameters
    ----------
    model_name : str
        Registry alias or raw model ID.
    source : str
        Backend source: ``"timm"``, ``"transformers"``, or
        ``"thingsvision"``.
    architecture : str
        Architecture family, e.g. ``"vit"``, ``"resnet"``.
    learning_paradigm : str
        Training objective: ``"supervised"``, ``"selfsupervised"``, or
        ``"contrastive"``.
    encoder_type : str
        Role of this encoder, e.g. ``"vision"``.
    embedding_policy : str
        Name of the active
        :class:`~vneurotk.vision.extractor.policy.EmbeddingPolicy`.
    """

    model_name: str
    source: str
    architecture: str
    learning_paradigm: str
    encoder_type: str
    embedding_policy: str
