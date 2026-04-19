"""Embedding policy definitions.

:class:`EmbeddingPolicy` controls how the final embedding is derived
from a model's raw output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

import torch
from torch import Tensor

__all__ = ["EmbeddingPolicy"]


class EmbeddingPolicy(str, Enum):
    """Strategy for computing the final embedding from model output.

    Attributes
    ----------
    CLS_TOKEN
        Take the [CLS] token (index 0 of the sequence dimension).
        Typical for ViT models trained with a CLS token.
    MEAN_POOL
        Average-pool all spatial or sequence positions.
    ALL_TOKENS
        Return the full token sequence including CLS and all patch tokens,
        shape ``(T, D)``.  Preferred for ViT representational analysis.
    PRE_HEAD
        Use the last captured hook activation, mean-pooled if needed.
    PROJECTION_OUT
        Use the projection head output, e.g. CLIP ``image_embeds``.
    BACKBONE_OUT
        Use the raw backbone output before any head.
    CUSTOM
        Call a user-supplied ``custom_fn(model_output, activations)``.
    """

    CLS_TOKEN = "cls_token"
    MEAN_POOL = "mean_pool"
    ALL_TOKENS = "all_tokens"
    PRE_HEAD = "pre_head"
    PROJECTION_OUT = "projection_out"
    BACKBONE_OUT = "backbone_out"
    CUSTOM = "custom"

    def apply(
        self,
        model_output: Any,
        activations: dict[str, Tensor],
        custom_fn: Callable | None = None,
    ) -> Tensor:
        """Compute the final embedding tensor.

        Parameters
        ----------
        model_output : Any
            Raw output from the model's forward pass.
        activations : dict[str, Tensor]
            Hook-captured activations keyed by layer name.
        custom_fn : Callable or None
            Required when policy is ``CUSTOM``.  Called as
            ``custom_fn(model_output, activations)`` and must return a
            Tensor.

        Returns
        -------
        Tensor
            Shape ``(D,)`` for most policies; ``(T, D)`` for
            ``ALL_TOKENS``.

        Raises
        ------
        ValueError
            If policy is ``CUSTOM`` but *custom_fn* is ``None``.
        NotImplementedError
            If the policy variant is not handled.
        """
        if self == EmbeddingPolicy.CUSTOM:
            if custom_fn is None:
                raise ValueError("EmbeddingPolicy.CUSTOM requires a custom_fn")
            return custom_fn(model_output, activations).squeeze()

        if self == EmbeddingPolicy.CLS_TOKEN:
            return _cls_token(model_output)

        if self == EmbeddingPolicy.MEAN_POOL:
            return _mean_pool(model_output)

        if self == EmbeddingPolicy.ALL_TOKENS:
            return _all_tokens(model_output)

        if self == EmbeddingPolicy.PRE_HEAD:
            return _pre_head(model_output, activations)

        if self == EmbeddingPolicy.PROJECTION_OUT:
            return _projection_out(model_output)

        if self == EmbeddingPolicy.BACKBONE_OUT:
            return _backbone_out(model_output)

        raise NotImplementedError(f"EmbeddingPolicy {self!r} not implemented")


# ------------------------------------------------------------------
# Policy implementations
# ------------------------------------------------------------------

def _to_tensor(x: Any) -> Tensor:
    """Coerce model output to a Tensor."""
    if isinstance(x, Tensor):
        return x
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if hasattr(x, "pooler_output") and x.pooler_output is not None:
        return x.pooler_output
    if hasattr(x, "image_embeds") and x.image_embeds is not None:
        return x.image_embeds
    raise TypeError(f"Cannot convert model output type {type(x)} to Tensor")


def _cls_token(model_output: Any) -> Tensor:
    t = _to_tensor(model_output)
    if t.ndim == 3:
        t = t.squeeze(0)  # (1, T, D) → (T, D)
    return t[0]  # CLS token


def _all_tokens(model_output: Any) -> Tensor:
    t = _to_tensor(model_output)
    if t.ndim == 3:
        t = t.squeeze(0)  # (1, T, D) → (T, D)
    return t  # (T, D): CLS + all patch tokens


def _mean_pool(model_output: Any) -> Tensor:
    t = _to_tensor(model_output)
    if t.ndim == 3:
        t = t.squeeze(0)  # (1, T, D) → (T, D)
    if t.ndim == 2:
        return t.mean(dim=0)
    return t.ravel()


def _pre_head(model_output: Any, activations: dict[str, Tensor]) -> Tensor:
    if activations:
        act = list(activations.values())[-1]
        if act.ndim == 3:
            act = act.squeeze(0)
        if act.ndim == 2:
            return act.mean(dim=0)
        return act.ravel()
    return _mean_pool(model_output)


def _projection_out(model_output: Any) -> Tensor:
    if hasattr(model_output, "image_embeds") and model_output.image_embeds is not None:
        return model_output.image_embeds.squeeze(0)
    t = _to_tensor(model_output)
    return t.squeeze(0).ravel() if t.ndim > 1 else t.ravel()


def _backbone_out(model_output: Any) -> Tensor:
    t = _to_tensor(model_output)
    if t.ndim == 3:
        return t.squeeze(0).mean(dim=0)
    if t.ndim == 2:
        return t.squeeze(0)
    return t.ravel()
