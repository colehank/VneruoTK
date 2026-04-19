"""plot_trial_sampling: batch anchor/positive/negative with time-coloured borders."""

from __future__ import annotations

from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps

from trial_cebra import TrialAwareDistribution

from ._utils import ImageSource, get_img

_BORDER_PX   = 12
_TARGET_SIZE = (224, 224)


# ── sampling ──────────────────────────────────────────────────────────────────

def _sample_batch(
    dist: TrialAwareDistribution,
    batch_size: int,
    anchor_idx: Optional[int | torch.Tensor] = None,
):
    if anchor_idx is None:
        anchor_idx = dist.sample_prior(num_samples=1)
    elif not isinstance(anchor_idx, torch.Tensor):
        anchor_idx = torch.tensor([anchor_idx], device=dist.device)

    pos_idx = torch.stack(
        [dist.sample_conditional(anchor_idx) for _ in range(batch_size)]
    ).squeeze(1).to(dist.device)
    neg_idx = dist.sample_prior(num_samples=batch_size)

    def _info(idx: torch.Tensor):
        return (
            dist.timepoint_to_trial[idx].cpu().numpy(),
            dist.timepoint_rel_pos[idx].cpu().numpy(),
        )

    return _info(anchor_idx), _info(pos_idx), _info(neg_idx)


# ── image helpers ─────────────────────────────────────────────────────────────

def _make_sample_img(
    img: Image.Image,
    tid: int,
    rgba,
    width: int = _BORDER_PX,
) -> Image.Image:
    img = img.convert("RGB").resize(_TARGET_SIZE, Image.Resampling.LANCZOS)
    fill = (0, 0, 0) if tid < 0 else (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
    return ImageOps.expand(img, border=width, fill=fill)


def _compose_grid(imgs: list[Image.Image], padding: int = 3) -> Image.Image:
    if len(imgs) == 1:
        return imgs[0]
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = int(np.ceil(len(imgs) / cols))
    w = max(im.width  for im in imgs)
    h = max(im.height for im in imgs)
    canvas = Image.new(
        "RGB",
        (cols * (w + padding) + padding, rows * (h + padding) + padding),
        (255, 255, 255),
    )
    for i, im in enumerate(imgs):
        canvas.paste(im, (padding + (i % cols) * (w + padding),
                          padding + (i // cols) * (h + padding)))
    return canvas


# ── public API ────────────────────────────────────────────────────────────────

def plot_trial_sampling(
    dist: TrialAwareDistribution,
    trial_labels,
    images: dict[str, ImageSource],
    *,
    batch_size: int = 20,
    anchor_idx: Optional[int] = None,
    sfreq: Optional[float] = None,
    pre_len: int = 0,
    cmap: str = "summer",
    figsize: tuple = (9, 3),
) -> plt.Figure:
    """Plot anchor / positive / negative groups with in-trial-time coloured borders.

    Args:
        dist:         Fitted :class:`TrialAwareDistribution`.
        trial_labels: Sequence mapping trial_id → image_id (str).
        images:       Dict mapping image_id → file path, Path, or PIL Image.
        batch_size:   Number of positive and negative samples to draw.
        anchor_idx:   Timepoint index to use as anchor; sampled at random if None.
        sfreq:        Sampling frequency in Hz; if given, time labels are in ms.
        pre_len:      Pre-stimulus frames at the start of each trial; used to
                      align ``rel_pos == pre_len`` with t = 0.
        cmap:         Matplotlib colormap name for the in-trial time axis.
        figsize:      Figure size passed to :func:`matplotlib.pyplot.subplots`.

    Returns:
        :class:`matplotlib.figure.Figure` (1 × 3: anchor | positives | negatives)
        with a colorbar on the right showing in-trial time.
    """
    trial_labels = np.asarray(trial_labels)
    (anc_tids, anc_rels), (pos_tids, pos_rels), (neg_tids, neg_rels) = \
        _sample_batch(dist, batch_size, anchor_idx)

    # colormap spanning the full trial length
    trial_len = int((dist.trial_ends - dist.trial_starts).float().mean().item())
    if sfreq is not None:
        t_min = (0         - pre_len) * 1000.0 / sfreq
        t_max = (trial_len - pre_len) * 1000.0 / sfreq
    else:
        t_min, t_max = 0.0, float(trial_len)

    _cmap = plt.cm.get_cmap(cmap)
    norm  = mcolors.Normalize(vmin=t_min, vmax=t_max)

    def _color(rel: int):
        t = (rel - pre_len) * 1000.0 / sfreq if sfreq else float(rel)
        return _cmap(norm(t))

    def _fmt_avg(avg_rel: float) -> str:
        if sfreq is None:
            return f"avg {avg_rel:.1f} fr"
        return f"avg {(avg_rel - pre_len) * 1000.0 / sfreq:.0f} ms"

    def _group_arr(tids, rels) -> np.ndarray:
        imgs = []
        for tid, rel in zip(tids.ravel(), rels.ravel()):
            img_id = str(trial_labels[int(tid)]) if tid >= 0 else "None"
            rgba   = _color(int(rel)) if tid >= 0 else (0, 0, 0, 1)
            imgs.append(_make_sample_img(get_img(img_id, images), int(tid), rgba))
        return np.array(_compose_grid(imgs))

    def _avg_rel(rels) -> float:
        valid = rels[rels >= 0]
        return float(valid.mean()) if len(valid) > 0 else 0.0

    anc_tid  = int(anc_tids[0])
    anc_rgba = _color(int(anc_rels[0])) if anc_tid >= 0 else (0, 0, 0, 1)
    anc_arr  = np.array(_make_sample_img(
        get_img(str(trial_labels[anc_tid]) if anc_tid >= 0 else "None", images),
        anc_tid, anc_rgba,
    ))
    pos_arr = _group_arr(pos_tids, pos_rels)
    neg_arr = _group_arr(neg_tids, neg_rels)

    # resize +/- grids to match anchor panel
    target_wh = (anc_arr.shape[1], anc_arr.shape[0])
    pos_arr = np.array(Image.fromarray(pos_arr).resize(target_wh, Image.Resampling.LANCZOS))
    neg_arr = np.array(Image.fromarray(neg_arr).resize(target_wh, Image.Resampling.LANCZOS))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(right=0.82, bottom=0.14, wspace=0.08)

    for ax, arr, title, avg_rel in zip(
        axes,
        [anc_arr, pos_arr, neg_arr],
        ["R", "+", "-"],
        [_avg_rel(anc_rels), _avg_rel(pos_rels), _avg_rel(neg_rels)],
    ):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10, pad=2)
        ax.text(0.5, -0.06, _fmt_avg(avg_rel),
                transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_box_aspect(1)
        ax.axis("off")

    sm   = plt.cm.ScalarMappable(cmap=_cmap, norm=norm)
    sm.set_array([])
    unit = "ms" if sfreq else "frames"
    cax  = fig.add_axes([0.85, 0.18, 0.025, 0.65])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(f"In-trial time ({unit})", fontsize=8)
    ticks = np.linspace(t_min, t_max, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{v:.0f}" for v in ticks], fontsize=7)

    return fig
