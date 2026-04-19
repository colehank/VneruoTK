"""plot_sampling: visualise a single anchor/positive/negative triplet."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from trial_cebra import TrialAwareDistribution

from ._utils import ImageSource, fmt_time, get_img

# Conditionals that constrain the positive within ±time_offset of the
# anchor's relative position — these show the green window on the timeline.
_TIME_CONSTRAINED = frozenset({"trialTime", "trialTime_delta", "trialTime_trialDelta"})


# ── sampling ──────────────────────────────────────────────────────────────────

def _sample_triplet(
    dist: TrialAwareDistribution,
    anchor_idx: Optional[int | torch.Tensor] = None,
):
    if anchor_idx is None:
        anchor_idx = dist.sample_prior(num_samples=1)
    elif not isinstance(anchor_idx, torch.Tensor):
        anchor_idx = torch.tensor([anchor_idx], device=dist.device)

    pos_idx = dist.sample_conditional(anchor_idx)
    neg_idx = dist.sample_prior(num_samples=1)

    N = len(dist.continuous)
    results = []
    for idx in [anchor_idx, pos_idx, neg_idx]:
        tid = dist.timepoint_to_trial[idx].item()
        if tid >= 0:
            start = dist.trial_starts[tid].item()
            end   = dist.trial_ends[tid].item()
        else:
            start, end = 0, N
        results.append({"trial_id": tid, "rel_pos": idx.item() - start, "trial_len": end - start})

    return results, results[0]["trial_len"]


# ── drawing helpers ───────────────────────────────────────────────────────────

def _draw_images(axes, samples, trial_labels, images, colors, sfreq):
    for ax, s, color, label in zip(axes, samples, colors, ["R", "+", "-"]):
        img_id = str(trial_labels[s["trial_id"]]) if s["trial_id"] >= 0 else "None"
        ax.imshow(np.array(get_img(img_id, images)))
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(
            f"trial {s['trial_id']}: {fmt_time(s['rel_pos'], sfreq)}",
            fontsize=10, labelpad=3,
        )


def _draw_timeline(ax, samples, anchor_trial_len, time_offset, sfreq,
                   colors, c_window, c_bar, show_window: bool):
    anchor_rel = samples[0]["rel_pos"]
    win_lo = max(0, anchor_rel - time_offset)
    win_hi = min(anchor_trial_len, anchor_rel + time_offset)

    ax.barh(0, anchor_trial_len, left=0, height=0.35, color=c_bar, edgecolor="none")

    if show_window:
        ax.barh(0, win_hi - win_lo, left=win_lo, height=0.35,
                color=c_window, alpha=0.2, edgecolor=c_window, linewidth=1.2)
        ax.text(
            (win_lo + win_hi) / 2, 0.35,
            f"Time offset({time_offset}): {fmt_time(win_lo, sfreq)} ~ {fmt_time(win_hi, sfreq)}",
            ha="center", va="bottom", color=c_window, fontsize=8.5, fontstyle="italic",
        )

    for s, color in zip(samples, colors):
        ax.plot([s["rel_pos"]] * 2, [-0.22, 0.22],
                color=color, linewidth=2.5, solid_capstyle="round")

    ax.text(0, -0.4, fmt_time(0, sfreq), ha="left",  va="bottom", fontsize=8.5, color="gray")
    ax.text(anchor_trial_len, -0.4, fmt_time(anchor_trial_len, sfreq),
            ha="right", va="bottom", fontsize=8.5, color="gray")
    ax.text(anchor_trial_len / 2, -0.5, "In-trial time", ha="center", va="bottom", fontsize=10)

    ax.set_xlim(-3, anchor_trial_len + 3)
    ax.set_ylim(-0.45, 0.6)
    ax.axis("off")


# ── public API ────────────────────────────────────────────────────────────────

def plot_sampling(
    dist: TrialAwareDistribution,
    trial_labels,
    images: dict[str, ImageSource],
    *,
    sfreq: Optional[float] = None,
    anchor_idx: Optional[int] = None,
    figsize: tuple = (6, 3.5),
    c_anchor: str = "black",
    c_pos: str = "green",
    c_neg: str = "darkred",
    c_window: str = "green",
    c_bar: str = "lightgray",
) -> plt.Figure:
    """Plot one anchor / positive / negative triplet with an in-trial timeline.

    Args:
        dist:         Fitted :class:`TrialAwareDistribution`.
        trial_labels: Sequence mapping trial_id → image_id (str).
        images:       Dict mapping image_id → file path, Path, or PIL Image.
        sfreq:        Sampling frequency in Hz; if given, time labels are in ms.
        anchor_idx:   Timepoint index to use as anchor; sampled at random if None.
        figsize:      Figure size passed to :func:`matplotlib.pyplot.figure`.
        c_anchor / c_pos / c_neg: Spine and marker colours for each role.
        c_window:     Colour of the positive-sampling window on the timeline.
        c_bar:        Background colour of the timeline bar.

    Returns:
        :class:`matplotlib.figure.Figure` (2 × 3 grid: images on top, timeline below).
    """
    samples, anchor_trial_len = _sample_triplet(dist, anchor_idx)
    colors = [c_anchor, c_pos, c_neg]

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 3)

    img_axes = [fig.add_subplot(gs[0, col]) for col in range(3)]
    _draw_images(img_axes, samples, trial_labels, images, colors, sfreq)

    ax_bar = fig.add_subplot(gs[1, :])
    _draw_timeline(ax_bar, samples, anchor_trial_len, dist.time_offset,
                   sfreq, colors, c_window, c_bar,
                   show_window=dist.conditional in _TIME_CONSTRAINED)
    return fig
