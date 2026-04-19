"""Info summary object for VneuroTK data containers."""

from __future__ import annotations

from typing import Any


_STYLE = (
    "<style scoped>"
    ".vtk-info{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
    "Roboto,sans-serif;font-size:13px;max-width:480px}"
    ".vtk-info summary{cursor:pointer;padding:6px 0;font-size:14px}"
    ".vtk-info table{width:100%;border-collapse:collapse;margin:0 0 8px 0}"
    ".vtk-info th,.vtk-info td{padding:4px 12px;border-bottom:1px solid currentColor;border-bottom-opacity:0.2}"
    ".vtk-info th{text-align:left;width:50%;font-weight:500;opacity:0.75}"
    ".vtk-info td{text-align:right;width:50%}"
    ".vtk-info tr:last-child th,.vtk-info tr:last-child td{border-bottom:none}"
    ".vtk-info .vtk-na{opacity:0.5;font-style:italic}"
    "</style>"
)


class Info:
    """Summary object returned by :attr:`BaseData.info`.

    Renders rich HTML in Jupyter via ``_repr_html_``, similar to
    :class:`mne.Info`.

    Parameters
    ----------
    neuro : dict
        Dict with keys ``n_time``, ``n_neuro``, ``sfreq``, ``highpass``,
        ``lowpass``.
    visual : dict or None
        Dict with key ``n_stim``.
    trial : dict or None
        Dict with keys ``baseline``, ``trial_window``.
    configured : bool
        Whether the parent :class:`BaseData` has been configured.
    crop_mode : str or None
        Current crop mode (``"continues"``, ``"epochs"``, or ``None``).
    """

    def __init__(
        self,
        neuro: dict[str, Any],
        visual: dict[str, Any] | None,
        trial: dict[str, Any] | None,
        configured: bool,
        crop_mode: str | None = None,
        data_level: str = "timepoint",
    ) -> None:
        self._neuro = neuro
        self._visual = visual
        self._trial = trial
        self._configured = configured
        self._crop_mode = crop_mode
        self._data_level = data_level

    # --- HTML helpers ---------------------------------------------------

    @staticmethod
    def _table(rows: list[tuple[str, str]]) -> str:
        trs = "".join(
            f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows
        )
        return f"<table>{trs}</table>"

    @staticmethod
    def _section(title: str, body: str) -> str:
        return (
            f"<details open>"
            f"<summary><strong>{title}</strong></summary>"
            f"{body}</details>"
        )

    @staticmethod
    def _na(text: str = "Not configured") -> str:
        return f'<span class="vtk-na">{text}</span>'

    # --- public repr ------------------------------------

    def _repr_html_(self) -> str:
        n = self._neuro
        sfreq = n.get("sfreq")
        hp = n.get("highpass")
        lp = n.get("lowpass")
        neuro_rows = [
            ("Time points", str(n["n_time"])),
            ("Neuros", str(n["n_neuro"])),
            ("Sampling frequency", f"{sfreq:.2f} Hz" if sfreq is not None else "N/A"),
            ("Highpass", f"{hp:.2f} Hz" if hp is not None else "N/A"),
            ("Lowpass", f"{lp:.2f} Hz" if lp is not None else "N/A"),
        ]
        if self._data_level != "timepoint":
            neuro_rows.append(("Data level", self._data_level))
        parts = [self._section("Neuro", self._table(neuro_rows))]

        if self._configured and self._visual is not None:
            parts.append(self._section(
                "Visual",
                self._table([("n_visual", str(self._visual["n_stim"]))]),
            ))
        else:
            parts.append(self._section(
                "Visual",
                self._table([("Status", self._na())]),
            ))

        if self._configured and self._trial is not None:
            t = self._trial
            index_unit = "in segmented" if self._crop_mode else "in raw"
            data_mode = self._crop_mode if self._crop_mode else self._na("N/A")
            parts.append(self._section(
                "Trial",
                self._table([
                    ("Baseline", str(t["baseline"])),
                    ("Trial window", str(t["trial_window"])),
                    ("Index unit", index_unit),
                    ("Data mode", data_mode),
                ]),
            ))
        else:
            parts.append(self._section(
                "Trial",
                self._table([("Status", self._na())]),
            ))

        body = "".join(parts)
        return f'{_STYLE}<div class="vtk-info">{body}</div>'

    def __repr__(self) -> str:
        n = self._neuro
        sfreq = n.get("sfreq")
        hp = n.get("highpass")
        lp = n.get("lowpass")
        lines = [
            "Info",
            f"  Neuro: Time points={n['n_time']}, Neuros={n['n_neuro']}, "
            f"sfreq={sfreq}, highpass={hp}, "
            f"lowpass={lp}",
        ]
        if self._data_level != "timepoint":
            lines[1] += f", data_level={self._data_level}"
        if self._configured and self._visual is not None:
            lines.append(
                f"  Visual: n_visual={self._visual['n_stim']}"
            )
        else:
            lines.append("  Visual: Not configured")
        if self._configured and self._trial is not None:
            t = self._trial
            index_unit = "in segmented" if self._crop_mode else "in raw"
            data_mode = self._crop_mode or "N/A"
            lines.append(
                f"  Trial: baseline={t['baseline']}, "
                f"trial_window={t['trial_window']}, "
                f"Index unit={index_unit}, Data mode={data_mode}"
            )
        else:
            lines.append("  Trial: Not configured")
        return "\n".join(lines)
