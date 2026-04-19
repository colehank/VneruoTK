# %%
import mne
from numpy.typing import NDArray


def get_event_samples(raw: mne.io.BaseRaw, event_name="stim_on") -> NDArray:
    ev, evid = mne.events_from_annotations(raw)
    event_id = evid[event_name]
    event_ev = ev[ev[:, 2] == event_id]
    return event_ev[:, 0]
