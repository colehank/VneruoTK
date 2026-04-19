# %%
import vneurotk as vnt
from vneurotk.io import VTKPath
from api_path import ephys_path, mne_path, bids_path
import numpy as np
from pathlib import Path
SAVE_ROOT = Path("/nfs/z1/userhome/zzl-zhangguohao/workingdir/projs/VneuroTK/DB/test_save")

ephys_data = vnt.load(ephys_path)
mne_data = vnt.load(mne_path)
bids_data = vnt.load(bids_path)

mne_data_trial_window = [-0.2, 0.8]  # 以秒为单位的试次窗口
mne_data_visual_onsets = np.load("/nfs/z1/userhome/zzl-zhangguohao/workingdir/projs/VneuroTK/DB/mne/NOD-MEG/visual_onsets.npy", allow_pickle=True)
mne_data_visual_ids = np.load("/nfs/z1/userhome/zzl-zhangguohao/workingdir/projs/VneuroTK/DB/mne/NOD-MEG/visual_ids.npy", allow_pickle=True)
mne_data.configure(
    trial_window=mne_data_trial_window,
    vision_onsets=mne_data_visual_onsets,
    visual_ids=mne_data_visual_ids,
)

bids_data_trial_window = [-0.2, 0.8]  # 以秒为单位的试次窗口
bids_data_visual_onsets = np.load("/nfs/z1/userhome/zzl-zhangguohao/workingdir/projs/VneuroTK/DB/bids/NOD-MEG/visual_onsets.npy", allow_pickle=True)
bids_data_visual_ids = np.load("/nfs/z1/userhome/zzl-zhangguohao/workingdir/projs/VneuroTK/DB/bids/NOD-MEG/visual_ids.npy", allow_pickle=True)
bids_data.configure(
    trial_window=bids_data_trial_window,
    vision_onsets=bids_data_visual_onsets,
    visual_ids=bids_data_visual_ids,
)

ephys_vtk_path = VTKPath(
    SAVE_ROOT / "ephys",
    session=TestEphysPath.session,
    desc=TestEphysPath.desc,
    probe=TestEphysPath.probe,
    modality=TestEphysPath.modality,
)
mne_vtk_path = VTKPath(
    SAVE_ROOT / "mne",
    subject=TestMNEPath.subject,
    session=TestMNEPath.session,
    task=TestMNEPath.task,
    run=TestMNEPath.run,
)
bids_vtk_path = VTKPath(
    SAVE_ROOT / "bids",
    subject=TestBIDSPath.subject,
    session=TestBIDSPath.session,
    task=TestBIDSPath.task,
    run=TestBIDSPath.run,
)

ephys_data.save(ephys_vtk_path) # 暂不完成
mne_data.save(mne_vtk_path)
bids_data.save(bids_vtk_path)