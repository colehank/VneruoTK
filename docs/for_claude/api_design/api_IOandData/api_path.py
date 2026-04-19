from api_cons import TestEphysPath, TestMNEPath, TestBIDSPath
from vneurotk.io import EphysPath, MNEPath, VTKPath

ephys_path = EphysPath(
    root=TestEphysPath.root,
    session=TestEphysPath.session,
    desc=TestEphysPath.desc,
    probe=TestEphysPath.probe,
    modality=TestEphysPath.modality,
)

mne_path = MNEPath(
    root=TestMNEPath.root,
    subject=TestMNEPath.subject,
    session=TestMNEPath.session,
    task=TestMNEPath.task,
    run=TestMNEPath.run,
    extension=TestMNEPath.extension,
    modality=TestMNEPath.modality,
)

bids_path = BIDSPath(
    root=TestBIDSPath.root,
    subject=TestBIDSPath.subject,
    session=TestBIDSPath.session,
    task=TestBIDSPath.task,
    run=TestBIDSPath.run,
    extension=TestBIDSPath.extension,
    suffix=TestBIDSPath.suffix,
)

ephys_vtk_path = VTKPath(
    root=TestEphysPath.root,
    session=TestEphysPath.session,
    desc=TestEphysPath.desc,
    probe=TestEphysPath.probe,
    modality=TestEphysPath.modality,
)

mne_vtk_path = VTKPath(
    root=TestMNEPath.root,
    subject=TestMNEPath.subject,
    session=TestMNEPath.session,
    task=TestMNEPath.task,
    run=TestMNEPath.run,
    extension=TestMNEPath.extension,
    modality=TestMNEPath.modality,
)

bids_vtk_path = VTKPath(
    root=TestBIDSPath.root,
    subject=TestBIDSPath.subject,
    session=TestBIDSPath.session,
    task=TestBIDSPath.task,
    run=TestBIDSPath.run,
    extension=TestBIDSPath.extension,
    suffix=TestBIDSPath.suffix,
)

