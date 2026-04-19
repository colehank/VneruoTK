# vneurotk API Design: vneurotk.io, vneurotk.data

## Overview
vneurotk支持从ephys、mne、bids等不同数据格式的文件中读取数据，并将其转换为vneurotk的数据结构。用户可以直接使用`vneurotk.load()`函数加载这些数据，无需关心底层的文件路径和格式细节。加载后的数据保存为vneurotk.data.BaseData对象，用户可以使用vneurotk提供的各种方法对数据进行处理和分析。

## vneurotk.io.VTKPath
VTKPath是所有数据类型的路径类的基类，定义了基本的路径属性和方法。它包含以下属性：
- root: 数据的根目录
- session: 数据的会话信息
- subject: 数据的受试者信息
- task: 数据的任务信息
- run: 数据的运行信息
- desc: 数据的描述信息
- probe: 数据的探针信息，仅针对ephys数据
- suffix: 数据的后缀信息
- extension: 数据的扩展名信息
- modality: 数据的模态信息, 包括ephys、mne、bids

### vneurotk.io.EphysPath, MNEPath, BIDSPath
EphysPath、MNEPath和BIDSPath分别继承自VTKPath。
它们根据不同的数据格式定义了特定的属性和方法。例如，EphysPath包含probe属性，而MNEPath和BIDSPath则没有。

### vneurotk.io.load
load函数是vneurotk的核心函数之一，用于加载数据。用户只需传入一个路径对象（如EphysPath、MNEPath或BIDSPath），load函数会根据路径对象的属性自动识别数据类型并加载数据。

- MNEPath: 使用mne.io.read_raw加载数据
- BIDSPath: 使用mne_bids.read_raw_bids加载数据
- EphysPath: 保留接口，暂不实现

vneurotk.io.load函数的返回值是一个vneurotk.data.BaseData对象，包含了加载的数据和相关的元信息。

## vneurotk.data.BaseData
### BaseData是vneurotk中所有数据类型的基类，定义了基本的数据结构和方法。它包含以下属性：
- neuro: 数据的主要内容，通常是一个`ntime x nchan`的numpy数组
- neuro_info: 神经数据的元信息，通常是一个字典，包含：
  - 采样率：数据的采样率，单位为Hz
  - 通道名称：数据的通道名称列表
  - 高通滤波：数据的高通滤波频率，单位为Hz
  - 低通滤波：数据的低通滤波频率，单位为Hz
  - 源文件: 数据的原始文件路径
  - 对于MNEPath与BIDSPath数据，直接从mne.io.Raw对象中提取
    - 采样率：raw.info['sfreq']
    - 通道名称：raw.info['ch_names']
    - 高通滤波：raw.info['highpass']
    - 低通滤波：raw.info['lowpass']
    - 源文件: MNEPath/BIDSPath.fpath
  - 对于EphysPath数据，暂不实现

- visual: 神经数据对应的视觉刺激信息，通常是一个`ntime,`的numpy数组，对应每个时间点的刺激标签，刺激时间点为刺激ID，非刺激事件点为np.nan
- visual_info: 视觉刺激信息的元信息，通常是一个字典，包含：
  - 刺激数量：数据中包含的unique刺激ID刺激数量
  - 刺激IDs：数据中包含的unique刺激ID列表


- trial: 神经数据对应的试次信息，通常是一个`ntime,`的numpy数组，对应每个时间点的trial_id, 试次从0开始编号，试次时间点为trial_id，非试次事件点为np.nan
- trial_info: 试次信息的元信息，通常是一个字典，包含：
  - baseline: 以visual_onset为0点，trial_start到visual_onset的偏移，如[-20, 0]
  - trial_window: 以visual_onset为0点，trial_start到trial_end的偏移，如[-20, 80]

- trial_starts: 试次开始的时间点索引，通常是一个`n_trials,`的numpy数组, 如 [100, 500, 900]
- trial_ends: 试次结束的时间点索引，通常是一个`n_trials,`的numpy数组, 如 [200, 600, 1000]
- visual_onsets: 刺激事件的时间点索引，通常是一个`n_trials,`的numpy数组, 如 [120, 520, 920]

### BaseData.configure()：
configure方法用于配置试次信息。
用户需要传入`trial_window`和`visual_onsets`, `visual_ids`三个参数，trial_window支持以秒为单位或以采样点位单位。
分别表示试次持续时间和刺激事件的时间点以及对应的刺激ID。而BaseData会根据这些信息自动生成trial、trial_info、trial_starts和trial_ends等等属性。用户可以通过这些属性方便地访问试次信息。此外，再接收两个默认参数，crop = False，当crop 为True时，configure方法会自动调用crop方法裁剪数据，以及mode，表示裁剪后数据的返回形状，支持"continues"和"epochs"两种模式。默认为"continues"模式，即返回连续的形状。

### BaseData.crop(mode:str)：
crop方法用于裁剪数据。mode可选"continues"或"epochs"。
该方法裁剪掉那些非trial的时间点。即仅保留trial_starts到trial_ends之间的时间点。
对于神经数据，裁剪后的数据可以返回为两种不同的形状：
  - mode="continues"表示将数据裁剪为连续的形状，即`ntime x nchan`的二维数组。
  - mode="epochs"表示将数据裁剪为试次的形状，即`n_trials x n_timepoints x n_chan`的三维数组，n_timepoints是trial_window的长度。用户可以根据需要选择不同的形状来处理和分析数据。
对于视觉数据和试次数据，同样进行裁剪，确保它们与神经数据保持一致的时间点。
  

### BaseData.save:
save方法用于保存数据。用户需要传入一个路径，可以是VTKPath或pathlib.Path或字符串。BaseData只会保存配置过试次信息的BaseData对象，以h5格式高效保存数据。保存的数据包含neuro、visual、trial以及它们的元信息。用户可以使用vneurotk.load函数加载保存的数据，加载后的数据仍然是一个BaseData对象，包含了neuro、visual、trial以及它们的元信息。

### 直接建立vneurotk.data.BaseData
vneurotk.data.BaseData也可以由用户直接创建。用户需要传入neuro、visual、trial以及它们的元信息。用户可以通过这种方式创建一个BaseData对象，而不需要从文件中加载数据。

## 部分api接口设计示例
@api_apth.py：所有Path类
@api_load_save.py：load和save函数