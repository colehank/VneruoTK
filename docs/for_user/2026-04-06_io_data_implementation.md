# VneuroTK IO & Data 模块实现报告

**日期**: 2026-04-06  
**任务**: 根据 `docs/for_claude/api_design/api_IOandData/spec.md` 实现 vneurotk.io 和 vneurotk.data 模块

---

## 完成内容

### 1. vneurotk.io 模块

实现了完整的路径管理和数据加载系统：

#### Path 类层次结构
- **VTKPath**: 基础路径类
  - 属性: root, session, subject, task, run, desc, probe, suffix, extension, modality
  - `fpath` 属性自动构建文件路径
  - 支持位置参数和关键字参数构造

- **EphysPath(VTKPath)**: Ephys 数据路径
  - 包含 probe 属性
  - modality 自动设为 "ephys"

- **MNEPath(VTKPath)**: MNE 数据路径
  - 自定义 fpath 构建 MNE 风格路径
  - 格式: `sub-{subject}_ses-{session}_task-{task}_run-{run}_{suffix}.{extension}`

- **BIDSPath(VTKPath)**: BIDS 数据路径
  - 内部封装 mne_bids.BIDSPath
  - 提供 bids_path 属性访问底层对象

#### load() 函数
通用数据加载器，支持：
- **MNE 格式**: 使用 `mne.io.read_raw()` 加载
- **BIDS 格式**: 使用 `mne_bids.read_raw_bids()` 加载
- **H5 格式**: 加载已保存的 VTK 数据
- **Ephys 格式**: 预留接口（NotImplementedError）

自动提取元信息：采样率、通道名、滤波参数、源文件路径

### 2. vneurotk.data.BaseData 类

核心数据容器，包含：

#### 数据属性
- `neuro`: (ntime, nchan) 神经数据数组
- `neuro_info`: 元信息字典（采样率、通道名、滤波参数等）
- `visual`: (ntime,) 视觉刺激标签数组
- `visual_info`: 刺激元信息（刺激数量、ID列表）
- `trial`: (ntime,) 试次标签数组
- `trial_info`: 试次元信息（baseline、trial_window）
- `trial_starts`, `trial_ends`, `visual_onsets`: 试次时间索引

#### 核心方法
- **configure(trial_window, visual_onsets, visual_ids)**
  - 配置试次结构
  - 支持秒（float）或采样点（int）为单位的 trial_window
  - 自动生成 visual、trial 数组和相关元信息

- **save(path)**
  - 保存为 H5 格式
  - 仅保存已配置的数据
  - 支持 VTKPath、Path 或字符串路径

#### 便捷属性
- `ntime`, `nchan`: 数据维度
- `configured`: 是否已配置试次
- `n_trials`: 试次数量

### 3. 测试覆盖

编写了 10 个单元测试，覆盖：
- 所有 Path 类的构造和属性
- BaseData 的创建、配置、保存、加载
- 错误处理（未配置保存、未实现的加载器）

**测试结果**: ✅ 10/10 通过

---

## API 使用示例

### 加载 MNE 数据
```python
import vneurotk as vnt
from vneurotk.io import MNEPath

path = MNEPath(
    root="/data/mne",
    subject="01",
    session="ImageNet01",
    task="ImageNet",
    run="01",
    suffix="clean",
    extension=".fif",
)
data = vnt.load(path)
```

### 配置试次
```python
import numpy as np

visual_onsets = np.array([100, 400, 700])  # 刺激出现时间点
visual_ids = np.array([1, 2, 1])           # 对应的刺激ID
trial_window = [-0.2, 0.8]                 # 试次窗口（秒）

data.configure(
    trial_window=trial_window,
    visual_onsets=visual_onsets,
    visual_ids=visual_ids,
)
```

### 保存数据
```python
from vneurotk.io import VTKPath
from pathlib import Path

save_path = VTKPath(
    Path("/save/dir"),
    subject="01",
    session="test",
    task="task1",
)
data.save(save_path)
```

### 加载已保存数据
```python
loaded_data = vnt.load(save_path)
```

---

## 技术细节

- **Python 版本**: ≥ 3.10
- **类型注解**: 完整覆盖
- **文档**: Numpy 风格 docstring
- **日志**: 使用 loguru
- **数据格式**: HDF5 (h5py)
- **依赖**: numpy, h5py, mne, mne-bids, loguru

---

## 文件结构

```
src/vneurotk/
├── __init__.py              # 导出 load
├── io/
│   ├── __init__.py          # 导出 Path 类和 load
│   ├── path.py              # VTKPath, EphysPath, MNEPath, BIDSPath
│   └── loader.py            # load 函数实现
└── data/
    ├── __init__.py          # 导出 BaseData
    └── base.py              # BaseData 实现

tests/
└── test_io_and_data.py      # 单元测试
```

---

## 下一步

根据 spec.md，当前实现已完成：
- ✅ vneurotk.io 模块
- ✅ vneurotk.data.BaseData 类
- ✅ 完整的 API 示例验证

待实现：
- vneurotk.meta 模块（如有需求）
- EphysPath 的实际加载逻辑（当前为占位符）
