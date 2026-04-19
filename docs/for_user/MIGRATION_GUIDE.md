# VTKPath 基类架构重构 - 迁移指南

## 概述

VneuroTK 项目已成功重构了 VTKPath 架构，现在具有统一的基类设计，支持多种神经科学数据类型的路径管理。

## 架构变更

### 新增文件

1. **基类模块**
   - `src/vneurotk/io/base.py` - `BaseVTKPath` 抽象基类

2. **包初始化文件**
   - `src/vneurotk/__init__.py`
   - `src/vneurotk/io/__init__.py`
   - `src/vneurotk/io/ephys/__init__.py`
   - `src/vneurotk/io/mne/__init__.py`

3. **测试文件**
   - `examples/test_vtkpath.py` - 全面的功能测试

### 修改文件

1. **Ephys 模块** (`src/vneurotk/io/ephys/path.py`)
   - 类名: `VTKPath` → `EphysVTKPath`
   - 继承自 `BaseVTKPath`
   - 使用类属性存储配置

2. **MNE 模块** (`src/vneurotk/io/mne/path.py`)
   - 从 BIDSPath 包装器重写为自主实现
   - 类名: `VTKPath` → `MneVTKPath`
   - 继承自 `BaseVTKPath`
   - 实现完整的 BIDS 标准路径生成

3. **示例代码** (`examples/path.py`)
   - 更新导入语句
   - 使用新的类名

## 使用指南

### 导入方式

**推荐的新方式：**

```python
from vneurotk.io.ephys import EphysVTKPath
from vneurotk.io.mne import MneVTKPath
from vneurotk.io import BaseVTKPath
```

### Ephys 数据路径

```python
from vneurotk.io.ephys import EphysVTKPath

# 创建路径对象
path = EphysVTKPath(
    root="DB/project",
    session="251024_FanFan_nsd1w_MSB",
    desc="TrialRaster",
    probe=1
)

# 访问属性
print(path.fpath)       # 完整路径
print(path.basename)    # 文件名
print(path.directory)   # 目录路径
print(path.extension)   # 扩展名（自动推断为 .h5）

# 链式更新
base_path = EphysVTKPath(root="DB/project", session="ses-01")
trial_raster = base_path.update(desc="TrialRaster", probe=1)
unit_prop = base_path.update(desc="UnitProp", probe=1)

# 从路径解析
from pathlib import Path
parsed = EphysVTKPath.from_path(
    Path("DB/project/sessions/ses-01/UnitProp_ses-01_probe1.csv")
)
```

### MNE 数据路径（BIDS 风格）

```python
from vneurotk.io.mne import MneVTKPath

# 创建路径对象
path = MneVTKPath(
    root="/data/NOD-MEEG",
    subject="01",
    session="ImageNet01",
    task="ImageNet",
    run="01",
    datatype="meg",
    suffix="meg",
    extension=".fif"
)

# 生成的路径符合 BIDS 标准
print(path.fpath)
# 输出: /data/NOD-MEEG/sub-01/ses-ImageNet01/meg/sub-01_ses-ImageNet01_task-ImageNet_run-01_meg.fif

# 链式更新
base_path = MneVTKPath(root="/data", subject="01", session="ses-01", datatype="meg")
raw = base_path.update(task="RestingState", run="01", suffix="meg")
events = base_path.update(task="RestingState", suffix="events", extension=".tsv")
```

### 扩展新数据类型

继承 `BaseVTKPath` 可以轻松添加新的数据类型：

```python
from vneurotk.io.base import BaseVTKPath
from pathlib import Path
from typing import Optional, Union, Dict, Set

class FmriVTKPath(BaseVTKPath):
    """fMRI 数据路径管理"""
    
    # 配置扩展名映射
    _extension_registry: Dict[str, str] = {
        "bold": ".nii.gz",
        "T1w": ".nii.gz",
        "T2w": ".nii.gz",
    }
    _shared_types: Set[str] = set()
    
    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[Union[int, str]] = None,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        self._subject = subject
        self._task = task
        self._run = run
        self._suffix = suffix
        
        super().__init__(
            root=root,
            session=session,
            extension=extension,
            subject=subject,
            task=task,
            run=run,
            suffix=suffix,
        )
    
    @property
    def basename(self) -> str:
        """实现 BIDS 风格文件命名"""
        parts = [f"sub-{self._subject}"]
        if self._session:
            parts.append(f"ses-{self._session}")
        if self._task:
            parts.append(f"task-{self._task}")
        if self._run:
            parts.append(f"run-{self._run:02d}" if isinstance(self._run, int) else f"run-{self._run}")
        if self._suffix:
            parts.append(self._suffix)
        
        name = "_".join(parts)
        if self._extension:
            name += self._extension
        return name
    
    @property
    def directory(self) -> Path:
        """实现 BIDS 风格目录结构"""
        path = self._root / f"sub-{self._subject}"
        if self._session:
            path = path / f"ses-{self._session}"
        path = path / "func"
        return path
```

## 关键特性

### 1. 配置注册系统

每个子类维护自己的扩展名映射和共享类型：

```python
# 查看配置
print(EphysVTKPath._extension_registry)
# {'TrialRaster': '.h5', 'UnitProp': '.csv', ...}

print(EphysVTKPath._shared_types)
# {'TrialRecord', 'ChTrialRecord'}
```

### 2. 自动扩展名推断

根据 `desc`、`suffix` 或 `datatype` 自动推断扩展名：

```python
# Ephys: 根据 desc 推断
path = EphysVTKPath(root="DB", session="ses-01", desc="TrialRaster")
print(path.extension)  # .h5 (自动推断)

# MNE: 根据 suffix 推断
path = MneVTKPath(root="/data", subject="01", session="ses-01", 
                  datatype="meg", suffix="meg")
print(path.extension)  # .fif (自动推断)
```

### 3. 链式更新（不可变模式）

所有更新操作都返回新实例，原对象不变：

```python
base = EphysVTKPath(root="DB", session="ses-01")
path1 = base.update(desc="TrialRaster", probe=1)
path2 = base.update(desc="UnitProp", probe=2)

# base 保持不变
print(base.desc)   # None
print(path1.desc)  # TrialRaster
print(path2.desc)  # UnitProp
```

### 4. 路径反向解析

从文件路径反向构建路径对象：

```python
from pathlib import Path

path_obj = EphysVTKPath.from_path(
    Path("DB/sessions/ses-01/UnitProp_ses-01_probe1.csv")
)

print(path_obj.desc)    # UnitProp
print(path_obj.session) # ses-01
print(path_obj.probe)   # 1
```

## 验证

运行测试脚本验证所有功能：

```bash
uv run python examples/test_vtkpath.py
```

运行示例代码：

```bash
uv run python examples/path.py
```

## 迁移检查清单

- [ ] 更新导入语句（`from vneurotk.io.ephys import EphysVTKPath`）
- [ ] 将 `VTKPath` 替换为 `EphysVTKPath` 或 `MneVTKPath`
- [ ] 对于 MNE，添加 `suffix` 参数（如 `suffix="meg"`）
- [ ] 验证扩展名自动推断是否正确
- [ ] 运行测试确保功能正常

## 常见问题

### Q: 旧代码还能用吗？

A: 可以。我们提供了向后兼容别名 `VTKPath`，但建议尽快迁移到新的类名。别名将在 v1.0.0 中移除。

### Q: 如何添加新的文件类型？

A: 直接在子类的 `_extension_registry` 中添加映射：

```python
EphysVTKPath._extension_registry["NewType"] = ".ext"
```

### Q: MNE 模块为什么不再使用 BIDSPath？

A: 为了保持架构一致性和更好的控制能力。新实现完全遵循 BIDS 标准，同时与其他数据类型共享相同的基类接口。

### Q: 如何自定义文件命名规则？

A: 重写子类的 `basename` 属性：

```python
class CustomVTKPath(BaseVTKPath):
    @property
    def basename(self) -> str:
        # 自定义命名逻辑
        return f"custom_{self._session}.ext"
```

## 总结

新的基类架构提供了：

✅ 统一的接口和通用功能  
✅ 灵活的配置注册系统  
✅ 链式更新和不可变模式  
✅ 自动扩展名推断  
✅ 轻松扩展新数据类型  
✅ 向后兼容性  

这个架构为未来添加更多神经科学数据类型（如 fMRI、ECoG、iEEG 等）奠定了坚实的基础。
