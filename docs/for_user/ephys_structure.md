# Neuro-Ephys 数据结构规范（通用版）

> **版本**：1.0
> **制定日期**：2026-03-20
> **适用范围**：所有神经电生理项目的数据整理与归档
> **维护原则**：本文件是所有项目的上位规范；项目级规范（`DATA_STRUCTURE_{project}.md`）应与本文件保持一致，仅可在本文件允许的范围内做项目特化。

---

## 1. 顶层布局

```
F:\
├── #Datasets\                          # 所有项目数据，每个项目完全自包含
│   ├── {project_id}\
│   │   ├── raw\
│   │   ├── nwb\
│   │   ├── sessions\
│   │   ├── stimuli\
│   │   ├── embeddings\
│   │   ├── results\
│   │   └── DATA_STRUCTURE_{project_id}.md
│   └── {project_id}\
│       └── ...
│
└── SharedModels\                       # 跨项目共享的预训练权重（唯一全局共享目录）
    ├── robust\
    └── vvs\
```

### 设计原则

- **项目是第一级组织单位**：每个项目完全自包含，数据流 raw → nwb → sessions → results 全部在项目目录内完成
- **只有预训练权重跨项目共享**：`F:\SharedModels\` 是唯一的全局目录
- **stimuli / embeddings 归属项目**：若两个项目确实复用相同文件，用 symlink 指向源项目，不做物理拷贝
- **扁平且可预测**：所有项目内部子目录名称保持一致，代码只需一个 `PROJECT_ROOT` 变量即可定位一切

---

## 2. 命名规范

### 2.1 Project ID

自由命名，建议 PascalCase 或 snake_case，能一眼看出实验内容即可。

示例：`TripleN10k`、`GratingOrientation`、`TextureSegmentation`

### 2.2 Session ID 格式

```
{date}_{subject}_{paradigm}{block}_{region}
```

| 字段 | 格式 | 示例 | 说明 |
|---|---|---|---|
| `date` | YYMMDD | `251024` | 记录日期 |
| `subject` | PascalCase | `FanFan` | 动物/被试名 |
| `paradigm` | 小写缩写 | `nsd` | 实验范式 |
| `block` | 数字 + 可选字母后缀 | `1w` | session 序号 + 可选标记 |
| `region` | 大写缩写 | `MSB` | 目标脑区，便于多脑区项目扩展 |

**完整示例**：`251024_FanFan_nsd1w_MSB`

> `region` 字段自本规范起为必填项，以支持后续多脑区扩展。

### 2.3 多电极（Multi-Probe）命名

同一 session 使用多根电极时，通过文件名中的 **probe 标签** 区分：

```
probe 标签格式: _probe{N}     N = 0, 1, 2 ...
```

- 单电极 session：文件名中**不带** probe 标签（向后兼容）
- 多电极 session：每根 probe 的独立文件**必须带** probe 标签

### 2.4 文件命名模板

```
{FileType}_{session_id}[_probe{N}].{ext}
```

| 字段 | 是否必须 | 说明 |
|---|---|---|
| `{FileType}` | 是 | PascalCase，见 §4 文件类型清单 |
| `_{session_id}` | 是 | 保证文件脱离目录后仍可识别来源 |
| `_probe{N}` | 条件必须 | 仅多电极 session 必须；单电极省略 |
| `.{ext}` | 是 | `h5` / `csv` / `nwb` |

---

## 3. 项目内部结构

以下路径均相对于 `#Datasets/{project_id}/`。

### 3.1 raw/ — 原始数据归档

```
raw/
└── {session_id}/
    ├── spikeglx/               # 原始 Neuropixels 数据（或 symlink → 外部驱动器）
    │   ├── probe0/             # 多电极时按 probe 分子目录
    │   └── probe1/
    └── bhv2/                   # 行为数据
```

**规则**：只读，永不修改。原始数据体积大时可保留在外部驱动器，此处创建 symlink。

### 3.2 nwb/ — 预处理中间产物

```
nwb/
├── {session_id}.nwb                    # 单电极
├── {session_id}_probe0.nwb             # 多电极
└── {session_id}_probe1.nwb
```

**规则**：一根电极对应一个 NWB 文件。日常不直接读取，仅用于重建下游文件。

### 3.3 sessions/ — 核心工作区 ★

日常分析代码唯一需要接触的数据目录。

#### 单电极 session

```
sessions/
└── {session_id}/
    ├── TrialRaster_{session_id}.h5         # NWB 拆出
    ├── TrialRecord_{session_id}.csv        # NWB 拆出 → 原地 enrich
    ├── UnitProp_{session_id}.csv           # NWB 拆出 → 原地 enrich
    ├── MeanFr_{session_id}.h5              # 派生：时间窗均值发放率
    └── AvgPsth_{session_id}.h5             # 派生：跨试次平均 PSTH
```

#### 多电极 session

```
sessions/
└── {session_id}/
    ├── TrialRaster_{session_id}_probe0.h5
    ├── TrialRaster_{session_id}_probe1.h5
    ├── TrialRecord_{session_id}.csv         # 跨 probe 共享，只有一份
    ├── UnitProp_{session_id}_probe0.csv
    ├── UnitProp_{session_id}_probe1.csv
    ├── MeanFr_{session_id}_probe0.h5
    ├── MeanFr_{session_id}_probe1.h5
    ├── AvgPsth_{session_id}_probe0.h5
    └── AvgPsth_{session_id}_probe1.h5
```

#### Channel-Level 文件（项目特有，按需添加）

部分项目需要 channel-level MUA 分析，使用 `Ch` 前缀：

```
sessions/{session_id}/
    ├── ChTrialRaster_{session_id}.h5
    ├── ChTrialRecord_{session_id}.csv
    ├── ChMeanFr_{session_id}.h5
    └── ChProp_{session_id}.csv
```

### 3.4 stimuli/ — 刺激材料

```
stimuli/
├── {stimulus_set}.hdf5
└── {index_file}.tsv / .csv        # 刺激集索引、条件映射表等
```

若两个项目使用完全相同的刺激集，在第二个项目中用 symlink 指向源项目的对应文件。

### 3.5 embeddings/ — 特征向量

```
embeddings/
├── {prefix}_visual_embedding.h5
├── {prefix}_robust_embeddings.h5
└── ...
```

同 stimuli，跨项目共享时用 symlink。

### 3.6 results/ — 分析输出

`results/` 内部结构**不做强制要求**，以下为推荐参考：

```
results/
├── {analysis_type}/
│   └── {run_id}/
│       └── ...
└── ...
```

- `{analysis_type}` 示例：`pls`、`encoding`、`pgd`
- `{run_id}` 建议与脚本中的 `RUN_NAME` 或实验批次保持一致

---

## 4. 文件类型清单

| FileType | 扩展名 | 来源 | 说明 | Probe 独立？ |
|---|---|---|---|---|
| `TrialRaster` | `.h5` | NWB 拆出 | 每 trial 的 spike raster | ✓ |
| `TrialRecord` | `.csv` | NWB 拆出 → enrich | 试次信息（刺激、condition 等） | ✗ 共享 |
| `UnitProp` | `.csv` | NWB 拆出 → enrich | 单元属性（reliability 等） | ✓ |
| `MeanFr` | `.h5` | TrialRaster 派生 | 时间窗均值发放率 | ✓ |
| `AvgPsth` | `.h5` | TrialRaster 派生 | 跨试次平均 PSTH | ✓ |
| `ChTrialRaster` | `.h5` | NWB 拆出 | Channel-level MUA raster | ✓ |
| `ChTrialRecord` | `.csv` | NWB 拆出 → enrich | Channel-level 试次信息 | ✗ 共享 |
| `ChMeanFr` | `.h5` | ChTrialRaster 派生 | Channel-level 窗均值 | ✓ |
| `ChProp` | `.csv` | 派生计算 | Channel 属性（reliability 等） | ✓ |

**TrialRecord 共享原则**：同一 session 的多根电极共享同一套行为任务，`TrialRecord` 和 `ChTrialRecord` 各只有一份，不按 probe 拆分。

---

## 5. 数据流

```
raw/spikeglx + raw/bhv2
        │
        ▼  [预处理: Kilosort + NWB 打包]
    nwb/{session_id}[_probe{N}].nwb
        │
        ▼  [NWB 后处理: 拆分]
    sessions/{session_id}/
        ├── TrialRaster_*.h5
        ├── TrialRecord_*.csv
        └── UnitProp_*.csv
              │
              ▼  [派生计算]
        ├── MeanFr_*.h5            ← TrialRaster 取窗均值
        ├── AvgPsth_*.h5           ← TrialRaster 跨试次平均
        ├── TrialRecord_*.csv      ← 原地 enrich（condition mapping）
        └── UnitProp_*.csv         ← 原地 enrich（reliability, selectivity…）
```

**处理规则**：
- `TrialRecord` / `UnitProp` 原地覆盖更新，不保留历史版本
- 如需重建下游文件，从 `nwb/` 重新拆分
- `raw/` 是终极备份，永不修改

---

## 6. 代码中的路径约定

```python
# config.py — 所有项目共用模板
from pathlib import Path

DATASETS_ROOT = Path("F:/#Datasets")
SHARED_MODELS = Path("F:/SharedModels")

def project_root(project_id: str) -> Path:
    return DATASETS_ROOT / project_id

def session_dir(project_id: str, session_id: str) -> Path:
    return project_root(project_id) / "sessions" / session_id

def session_file(
    project_id: str,
    session_id: str,
    file_type: str,
    ext: str,
    probe: int | None = None,
) -> Path:
    name = f"{file_type}_{session_id}"
    if probe is not None:
        name += f"_probe{probe}"
    return session_dir(project_id, session_id) / f"{name}.{ext}"

def results_dir(project_id: str, analysis_type: str, run_id: str) -> Path:
    return project_root(project_id) / "results" / analysis_type / run_id
```

---

## 7. 跨项目共享策略

| 资源类型 | 存放位置 | 跨项目共享方式 |
|---|---|---|
| 预训练权重 | `F:\SharedModels\{family}\` | 所有项目直接引用同一路径 |
| 刺激材料 | 各项目 `stimuli\` | symlink 指向源项目的对应文件 |
| 特征向量 | 各项目 `embeddings\` | symlink 指向源项目的对应文件 |
| 电生理数据 | 各项目 `raw\`, `nwb\`, `sessions\` | 不共享，各项目独立 |
| 分析结果 | 各项目 `results\` | 不共享，各项目独立 |

---

## 8. 扩展指南

### 添加新项目

1. 在 `#Datasets\` 下创建新目录，内部结构与本规范模板一致
2. 将本文件 symlink 或复制为 `DATA_STRUCTURE_{project_id}.md`，填入项目特化内容
3. 若与已有项目共享刺激/特征，在对应目录下创建 symlink

### 添加新文件类型

1. 在 §4 文件类型清单中增加一行
2. 遵循命名模板 `{FileType}_{session_id}[_probe{N}].{ext}`
3. 在 §5 数据流中标注来源和依赖
4. 更新版本号

### 迁移到多电极

- 已有单电极 session 无需改动（无 probe 后缀 = 单电极）
- 新增多电极 session 时，probe 独立文件必须带 `_probe{N}`
- `TrialRecord` 始终共享，不按 probe 拆分
