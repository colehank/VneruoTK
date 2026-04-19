# vneurotk API Design: vneurotk.io.EphysPath
EphysPath是vneurotk.io.VTKPath的子类，专门用于表示ephys数据的路径信息。

## Ephys目录结构
```
MonkeyVision/  
┣ embeddings/  
┣ nwb/  
┣ raw/  
┣ sessions/  
┗ stimuli/
```
Ephys数据以项目为单位组织，每个项目包含上述五个子目录：
- `raw/`：原始数据文件，包含未经处理的电生理数据
- `nwb/`：预处理中间产物
- `sessions/`：**最终整理好的session-level数据文件，供分析使用**
- `stimuli/`：实验刺激相关文件,此处不做过多规范，允许项目特化
- `embeddings/`：试验刺激的embedding文件，此处不做过多规范，允许项目特化

## `raw/`
```
raw/
└── {session_id}/
    ├── spikeglx/    # 原始 Neuropixels 数据
    │   ├── probe0/  # 多电极时按 probe 分子目录
    │   └── probe1/
    └── bhv2/        # 行为数据
```

## `nwb/`
```
nwb/
├── {session_id}.nwb                    # 单电极
├── {session_id}_probe0.nwb             # 多电极
└── {session_id}_probe1.nwb
```
## `sessions/`
总体命名规范: **`{root}/sessions/{session_id}/{dtype}_{session_id}{_probe{N}}.{ext}`**

- `root`: 数据的根目录，通常是一个字符串，指向数据存储的根路径。如：`DB/ephys/MonkeyVision`
- `session_id`: session的唯一标识符，通常包含日期、受试者、实验范式等信息
  - 格式为 `{date}_{subject}_{paradigm}{block}_{region}`
  - eg., `251024_FanFan_nsd1w_MSB`。其中， `nsd` 表示实验范式，`1w`表示session序号和可选标记，`MSB`表示记录脑区。
- `dtype`: 数据类型，包含：
  - `TrialRaster` 
  - `TrialRecord`
  - `UnitProp`
  - `MeanFr`
  - `AvgPsth`
  - `ChTrialRaster`
  - `ChTrialRecord`
  - `ChMeanFr`
  - `ChProp`  
  其中，以`Ch`开头的数据类型表示channel-level MUA数据，其他表示unit-level MUA数据。

- `probe{N}`: 可选项，仅当同一session使用多根电极时需要添加，格式为`_probe{N}`，N从0开始编号

  - 单电极 session：文件名中**不带** probe 标签
  - 多电极 session：每根 probe 的独立文件**必须带** probe 标签

- `ext`: 文件扩展名，支持 `h5`、`csv` 和 `nwb` 格式

## 示例目录路径：  
```
DB/ephys/MonkeyVision/sessions/251024_FanFan_nsd1w_MSB_fake/   
┣ AvgPsth_251024_FanFan_nsd1w_MSB.h5  
┣ AvgPsth_251024_FanFan_nsd1w_MSB_probe0.h5  
┣ AvgPsth_251024_FanFan_nsd1w_MSB_probe1.h5  
┣ ChMeanFr_251024_FanFan_nsd1w_MSB.h5  
┣ ChProp_251024_FanFan_nsd1w_MSB.csv  
┣ ChTrialRaster_251024_FanFan_nsd1w_MSB.csv  
┣ ChTrialRaster_251024_FanFan_nsd1w_MSB.h5  
┣ MeanFr_251024_FanFan_nsd1w_MSB.h5  
┣ MeanFr_251024_FanFan_nsd1w_MSB_probe0.h5  
┣ MeanFr_251024_FanFan_nsd1w_MSB_probe1.h5  
┣ TrialRaster_251024_FanFan_nsd1w_MSB.csv  
┣ TrialRaster_251024_FanFan_nsd1w_MSB.h5  
┣ TrialRaster_251024_FanFan_nsd1w_MSB_probe0.csv  
┣ TrialRaster_251024_FanFan_nsd1w_MSB_probe0.h5  
┣ TrialRaster_251024_FanFan_nsd1w_MSB_probe1.csv  
┣ TrialRaster_251024_FanFan_nsd1w_MSB_probe1.h5  
┣ UnitProp_251024_FanFan_nsd1w_MSB.csv  
┣ UnitProp_251024_FanFan_nsd1w_MSB_probe0.csv  
┗ UnitProp_251024_FanFan_nsd1w_MSB_probe1.csv  
```
