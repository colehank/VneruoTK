# TODO list:
- [x] vneurotk.io - Completed 2026-04-06
- [x] vneurotk.data - Completed 2026-04-06
- [x] vneurotk.io.EphysPath 重设计 - Completed 2026-04-09
- [x] vneurotk ephys 加载 + 测试 - Completed 2026-04-09
- [] vneurotk.meta

## Task notes:

### 2026-04-09: EphysPath 重设计
**完成内容：**
1. 新增 `EPHYS_DTYPES`、`EPHYS_EXTENSIONS` 模块级常量（`path.py`）
2. 重写 `EphysPath`：
   - 新增字段：`session_id: str | None`，`dtype: str | None`
   - `__post_init__` 验证 dtype 和 extension 合法性
   - 覆盖 `fpath`：`{root}/sessions/{session_id}/{dtype}_{session_id}[_probe{N}].{ext}`
   - 新增属性：`session_dir`、`raw_dir`、`nwb_path`
   - 新增类方法：`from_components(root, date, subject, paradigm, region, ...)`
3. 修复 `loader.py`：将 `EphysPath` 判断提前至 `.h5` 扩展名检查之前
4. 导出 `EPHYS_DTYPES`、`EPHYS_EXTENSIONS` 至 `vneurotk.io.__init__`
5. 更新 `TestEphysPath` 测试类（12 个测试用例），所有 28 个测试全部通过

### 2026-04-09: Ephys 加载实现 + 测试
**完成内容：**
1. `BaseData` 扩展：`neuro` 改为 property 支持 lazy loading；新增 `trial_meta`、`data_level` 参数
2. `Info` 适配：处理 sfreq/highpass/lowpass 为 None 的情况；显示 data_level
3. `loader.py` 全面重写：
   - `_load_ephys_raster`: COO 稀疏→dense 延迟加载，自动读取 TrialRecord/UnitProp
   - `_load_ephys_mean_fr`: 直接加载，data_level="trial"
   - `_load_ephys_stim_fr`: 直接加载，data_level="stimulus"
   - AvgPsth → NotImplementedError 占位
4. 修复 `save()`: stim_ids 改用 h5 dataset 存储（解决大数组超出 attribute 限制）
5. 修复 `_load_from_h5`: 支持从 dataset 读取 stim_ids；支持 trial_meta/data_level 恢复
6. **44 个测试全部通过**，新增：
   - `TestLoadFunction`：AvgPsth NotImplementedError、metadata dtype ValueError
   - `TestLoadEphysRaster`：lazy loading、access、shape、trial_meta、visual_info、to_continues
   - `TestLoadEphysMeanFr`：MeanFr/ChMeanFr 加载、configure/crop 拦截
   - `TestLoadEphysStimFr`：ChStimFr stimulus-level 加载
   - `TestEphysSaveLoadRoundtrip`：save/load 往返 + trial_meta 保留

### 2026-04-06: vneurotk.io and vneurotk.data implementation
**Completed modules:**
1. `vneurotk.io` - Path classes and data loading
   - `VTKPath`: Base path class with all attributes (root, session, subject, task, run, desc, probe, suffix, extension, modality)
   - `EphysPath`: Ephys-specific path with probe attribute
   - `MNEPath`: MNE-specific path with custom fpath construction
   - `BIDSPath`: BIDS-specific path wrapping mne_bids.BIDSPath
   - `load()`: Universal loader supporting MNE, BIDS, and h5 formats (EphysPath placeholder)

2. `vneurotk.data.BaseData` - Core data container
   - Attributes: neuro, neuro_info, visual, visual_info, trial, trial_info, trial_starts, trial_ends, visual_onsets
   - `configure()`: Configures trial structure from trial_window, visual_onsets, visual_ids
   - `save()`: Saves configured data to h5 format
   - Properties: ntime, nchan, configured, n_trials

**Tests:** 10 unit tests covering all core functionality, all passing

**API verified:** Matches spec.md examples from api_path.py and api_load_save.py