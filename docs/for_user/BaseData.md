# Data Structures

## Data types
- Neural data: electrophysiology (ephys) and magnetoencephalography/electroencephalography (MEEG) and so on.
- Stimuli data: visual stimuli, embeddings/labels of visual stimuli.

## Neural data
### Data types
1. WithTime: ntrial x n_neuro x n_time
2. WithoutTime: ntrial x n_neuro
### Data source
1. Ephys: monkey MUA(multi-unit activity), n_neuro -> n_unit
   - Raster: ntrial x n_unit x n_timebin
   - PSTH: ntrial  x n_unit x n_timebin
   - MeanFR: ntrial x n_unit
     Raster --sliding smooth window--> PSTH --mean firing rate (FR)--> MeanFR
2. MEEG: human MEG/EEG, n_neuro -> n_channel
   - BaseRaw: ntime x n_channel
   - Epochs: ntrial x n_channel x n_time

## Stimuli data
### Data types
1. raw data: the .PNG/.JPEG/... files of visual stimuli
2. embeddings: the vector representations of visual stimuli, e.g., from a pretrained vision model. ntrial x embedding_dim
3. labels: the id of visual stimuli. ntrial

## Metadata connecting neural data and stimuli data
Using a DataFrame-like, columns must contain at least:
- subject: the unique identifier of a subject.
- session: the unique identifier of a recording session.
- trial: the unique identifier of a trial.
- stimulus: the unique identifier of a stimulus.




2层级任务：
	1. autoreasearch --> vneuron optimization 
  1.   ( train.py ) ==> solid toolbox  ==> VneuroTK (blue print \ prompt) [GZX: encoder + ephys data specification] [ZGH: cebra + MEEG data spec]
      1.   harness engineering, 脱手 cc 全程，vibe coding 实践，学会撬动AI编程工具 ==> VneuroTK [encoder][cebra] == {encoding+decoding}
      2.   cebra & encoder -can handle- {ephys + MEEG} data spec  

分工上：1. 讨论清楚 数据规范+工具包架构 基于 蓝图.md ==> 蓝图V2.md ==> cc 实现； 2. How to use/apply autoreseach;
- 各自业务需求：[cebra] 模型已有；数据： cebra X : time x feature , time-samples[ephys: 30000hz --> 1000hz (1ms); MEG: 1200hz --> 100hz (10ms)]; feature [ephys: unit/channel(spike sorting, ICA 394channel --> 800 unit --> 200 unit); MEEG: channel] ; 对齐？ 
encoder Y：
- Raster data: ntrial x n_unit x n_timebin (time == ntrial x n_timebin). eg., 0010001002
- PSTH data: ntrial  x n_unit x n_timebin (25ms smooth sliding window FR) 
- Mean firing rate data: ntrial x n_unit
encoder X : ntrial x embedding_dim; ｛关于 embdding 的管理设计。提前写入有一个集合，需要设计协议｝。
TripleN10k (NSD 73000) fullNSD_embeddings.h5/fullNSD_embedding.h5/fullNSD_robust_embedding.h5 : dataset : network / layer = (73000, n_feature)
								pytorch, timm, huggingface(pytorch),  用法：提前存好，通过索引找到需要embedding, 需要设计 config 管理；问题：1. 很大，可能压缩算法能解决；2. 刺激混合情况，选择性locaizer, FOB; 3. 可拓展性，引入新刺激，新的embedding。
	1. 蓝图文件搞定好+学习最先进的 vibe coding （harness engineering）经验，项目规范， 蓝图==>
- 实现细节设计

