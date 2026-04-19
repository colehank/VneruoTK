# vneurotk.vision — DNN 视觉特征提取使用说明

## 整体目标

`vneurotk.vision` 解决视觉神经科学中**最核心的特征对齐问题**：

> 对于每张实验刺激图片，提取预训练 DNN 各层的中间激活，以供后续 encoding（用 DNN 激活预测神经响应）和 decoding（用神经响应重建刺激）任务使用。

---

## 设计思路

### 整体架构

```
PIL.Image / np.ndarray / pathlib.Path
         │
         ▼
  ┌─────────────┐    ┌────────────────┐    ┌──────────────────┐
  │   Backend   │◄───│  LayerSelector │    │ EmbeddingPolicy  │
  │  (加载/推理) │    │ (选哪些层 hook) │    │ (如何得到嵌入向量)│
  └──────┬──────┘    └────────────────┘    └────────┬─────────┘
         │ forward()                                │
         │ collect_activations()                    │ apply()
         ▼                                          ▼
  dict[layer, Tensor]  ─────────────────►  final_embedding Tensor
         │                                          │
         └──────────────────┬───────────────────────┘
                            ▼
               VisualRepresentations
               (n_stim, ...)   ← numpy / tensor 两种视图
```

### 为什么这样设计

| 决策 | 原因 |
|---|---|
| Hook 而非修改模型 | 任意 PyTorch 模型开箱即用，不需要修改 forward 方法 |
| 统一返回 `VisualRepresentations` | 单张与批量接口一致，n_stim=1 是批量的特例，下游代码无需 isinstance 判断 |
| 激活以 numpy 存储 | 科学计算（RSA、线性回归）均以 numpy 为主；tensor 视图按需转换 |
| Backend 分层 | timm / transformers / thingsvision 各有不同的加载与预处理逻辑，隔离后各自维护，不相互污染 |
| stim_id 索引 | 实验中图片 ID 是核心标识符，`select(ids)` 直接对齐 `BaseData.trial_stim_ids` |

---

## 快速开始

```python
from vneurotk.vision import VisionExtractor

# 内置模型，一行加载
ext = VisionExtractor("dinov2-vit-b", device="cuda")

# 单张图片 → n_stim=1
from PIL import Image
img = Image.open("cat.jpg")
vr = ext.extract(img)
print(vr.final_embedding.shape)     # (1, 197, 768) — all tokens
print(vr["encoder.layer.0"].shape)  # (1, 197, 768) — 第 0 层激活

# 实验全部刺激 → n_stim=N
images = {i: Image.open(f"stim_{i:04d}.jpg") for i in range(100)}
vr = ext.extract(images)
print(vr.final_embedding.shape)     # (100, 197, 768)
print(vr.n_stim)                    # 100
```

---

## 内置模型（REGISTRY）

```python
from vneurotk.vision import REGISTRY
print(REGISTRY.list())   # 所有已注册的短名
```

| 短名 | 架构 | 范式 | 嵌入策略 | Backend |
|---|---|---|---|---|
| `vit-b-16-imagenet` | ViT-B/16 | supervised | all_tokens | timm |
| `vit-b-16-in21k` | ViT-B/16 IN-21k | supervised | all_tokens | timm |
| `vit-b-16-dino` | ViT-B/16 DINO | selfsupervised | all_tokens | timm |
| `resnet50` | ResNet-50 | supervised | pre_head | timm |
| `resnet50-a1` | ResNet-50 A1 | supervised | pre_head | timm |
| `resnetv2-50` | ResNetV2-50 | supervised | pre_head | timm |
| `dinov2-vit-b` | DINOv2 ViT-B | selfsupervised | all_tokens | transformers |
| `dinov3-vit-b` | DINOv3 ViT-B | selfsupervised | all_tokens | transformers |
| `dinov3-vit-s` | DINOv3 ViT-S | selfsupervised | all_tokens | transformers |
| `clip-vit-b-32` | CLIP ViT-B/32 | contrastive | projection_out | transformers |
| `siglip-b-16` | SigLIP ViT-B/16 | contrastive | mean_pool | transformers |
| `siglip2-b-16` | SigLIP2 ViT-B/16 | contrastive | mean_pool | transformers |

查看某模型的详细配置：

```python
cfg = REGISTRY.get("dinov3-vit-b")
# ModelConfig(source='transformers',
#             model_id='facebook/dinov3-vitb16-pretrain-lvd1689m',
#             policy='all_tokens', paradigm='selfsupervised')
```

---

## 注册自定义模型

```python
from vneurotk.vision import REGISTRY, ModelConfig

REGISTRY.register("my-vit", ModelConfig(
    source="timm",
    model_id="vit_large_patch16_224",
    policy="all_tokens",
    paradigm="supervised",
))

ext = VisionExtractor("my-vit", device="cuda")
```

---

## Backend 详解

Backend 封装了**模型加载**、**图片预处理**、**forward 推理**三个步骤。通常你不需要直接使用 backend，由 `VisionExtractor` 内部管理。需要直接使用的场景：已有模型对象、需要离线加载权重、或需要精细控制推理过程。

### TimmBackend

基于 [timm](https://github.com/huggingface/pytorch-image-models)，支持 1000+ 预训练模型。

```python
from vneurotk.vision.extractor.backend import TimmBackend

backend = TimmBackend(device="cuda")
backend.load("vit_base_patch16_224", pretrained=True)

# 查看模型结构
for info in backend.enumerate_layers():
    print(f"{info.name:40s} {info.module_type:20s} depth={info.depth}")

# 直接推理
from PIL import Image
img = Image.open("cat.jpg")
inputs = backend.preprocess(img)   # {'pixel_values': Tensor(1,3,224,224)}
out = backend.forward(inputs)      # Tensor(1, num_classes)
```

预处理由 `timm.data.create_transform` 自动匹配模型配置（分辨率、归一化参数）。

### TransformersBackend

基于 HuggingFace `transformers`，支持所有 `AutoModel` 兼容模型，CLIP 自动识别。

```python
from vneurotk.vision.extractor.backend import TransformersBackend

# 普通 ViT / DINOv2 / SigLIP
backend = TransformersBackend(device="cuda", learning_paradigm="selfsupervised")
backend.load("facebook/dinov2-base")

# CLIP 自动识别，forward 返回带 image_embeds 的输出
backend_clip = TransformersBackend(device="cuda", learning_paradigm="contrastive")
backend_clip.load("openai/clip-vit-base-patch32")

# 推理
inputs = backend.preprocess(img)   # AutoProcessor 输出
out = backend.forward(inputs)      # HuggingFace ModelOutput
```

CLIP 的 hook 自动挂在 `vision_model` 上，forward 返回的 `out.image_embeds` 供 `projection_out` 策略使用。

### ThingsVisionBackend

基于 [thingsvision](https://github.com/ViCCo-Group/thingsvision)（需额外安装：`uv add thingsvision`）。

```python
from vneurotk.vision.extractor.backend import ThingsVisionBackend

backend = ThingsVisionBackend(source="torchvision", device="cuda")
backend.load("resnet50", pretrained=True)
```

`source` 对应 thingsvision 支持的模型来源（`"torchvision"`, `"timm"`, `"keras"` 等）。

### 从已有模型对象构建（from_model）

适合已有自定义权重的场景：

```python
import timm
import torch
from vneurotk.vision import VisionExtractor
from vneurotk.vision.extractor.backend import TimmBackend
from vneurotk.vision.extractor.selector import CustomSelector

# 加载自定义权重
model = timm.create_model("vit_base_patch16_224", pretrained=False)
model.load_state_dict(torch.load("my_weights.pth"))

backend = TimmBackend(device="cuda")
backend.load("vit_base_patch16_224", pretrained=False)
backend.model = model  # 替换为自定义权重

ext = VisionExtractor.from_model(
    model=model,
    backend=backend,
    embedding_policy="all_tokens",
    selector=CustomSelector(["blocks.6", "blocks.11"]),
)
```

---

## Hook 设置详解

Hook 决定**在哪些层捕获激活**。有三种设置方式：

### 方式 1：LayerSelector（推荐）

由 `VisionExtractor` 构造时自动调用，无需手动操作。

```python
from vneurotk.vision import VisionExtractor, BlockLevelSelector, AllLeafSelector, CustomSelector

# BlockLevelSelector（默认）：按架构自动识别主要 block
# ViT → blocks.0 ~ blocks.11
# ResNet → layer1 ~ layer4
# DINOv2 → encoder.layer.0 ~ encoder.layer.11
# DINOv3 → model.layer.0 ~ model.layer.11
ext = VisionExtractor("vit-b-16-in21k", selector=BlockLevelSelector())

# AllLeafSelector：所有叶子模块（跳过 Dropout/ReLU/Identity 等）
ext = VisionExtractor("resnet50", selector=AllLeafSelector())

# CustomSelector：精确指定层名（必须与 named_modules() 一致）
ext = VisionExtractor(
    "vit-b-16-in21k",
    selector=CustomSelector(["blocks.6", "blocks.11"]),
)
```

**BlockLevelSelector 支持的架构**：

| pattern | 典型模型 |
|---|---|
| `blocks.N` | timm ViT 系列 |
| `encoder.layers.N` | HF ViT |
| `encoder.layer.N` | HF DINOv2 |
| `model.layer.N` | HF DINOv3 |
| `layerN.M` | ResNet |
| `features.N` | VGG / EfficientNet |
| `stages.N` | ConvNeXt |
| `layers.N` | Swin Transformer |

对于不在列表中的模型，BlockLevelSelector 会 fallback 到 top-level children。可通过 `include_patterns` 追加 regex：

```python
BlockLevelSelector(include_patterns=[r"^custom_encoder\.\d+$"])
```

### 方式 2：先枚举层名，再精确 hook

```python
from vneurotk.vision.extractor.backend import TimmBackend

backend = TimmBackend(device="cpu")
backend.load("vit_base_patch16_224", pretrained=False)

# 查看所有可 hook 的层
for info in backend.enumerate_layers():
    if "block" in info.name:
        print(info.name, info.module_type)
# blocks.0   Block
# blocks.0.norm1   LayerNorm
# ...

# 指定感兴趣的层
backend.register_hooks(["blocks.3", "blocks.7", "blocks.11"])
```

### 方式 3：直接调用 register_hooks / remove_hooks

```python
# 注册（清除旧 hook 并重新注册）
backend.register_hooks(["blocks.0", "blocks.11"])

# 推理
out = backend.forward(inputs)

# 取出激活（同时清空内部缓冲区）
acts = backend.collect_activations()
# OrderedDict: {"blocks.0": Tensor(197,768), "blocks.11": Tensor(197,768)}

# 清除所有 hook（如不再需要）
backend.remove_hooks()
```

**注意**：`collect_activations()` 每次调用后会清空缓冲区，每次 forward 后必须调用一次，否则下一次 forward 的激活会直接覆盖。

---

## 嵌入策略（EmbeddingPolicy）

控制如何从模型输出提取最终嵌入向量（`final_embedding`）。

| 策略 | 输出形状 | 说明 |
|---|---|---|
| `all_tokens` | `(T, D)` | 全部 token（CLS + register + patch），ViT 默认 |
| `cls_token` | `(D,)` | 只取 CLS token（position 0） |
| `mean_pool` | `(D,)` | 对 T 维取均值，SigLIP 使用 |
| `projection_out` | `(D,)` | 投影头输出（`image_embeds`），CLIP 使用 |
| `pre_head` | `(D,)` | 最后一个 hook 层的均值，ResNet 使用 |
| `backbone_out` | `(D,)` | `last_hidden_state` 均值 |
| `custom` | 任意 | 用户自定义函数 |

覆盖默认策略：

```python
# ViT 默认 all_tokens，改为只取 CLS
ext = VisionExtractor("vit-b-16-in21k", embedding_policy="cls_token")

# 自定义：取第 11 层所有 token 均值
def my_policy(model_output, activations):
    return activations["blocks.11"].mean(dim=0)

ext = VisionExtractor(
    "vit-b-16-in21k",
    embedding_policy="custom",
    custom_fn=my_policy,
)
```

---

## 输出：VisualRepresentations

`extract()` 始终返回 `VisualRepresentations`。单张图片 `n_stim=1`，批量 `n_stim=N`。

### 基本属性

```python
vr = ext.extract(img_or_dict)

vr.n_stim                  # 图片数量（单张=1）
vr.layer_names             # ['blocks.0', ..., 'blocks.11']
vr.model_meta              # ModelMeta(model_name=..., source=..., ...)
vr.final_embedding         # np.ndarray (n_stim, ...)
```

### 获取各层激活（numpy）

```python
# 方式 A：索引访问（等同于 numpy()）
arr = vr["blocks.11"]           # np.ndarray (n_stim, 197, 768)

# 方式 B：numpy() 方法（None → final_embedding）
arr = vr.numpy("blocks.11")     # np.ndarray (n_stim, 197, 768)
emb = vr.numpy()                # np.ndarray (n_stim, ...)  ← final_embedding
```

### 获取各层激活（PyTorch Tensor）

```python
t = vr.to_tensor("blocks.11")  # torch.Tensor (n_stim, 197, 768)
t = vr.to_tensor()             # torch.Tensor (n_stim, ...)  ← final_embedding
```

### ViT Token 结构

```python
emb = vr.final_embedding        # (n_stim, T, D)

# ViT-B/16 @ 224px：T = 197 = 1 CLS + 196 patch tokens
emb[:, 0, :]    # (n_stim, 768) — CLS token
emb[:, 1:, :]   # (n_stim, 196, 768) — patch tokens（14×14 栅格扫描）

# DINOv2/v3：T = 201 = 1 CLS + 4 register tokens + 196 patch tokens
emb[:, 0, :]    # CLS token
emb[:, 1:5, :]  # register tokens（存储全局信息，不对应图片区域）
emb[:, 5:, :]   # patch tokens
```

SigLIP 无 CLS token，`mean_pool` 后 `final_embedding` 形状为 `(n_stim, 768)`。

### 按 stim_id 选取子集

```python
sub = vr.select([10, 20, 30])       # 按 stim_id，VisualRepresentations，n_stim=3
sub = vr.select_by_index([0, 1, 2]) # 按位置索引
```

---

## 完整激活获取示例

```python
from vneurotk.vision import VisionExtractor
from PIL import Image

ext = VisionExtractor("dinov2-vit-b", device="cuda")

# 提取所有刺激
images = {i: Image.open(f"stim_{i:04d}.jpg") for i in range(100)}
vr = ext.extract(images)

# 查看所有层名
print(vr.layer_names)
# ['encoder.layer.0', 'encoder.layer.1', ..., 'encoder.layer.11']

# 取出所有层的激活（字典）
all_acts = {layer: vr[layer] for layer in vr.layer_names}
# all_acts["encoder.layer.11"].shape → (100, 197, 768)

# 最终嵌入
final = vr.final_embedding          # (100, 197, 768)，all_tokens 策略
cls_only = final[:, 0, :]           # (100, 768)

# 转 tensor 用于模型训练
import torch
t = vr.to_tensor("encoder.layer.11")  # torch.Tensor (100, 197, 768)
cls_feat = t[:, 0, :]                  # (100, 768)
```

---

## 与 BaseData 对接

`BaseData.trial_stim_ids` 返回每个 trial 对应的 stim_id，与 `VisualRepresentations` 的 `stim_ids` 共享同一命名空间。

```python
from vneurotk.vision import VisionExtractor

ext = VisionExtractor("dinov2-vit-b", device="cuda")

# 提取全部刺激（images: {stim_id: PIL.Image}）
vr = ext.extract(images)

# 从神经数据取 trial 对应的 stim_id
stim_ids = neural_data.trial_stim_ids   # np.ndarray (n_trials,)

# 按 trial 顺序对齐
vr_aligned = vr.select(stim_ids)       # VisualRepresentations, n_stim=n_trials

vr_aligned.final_embedding.shape       # (n_trials, 197, 768)
vr_aligned["encoder.layer.11"].shape   # (n_trials, 197, 768)
```

---

## 注意事项

**模型缓存位置**

| Backend | 路径 |
|---|---|
| timm | `~/.cache/huggingface/hub/` |
| transformers | `~/.cache/huggingface/hub/` |
| thingsvision | `~/.cache/torch/hub/` |

**thingsvision 需要单独安装**：`uv add thingsvision`。实例化 `ThingsVisionBackend` 时立即检查，不延迟到 `load()`。

**collect_activations() 清空缓冲区**：每次 `forward()` 后需调用 `collect_activations()`；如果多次 forward 之间未取出激活，旧的激活会被覆盖丢失。

**batch_size=1**：当前 backend 均以 batch=1 运行，hook 会自动 squeeze 掉 batch 维度。如需批量推理以提速，需自行修改 backend。
