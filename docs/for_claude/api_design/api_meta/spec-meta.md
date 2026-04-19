# VneuroTK · Vision 模块架构设计

---

## 1. 总体架构

```
VneuroTK.vision
│
├── extractor/
│   ├── VisionExtractor          # 统一入口，唯一对外暴露的调用接口
│   ├── backend/
│   │   ├── BaseBackend          # 抽象基类，定义 Backend 契约
│   │   ├── TimmBackend
│   │   ├── TransformersBackend
│   │   └── ThingsVisionBackend
│   ├── selector/
│   │   ├── LayerSelector        # 抽象基类
│   │   ├── BlockLevelSelector   # 默认：每个 block 输出一层
│   │   ├── AllLeafSelector      # 所有叶子层
│   │   └── CustomSelector       # 用户指定层名列表
│   └── policy/
│       └── EmbeddingPolicy      # 声明 final_embedding 从哪里取
│
├── representation/
│   └── VisualRepresentation     # 数据容器，不含业务逻辑
│
└── registry/
    └── ModelRegistry            # 模型名 → (source, policy, selector) 映射表
```

设计核心原则：
- VisionExtractor 只知道图片进、VisualRepresentation 出，不知道底层是哪个库
- Backend 只负责「怎么跑这个模型、怎么挂 hook」，不知道神经数据是什么
- VisualRepresentation 只记录、不决策，所有形状/语义问题留给上层

---

## 2. 各模块职责与实现逻辑

---

### 2.1 VisualRepresentation（数据容器）

**职责**：忠实保存所有层激活及其 metadata，不做任何 pooling、reshape 或语义推断。

**核心字段**：

```python
@dataclass
class LayerMeta:
    name: str                  # 原始层名，如 "blocks.5"
    normalized_name: str       # 归一化层名，如 "block_5"
    module_type: str           # 如 "Block", "Linear", "Conv2d"
    shape: tuple               # 激活形状，如 (1, 197, 768)
    shape_type: str            # 'spatial'(B,C,H,W) | 'sequential'(B,T,D) | 'vector'(B,D)
    depth: int                 # 在模型中的嵌套深度
    is_final: bool             # 是否是 final_embedding 来源层


@dataclass
class ModelMeta:
    model_name: str            # 如 "vit_base_patch16_224"
    source: str                # 'timm' | 'transformers' | 'thingsvision'
    architecture: str          # 'ViT' | 'ResNet' | 'Swin' | ...
    learning_paradigm: str     # 'supervised' | 'contrastive' | 'selfsupervised'
    encoder_type: str          # 'single' | 'dual' | 'with_projector'
    embedding_policy: str      # 'cls_token' | 'mean_pool' | 'pre_head' | 'projection_output'


class VisualRepresentation:
    activations: OrderedDict[str, Tensor]   # normalized_name → 激活 tensor，保留原始形状
    layer_metas: dict[str, LayerMeta]       # normalized_name → LayerMeta
    model_meta: ModelMeta
    final_embedding: Tensor                  # 按 EmbeddingPolicy 取出，shape 视模型而定
    input_shape: tuple
```

**设计说明**：
- `activations` 的 key 统一用 `normalized_name`（如 `block_0` 而不是 `encoder.layer.0`），
  屏蔽不同库的命名差异，但 `layer_meta.name` 里保留原始名，可随时追溯
- `final_embedding` 单独暴露是为了方便编解码模块直接取，
  但 `layer_metas` 里的 `is_final=True` 那一层的 `activations` 里也有完整值
- 不提供任何 `.pool()`、`.flatten()` 方法，形状转换是编解码模块的事

---

### 2.2 BaseBackend（适配层抽象）

**职责**：隔离各个视觉库的差异，统一成三个操作：加载模型、预处理图片、带 hook 前向。

**需要实现的契约**：

```python
class BaseBackend(ABC):

    @abstractmethod
    def load(self, model_name: str, pretrained: bool) -> None:
        """加载模型到 self.model，初始化预处理器"""

    @abstractmethod
    def preprocess(self, image: Image | Tensor) -> dict:
        """图片 → 可直接传入 model 的 inputs dict"""

    @abstractmethod
    def forward(self, inputs: dict) -> Any:
        """带 hook 的前向，hook 结果写入 self._activations"""

    @abstractmethod
    def enumerate_layers(self) -> list[LayerInfo]:
        """枚举所有可提取层的信息"""

    @abstractmethod
    def get_model_meta(self) -> ModelMeta:
        """返回该模型的 ModelMeta，包括 learning_paradigm 等"""

    # 公共实现（基类提供）
    def register_hooks(self, layer_names: list[str]) -> None: ...
    def remove_hooks(self) -> None: ...
    def normalize_layer_name(self, raw_name: str) -> str: ...
```

**各 Backend 关键实现差异**：

`TimmBackend`
- `preprocess`：读 `model.default_cfg` 自动构建 transform，不硬编码 ImageNet 均值
- `get_model_meta`：从模型类名推断 architecture，从 pretrained tag 推断 learning_paradigm
  （如 `vit_base_patch16_224.dino` 判定为 selfsupervised）
- hook 目标：`named_modules()` 枚举，BlockLevelSelector 默认取 `blocks.*`

`TransformersBackend`
- `forward`：调用时传 `output_hidden_states=True`，把 `hidden_states` tuple 直接
  作为 block 级激活，不依赖 hook（更稳定）；对非标准层补充 hook
- `get_model_meta`：从 `config.model_type` 推断 architecture

`ThingsVisionBackend`
- 单张/小批量：内部走 hook，不调 ThingsVision 的 DataLoader
- 大批量（BatchExtractor）：委托给 ThingsVision 的 `extract_features()`，
  结果格式化后填入 VisualRepresentation
- 这个 Backend 主要价值是复用 ThingsVision 已有的模型列表和权重管理

---

### 2.3 LayerSelector（层选择策略）

**职责**：给定一个模型的层列表，决定提取哪些层。

```python
class BlockLevelSelector(LayerSelector):
    """
    默认策略：每个主要 block 取一层输出。
    ViT → blocks.0 ~ blocks.11
    ResNet → layer1 ~ layer4
    逻辑：取 depth <= 2 且 module_type 在预设的 block 类型列表里的层
    """

class AllLeafSelector(LayerSelector):
    """
    取所有叶子层（无子模块），适合细粒度 probing 实验
    """

class CustomSelector(LayerSelector):
    """
    用户直接传层名列表，支持原始名或 normalized_name
    """
    def __init__(self, layer_names: list[str]): ...
```

---

### 2.4 EmbeddingPolicy（最终表征策略）

**职责**：声明 `final_embedding` 从哪一层、用什么方式取。

```python
class EmbeddingPolicy(Enum):
    CLS_TOKEN        = "cls_token"         # ViT: 取最后一层 [:, 0, :]
    MEAN_POOL        = "mean_pool"         # 对所有 token mean，某些 ViT
    PRE_HEAD         = "pre_head"          # 分类模型: head 之前那层
    PROJECTION_OUT   = "projection_out"    # CLIP/SimCLR: 投影层输出
    BACKBONE_OUT     = "backbone_out"      # 对比学习: 不含 projection head
    CUSTOM           = "custom"            # 用户指定层名
```

每个 Backend 在 `get_model_meta()` 里声明默认 policy，用户可以在创建 VisionExtractor 时覆盖。

---

### 2.5 ModelRegistry（模型注册表）

**职责**：维护已知模型的配置，让调用者不需要知道模型来自哪个库。

```python
# 内置条目示例
REGISTRY = {
    "vit-b-16-imagenet":  ModelConfig(source="timm",
                                       model_id="vit_base_patch16_224",
                                       policy=EmbeddingPolicy.CLS_TOKEN,
                                       paradigm="supervised"),

    "vit-b-16-dino":      ModelConfig(source="timm",
                                       model_id="vit_base_patch16_224.dino",
                                       policy=EmbeddingPolicy.CLS_TOKEN,
                                       paradigm="selfsupervised"),

    "clip-vit-b-32":      ModelConfig(source="transformers",
                                       model_id="openai/clip-vit-base-patch32",
                                       policy=EmbeddingPolicy.PROJECTION_OUT,
                                       paradigm="contrastive",
                                       encoder_type="dual"),

    "resnet50":           ModelConfig(source="timm",
                                       model_id="resnet50",
                                       policy=EmbeddingPolicy.PRE_HEAD,
                                       paradigm="supervised"),
}

# 支持用户注册自定义模型
ModelRegistry.register("my-model", model_config)
```

---

### 2.6 VisionExtractor（统一入口）

**职责**：组装上面所有模块，对外暴露唯一接口。

**实现逻辑**：

```
初始化时：
  1. 查 Registry 或直接接收 model/backend 参数
  2. 实例化对应 Backend，调用 backend.load()
  3. 实例化 LayerSelector（默认 BlockLevelSelector）
  4. 确定 EmbeddingPolicy

extract() 调用时：
  1. backend.preprocess(image) → inputs
  2. selector.select(backend.enumerate_layers()) → target_layer_names
  3. backend.register_hooks(target_layer_names)
  4. backend.forward(inputs) → raw_output
  5. backend.remove_hooks()
  6. 从 _activations 构建 OrderedDict，key 做 normalize
  7. 按 EmbeddingPolicy 从 activations 里取 final_embedding
  8. 打包成 VisualRepresentation 返回
```

---

## 3. 接口示例

---

### 最简调用

```python
from vneuroTK.vision import VisionExtractor

# 通过 Registry 名称创建，不需要关心底层库
extractor = VisionExtractor("vit-b-16-dino")

image = Image.open("stimulus.jpg")
rep = extractor.extract(image)
```

---

### 访问 VisualRepresentation

```python
# 查看所有层
for name, meta in rep.layer_metas.items():
    print(f"{name:20s}  shape={meta.shape}  type={meta.shape_type}")

# block_0              shape=(1, 197, 768)  type=sequential
# block_1              shape=(1, 197, 768)  type=sequential
# ...
# block_11             shape=(1, 197, 768)  type=sequential

# 取某一层激活
act = rep.activations["block_6"]           # Tensor (1, 197, 768)

# 取最终表征（按 Policy 已经确定好的那一层）
emb = rep.final_embedding                  # Tensor (1, 768)，CLS token

# 查看这个表征是怎么来的
print(rep.model_meta.embedding_policy)     # "cls_token"
print(rep.model_meta.learning_paradigm)    # "selfsupervised"

# 找到 final_embedding 对应的完整激活（含所有 token）
final_layer = next(m for m in rep.layer_metas.values() if m.is_final)
full_act = rep.activations[final_layer.normalized_name]  # (1, 197, 768)
```

---

### 覆盖默认策略

```python
from vneuroTK.vision import VisionExtractor, EmbeddingPolicy, CustomSelector

# 覆盖 embedding policy
extractor = VisionExtractor(
    "clip-vit-b-32",
    embedding_policy=EmbeddingPolicy.BACKBONE_OUT,   # 不要投影层，要 backbone 输出
)

# 自定义提取层
extractor = VisionExtractor(
    "resnet50",
    selector=CustomSelector(["layer1", "layer2", "layer3", "layer4", "global_pool"]),
)

rep = extractor.extract(image)
print(rep)
# VisualRepresentation
#   model      : resnet50  (supervised / single)
#   policy     : pre_head
#   layers     : 5
#   layer1     : (1, 256, 56, 56)  spatial
#   layer2     : (1, 512, 28, 28)  spatial
#   layer3     : (1, 1024, 14, 14) spatial
#   layer4     : (1, 2048, 7, 7)   spatial
#   global_pool: (1, 2048)         vector   ← final_embedding 来源
```

---

### 直接传入模型（不走 Registry）

```python
import timm
from vneuroTK.vision import VisionExtractor
from vneuroTK.vision.backend import TimmBackend
from vneuroTK.vision.policy import EmbeddingPolicy

model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

extractor = VisionExtractor.from_model(
    model=model,
    backend=TimmBackend,
    embedding_policy=EmbeddingPolicy.PRE_HEAD,
    learning_paradigm="supervised",
)

rep = extractor.extract(image)
```

---

### 注册自定义模型

```python
from vneuroTK.vision.registry import ModelRegistry, ModelConfig
from vneuroTK.vision.policy import EmbeddingPolicy

ModelRegistry.register(
    name="my-custom-vit",
    config=ModelConfig(
        source="timm",
        model_id="vit_large_patch14_dinov2",
        policy=EmbeddingPolicy.CLS_TOKEN,
        paradigm="selfsupervised",
        encoder_type="single",
    )
)

extractor = VisionExtractor("my-custom-vit")
```

---

### 在编解码模块中使用（上游视角）

```python
# 编解码模块只依赖 VisualRepresentation，不关心模型细节
def encode(rep: VisualRepresentation, target_roi: str) -> Tensor:

    # 根据实验需要自己决定用哪一层、怎么 pool
    if rep.model_meta.architecture == "ViT":
        # 取中间某层，对 patch token 做 mean（跳过 CLS）
        act = rep.activations["block_6"]      # (1, 197, 768)
        feat = act[:, 1:, :].mean(dim=1)      # (1, 768)

    elif rep.model_meta.architecture in ("ResNet", "ConvNeXt"):
        # 取空间特征，做 adaptive pool
        act = rep.activations["block_3"]      # (1, 1024, 14, 14)
        feat = act.mean(dim=[-2, -1])         # (1, 1024)

    # 送入线性编码器
    return linear_encoder[target_roi](feat)
```

---

## 4. 扩展性说明

**加新模型库**：继承 `BaseBackend`，实现 5 个抽象方法，在 Registry 里加条目，其余不动。

**加新 Policy**：在 `EmbeddingPolicy` 枚举里加一项，在 `VisionExtractor._apply_policy()` 
里加对应分支，Backend 层不需要改动。

**加新 Selector**：继承 `LayerSelector`，实现 `select(layer_infos) -> list[str]`，一个方法。

**BatchExtractor**（大规模数据集）：复用同一个 Backend 和 Selector，
只是把单张 `extract()` 换成 DataLoader 循环，输出从单个 VisualRepresentation 
变成 `list[VisualRepresentation]` 或按层堆叠的 tensor，视下游需要。
