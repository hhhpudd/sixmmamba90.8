# SixMamba 项目摘要

> **基于 Vision Mamba 与六边形方向感知融合的皮肤病变分割网络**
>
> **核心创新**：`SixMambaFuse` 内含 `HexMambaProcessorV3` — 沿六边形六个方向用沙漏掩码提取特征，经 Vision Mamba 处理后融合，实现对不规则病变区域的多方向上下文感知。

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目文件结构](#2-项目文件结构)
3. [数据预处理 (process.py)](#3-数据预处理-processpy)
4. [训练入口 — train_isic.py main() 完整流程](#4-训练入口--train_isicpy-main-完整流程)
5. [模型核心 — FAFuse_B 类完整解析](#5-模型核心--fafuse_b-类完整解析)
6. [融合模块 — FAFusion_block 类完整解析](#6-融合模块--fafusion_block-类完整解析)
7. [核心创新 — SixMambaFuse 类完整解析](#7-核心创新--sixmambafuse-类完整解析)
8. [核心创新 — HexMambaProcessorV3 类完整解析](#8-核心创新--hexmambaprocessorv3-类完整解析)
9. [辅助工具 — SixTransform 类完整解析](#9-辅助工具--sixtransform-类完整解析)
10. [Mamba 主干 — VisionMamba 类](#10-mamba-主干--visionmamba-类)
11. [解码器输出与损失计算](#11-解码器输出与损失计算)
12. [完整运行路径一览](#12-完整运行路径一览)
13. [评估与测试 (test_isic.py)](#13-评估与测试-test_isicpy)
14. [核心创新原理解析](#14-核心创新原理解析)
15. [项目演进史](#15-项目演进史)

---

## 1. 项目概述

**SixMamba** 面向 **ISIC 2018** 皮肤镜图像分割任务。整体架构继承自 TransFuse 的双编码器思路，但做了两项关键替换：

- **Transformer 编码器** → **Vision Mamba**（状态空间模型 SSM，线性复杂度 O(n) 替代 O(n²)）
- **传统融合模块中的轴向注意力** → **SixMambaFuse 内含 HexMambaProcessorV3**（六边形沙漏掩码 + Mamba 方向感知处理）

最终模型输出四个分割预测图（一个主输出 + 三个辅助输出），通过边界感知加权损失联合优化，在测试时取 Dice + IoU 作为模型保存依据。

---

## 2. 项目文件结构

```
sixmamba/
│
├── train_isic.py                    # ⭐ 训练入口（主脚本）
├── test_isic.py                     # 测试/评估入口
├── process.py                       # 数据预处理（ISIC2018 → .npy）
├── random_division.py               # 训练/验证集随机划分
│
├── PROJECT_SUMMARY.md               # 本文档
│
├── data/                            # 预处理后的数据
│   ├── data_train.npy               # 训练图像 [N, 352, 352, 3]
│   ├── mask_train.npy               # 训练掩码 [N, 352, 352]
│   ├── data_test.npy                # 测试图像
│   └── mask_test.npy                # 测试掩码
│
├── utils/
│   ├── dataloader.py                # DataLoader + 数据增强
│   └── utils.py                     # AvgMeter 平均损失追踪
│
├── pretrained/                      # 预训练权重
│   ├── resnet34-43635321.pth        # ResNet-34 ImageNet 预训练
│   └── vim_s_midclstok_ft_81p6acc.pth  # Vision Mamba 预训练
│
├── snapshots/                       # 训练检查点 .pth
│
├── logs/                            # TensorBoard 日志
│
└── lib/logs/sixmamba/lib/           # ⭐ 核心模型代码（Python 包路径）
    ├── FAFuse.py                    # ⭐ FAFuse_B, FAFusion_block, SixMambaFuse
    ├── sixmambaV.py                 # ⭐ HexMambaProcessorV3
    ├── sixtransform.py              # SixTransform（沙漏掩码生成与裁剪）
    ├── models_mamba.py              # VisionMamba（Mamba SSM 架构）
    ├── circle_transform.py          # CartToPolarTensor（极坐标变换，存档模块使用）
    ├── rope.py                      # RoPE 旋转位置编码
    ├── CoTAttention.py              # 上下文 Transformer 注意力（存档）
    ├── DeiT.py                      # DeiT 封装（存档）
    ├── vision_transformer.py        # ViT 基础实现（存档）
    └── ... 其他存档文件
```

> **注意**：`lib/logs/sixmamba/lib/` 实际就是活动 Python 包。`train_isic.py` 中 `from lib.FAFuse import FAFuse_B` 解析到此目录。
>
> 根目录下没有独立的 `lib/` 文件夹，Python 的 import 路径恰好能找到 `lib/logs/sixmamba/lib/`。

---

## 3. 数据预处理 (process.py)

```python
# 流程：
ISIC 2018 原始图片文件夹
  → 遍历所有 .jpg 图像
  → OpenCV 读取 → 转为 RGB
  → Resize 到 352×352（cv2.INTER_LINEAR）
  → 归一化到 [0,1]
  → 保存为 data_train.npy / data_test.npy
  → 掩码同理，保存为 mask_train.npy / mask_test.npy
```

`random_division.py` 从训练集中随机抽 520 张作为验证集。

---

## 4. 训练入口 — train_isic.py main() 完整流程

### 4.1 启动入口

```bash
# 命令行启动
python train_isic.py \
  --epoch 300 \
  --lr 1e-4 \
  --batchsize 16 \
  --train_save "FAFuse_R" \
  --radius1 7 --radius2 5 --radius3 3 \
  --width1 2 --width2 2 --width3 3
```

### 4.2 main() 函数逐行执行流程

```python
# train_isic.py 第 177 行 ── if __name__ == '__main__':

# ─── 第1步：解析参数 ───
opt = parser.parse_args()
radiuslist = [opt.radius1, opt.radius2, opt.radius3]  # [7, 5, 3]
widthlist = [opt.width1, opt.width2, opt.width3]      # [2, 2, 3]

# ─── 第2步：创建 TensorBoard 日志记录器 ───
writer = SummaryWriter('logs/R7_5_3_W2-2-3' + time + "/")

# ─── 第3步：创建模型 ⭐ ───
model = FAFuse_B(
    pretrained=True,          # 加载两个预训练权重
    ring_radius=[7,5,3],      # 传给 RAFusion_block（内部透传，当前版本实际未使用）
    ring_width=[2,2,3]        # 同上，存档兼容参数
).cuda()

# ─── 第4步：配置优化器 ───
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.5, 0.999)
)

# ─── 第5步：加载数据 ───
train_loader = get_loader("data/data_train.npy", "data/mask_train.npy", batchsize=16)
# 内部: SkinDataset → albumentations 增强 → ImageNet归一化 → DataLoader

# ─── 第6步：配置调度器 ───
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

# ─── 第7步：主训练循环 ───
best_loss = 0
for epoch in range(1, 301):
    best_loss = train(train_loader, model, optimizer, epoch, best_loss,
                      radiuslist, widthlist, scheduler)
    scheduler.step()
```

### 4.3 train() 函数 — 每个 epoch 的执行体

```python
def train(train_loader, model, optimizer, epoch, best_loss, radiuslist, widthlist, scheduler):

    model.train()  # 设置为训练模式

    for i, pack in enumerate(train_loader, start=1):
        # ── 取数据 ──
        images, gts = pack
        images = Variable(images).cuda()     # [16, 3, 224, 224]
        gts    = Variable(gts).cuda()        # [16, 1, 224, 224]

        # ⭐ ── 前向传播 ──
        lateral_map_4, lateral_map_3, lateral_map_2, resmap = model(images)
        #   map_x  (主输出)        — CNN + Transformer 联合路径 → [16, 1, 224, 224]
        #   map_1  (Transformer辅助) — 纯 Mamba 分支         → [16, 1, 224, 224]
        #   map_2  (联合辅助)       — 中间融合路径           → [16, 1, 224, 224]
        #   resmap (CNN辅助)        — 纯 CNN 分支            → [16, 1, 224, 224]

        # ── 损失计算 ──
        loss4 = structure_loss(lateral_map_4, gts)   # 主输出
        loss3 = structure_loss(lateral_map_3, gts)   # Transformer 辅助
        loss2 = structure_loss(lateral_map_2, gts)   # 联合辅助
        loss1 = structure_loss(resmap, gts)          # CNN 辅助

        loss = 0.4*loss1 + 0.3*loss4 + 0.2*loss2 + 0.1*loss3
        #   CNN 辅助权重最高(0.4)，主输出第二(0.3)

        # ── 反向传播 ──
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step(epoch + i/iters)   # 每个 step 更新学习率
        optimizer.zero_grad()

    # ── epoch 结束：测试 ──
    meanloss = test(model, opt.test_path, epoch)
    if meanloss > best_loss:  # meanloss = Dice + IoU
        best_loss = meanloss
        torch.save(model.state_dict(), save_path + 'FAFuse-%d.pth' % epoch)
        # 同时清理旧检查点
```

### 4.4 structure_loss() — 边界感知加权损失

```python
def structure_loss(pred, mask):
    # ⭐ 边界权重图: 平坦区域≈1, 边界区域≈6
    # 原理: avg_pool 会模糊边界, |原始-模糊| 在边界处大
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )

    # 加权二值交叉熵
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))

    # 加权 IoU 损失
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2,3))
    union = ((pred + mask) * weit).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()
```

---

## 5. 模型核心 — FAFuse_B 类完整解析

### 5.1 类定义与初始化

**位置**：`FAFuse.py:513-717`

```python
class FAFuse_B(nn.Module):
    def __init__(self, num_classes=1, drop_rate=0.2, normal_init=True,
                 pretrained=False, level=1,
                 ring_radius=[7,5,3], ring_width=[2,2,3]):
```

### 5.2 __init__() 初始化 — 每个子模块的创建过程

```
══════════════════════════════════════════════════════════════════════
FAFuse_B.__init__()  开始创建内部所有子模块
══════════════════════════════════════════════════════════════════════

┌── [模块1] CNN 编码器（ResNet-34）─────────────────────────────────┐
│                                                                     │
│  self.resnet = resnet()                            ← torchvision    │
│  if pretrained:                                                     │
│      self.resnet.load_state_dict(                                   │
│          torch.load('pretrained/resnet34-43635321.pth'))            │
│                                                                     │
│  self.resnet.fc = nn.Identity()       # 去掉全连接层                │
│  self.resnet.layer4 = nn.Identity()   # 只用 layer3 及之前          │
│                                                                     │
│  前向时提取 3 个尺度的特征:                                          │
│    x_u2 ← layer1   [64, 56, 56]     # 浅层细节                     │
│    x_u1 ← layer2   [128, 28, 28]    # 中层                          │
│    x_u  ← layer3   [256, 14, 14]    # 深层语义                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌── [模块2] Mamba 编码器（Vision Mamba）─────────────────────────────┐
│                                                                     │
│  self.transformer = deit(pretrained=pretrained)                     │
│  # deit 是别名, 实际导入的是:                                       │
│  # vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_   │
│  #   with_midclstok_div2                                            │
│  # 来自 models_mamba.py 的预配置 VisionMamba 变体                   │
│  #                                                                │
│  # 配置:                                                            │
│  #   img_size=224, patch_size=16, stride=16                         │
│  #   embed_dim=384, depth=12, bimamba_type=v2                       │
│  #   use_middle_cls_token=True                                      │
│  #   head = Linear(384, 196*768)    ← 生成密集特征图                │
│                                                                     │
│  前向时:                                                            │
│  输入: [B,3,224,224]                                                │
│    → Patch Embed: 16×16 stride=16 → 196=14×14 patches              │
│    → 12层双向Mamba → reshape                                     │
│    → x_b = [B, 768, 14, 14]     ← 密集特征图                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌── [模块3] Mamba 特征上采样通道压缩 ──────────────────────────────────┐
│                                                                     │
│  self.up1 = Up(in_ch1=768, out_ch=128)    # 768ch → 128ch          │
│  self.up2 = Up(128, 64)                    # 128ch → 64ch           │
│                                                                     │
│  前向:                                                              │
│    x_b  [768,14,14] → up1 → x_b_1 [128,28,28]  ← T1特征           │
│    x_b_1 [128,28,28] → up2 → x_b_2 [64,56,56]   ← T2特征           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌── [模块4] 多尺度融合模块（3个FAFusion_block + 2个注意力Up）★★★★   │
│                                                                     │
│  ◆ FAFusion_block 两个输入: 同尺度的CNN特征 + Mamba特征            │
│  ◆ 内部: 拼接 → 可变形卷积 → SixMambaFuse → HexMambaProcessorV3   │
│                                                                     │
│  --- 尺度1: 14×14 (最粗) ---                                       │
│  self.up_c = FAFusion_block(                                       │
│      ch_1=256, ch_2=768, ch_int=256, ch_out=256, imgsize=14,       │
│      radius1_base=7, radius2_base=5, radius3_base=3,               │
│      width1_base=2, width2_base=2, width3_base=3                   │
│  )                                                                  │
│  # 输入: x_u [256,~4,~4] + x_b [768,14,14]                        │
│  # 输出: [256,14,14]                                               │
│                                                                     │
│  --- 尺度2: 28×28（中等）---                                       │
│  self.up_c_1_1 = FAFusion_block(                                   │
│      ch_1=128, ch_2=128, ch_int=128, ch_out=128, imgsize=28,       │
│      ...radius/width参数...                                         │
│  )                                                                  │
│  # 输入: x_u_1 [128,7,7] + x_b_1 [128,28,28]                     │
│  # 输出: [128,28,28]                                               │
│                                                                     │
│  --- 注意力门控融合 up_c + up_c_1_1 ---                             │
│  self.up_c_1_2 = Up(in_ch1=256, out_ch=128, in_ch2=128, attn=True) │
│  # Attention_block: W_g(g) + W_x(x) → Sigmoid → gate × x          │
│  # 输出: [128,28,28]                                               │
│                                                                     │
│  --- 尺度3: 56×56（最细）---                                       │
│  self.up_c_2_1 = FAFusion_block(                                   │
│      ch_1=64, ch_2=64, ch_int=64, ch_out=64, imgsize=56,           │
│      ...radius/width参数...                                         │
│  )                                                                  │
│  # 输入: x_u_2 [64,14,14] + x_b_2 [64,56,56]                     │
│  # 输出: [64,56,56]                                                │
│                                                                     │
│  --- 注意力门控融合 up_c_1 + up_c_2_1 ---                           │
│  self.up_c_2_2 = Up(128, 64, 64, attn=True)                       │
│  # 输出: [64,56,56]                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌── [模块5] 解码器（四个分割输出头）─────────────────────────────────┐
│                                                                     │
│  ┌─ map_x 主输出 ─────────────────────────────────────────────────┐ │
│  │  self.final_x = Sequential(Conv(256,64,1), Conv(64,64,3),      │ │
│  │                            Conv(64,1,3))                       │ │
│  │  self.final_x1 = Conv(64,64,1)      # 中间层                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ map_1 (Transformer 辅助) ─────────────────────────────────────┐ │
│  │  self.final_1 = Sequential(Conv(64,64,3), Conv(64,1,3))        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ map_2 (联合辅助) ─────────────────────────────────────────────┐ │
│  │  同 map_1: self.final_2                                        │ │
│  │  输出 convv = Conv(64, 1, 1)    # 共享卷积头                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ resmap (CNN 辅助) ────────────────────────────────────────────┐ │
│  │  输出 convv = Conv(64, 1, 1)    # 共享卷积头                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌── [模块6] 其他辅助模块 ────────────────────────────────────────────┐
│                                                                     │
│  self.drop = nn.Dropout2d(drop_rate=0.2)                          │
│                                                                     │
│  self.resize_224_224 = Resize([224,224])                          │
│  self.C2P = CartToPolarTensor(radius=112, img_size=224)  # 存档    │
│  self.P2C = PolarToCartTensor(radius=112, img_size=224)  # 存档    │
│  self.rotate_transform = transforms.functional.rotate    # 存档    │
│  self.mamba_features_conv = odconv3x3(9,3)  # 存档(极坐标融合用)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 forward() 前向传播 — 完整数据流

```python
def forward(self, imgs, labels=None):
    """
    输入: imgs [B, 3, 224, 224]
    输出: (map_x, map_1, map_2, resmap)
          每个都是 [B, 1, 224, 224]
    """
```

```
══════════════════════════════════════════════════════════════════════
FAFuse_B.forward(imgs)  完整前向数据流
══════════════════════════════════════════════════════════════════════

输入图片: imgs [B, 3, 224, 224]
  │
  ├── ═══════ [A] Mamba 分支 ═══════
  │   │
  │   feature_Car = self.transformer(imgs)       ← VisionMamba 12层
  │   # models_mamba.py: 双向 Mamba SSM
  │   # 输入 [B,3,224] → PatchEmbed 16×16 → [B,196,384]
  │   # 12层 bimamba_v2 → 中间 CLS token
  │   # Head Linear(384, 196*768) → [B,196,768]
  │   │
  │   x_b = feature_Car.view(B, 196, 768)
  │   x_b = torch.transpose(x_b, 1, 2)           # [B, 768, 196]
  │   x_b = x_b.view(B, -1, 14, 14)              # [B, 768, 14, 14]
  │   │
  │   x_b = self.drop(x_b)                        # Dropout2d(0.2)
  │   │
  │   x_b_1 = self.up1(x_b)   # Up(768→128, ×2)   → [B, 128, 28, 28]
  │   x_b_1 = self.drop(x_b_1)                    # T1 特征
  │   │
  │   x_b_2 = self.up2(x_b_1) # Up(128→64, ×2)    → [B, 64, 56, 56]
  │   x_b_2 = self.drop(x_b_2)                    # T2 特征
  │
  ├── ═══════ [B] CNN 分支 ═══════
  │   │
  │   x_u = self.resnet.conv1(imgs)    # Conv7×7, /2   → [B, 64, 112, 112]
  │   x_u = self.resnet.bn1(x_u)
  │   x_u = self.resnet.relu(x_u)
  │
  │   x_u1 = x_u                                     # [B, 64, 112, 112]
  │   x_u  = self.resnet.maxpool(x_u)  # /2          → [B, 64, 56, 56]
  │
  │   x_u_2 = self.resnet.layer1(x_u)                # → [B, 64, 56, 56]  ← G2
  │   x_u_2 = self.drop(x_u_2)
  │
  │   x_u_1 = self.resnet.layer2(x_u_2)              # → [B, 128, 28, 28] ← G1
  │   x_u_1 = self.drop(x_u_1)
  │
  │   x_u = self.resnet.layer3(x_u_1)                # → [B, 256, 14, 14] ← G0
  │   x_u = self.drop(x_u)
  │
  ├── ═══════ [C] 多尺度融合路径 ═══════
  │   │
  │   ╔══════════════════════════════════════════════════╗
  │   ║ ★ 每个 FAFusion_block 内部:                     ║
  │   ║   拼接 → 可变形卷积 → SixMambaFuse             ║
  │   ║     → HexMambaProcessorV3                       ║
  │   ╚══════════════════════════════════════════════════╝
  │   │
  │   ┌──────────────────────────────────────────────────────┐
  │   │ 尺度1: 14×14 (最深)                                  │
  │   │ x_c = self.up_c(x_u, x_b)                           │
  │   │   x_u [B,256,~4,~4]  (实际是layer3输出, 经插值对齐) │
  │   │   x_b [B,768,14,14]                                  │
  │   │   输出: [B,256,14,14]                                │
  │   └──────────────────────────────────────────────────────┘
  │   │
  │   ┌──────────────────────────────────────────────────────┐
  │   │ 尺度2: 28×28                                         │
  │   │ x_c_1_1 = self.up_c_1_1(x_u_1, x_b_1)              │
  │   │   x_u_1 [B,128,7,7]                                  │
  │   │   x_b_1 [B,128,28,28]                                │
  │   │   输出: [B,128,28,28]                                │
  │   │                                                      │
  │   │ x_c_1 = self.up_c_1_2(x_c, x_c_1_1)   ← 注意力门控  │
  │   │   x_c [B,256,14,14] → Up×2 → [B,256,28,28]          │
  │   │   + x_c_1_1 [B,128,28,28] 通过 Attention_gate       │
  │   │   拼接 → DoubleConv → [B,128,28,28]                 │
  │   └──────────────────────────────────────────────────────┘
  │   │
  │   ┌──────────────────────────────────────────────────────┐
  │   │ 尺度3: 56×56 (最浅)                                  │
  │   │ x_c_2_1 = self.up_c_2_1(x_u_2, x_b_2)              │
  │   │   x_u_2 [B,64,14,14]                                 │
  │   │   x_b_2 [B,64,56,56]                                 │
  │   │   输出: [B,64,56,56]                                 │
  │   │                                                      │
  │   │ x_c_2 = self.up_c_2_2(x_c_1, x_c_2_1)  ← 注意力门控 │
  │   │   x_c_1 [B,128,28,28] → Up×2 → [B,128,56,56]        │
  │   │   + x_c_2_1 [B,64,56,56] 通过 Attention_gate        │
  │   │   拼接 → DoubleConv → [B,64,56,56]                  │
  │   └──────────────────────────────────────────────────────┘
  │
  ├── ═══════ [D] 解码器 — 四个分割输出 ═══════
  │   │
  │   ┌─ map_x（主输出）──────────────────────────────────────┐
  │   │ x_c_2 [64,56,56]                                      │
  │   │ → final_x1 Conv(64,64,1)                              │
  │   │ → + x_u_2 [64,56,56]    ← 残差连接浅层CNN细节         │
  │   │ → ReLU → interpolate ×2 → [64,112,112]               │
  │   │ → + x_u1 [64,112,112]   ← 残差连接更浅层细节          │
  │   │ → ReLU → interpolate ×2 → [64,224,224]               │
  │   │ → convv Conv(64,1,1) → [1,224,224]                   │
  │   └──────────────────────────────────────────────────────┘
  │   │
  │   ┌─ map_1（Transformer 辅助输出）────────────────────────┐
  │   │ x_b_2 [64,56,56]                                      │
  │   │ → final_1 Conv→Conv → [64,224,224]                   │
  │   │ → interpolate ×4 → [1,224,224]                       │
  │   └──────────────────────────────────────────────────────┘
  │   │
  │   ┌─ map_2（联合辅助输出）────────────────────────────────┐
  │   │ x_c_2 + x_u_2 → ReLU → interpolate ×2 → [64,112,112] │
  │   │ → + x_u1 → ReLU → interpolate ×2 → [64,224,224]       │
  │   │ → convv → [1,224,224]                                 │
  │   └──────────────────────────────────────────────────────┘
  │   │
  │   ┌─ resmap（CNN 辅助输出）───────────────────────────────┐
  │   │ x_u1 [64,112,112]                                     │
  │   │ → convv Conv(64,1,1) → [1,112,112]                   │
  │   │ → interpolate ×2 → [1,224,224]                       │
  │   └──────────────────────────────────────────────────────┘
  │
  输出: (map_x, map_1, map_2, resmap)
        每个 [B, 1, 224, 224]
```

---

## 6. 融合模块 — FAFusion_block 类完整解析

**位置**：`FAFuse.py:429-463`

### 6.1 类定义与初始化

```python
class FAFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, ch_int, ch_out,
                 drop_rate=0., imgsize=224, level=1, **kwargs):
```

### 6.2 __init__() 创建的子模块

```python
# 参数示例: ch_1=256, ch_2=768, ch_int=256, imgsize=14
# 注: 实际调用时传入了 radius1_base/width1_base 等 kwargs,
#    但这些参数被 FAFusion_block 忽略了（内部不消费）, 只透传

# ── 可变形卷积的偏移量生成器 ──
self.offset_conv = nn.Conv2d(
    in_channels,          # ch_1 + ch_2 = 256+768 = 1024
    2 * 3 * 3,           # 每个点 2个方向 × 3×3 核 = 18个偏移量
    kernel_size=3, padding=1, bias=True
)
# 权重初始化为0（开始时无变形）

# ── 可变形卷积 ──
self.deform_conv = nn.Conv2d(
    in_channels, in_channels, kernel_size=3, padding=1, bias=False
)

# ⭐ ── 核心创新模块 ──
self.sixmamba = SixMambaFuse(
    ch1=ch_1,
    ch2=ch_2,
    ch_int=ch_int,
    imgsize=imgsize
)
# 注意: FAFusion_block 只管"组合", SixMambaFuse 才是核心
```

### 6.3 forward() 调用链

```python
def forward(self, g, t):
    """
    输入:
      g — CNN 特征 (如 [B,256,~4,~4])
      t — Mamba 特征 (如 [B,768,14,14])
    输出: 融合后特征 [B, ch_int, imgsize, imgsize]
    """

    # ═══ 第1步：拼接 ═══
    bp = torch.cat([g, t], dim=1)
    # bp [B, 256+768, 14, 14] = [B, 1024, 14, 14]
    # 注: 如果 g 和 t 空间尺寸不同, 在外部已通过 interpolate 对齐

    # ═══ 第2步：可变形卷积对齐 ═══
    offsets = self.offset_conv(bp)
    # offsets [B, 18, 14, 14]  每个位置 3×3 核的 (dx,dy)

    bp_deformed = ops.deform_conv2d(
        bp, offsets, self.deform_conv.weight,
        padding=(1, 1)
    )
    # 可变形卷积让 CNN 特征自适应地在 Mamba 特征图上做空间变形对齐

    # ═══ 第3步：SixMambaFuse ⭐ ═══
    fuse = self.sixmamba(bp_deformed)
    # └─→ SixMambaFuse.forward() 内部:
    #       深度可分离卷积 → HexMambaProcessorV3 → 残差 → ReLU6
    #       └─→ HexMambaProcessorV3.forward() 内部:
    #             3个角度(0°,60°,120°)的沙漏掩码 + Mamba处理 + max融合
    #
    # 输出: [B, ch_int, imgsize, imgsize]
    #     如 [B, 256, 14, 14]

    return fuse
```

### 6.4 三个FAFusion_block实例化与调用的对应关系

| 实例变量 | CNN输入 | Mamba输入 | ch_int | 输出尺寸 | 在forward中被调用 |
|---|---|---|---|---|---|
| `self.up_c` | `x_u` [256,~4,~4] | `x_b` [768,14,14] | 256 | 14×14 | `up_c(x_u, x_b)` |
| `self.up_c_1_1` | `x_u_1` [128,7,7] | `x_b_1` [128,28,28] | 128 | 28×28 | `up_c_1_1(x_u_1, x_b_1)` |
| `self.up_c_2_1` | `x_u_2` [64,14,14] | `x_b_2` [64,56,56] | 64 | 56×56 | `up_c_2_1(x_u_2, x_b_2)` |

---

## 7. 核心创新 — SixMambaFuse 类完整解析

**位置**：`FAFuse.py:401-426`

### 7.1 类定义与初始化

```python
class SixMambaFuse(nn.Module):
    def __init__(self, ch1=128, ch2=128, drop_path=0., ch_int=1, imgsize=14):
        super().__init__()
        self.ch_int = ch_int

        # ── 深度可分离卷积降维 ──
        self.dwconv = ConvBN(
            ch1+ch2,          # 输入通道 = CNN通道 + Mamba通道
            ch_int,           # 输出通道 = 中间通道数
            3, 1,             # 3×3卷积, stride=1
            (3-1)//2,         # padding=1
            groups=1,
            with_bn=True
        )

        # ⭐ ── 核心创新：六边形Mamba处理器 ──
        self.processor = HexMambaProcessorV3(
            square_size=imgsize,   # 如 14 (token网格尺寸)
            ch=ch_int              # 如 256 (输入通道)
        )

        self.act = nn.ReLU6()
```

### 7.2 forward() 调用链

```python
def forward(self, x):
    """
    输入: x [B, ch1+ch2, H, W]
         如 [B, 1024, 14, 14]（来自 FAFusion_block 的可变形卷积输出）
    输出: [B, ch_int, H, W]
         如 [B, 256, 14, 14]
    """
    # ═══ 第1步：深度可分离卷积降维 ═══
    x = self.dwconv(x)
    # ConvBN: Conv2d(ch1+ch2→ch_int, 3×3) + BN
    # 输出 [B, ch_int, 14, 14]  例如 [B, 256, 14, 14]

    # ═══ 第2步：保存残差 ═══
    identity = x

    # ═══ 第3步：核心 — HexMambaProcessorV3 ⭐ ═══
    x = self.processor(x)
    # └─→ HexMambaProcessorV3.forward():
    #  对 batch 中每张图、每个角度(0/60/120)：
    #    沙漏掩码 → 裁剪 → resize方形 → VisionMamba → 还原 → 旋转回正
    #  三个角度 max 融合 → center pad 回原尺寸
    # 输出 [B, ch_int, 14, 14]

    # ═══ 第4步：残差连接 ═══
    x = x + identity
    # 经典残差设计, 保证梯度流通

    # ═══ 第5步：激活 ═══
    x = self.act(x)   # ReLU6

    return x
```

### 7.3 架构设计意图

```
SixMambaFuse 内部结构:
┌──────────────────────────────────────┐
│       输入 x [B, C_in, H, W]         │
│              │                       │
│      ┌───────▼───────┐               │
│      │  dwconv(3×3)  │  ← 降维+特征变换│
│      └───────┬───────┘               │
│              │                       │
│         ┌────▼────┐                  │
│         │ identity│  ← 保存残差      │
│         └────┬────┘                  │
│              │                       │
│      ┌───────▼───────────┐           │
│      │ HexMambaProcessor │  ← ⭐    │
│      │   V3              │    六边形  │
│      │   0°→Mamba        │    方向    │
│      │  60°→Mamba        │    感知    │
│      │ 120°→Mamba        │           │
│      │   max融合         │           │
│      └───────┬───────────┘           │
│              │                       │
│         ┌────▼────┐                  │
│         │ +identity│  ← 残差连接    │
│         └────┬────┘                  │
│              │                       │
│      ┌───────▼───────┐               │
│      │   ReLU6激活   │               │
│      └───────┬───────┘               │
│              │                       │
│       输出 [B, C_out, H, W]          │
└──────────────────────────────────────┘
```

---

## 8. 核心创新 — HexMambaProcessorV3 类完整解析

**位置**：`sixmambaV.py:11-152`

### 8.1 类定义与初始化

```python
class HexMambaProcessorV3(nn.Module):
    def __init__(self, square_size=14, ch=1024):
```

初始化时创建 **一个新的 VisionMamba 实例**，注意这是项目中**第二个** VisionMamba 实例：

```python
self.vim = VisionMamba(
    img_size=self.square_size,   # 14 (注意: 不是224, 而是token网格尺寸)
    patch_size=7,                # patch大小
    embed_dim=384,
    depth=8,                     # 8层 Mamba (比主干的12层浅)
    rms_norm=True,
    channels=ch,                 # 输入通道数自适应 (例如256)
    stride=7                     # stride=patch_size (非重叠patch)
).cuda()
```

### 8.2 forward() 完整执行过程

这是整个项目最核心的函数。下面逐行跟踪：

```python
def forward(self, x):
    """
    输入: x [B, C, H, W]
         如 [16, 256, 14, 14] （来自 SixMambaFuse 的 dwconv 输出）
    输出: [B, C, orig_h, orig_w]
         如 [16, 256, 14, 14]
    """

    B, C, H, W = x.shape           # B=16, C=256, H=14, W=14
    angles = [0, 60, 120]          # 三个旋转角度
    outputs = []                   # 收集 batch 中每张图的结果
```

```
══════════════════════════════════════════════════════════════════════
对 batch 中每张图 b:
══════════════════════════════════════════════════════════════════════

for b in range(B):                     # 遍历 16 张图
    img = x[b]                          # [C=256, H=14, W=14]
    restored_imgs = []                 # 存储3个角度的结果

    for angle in [0, 60, 120]:          # ← 每个角度循环
        │
        │  ═══════ 步骤①: 生成沙漏掩码 ═══════
        │  mask = SixTransform.Mask.create(H, W, angle=angle, device=img.device)
        │  # → 创建沙漏形状掩码 [14, 14]
        │  #   angle=0:   沙漏垂直方向
        │  #   angle=60:  沙漏旋转60°
        │  #   angle=120: 沙漏旋转120°
        │  #
        │  #   Three masks together cover all six directions:
        │  #     0°:   ↕ + ↗↙
        │  #     60°:  ↗ + ↘↖
        │  #     120°: ↘ + ↖↙
        │  #   → 六边形六个方向全覆盖
        │
        │  ═══════ 步骤②: 应用掩码 ═══════
        │  masked = img * mask.unsqueeze(0)   ← 广播到 [C,14,14]
        │  # 只有沙漏区域保留, 其余为0
        │
        │  ═══════ 步骤③: 裁剪沙漏区域 ═══════
        │  cropped = SixTransform.Crop.crop(masked, mask)
        │  # 内部:
        │  # 1) 找沙漏的4个角点 (top/right/bottom/left)
        │  # 2) 计算旋转角度 (使长边水平)
        │  # 3) warp_affine 旋转整张图 + 扩展画布
        │  # 4) 找有效区域边界, 紧密裁剪
        │  # 5) 若宽>高则转置使短边在下
        │  # 输出: [C, crop_h, crop_w]   矩形区域
        │
        │  ═══════ 步骤④: resize 到正方形 ═══════
        │  square = self.resize_tensor(cropped, [14, 14])
        │  # 双线性插值, 统一到 14×14
        │
        │  ═══════ 步骤⑤: Vision Mamba ⭐ ═══════
        │  colored_square = square.unsqueeze(0)     # [1, C, 14, 14]
        │  # 注意: 当前代码中实际用了一行注释掉的调用:
        │  #   colored_square = self.vim(square.unsqueeze(0))
        │  # 实际上这行被注释了, 直接用了 square 本身。
        │  # 这意味着当前版本中 Mamba 处理是占位的。
        │  # 输出: [C, 14, 14]
        │
        │  ═══════ 步骤⑥: resize 回裁剪矩形尺寸 ═══════
        │  compressed = self.resize_tensor(colored_square, [crop_h, crop_w])
        │  # 从 14×14 恢复到裁剪前的矩形尺寸
        │
        │  ═══════ 步骤⑦: 旋转回原方向 ═══════
        │  if angle == 0:
        │      compressed = compressed.flip(1).flip(2)   # 特殊处理0°方向
        │  rotated_back = self.rotate_tensor(compressed, angle)
        │  # 用仿射变换旋转 -angle 角度, 回到原始方向
        │  # 输出: [C, H_rot, W_rot]
        │
        │  ═══════ 步骤⑧: 紧裁 (去除黑边) ═══════
        │  valid_mask = rotated_back.sum(0) > 0    ← 找到非零区域
        │  if valid_mask.any():
        │      ys, xs = valid_mask.nonzero(as_tuple=True)
        │      rmin, rmax = ys.min(), ys.max()+1
        │      cmin, cmax = xs.min(), xs.max()+1
        │      rotated_back = rotated_back[:, rmin:rmax, cmin:cmax]
        │  # 去除旋转引入的黑色填充区域
        │
        │  restored_imgs.append(rotated_back)   ← 保存单个角度的结果
        │
        └──── 结束角度循环 ────

    ═══════ 合并三个角度 ═══════
    # ① 找到三个角度输出中的最大宽高
    max_h = max(im.shape[1] for im in restored_imgs)
    max_w = max(im.shape[2] for im in restored_imgs)

    # ② 将所有角度 pad 到相同尺寸
    merged_imgs = []
    for im in restored_imgs:
        C, H_im, W_im = im.shape
        canvas = torch.zeros(C, max_h, max_w, device=im.device)
        y0 = (max_h - H_im) // 2
        x0 = (max_w - W_im) // 2
        canvas[:, y0:y0+H_im, x0:x0+W_im] = im
        merged_imgs.append(canvas)

    # ③ ⭐ 沿角度维度最大池化融合
    merged = torch.stack(merged_imgs, 0).max(0)[0]
    # stack → [3, C, max_h, max_w]
    # max(0) → 每个位置取三个角度中的最大值

    ═══════ center pad 回原始尺寸 ═══════
    canvas = torch.zeros(C, self.orig_h, self.orig_w, device=merged.device)
    # orig_h = orig_w = 14
    y0 = (self.orig_h - H_m) // 2
    x0 = (self.orig_w - W_m) // 2
    canvas[:, y0:y0+H_m, x0:x0+W_m] = merged

    outputs.append(canvas)   # 保存单张图的结果

══════════════════════════════════════════════════════════════════════
batch 循环结束
══════════════════════════════════════════════════════════════════════

return torch.stack(outputs, 0)
# [B, C, orig_h=14, orig_w=14]  ← 输出尺寸与输入一致
```

### 8.3 辅助方法

```python
@staticmethod
def resize_tensor(x, size):
    """双线性插值resize, 支持 [C,H,W] 或 [1,C,H,W]"""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x_resized = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return x_resized.squeeze(0)

@staticmethod
def rotate_tensor(x, angle_deg):
    """仿射旋转, 支持 [C,H,W]"""
    theta = torch.tensor([[cos,-sin,0], [sin,cos,0]], device=x.device).unsqueeze(0)
    grid = F.affine_grid(theta, x_batch.size(), align_corners=False)
    rotated = F.grid_sample(x_batch, grid, mode='bilinear', padding_mode='zeros')
    return rotated.squeeze(0)
```

### 8.4 处理流程可视化

```
原始特征 [B,C,14,14]
      │
      ├── 角度0° ── 沙漏掩码 ── 裁剪 ── resize ── Mamba ── 还原 ── 旋转回正 ── 紧裁 ──┐
      ├── 角度60° ─ 沙漏掩码 ── 裁剪 ── resize ── Mamba ── 还原 ── 旋转回正 ── 紧裁 ──┼── max融合 ── center pad ── [B,C,14,14]
      └── 角度120°─ 沙漏掩码 ── 裁剪 ── resize ── Mamba ── 还原 ── 旋转回正 ── 紧裁 ──┘

三个角度覆盖六方向:
  0°:    △        60°:   △         120°:    △
        / \             / \                / \
       /   \           /   \              /   \
      /     \         /     \            /     \
     /       \       /       \          /       \
     \       /       \       /          \       /
      \     /         \     /            \     /
       \   /           \   /              \   /
        \ /             \ /                \ /
         ▼               ▼                  ▼

  覆盖方向:          覆盖方向:           覆盖方向:
  ↑ ↓ ↗ ↙           ↗ ↘ ↖              ↘ ↖ ↙

  合并 = 六方向全覆盖:
         ↖↑↗
         ←●→
         ↙↓↘
```

---

## 9. 辅助工具 — SixTransform 类完整解析

**位置**：`sixtransform.py:1-187`

### 9.1 SixTransform.Mask — 沙漏掩码生成

```python
class SixTransform.Mask:

    @staticmethod
    def _build_mask(H, W, device):
        """在 (H,W) 网格上构建角度0°的沙漏掩码"""

        # 两个共顶点的等边三角形
        # 三角形边长 a = W/2
        a = W / 2
        h_tri = (√3 / 2) * a           # 等边三角形高
        center_x, center_y = W//2, H//2

        # 上三角形 (△)
        pts_up = [
            [center_x, center_y],               # 顶点：中心
            [center_x + a/2, center_y - h_tri],  # 右上
            [center_x - a/2, center_y - h_tri],  # 左上
        ]

        # 下三角形 (倒▼)
        pts_down = [
            [center_x, center_y],               # 顶点：中心（共享）
            [center_x - a/2, center_y + h_tri],  # 左下
            [center_x + a/2, center_y + h_tri],  # 右下
        ]

        # 用重心坐标法判断每个像素是否在三角形内
        def point_in_triangle(pt, tri):
            A, B, C = tri
            v0 = C - A
            v1 = B - A
            v2 = pt - A
            dot00 = (v0 * v0).sum(-1)
            dot01 = (v0 * v1).sum(-1)
            dot02 = (v0 * v2).sum(-1)
            dot11 = (v1 * v1).sum(-1)
            dot12 = (v1 * v2).sum(-1)
            invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6)
            u = (dot11 * dot02 - dot01 * dot12) * invDenom
            v = (dot00 * dot12 - dot01 * dot02) * invDenom
            return (u >= 0) & (v >= 0) & (u + v <= 1)

        mask = mask_up | mask_down   # 两个三角形的并集
        return mask.float()          # [H, W]

    @staticmethod
    def create(H, W, angle=0, device='cpu'):
        """生成指定角度的沙漏掩码（带缓存）"""
        key = (H, W, angle)
        if key not in _mask_cache:
            base_mask = _build_mask(H, W, 'cpu')       # 先生成0°的
            if angle != 0:
                mask = kornia.rotate(base_mask, angle)  # 用kornia旋转
            else:
                mask = base_mask
            _mask_cache[key] = mask                     # 缓存
        return _mask_cache[key].to(device)
```

### 9.2 SixTransform.Crop — 沙漏区域裁剪

```python
class SixTransform.Crop:

    @staticmethod
    def get_corners(mask):
        """找到沙漏掩码的四个角点"""
        ys, xs = (mask > 0).nonzero(as_tuple=True)

        # 最上面的点 (top)
        top_idx = ys.argmin()
        top = [xs[top_idx], ys[top_idx]]

        # 最下面的点 (bottom)
        bottom_idx = ys.argmax()
        bottom = [xs[bottom_idx], ys[bottom_idx]]

        # 中间行最左和最右的点 (left, right)
        mid_y = (top[1] + bottom[1]) / 2
        mid_mask = (ys > top[1]) & (ys < bottom[1])
        mid_xs = xs[mid_mask]
        mid_ys = ys[mid_mask]
        left_idx = mid_xs.argmin()
        right_idx = mid_xs.argmax()
        left = [mid_xs[left_idx], mid_ys[left_idx]]
        right = [mid_xs[right_idx], mid_ys[right_idx]]

        return [top, right, bottom, left]   # 4个点

    @staticmethod
    def crop(img_tensor, mask):
        """裁剪旋转后的沙漏区域"""
        # 获取四个角点
        corners = get_corners(mask)
        tl, tr, br, bl = corners

        # 计算旋转角度（使沙漏长边水平）
        vec_diag = br - tl
        angle = -atan2(vec_diag[1], vec_diag[0])   # → 角度

        # 旋转图像和掩码（同时自动扩展画布）
        img_rot = _rotate_expand(img_tensor, angle)
        mask_rot = _rotate_expand(mask, angle, mode='nearest')

        # 找到有效区域的边界框
        ys, xs = (mask_rot > 0).nonzero(as_tuple=True)
        rmin, rmax = ys.min(), ys.max()+1
        cmin, cmax = xs.min(), xs.max()+1

        # 紧密裁剪
        cropped = img_rot[:, rmin:rmax, cmin:cmax]

        # 如果宽>高，转置使短边朝下
        C, h, w = cropped.shape
        if w > h:
            cropped = cropped.permute(0, 2, 1).flip(2)

        return cropped
```

### 9.3 沙漏裁剪效果示意图

```
原始特征图(14×14)    沙漏掩码(angle=0°)    掩码后          裁剪结果
    ┌──────┐              ┌──────┐          ┌──────┐        ┌────┐
    │      │              │  /\  │          │  /\  │        │ /\  │
    │      │              │ /  \ │          │ /  \ │        │/  \ │
    │      │      ×       │/    \│    =     │/    \│   →    │\  / │
    │      │              │\    /│          │\    /│        │ \/  │
    │      │              │ \  / │          │ \  / │        └────┘
    │      │              │  \/  │          │  \/  │
    └──────┘              └──────┘          └──────┘

   angle=60°:               ↓ 旋转60°后裁剪
   angle=120°:              ↓ 旋转120°后裁剪
```

---

## 10. Mamba 主干 — VisionMamba 类

**位置**：`models_mamba.py:693-1014`

### 10.1 项目中使用的是两个不同配置的 VisionMamba 实例

```
══════════════════════════════════════════════════════════════════
实例1: FAFuse_B 中的主干编码器
══════════════════════════════════════════════════════════════════
变体名称: vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
  img_size=224, patch_size=16, stride=16
  embed_dim=384, depth=12
  bimamba_type=v2          ← 双向Mamba
  use_middle_cls_token=True ← 中间位置CLS token
  if_abs_pos_embed=True    ← 绝对位置编码
  final_pool_type='none'
  head = Linear(384, 196*768)  ← 生成密集特征图
  预训练: vim_s_midclstok_ft_81p6acc.pth

forward:
  [B,3,224,224] → PatchEmbed 16×16 stride=16 → [B,196,384]
  → 12层双向Mamba v2 → head → [B,196,768]
  → reshape → [B,768,14,14]

══════════════════════════════════════════════════════════════════
实例2: HexMambaProcessorV3 内部的处理用 Mamba
══════════════════════════════════════════════════════════════════
  img_size=14, patch_size=7, stride=7
  embed_dim=384, depth=8
  channels=C (自适应, 如256)  ← 支持任意输入通道
  rms_norm=True

forward:
  [1,C,14,14] → PatchEmbed 7×7 stride=7 → [1,4,384] (14/7=2, 2×2=4)
  → 8层Mamba → [1,4,384] → 还原
  (注: 当前代码中这个调用被注释, 实际直接pass through)
```

### 10.2 Mamba SSM 核心结构

```
每个 Mamba 层:
输入 x
  │
  ├──→ norm(x) → SSM (状态空间模型, 选择性扫描算法)
  │     │
  │     ├── 扩展维度: Linear(x, 2×embed_dim)
  │     ├── 1D卷积 → SiLU激活
  │     ├── 选择性SSM (A, B, C, Δ 参数由输入动态预测)
  │     ├── 乘回门控信号
  │     └── Linear(embed_dim, embed_dim)
  │
  └──→ + 残差连接 → 输出
```

**双向实现 (bimamba_v2)**：每层内做两次 SSM — 一次正向序列，一次反向序列，结果拼接。

---

## 11. 解码器输出与损失计算

### 11.1 四个输出的各自角色

| 输出变量 | 名称 | 来源路径 | 损失权重 | 作用 |
|---|---|---|---|---|
| `map_x` | 主输出 | 联合路径 (CNN+Mamba融合) | 0.3 | 最终推理用的主预测 |
| `map_1` | Transformer辅助 | 纯 Mamba 分支上采样 | 0.1 | 辅助训练, 防止Mamba分支退化 |
| `map_2` | 联合辅助 | 中间融合路径 | 0.2 | 辅助训练, 提供中层监督 |
| `resmap` | CNN辅助 | 纯 CNN 分支上采样 | **0.4** | 权重最高, 提供强CNN梯度监督 |

### 11.2 解码细节

```
map_x 构建过程:
x_c_2 [64,56,56]
  → Conv(64→64,1) + BN + ReLU
  → + x_u_2 [64,56,56]         (CNN浅层细节跳跃连接)
  → ReLU → interpolate ×2
  → + x_u1 [64,112,112]        (CNN最浅层细节跳跃连接)
  → ReLU → interpolate ×2
  → Conv(64→1,1) → [1,224,224]  ← 最终分割图
```

---

## 12. 完整运行路径一览

### 12.1 一站式调用链

```
train_isic.py
  └── main()
        │
        │  [模型创建]
        ├── FAFuse_B(pretrained=True, ring_radius=[7,5,3], ring_width=[2,2,3])
        │     │
        │     ├── ResNet34 (CNN编码器)
        │     │     └── 加载 pretrained/resnet34-43635321.pth
        │     │
        │     ├── VisionMamba (Mamba编码器, 实例1)
        │     │     └── 加载 pretrained/vim_s_midclstok_ft_81p6acc.pth
        │     │
        │     ├── Up×2 (通道压缩 768→128→64)
        │     │
        │     ├── [FAFusion_block] ×3 ⭐
        │     │     │
        │     │     └── offset_conv + deform_conv (可变形卷积对齐)
        │     │           │
        │     │           └── SixMambaFuse ⭐
        │     │                 │
        │     │                 ├── dwconv (深度可分离卷积降维)
        │     │                 │
        │     │                 └── HexMambaProcessorV3 ⭐
        │     │                       │
        │     │                       ├── 角度0°: Mask→Crop→Mamba→Rotate
        │     │                       ├── 角度60°: Mask→Crop→Mamba→Rotate
        │     │                       ├── 角度120°: Mask→Crop→Mamba→Rotate
        │     │                       ├── max融合三个角度
        │     │                       │
        │     │                       └── VisionMamba (实例2, 8层)
        │     │
        │     └── Decoder (4个输出头)
        │
        │  [训练循环]
        └── for epoch in 1..300:
              │
              └── train()
                    │
                    ├── model(images) → (map_x, map_1, map_2, resmap)
                    │     ↑
                    │     └── FAFuse_B.forward()
                    │           ├── Mamba编码 → x_b, x_b_1, x_b_2
                    │           ├── CNN编码 → x_u, x_u_1, x_u_2
                    │           ├── 3×FAFusion_block → 3尺度融合特征
                    │           └── 解码器 → 4个分割图
                    │
                    ├── structure_loss × 4 (边界感知加权损失)
                    ├── 加权求和 → 反向传播
                    └── 每epoch末测试 → Dice+IoU → 创新高则保存
```

### 12.2 类的嵌套关系（缩进表示包含关系）

```
FAFuse_B                             ← 顶部模型
│
├── ResNet                           ← CNN 编码器
│
├── VisionMamba (12层)               ← Mamba 编码器 (实例1)
│
├── Up (768→128)                     ← 通道压缩
├── Up (128→64)                      ← 通道压缩
│
├── FAFusion_block (up_c)            ← 14×14尺度融合
│   └── SixMambaFuse
│       └── HexMambaProcessorV3
│           └── VisionMamba (8层)    ← 实例2
│
├── FAFusion_block (up_c_1_1)        ← 28×28尺度融合
│   └── SixMambaFuse
│       └── HexMambaProcessorV3
│           └── VisionMamba (8层)    ← 实例3
│
├── FAFusion_block (up_c_2_1)        ← 56×56尺度融合
│   └── SixMambaFuse
│       └── HexMambaProcessorV3
│           └── VisionMamba (8层)    ← 实例4
│
├── Up (注意力门控, up_c_1_2)        ← 注意力融合14→28
├── Up (注意力门控, up_c_2_2)        ← 注意力融合28→56
│
└── Decoder (4个输出头)              ← 分割输出
```

---

## 13. 评估与测试 (test_isic.py)

**位置**：`test_isic.py:46-115`

```python
if __name__ == '__main__':
    # 加载模型
    model = FAFuse_B().cuda()
    model.load_state_dict(torch.load(opt.ckpt_path))  # 如 FAFuse-33.pth
    model.eval()

    # 加载测试数据
    test_loader = test_dataset("data/data_test.npy", "data/mask_test.npy")

    # 逐张图评估
    for i in range(test_loader.size):
        image, gt = test_loader.load_data()    # 取一张图+掩码
        image = image.cuda()

        with torch.no_grad():
            _, _, res, _ = model(image)        # 只用主输出

        # 二值化
        res = res.sigmoid().cpu().numpy().squeeze()
        res = 1*(res > 0.5)

        # 计算指标
        dice  = mean_dice_np(gt, res)
        iou   = mean_iou_np(gt, res)
        recall = recall_score(gt, res)
        precision = precision_score(gt, res)
        acc   = np.sum(res == gt) / (res.shape[0]*res.shape[1])

    # 输出平均指标
    print('Dice: {:.4f}, IoU: {:.4f}, Acc: {:.4f}, '
          'Recall: {:.4f}, Precision: {:.4f}')
```

**评估指标计算**（纯 numpy 实现）：

```python
def mean_dice_np(y_true, y_pred):
    intersection = |y_pred * y_true|   # 逐元素乘
    mask_sum = |y_true| + |y_pred|
    return 2 * intersection / mask_sum

def mean_iou_np(y_true, y_pred):
    intersection = |y_pred * y_true|
    union = |y_true| + |y_pred| - intersection
    return intersection / union
```

---

## 14. 核心创新原理解析

### 14.1 为什么用六边形沙漏掩码？

```
传统方形卷积/注意力:
  ┌──────┐     ┌──────┐     ┌──────┐
  │ ■■■■ │     │ ■■■■ │     │ ■■■■ │
  │ ■■■■ │     │ ■■■■ │     │ ■■■■ │   ← 所有方向权重相同
  │ ■■■■ │     │ ■■■■ │     │ ■■■■ │
  └──────┘     └──────┘     └──────┘
  感受野: 各向同性方形, 无法沿病变长轴定向

六边形沙漏掩码:
  0°:    △         60°:     △        120°:    △
        / \               / \               / \
       /   \             /   \             /   \
      /沙漏 \           /沙漏 \           /沙漏 \
     /       \         /       \         /       \
     \       /         \       /         \       /
      \沙漏 /           \沙漏 /           \沙漏 /
       \   /             \   /             \   /
        \ /               \ /               \ /
         ▼                 ▼                 ▼
  覆盖:↕↗↙          覆盖:↗↘↖         覆盖:↙↖↘
  ─────────────────────────────────────────────
  合并后六方向全覆盖 → 各向异性感知, 适合不规则病变
```

| 对比维度 | 传统方形方案 | SixMamba 六边形方案 |
|---|---|---|
| 感受野形状 | 各向同性方形 | **各向异性沙漏形** |
| 方向覆盖 | 所有方向均等 | **六个特定方向聚焦** |
| 对细长病变 | 边界信息易稀释 | **沿长轴方向可聚焦** |
| 计算复杂度 | Transformer O(n²) | **Mamba SSM O(n)** |
| 融合方式 | 自注意力加权和 | **沙漏裁剪+Mamba+max融合** |

### 14.2 为什么是三个角度而不是六个？

- 每个角度的沙漏是 **两个共顶点的对立三角形**
- 角度0° 覆盖垂直方向 (↑↓) + 两个斜向 (↗↙)
- 角度60° 覆盖另两个斜向 (↘↖) + 角度0°已覆盖的 ↗
- 角度120° 覆盖剩余方向 (↙↖↘)
- **三个角度恰好覆盖六边形全部六个方向，不冗余**

### 14.3 max 融合 vs 其他融合方式

```python
# 三个角度 → stack → max(0) 融合
merged = torch.stack([result_0, result_60, result_120], 0).max(0)[0]
```

- **max 融合**：取每个位置上三个角度中的最大响应 → 对病变最显著的方向被保留
- 替代方案（未采用）：平均（稀释方向特异性）、加权和（需学习权重）、注意力（计算复杂）

### 14.4 可变形卷积 + SixMambaFuse 的配合

```
FAFusion_block 内部:
  ┌───────────────────────────────────────────┐
  │  CNN特征 (g) ──┐                          │
  │                ├── 拼接 ── 可变形卷积 ── SixMambaFuse │
  │  Mamba特征 (t) ─┘         ↑                          │
  │                     CNN/Mamba特征                    │
  │                    空间自适应对齐                     │
  └───────────────────────────────────────────────────────┘

  可变形卷积: 学习空间偏移量, 让CNN特征在Mamba特徵图上"对齐"
  SixMambaFuse: 对齐后, 用六边形Mamba提取方向感知的融合特征
```

---

## 15. 项目演进史

项目存档文件记录了融合模块的完整演变过程：

```
版本1: FAFuse原版.py
  └─ 融合: 4个 AxialAttention (高/宽/主对角/副对角)
  └─ Transformer: DeiT Base
  └─ 思想: 四轴方向拆分注意力

       ↓

版本2: FAFuseCA.py
  └─ 融合: RingAttention (极坐标环形注意力)
  └─ + CoTAttention (上下文Transformer)
  └─ 思想: 极坐标下的环形特征提取

       ↓

版本3: FAFuse#Old.py
  └─ 融合: RingAttention + MambaTransformer
  └─ 首次引入 Mamba
  └─ 思想: 用SSM替代Transformer降低复杂度

       ↓

版本4: FAFuse.py (当前) ✅
  └─ 融合: 可变形卷积(DCN) + SixMambaFuse + HexMambaProcessorV3
  └─ 编码器: ResNet34 + VisionMamba (12层)
  └─ 处理器: 3角度沙漏掩码 → Mamba → max池化
  └─ 思想: 六边形方向感知 + Mamba线性复杂度
```

---

> **文档生成日期**：2026-06-26
>
> **核心文件路径**：
> - 主模型: `lib/logs/sixmamba/lib/FAFuse.py`（约1075行）
> - 六边形Mamba处理器: `lib/logs/sixmamba/lib/sixmambaV.py`（约152行）
> - 沙漏掩码工具: `lib/logs/sixmamba/lib/sixtransform.py`（约187行）
> - Mamba主干: `lib/logs/sixmamba/lib/models_mamba.py`（约1014行）
> - 训练入口: `train_isic.py`（约231行）
> - 测试入口: `test_isic.py`（约116行）
