# CIFAR-100 真实数据配对试验 — 2026-10-16

本轮未观察到 AFT 的测试集收益。无迁移 baseline 为 **18.78%**，修复版 RBF AFT 为 **18.53%**，差值 **−0.25 个百分点**。这是一个种子、5 epoch 的小规模训练试验，不是完整数据训练或原论文复现。

| 方法 | 最佳 epoch | 验证准确率 | 测试正确数 / 10,000 | 测试准确率 | 测试交叉熵 |
|---|---:|---:|---:|---:|---:|
| 同一 CNN，无迁移 | 5 | 18.05% | 1,878 | 18.78% | 3.3920 |
| RBF AFT，beta=1 | 5 | 18.65% | 1,853 | 18.53% | 3.3940 |

AFT 单独答对386张，baseline单独答对411张。对测试样本进行2,000次配对 bootstrap，准确率差的95%百分位区间为[−0.80, +0.28]个百分点，跨过零。此区间仅反映这两个固定模型的测试样本不确定性，不包含训练种子差异，也不能证明两种方法等效。

## 锁定的实验条件

- CIFAR-100 官方训练集的分层子集：每类100张训练、20张验证，合计10,000/2,000；没有交集。官方完整测试集10,000张，仅在每组按验证集选定checkpoint后各评估一次。
- seed=42；5 epochs；batch=64；Adam，学习率0.001；无数据增强。两组学生完全相同的初始状态、拆分、批次顺序、优化器和训练步数。
- 三层小型CNN学生；ImageNet预训练ResNet-18教师；最后一个stage；RBF sigma=1。AFT beta=1，baseline beta=0。教师始终冻结并处于eval，学生使用CIFAR归一化，教师使用224像素和ImageNet归一化。
- 无增强允许预计算冻结教师特征，避免重复前向。缓存与在线特征的loss/logits/selector梯度一致性回归测试通过。
- 数据为公开 `uoft-cs/cifar100` Parquet镜像；数据来源URL及SHA-256、源码哈希、完整拆分索引和初始化状态哈希保存在 `real_pilot_manifest.json`。权重来自torchvision官方ResNet-18 IMAGENET1K_V1。
- Apple MPS，PyTorch 2.14.1，Python 3.12。启用确定性算法warn_only；跨设备、软件版本不保证逐位重复。

总运行约79.9秒（不含下载），包含数据读取、约31.7秒时完成的教师缓存、两组训练、验证和测试。`results.json`中的`training_seconds`实际包含每组验证和最终测试：baseline约23.2秒，AFT约24.5秒；不能作为纯训练速度比较，缓存成本也不可忽略。

## 结论与下一步

代码修复使训练/评估口径有效，不保证迁移提高准确率。对齐损失从0.0252降至0.0093，仅说明该辅助目标被优化，不代表分类泛化改善。两组验证准确率仍随训练上升，本轮不足以判断充分训练后的效果。旧日志61.54%是不同实现/协议下的训练准确率，不能与这里测试准确率相减来评价修复。

下一轮应事先固定更长训练协议，在验证集选择beta和sigma，与相同训练条件的baseline做多个种子的比较。由于本次已看过官方测试成绩，后续结果应披露这一探索过程，不能把反复查看测试集当作全新盲测。本轮结果不适合在简历中写“提升准确率”；更合适的贡献是建立无迁移对照、修复冻结与选层问题、实现可复核评估。

## 复现

```bash
pip install -r requirements-pilot.txt
python prepare_pilot_data.py
python run_pilot.py --device mps --output runs/new-pilot
python analyze_pilot.py --run runs/new-pilot --output reports/new-pilot-metrics.json
```

CPU可用`--device cpu`，但耗时和数值可能不同。首次运行会下载预训练权重；不需要付费API。本地`runs/real-pilot-2026-10-16/`保留权重和缓存，发布包仅包含小型结果文件、逐样本预测和运行清单，不含数据或权重。`real_pilot_metrics.json`由预测重算，验证结果不依赖手工录入表格。
