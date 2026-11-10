# Which teacher targets help a small CNN on CIFAR-100?

In a fixed CIFAR-100 subset experiment, the student reached **38.53% mean validation accuracy** with each image's own teacher probabilities and **34.30%** when those probability vectors were shuffled within class. All three training seeds favored matched targets. The comparison asks whether individual images benefit from different soft labels, even within the same ground-truth class.

## Design

The Fall 2024 NYU Bayesian Machine Learning course project investigated kernel-based feature transfer. The AI-assisted October–November 2026 extension tested class-probability distillation. A frozen ImageNet ResNet-18 supplied features for a linear CIFAR-100 classifier. Five-fold stratified cross-fitting on 10,000 training images gave each image scores from a classifier trained on the other four folds. A compact CNN learned from hard labels and temperature-2 teacher probabilities with equally weighted cross-entropy and temperature-scaled KL losses.

Within each seed, all three groups shared the student initialization, batch order, hard labels, Adam learning rate (0.001), batch size (64) and 30-epoch budget. The shuffled condition used one fixed permutation within each class with no image mapped to itself. The class-mean condition assigned every image the average of the 100 training probability vectors in its true class.

## Epoch-30 validation accuracy

Each cell gives the number of correct predictions out of the same 2,000 validation images and its corresponding percentage.

| Training seed | Matched | Within-class shuffle | Class mean | Matched minus shuffled |
|---|---:|---:|---:|---:|
| 42 | 795 (39.75%) | 686 (34.30%) | 705 (35.25%) | +5.45 pp |
| 43 | 757 (37.85%) | 687 (34.35%) | 695 (34.75%) | +3.50 pp |
| 44 | 760 (38.00%) | 685 (34.25%) | 725 (36.25%) | +3.75 pp |
| Mean | **38.53%** | **34.30%** | 35.42% | **+4.23 pp** |

Under the same fixed-epoch validation protocol, the small CNN trained only on hard labels reached 30.80%, 29.60% and 30.40% on seeds 42, 43 and 44, respectively. Its mean was **30.27%**, giving matched distillation an **8.27 percentage point** advantage. [Seed 42 history](extended_30ep/evidence/extended-30ep-seed42/baseline_history.json) · [Seed 43 history](extended_30ep/evidence/extended-30ep-seed43/baseline_history.json) · [Seed 44 history](extended_30ep/evidence/extended-30ep-seed44/baseline_history.json).

Seed 42 was the initial screen. On the two subsequent seeds, the matched-minus-shuffled difference averaged **3.625 percentage points**. The three runs share the data split, out-of-fold teacher outputs and permutation. These figures describe exploratory validation performance on this subset.

## What the other controls showed

- With the true-class probability fixed to its class average, retaining each image's relative probabilities over the other 99 classes yielded 37.63% mean validation accuracy. Retaining only the image-specific true-class probability, with a class-average allocation across the other classes, yielded 36.25%.
- Retaining the teacher's five largest non-target probabilities and distributing the remaining mass in proportion to the class average yielded 36.30%. Permuting those five values among the same original teacher candidates yielded 35.92%. The 0.38-point mean difference changed sign across seeds.
- The October RBF feature-alignment comparison used validation-selected checkpoints and the 10,000-image **test** set. RBF transfer achieved 31.96% mean test accuracy versus 30.46% for the supervised baseline. The November probability-distillation table uses fixed-epoch **validation** accuracy. These experiments use different evaluation rules.
- The separate paper-scale ViT-S/BiT run stopped during baseline fine-tuning, before final baseline evaluation and AFT training.
