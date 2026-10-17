# Kernel Feature Transfer Study

A CIFAR-100 course experiment inspired by Adaptive Feature Transfer (AFT), with a corrected training harness. This is **not an exact reproduction of the AFT paper** and does not introduce AFT as a new method.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
python train.py --config configs/rbf_multilayer.json --smoke --output runs/smoke
```

Validated locally with Python 3.12 on macOS CPU. The smoke run uses 16 synthetic training images, 8 validation images, 8 test images and a randomly initialized teacher. It verifies execution and gradients, not classification quality. Hosted checks: [GitHub Actions](https://github.com/doorapen/aft-transfer-study/actions/workflows/tests.yml).

## Extended matched experiment (2026-10-16)

Three seeds (42/43/44), 30 epochs each, fixed 10k/2k/10k split. Mean test accuracy: baseline 30.46%, RBF AFT 31.96%. See [full results and limits](reports/EXTENDED_30EP_2026-10-16.md). This small-subset exploratory experiment is not the full training-set benchmark.

Recompute the published summary without training:
```bash
python summarize_extended.py --runs-dir reports/extended_30ep/evidence
```
Plot with `pip install -r requirements-plots.txt` then `python plot_extended.py`.

## Real-data pilot completed

A matched 5-epoch, seed-42 CIFAR-100 pilot used 10,000 training / 2,000 validation / 10,000 test images. Baseline test accuracy: **18.78%**; RBF AFT: **18.53%** (−0.25 percentage points). No improvement was demonstrated. See [protocol, limitations and reproduction](reports/REAL_PILOT_2026-10-16.md). This is separate from the full protocol below.

## Full CIFAR-100 experiment

```bash
python train.py --config configs/baseline.json --download --output runs/baseline-seed42
python train.py --config configs/linear.json --download --output runs/linear-seed42
python train.py --config configs/rbf.json --download --output runs/rbf-seed42
```

These commands download CIFAR-100 and, for the current harness, ResNet-18 weights, and may take substantial time. They have **not** been executed in this maintenance pass. Change device/seed in a copied configuration for GPU/multiple seeds. Do not tune configurations on the test results; choose them on validation and reserve test comparison for the final locked protocol. The baseline/linear/RBF configs share the seed, training protocol and selected layer. A separate rbf_multilayer.json explores multiple teacher layers; do not attribute its changes to the kernel alone.

The official training set is split into 45,000 training and 5,000 validation examples with stored indices. Augmentation, when enabled, applies only to training. Each run chooses its checkpoint using validation accuracy, then evaluates the official 10,000-image test set once. Report multiple seeds before claiming reliable improvement.

## Corrected implementation

- Frozen ResNet-18 remains in evaluation mode even when the containing model enters training mode; BatchNorm statistics cannot drift.
- Student uses CIFAR-100 normalization; teacher receives the same image/augmentation resized to 224 and ImageNet normalization.
- Negative layer indices are resolved against four explicit ResNet stages; empty, repeated or invalid selections raise errors.
- Learnable sigmoid feature weights and linear/RBF Gram matrices align teacher/student representations without requiring equal feature dimensions. The loss is the mean squared Gram difference, averaged across selected layers.
- Positive RBF bandwidth, finite losses, frozen teacher parameters and nonzero student/selector gradients are covered by tests.
- Metrics are sample-weighted means. Training/validation/test metrics are separate. Source configuration, split indices, best checkpoint and metrics are saved per run.

The corrected harness intentionally changes preprocessing, normalization and alignment-loss scaling from exploratory notebook variants. Historical results must not be compared as if these were identical runs. Legacy bi-level and B-tuning experiments remain historical material; they are not advertised as validated implementations here.

## Historical results and limits

See [historical audit](reports/HISTORICAL.md) and `reports/historical_training.csv`. The notebook's cell 4 recorded six configurations for 20 epochs, with training accuracies from 35.41% to 61.54%. The 61.54% figure is **training accuracy**, not test accuracy. No verified held-out result was recovered. Later handwritten plotting arrays are excluded from the evidence table.

The author reports the course project was largely independently completed in Fall 2024. Original method and upstream code belong to their authors: [Transferring Knowledge from Large Foundation Models to Small Downstream Models](https://arxiv.org/abs/2406.07337). The 2026 implementation repair was prepared with coding-assistant help. It is separate from historical course results. No paper PDF, downloaded upstream code bundle, model weights or dataset is included in this release. No open-source license has been assigned to this prepared release.


Resume statements and contribution limits: [evidence note](reports/RESUME_CLAIMS.md).

Public experiment manifests normalize local machine paths to repository-relative paths. Source hashes and measured results are unchanged; `reports/extended_30ep/evidence_sha256.json` checks the sanitized public copies. Original manifests remain in the local run archive.
