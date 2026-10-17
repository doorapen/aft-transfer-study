# Resume claims and evidence

Original Fall 2024 course project; the October 2026 repair and reproduction were AI-assisted. A pretrained ResNet-18 teacher and small CNN student are aligned through kernel similarity matrices. This is inspired by AFT, not a new method or an exact reproduction of the paper.

See [30-epoch report](EXTENDED_30EP_2026-10-16.md) and [summary](extended_30ep/summary.json). Fixed 10,000/2,000/10,000 train/validation/test split, three seeds and 30 epochs. Mean test accuracy: RBF transfer 31.9567%, matched no-transfer baseline 30.4600%. Checkpoints were selected on validation accuracy. The result is specific to this subset and protocol; AFT test cross-entropy was worse in all seeds. Earlier test inspection makes this exploratory, not a fresh blind test. The old 61.54% was training accuracy.
