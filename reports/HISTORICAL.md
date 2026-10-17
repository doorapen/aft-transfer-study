# Historical AFT audit

Source: BML_project.ipynb, cell index 4. SHA-256: `9b79728a4a0d30e7497763021f60d07133e27487daa431c4a0111661beba6b1d`. Parsed 60 printed epoch records, six configurations, ten logged checkpoints each (epochs 2 through 20). These are training-set metrics, not validation or test. Logged loss is a sum of batch means and is retained as historical data, not compared with the corrected sample-weighted mean.

| Configuration | Final training accuracy |
|---|---:|
| Basic implementation | 44.62% |
| RBF kernel only | 61.54% |
| Bi-level optimization only | 53.27% |
| Data augmentation only | 35.41% |
| BatchNorm & Dropout only | 54.39% |
| All improvements | 43.20% |

The notebook uses an existing AFT idea. Its later multilayer branch compared negative requested indices with nonnegative loop indices, selecting no layers. Parent train() also switched the frozen teacher into training mode, allowing BatchNorm state drift. Later hand-entered plotting arrays are not experimental observations and are excluded.

The repaired implementation is a new maintenance version, not a numerical reproduction: teacher preprocessing, layer resolution, kernel normalization and loss scaling are explicit and differ from notebook variants. Initial maintenance used synthetic smoke tests. A subsequent real-data pilot is documented separately in [the pilot report](REAL_PILOT_2026-10-16.md); it does not validate the historical numbers.
