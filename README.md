# cpath_conic

Code and experiment pipeline for the [CoNIC: Colon Nuclei Identification and Counting challenge](https://conic-challenge.grand-challenge.org/).

- **Interactive results dashboard:** https://cyrusmaher.github.io/conic/
- **Official segmentation and classification leaderboard:** https://conic-challenge.grand-challenge.org/evaluation/segmentation-and-classification-final-test/leaderboard/

## Recommended model

The final presentation uses one rare-class-trained HoVer-Net checkpoint:

1. Replace the initial CellViT++ baseline with HoVer-Net.
2. Show the model more training patches containing rare cell types.
3. Average six flipped and rotated views at inference.
4. Use standard HoVer-Net cell separation at its intended resolution.

On our locked 657-patch internal test split, this model reaches **mPQ+ 0.4614** and **count R² 0.8017**. These are internal-split results, not an official challenge submission; the dashboard labels evaluation provenance throughout.

## Repository layout

- `cpath_conic/`: reusable data, metrics, sampling, augmentation, TTA, and visualization code
- `scripts/`: training, inference, ablation, calibration, analysis, and dashboard-generation entry points
- `experiments/conic_matrix.json`: experiment registry and accumulated decisions
- `benchmarks/conic_published.json`: published benchmark context
- `tests/`: metric, pipeline, leakage-guard, visualization, and deployment tests

Datasets, model checkpoints, generated predictions, and dashboard image artifacts are intentionally excluded.

## External implementations

Several experiments integrate the upstream projects below. Clone them under `third_party/` when reproducing those paths:

```bash
git clone https://github.com/TissueImageAnalytics/CoNIC.git third_party/CoNIC
git clone https://github.com/TIO-IKIM/CellViT-plus-plus.git third_party/CellViT-plus-plus
git clone https://github.com/vqdang/hover_net.git third_party/hover_net
```

The principal Python dependencies include NumPy, pandas, PyTorch, OpenCV, Pillow, SciPy, scikit-image, scikit-learn, matplotlib, and imgaug. Upstream model-specific environment requirements also apply.

## Tests

From an environment containing the dependencies and upstream packages:

```bash
pytest tests/test_pipeline.py tests/test_prepare_github_pages.py -q
```

The publication build used 143 passing tests.
