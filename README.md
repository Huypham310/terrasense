# TerraSense

Land-use classification for Sentinel-2 satellite imagery. TerraSense trains convolutional neural networks on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, tracks experiments with MLflow, exports the best model to ONNX, and serves predictions through a FastAPI service with a Next.js frontend.

## Repository layout

```
ml/    PyTorch training package (data pipeline, models, training, evaluation, export)
api/   FastAPI inference service backed by ONNX Runtime and PostgreSQL
web/   Next.js + TypeScript frontend
```

## Dataset

EuroSAT RGB contains 27,000 labeled 64x64 Sentinel-2 tiles across 10 land-use classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

| Split | Images |
| --- | --- |
| Train | 21,600 |
| Validation | 2,700 |
| Test | 2,700 |

Splits are stratified (seed 42), so every class keeps the same proportion in each split. Normalization statistics are computed on the training split only.

```bash
python -m venv .venv
.venv/Scripts/activate        # source .venv/bin/activate on macOS/Linux
pip install -e "ml[dev]"
cd ml && python -m terrasense.data.prepare
```

This downloads the dataset (~90 MB) to `data/raw/` and writes `data/processed/splits.json` and `stats.json`.

## Roadmap

- [x] Data pipeline: download EuroSAT, stratified train/val/test split, augmentation
- [ ] Models: from-scratch CNN baseline and ResNet-18 transfer learning
- [ ] Experiment tracking with MLflow, evaluation report, ONNX export
- [ ] Inference API with prediction history
- [ ] Web frontend for uploading tiles and browsing results
- [ ] Tests, Docker Compose, and CI

## License

MIT
