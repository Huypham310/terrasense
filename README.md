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

## Training

Two models are compared:

- `simple_cnn`: a small VGG-style CNN trained from scratch (3 conv blocks + global average pooling)
- `resnet18`: ImageNet-pretrained ResNet-18, fully fine-tuned

```bash
cd ml
python -m terrasense.train --model simple_cnn --epochs 15 --patience 4
python -m terrasense.train --model resnet18 --epochs 8 --lr 5e-4
python -m terrasense.evaluate --model resnet18
python -m terrasense.export --model resnet18
```

Training uses AdamW, cosine LR decay, label smoothing 0.1 and early stopping on validation accuracy. Every run is logged to MLflow (`mlflow.db` + `mlruns/`); browse them with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Results

Test split (2,700 images never seen during training or model selection), CPU latency measured on a single 64x64 image after warm-up:

| Model | Test accuracy | Macro F1 | Params | PyTorch latency | ONNX Runtime latency |
| --- | --- | --- | --- | --- | --- |
| simple_cnn | 96.48% | 0.964 | 289K | 1.74 ms | 0.79 ms |
| resnet18 | **98.19%** | **0.981** | 11.2M | 4.83 ms | 1.48 ms |

Transfer learning cuts the test error roughly in half. Most remaining mistakes are between the agricultural classes (PermanentCrop vs HerbaceousVegetation / AnnualCrop), which look very similar at 64x64.

![ResNet-18 confusion matrix](reports/resnet18/confusion_matrix.png)

The exported ONNX model matches PyTorch outputs within 1e-6 and runs about 3x faster on CPU, so the API only needs `onnxruntime` instead of the full PyTorch stack.

## API

FastAPI service in `api/` that serves the exported ONNX model with ONNX Runtime (no PyTorch needed at inference time) and keeps a prediction history in SQLite locally or PostgreSQL via `TERRASENSE_DATABASE_URL`.

```bash
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive docs at http://localhost:8000/docs.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Model + database status |
| GET | `/model` | Model name, classes, input size, test accuracy |
| POST | `/predict` | Classify one tile (multipart `file`), returns top-3 |
| POST | `/predict/batch` | Classify up to 16 tiles in one call |
| GET | `/predictions` | History, newest first (`limit`, `offset`, `label`) |
| GET | `/predictions/{id}` | One prediction |
| DELETE | `/predictions/{id}` | Delete a prediction |
| GET | `/stats` | Counts and average confidence per label |

Uploads are limited to PNG/JPEG/TIFF/WebP and 5 MB (`415` / `413` otherwise); unreadable images return `422`. Settings use the `TERRASENSE_` prefix, see `api/.env.example`.

## Roadmap

- [x] Data pipeline: download EuroSAT, stratified train/val/test split, augmentation
- [x] Models: from-scratch CNN baseline and ResNet-18 transfer learning
- [x] Experiment tracking with MLflow, evaluation report, ONNX export
- [x] Inference API with prediction history
- [ ] Web frontend for uploading tiles and browsing results
- [ ] Tests, Docker Compose, and CI

## License

MIT
