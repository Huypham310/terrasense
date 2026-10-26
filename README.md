# TerraSense

Land-use classification for Sentinel-2 satellite imagery. TerraSense trains convolutional neural networks on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset, tracks experiments with MLflow, exports the best model to ONNX, and serves predictions through a FastAPI service with a Next.js frontend.

> Work in progress.

## Repository layout

```
ml/    PyTorch training package (data pipeline, models, training, evaluation, export)
api/   FastAPI inference service backed by ONNX Runtime and PostgreSQL
web/   Next.js + TypeScript frontend
```

## Roadmap

- [ ] Data pipeline: download EuroSAT, stratified train/val/test split, augmentation
- [ ] Models: from-scratch CNN baseline and ResNet-18 transfer learning
- [ ] Experiment tracking with MLflow, evaluation report, ONNX export
- [ ] Inference API with prediction history
- [ ] Web frontend for uploading tiles and browsing results
- [ ] Tests, Docker Compose, and CI

## License

MIT
