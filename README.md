# GhostPixel-AI

GhostPixel-AI is a production-oriented repository scaffold for automated steganography detection on the ALASKA2 dataset. The stack uses Python 3.12+, PyTorch with Lightning, FastAPI for inference, Albumentations for forensic-safe preprocessing, and Pydantic v2 for configuration and response validation.

## Repository Layout

```text
GhostPixel-AI/
├── api/                 # FastAPI app, schemas, and inference dependencies
├── data/                # Dataset module, transforms, and raw dataset mount point
│   └── raw/             # Symlink target for ALASKA2 root
├── models/              # Residual layer, backbone model, Lightning wrapper
├── scripts/             # Training and evaluation entrypoints
├── settings.py          # Shared configuration via Pydantic settings
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

## ALASKA2 Dataset Placement

The repository expects the ALASKA2 folders to exist under `data/raw/` with this structure:

```text
data/raw/
├── Cover/
├── JMiPOD/
├── JUNIWARD/
├── Test/
└── UERD/
```

`Cover`, `JMiPOD`, `JUNIWARD`, and `UERD` are used for labeled 4-class training and validation. `Test` is treated as an unlabeled Kaggle inference split and is exposed through the same dataset/data module pipeline via `split="test"`.

If the dataset lives on an external drive, create a symlink into `data/raw` instead of copying the files:

```bash
mkdir -p data
ln -s /Volumes/ExternalDrive/ALASKA2 data/raw
```

If `data/raw` already exists as a normal directory, remove or rename it first, then recreate it as a symlink.

## Install

```bash
python3.12 -m venv .ghostenv
source .ghostenv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Training

```bash
python scripts/train.py
```

Useful environment variables:

```bash
export GHOSTPIXEL_DATA_ROOT=data/raw
export GHOSTPIXEL_BACKBONE_NAME=efficientnet_v2_s
export GHOSTPIXEL_BATCH_SIZE=16
export GHOSTPIXEL_IMAGE_SIZE=512
```

## Evaluation

```bash
export GHOSTPIXEL_CHECKPOINT_PATH=checkpoints/your-model.ckpt
python scripts/evaluate.py
```

## API Server

Run locally:

```bash
uvicorn api.main:app --reload
```

Inference request example:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.jpg"
```

The response schema contains:

- `class_name`
- `confidence_score`
- `explainability_map` (currently a placeholder for a future saliency or localization artifact)

## Docker

```bash
docker compose up --build
```

Mount checkpoints into `./checkpoints` and dataset access into `./data`.
