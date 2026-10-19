# Módulo 8 - Deep Learning (Homework 8)

CNN en PyTorch (CPU, determinística) para pelo rizado vs lacio, con y sin data augmentation.

## Archivos

- **hw8_questions.py**: solución comentada paso a paso (reproducibilidad, transforms, modelo, loop de entrenamiento, Q1-Q6).
- **reference_train.py**: script oficial del curso (para contrastar).
- **requirements-cpu.txt**, **dataset_manifest.json**: versiones y checksum del dataset (release 2026).
- **runs/**: se genera al entrenar (`history_baseline.json`, `history_augmented.json`, `hair_model.pt`).
- **hw8_run.log** / **reference_run.log**: salidas de mis corridas.

## Preparación

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements-cpu.txt --index-strategy unsafe-best-match
curl -fL -o data.zip https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
sha256sum data.zip      # 9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67
unzip -q data.zip
```

## Ejecutar

```bash
.venv/Scripts/python hw8_questions.py      # Windows (~5-10 min en CPU)
.venv/bin/python hw8_questions.py          # Linux/Mac
# opcional, para comparar:
.venv/Scripts/python reference_train.py --data-dir data --output-dir runs/reference
```

## Salida esperada

```
Q1: nn.BCEWithLogitsLoss()
Q2: 20073473
Q3: mediana train_accuracy (baseline)      = 0.81
Q4: std(ddof=0) train_loss (baseline)      = 0.139
Q5: media evaluation_loss (augmented)      = 0.576
Q6: media últimas 5 eval_accuracy (augm.)  = 0.72
```
