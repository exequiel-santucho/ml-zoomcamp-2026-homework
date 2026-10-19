#!/usr/bin/env python3
"""ML Zoomcamp 2026 - Módulo 8: Redes neuronales y Deep Learning (Homework 8).

CNN mínima en PyTorch para clasificar pelo rizado (curly=0) vs lacio (straight=1).
Sigue al pie de la letra las reglas de reproducibilidad del enunciado 2026.

Preparación (una sola vez):
    uv venv --python 3.11 .venv
    uv pip install --python .venv -r requirements-cpu.txt --index-strategy unsafe-best-match
    curl -fL -o data.zip https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
    unzip -q data.zip

Ejecutar (CPU, ~10-30 min según la máquina):
    .venv/Scripts/python hw8_questions.py        # Windows
    .venv/bin/python hw8_questions.py            # Linux/Mac
"""
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "runs"
SEED = 42
EPOCHS = 10

# ---------------------------------------------------------------------------
# 1) Reproducibilidad: TODO antes de crear datasets, loaders, modelo u optimizer
# ---------------------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(2)  # igual que reference_train.py
device = torch.device("cpu")

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Transformación determinística: baseline de train y SIEMPRE para evaluación
evaluation_transform = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    NORMALIZE,
])

# Data augmentation: solo para la segunda fase de entrenamiento
train_transform_augmented = transforms.Compose([
    transforms.Resize((200, 200), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomRotation(50, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])


# ---------------------------------------------------------------------------
# 2) Modelo: Conv -> ReLU -> MaxPool -> Flatten -> Dense(64) -> ReLU -> Dense(1)
#    Devuelve LOGITS (sin sigmoid). 200x200 -conv3x3-> 198x198 -pool2-> 99x99
# ---------------------------------------------------------------------------
class HairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 99 * 99, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.hidden(x))
        return self.output(x)


def run_epoch(model, loader, criterion, optimizer=None):
    """Una pasada por el loader. Con optimizer -> entrena; sin optimizer -> evalúa."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            # promedios ponderados por cantidad de ejemplos, no por batches
            loss_sum += loss.item() * images.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train_phase(model, train_loader, eval_loader, criterion, optimizer, name):
    history = {"train_loss": [], "train_accuracy": [],
               "evaluation_loss": [], "evaluation_accuracy": []}
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer)
        el, ea = run_epoch(model, eval_loader, criterion)
        for k, v in zip(history, (tl, ta, el, ea)):
            history[k].append(v)
        print(f"[{name}] epoch {epoch:2d} train_loss={tl:.4f} train_acc={ta:.4f} "
              f"eval_loss={el:.4f} eval_acc={ea:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return history


# ---------------------------------------------------------------------------
baseline_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=evaluation_transform)
evaluation_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=evaluation_transform)
assert len(baseline_dataset) == 800 and len(evaluation_dataset) == 201
assert baseline_dataset.class_to_idx == {"curly": 0, "straight": 1}

evaluation_loader = DataLoader(evaluation_dataset, batch_size=20, shuffle=False, num_workers=0)
train_loader = DataLoader(baseline_dataset, batch_size=20, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED))

model = HairModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

print("Q1 -> loss: nn.BCEWithLogitsLoss() (1 logit + sigmoid+BCE numéricamente estable)")
n_params = sum(p.numel() for p in model.parameters())
print(f"Q2 -> parámetros entrenables: {n_params}")

baseline = train_phase(model, train_loader, evaluation_loader, criterion, optimizer, "baseline")

# Fase 2: mismo modelo y optimizer (¡no se recrean!), solo cambia el transform de train
augmented_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform_augmented)
augmented_loader = DataLoader(augmented_dataset, batch_size=20, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(SEED + 1))
augmented = train_phase(model, augmented_loader, evaluation_loader, criterion, optimizer, "augmented")

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "history_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
(OUT_DIR / "history_augmented.json").write_text(json.dumps(augmented, indent=2) + "\n", encoding="utf-8")
torch.save(model.state_dict(), OUT_DIR / "hair_model.pt")

print("\n" + "=" * 70)
print(f"Q1: nn.BCEWithLogitsLoss()")
print(f"Q2: {n_params}")
print(f"Q3: mediana train_accuracy (baseline)      = {statistics.median(baseline['train_accuracy']):.2f}")
print(f"Q4: std(ddof=0) train_loss (baseline)      = {np.std(baseline['train_loss'], ddof=0):.3f}")
print(f"Q5: media evaluation_loss (augmented)      = {statistics.mean(augmented['evaluation_loss']):.3f}")
print(f"Q6: media últimas 5 eval_accuracy (augm.)  = {statistics.mean(augmented['evaluation_accuracy'][-5:]):.2f}")
