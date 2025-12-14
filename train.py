"""
FireVision: UAV-based smoke and fire detection.
Training and inference pipeline using EfficientNet with K-Fold CV.
"""

import os
import random
import copy
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


class CFG:
    """Training configuration."""
    data_dir = "/kaggle/input/fire-vision"
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    output_sub = "submission.csv"

    model_name = "tf_efficientnet_b0_ns"
    img_size = 256
    num_classes = 3
    epochs = 5
    batch_size = 32
    num_workers = 4
    lr = 1e-4
    weight_decay = 1e-5
    n_splits = 5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms(img_size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_valid_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FireDataset(Dataset):
    """Dataset for fire/smoke classification."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transforms=None, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["rel_path"])
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        if self.is_test:
            return image, row["ID"]
        return image, int(row["label_idx"])


def build_train_df(train_dir: str) -> pd.DataFrame:
    """Build training dataframe from folder structure (train/class_name/image.jpg)."""
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in {train_dir}")

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    rows = []

    for cls in sorted(class_dirs):
        cls_path = os.path.join(train_dir, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(image_exts):
                rows.append({"rel_path": os.path.join(cls, fname), "label": cls})

    if not rows:
        raise RuntimeError(f"No images found in {train_dir}")

    return pd.DataFrame(rows)


def build_test_df(test_dir: str) -> pd.DataFrame:
    """Build test dataframe from flat folder structure."""
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    rows = []

    for fname in os.listdir(test_dir):
        if fname.lower().endswith(image_exts):
            rows.append({"ID": fname, "rel_path": fname})

    if not rows:
        raise RuntimeError(f"No images found in {test_dir}")

    return pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    return running_loss / len(loader.dataset), f1_score(all_targets, all_preds, average="macro")


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return running_loss / len(loader.dataset), f1_score(all_targets, all_preds, average="macro")


def train_fold(fold: int, train_df: pd.DataFrame, valid_df: pd.DataFrame, cfg: CFG):
    print(f"\nFold {fold}")
    set_seed(cfg.seed + fold)

    train_loader = DataLoader(
        FireDataset(train_df, cfg.train_dir, get_train_transforms(cfg.img_size)),
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        FireDataset(valid_df, cfg.train_dir, get_valid_transforms(cfg.img_size)),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    model = build_model(cfg.model_name, cfg.num_classes).to(cfg.device)

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(cfg.num_classes), y=train_df["label_idx"].values
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(cfg.device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_f1, best_state = -np.inf, None

    for epoch in range(cfg.epochs):
        start = time.time()
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)
        val_loss, val_f1 = validate(model, valid_loader, criterion, cfg.device)
        scheduler.step()

        elapsed = time.time() - start
        print(f"  Epoch {epoch+1}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model_path = f"{cfg.model_name}_fold{fold}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"  Best val F1: {best_f1:.4f} -> {model_path}")

    return model, best_f1


def predict(models, test_df: pd.DataFrame, cfg: CFG, idx2label: dict):
    test_loader = DataLoader(
        FireDataset(test_df, cfg.test_dir, get_valid_transforms(cfg.img_size), is_test=True),
        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    all_ids, all_probs = [], []

    for images, ids in test_loader:
        images = images.to(cfg.device)
        all_ids.extend(ids)

        with torch.no_grad():
            probs = np.mean([
                torch.softmax(m(images), dim=1).cpu().numpy()
                for m in models
            ], axis=0)
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    pred_labels = [idx2label[i] for i in all_probs.argmax(axis=1)]

    return all_ids, pred_labels


def main():
    cfg = CFG()

    # Prepare training data
    train_df = build_train_df(cfg.train_dir)
    unique_labels = sorted(train_df["label"].unique())
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}
    train_df["label_idx"] = train_df["label"].map(label2idx)

    print(f"Classes: {unique_labels}")
    print(f"Train samples: {len(train_df)}")

    # K-Fold split
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    train_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["label_idx"])):
        train_df.loc[val_idx, "fold"] = fold

    # Train all folds
    models, scores = [], []
    for fold in range(cfg.n_splits):
        train_fold_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        valid_fold_df = train_df[train_df["fold"] == fold].reset_index(drop=True)
        model, score = train_fold(fold, train_fold_df, valid_fold_df, cfg)
        models.append(model)
        scores.append(score)

    print(f"\nCV scores: {scores}")
    print(f"Mean F1: {np.mean(scores):.4f}")

    # Inference
    test_df = build_test_df(cfg.test_dir)
    test_ids, test_labels = predict(models, test_df, cfg, idx2label)

    submission = pd.DataFrame({"ID": test_ids, "label": test_labels})
    submission = submission.sort_values("ID").reset_index(drop=True)
    submission.to_csv(cfg.output_sub, index=False)
    print(f"Submission saved: {cfg.output_sub}")


if __name__ == "__main__":
    main()
