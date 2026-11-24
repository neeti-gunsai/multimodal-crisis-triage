"""Dataset and DataLoader module for Multimodal Crisis Intelligence.

Handles:
- Loading the multimodal CSV dataset.
- Event-held-out splits (Train, Val, Test).
- Text tokenization via Hugging Face tokenizers.
- Vision preprocessing and data augmentations via torchvision.
- Multi-label binary encoding for humanitarian needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # type: ignore

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

SCENARIO_TO_IDX = {
    "cyclone": 0,
    "urban_flooding": 1,
    "industrial_fire": 2,
    "not_disaster": 3,
}
IDX_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_IDX.items()}

RESOURCE_NEED_COLS = [
    "search_and_rescue",
    "medical_aid",
    "shelter_needed",
    "water_food",
    "evacuation_advisory",
]


def get_image_transforms(split: str = "train", img_size: int = 224):
    """Return appropriate data augmentations for image inputs."""
    if not TORCH_AVAILABLE:
        return None

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class CrisisMultimodalDataset(Dataset):
    """PyTorch Dataset yielding paired text tokens, image tensors, and multi-task labels."""

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        tokenizer_name: str = "vinai/bertweet-base",
        max_length: int = 96,
        img_size: int = 224,
        transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root_dir = self.csv_path.parent
        self.split = split
        self.max_length = max_length
        self.img_size = img_size

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at: {self.csv_path}")

        df_all = pd.read_csv(self.csv_path)
        self.df = df_all[df_all["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No records found for split '{split}' in {self.csv_path}")

        logger.info("Loaded %d records for '%s' split.", len(self.df), self.split)

        # Image transforms
        self.transform = transform or get_image_transforms(split, img_size)

        # Tokenizer setup
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
            except Exception as e:
                logger.warning("Could not download tokenizer '%s' (%s). Using fallback tokenizer.", tokenizer_name, e)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                except Exception:
                    self.tokenizer = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # 1. Process Text
        text = str(row["text"])
        if self.tokenizer is not None and TORCH_AVAILABLE:
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
        elif TORCH_AVAILABLE:
            # Fallback simple tensor representation if tokenizer is offline
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            input_ids = [0] * self.max_length
            attention_mask = [1] * self.max_length

        # 2. Process Image
        img_rel = str(row["image_path"])
        img_path = self.root_dir / img_rel
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                if self.transform is not None:
                    pixel_values = self.transform(img_rgb)
                elif TORCH_AVAILABLE:
                    pixel_values = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
                else:
                    pixel_values = None
        except Exception as e:
            logger.warning("Error reading image %s (%s). Using blank image tensor.", img_path, e)
            if TORCH_AVAILABLE:
                pixel_values = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            else:
                pixel_values = None

        # 3. Scenario Label (Single-label multi-class: 0..3)
        scenario_str = str(row["scenario"]).lower()
        scenario_idx = SCENARIO_TO_IDX.get(scenario_str, SCENARIO_TO_IDX["not_disaster"])

        # 4. Resource Needs Labels (Multi-label binary: 5 binary targets)
        needs_vals = [float(row[col]) for col in RESOURCE_NEED_COLS]
        if TORCH_AVAILABLE:
            scenario_tensor = torch.tensor(scenario_idx, dtype=torch.long)
            needs_tensor = torch.tensor(needs_vals, dtype=torch.float32)
        else:
            scenario_tensor = scenario_idx
            needs_tensor = needs_vals

        return {
            "id": str(row["id"]),
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "scenario_label": scenario_tensor,
            "needs_labels": needs_tensor,
            "event_id": str(row["event_id"]),
        }
