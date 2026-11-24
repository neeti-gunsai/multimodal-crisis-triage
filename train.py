"""Training and Evaluation Pipeline for Multimodal Crisis Intelligence.

Trains the Gated Multimodal Classifier with multi-task losses:
- CrossEntropyLoss for Disaster Scenario Classification.
- BCEWithLogitsLoss for Multi-Label Urgent Resource Needs.
Evaluates on event-held-out splits and writes structured JSON reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = CURRENT_DIR / "data" / "multimodal_crisis_dataset.csv"
CHECKPOINTS_DIR = CURRENT_DIR / "checkpoints"
REPORTS_DIR = CURRENT_DIR / "reports"


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("PyTorch is required for training. Install torch and torchvision via 'pip install -r requirements.txt'")
        return {"status": "failed", "error": "PyTorch not installed"}

    from dataset import CrisisMultimodalDataset, SCENARIO_TO_IDX, IDX_TO_SCENARIO, RESOURCE_NEED_COLS
    from models.fusion_model import MultimodalCrisisClassifier

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Datasets & Loaders
    train_dataset = CrisisMultimodalDataset(
        csv_path=args.csv_path,
        split="train",
        tokenizer_name=args.text_model,
        max_length=args.max_length,
    )
    val_dataset = CrisisMultimodalDataset(
        csv_path=args.csv_path,
        split="val",
        tokenizer_name=args.text_model,
        max_length=args.max_length,
    )
    test_dataset = CrisisMultimodalDataset(
        csv_path=args.csv_path,
        split="test",
        tokenizer_name=args.text_model,
        max_length=args.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Model Initialization
    logger.info("Instantiating MultimodalCrisisClassifier (%s + %s)...", args.text_model, args.vision_model)
    model = MultimodalCrisisClassifier(
        text_model_name=args.text_model,
        vision_model_name=args.vision_model,
        num_scenarios=len(SCENARIO_TO_IDX),
        num_needs=len(RESOURCE_NEED_COLS),
        fusion_dim=args.fusion_dim,
        pretrained=not args.no_pretrained,
    ).to(device)

    # 3. Loss Functions & Optimizer
    criterion_scenario = nn.CrossEntropyLoss()
    criterion_needs = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_f1 = -1.0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = CHECKPOINTS_DIR / "best_multimodal_model.pt"

    logger.info("Starting training for %d epochs...", args.epochs)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device) if batch["pixel_values"] is not None else None
            scenario_targets = batch["scenario_label"].to(device)
            needs_targets = batch["needs_labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            loss_scen = criterion_scenario(outputs["scenario_logits"], scenario_targets)
            loss_needs = criterion_needs(outputs["needs_logits"], needs_targets)
            loss = loss_scen + (1.2 * loss_needs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))

        # Evaluate on Validation Set
        val_metrics = evaluate(model, val_loader, device, criterion_scenario, criterion_needs)
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Scenario Macro-F1: %.4f | Val Needs Micro-F1: %.4f",
            epoch, args.epochs, avg_loss, val_metrics["scenario_macro_f1"], val_metrics["needs_micro_f1"]
        )

        # Track Best Model
        if val_metrics["scenario_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["scenario_macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "text_model": args.text_model,
                "vision_model": args.vision_model,
                "num_scenarios": len(SCENARIO_TO_IDX),
                "num_needs": len(RESOURCE_NEED_COLS),
            }, best_model_path)
            logger.info("Saved best model checkpoint to %s", best_model_path)

    # 5. Final Evaluation on Untouched Test Split (Held-out Events)
    logger.info("Loading best checkpoint for evaluation on held-out test events...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion_scenario, criterion_needs)

    report_data = {
        "text_model": args.text_model,
        "vision_model": args.vision_model,
        "epochs": args.epochs,
        "best_val_macro_f1": best_val_f1,
        "test_results": test_metrics,
    }

    report_path = REPORTS_DIR / "eval_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Evaluation report saved to %s", report_path)
    logger.info("--- FINAL HELD-OUT TEST PERFORMANCE ---")
    logger.info("Scenario Accuracy: %.4f", test_metrics["scenario_accuracy"])
    logger.info("Scenario Macro-F1: %.4f", test_metrics["scenario_macro_f1"])
    logger.info("Resource Needs Micro-F1: %.4f", test_metrics["needs_micro_f1"])

    return report_data


def evaluate(model, dataloader, device, criterion_scenario, criterion_needs) -> Dict[str, Any]:
    """Run full evaluation and compute classification metrics."""
    import torch

    model.eval()
    all_scenario_preds = []
    all_scenario_targets = []
    all_needs_preds = []
    all_needs_targets = []
    all_gate_scores = []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device) if batch["pixel_values"] is not None else None
            scenario_targets = batch["scenario_label"].to(device)
            needs_targets = batch["needs_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )

            loss_scen = criterion_scenario(outputs["scenario_logits"], scenario_targets)
            loss_needs = criterion_needs(outputs["needs_logits"], needs_targets)
            total_val_loss += (loss_scen + loss_needs).item()

            # Scenario predictions (argmax over softmax)
            scenario_preds = outputs["scenario_logits"].argmax(dim=-1).cpu().numpy()
            all_scenario_preds.extend(scenario_preds)
            all_scenario_targets.extend(scenario_targets.cpu().numpy())

            # Needs predictions (sigmoid threshold at 0.50)
            needs_probs = torch.sigmoid(outputs["needs_logits"]).cpu().numpy()
            needs_preds = (needs_probs >= 0.50).astype(int)
            all_needs_preds.extend(needs_preds)
            all_needs_targets.extend(needs_targets.cpu().numpy())

            all_gate_scores.extend(outputs["gate_score"].cpu().numpy().flatten())

    scen_acc = float(accuracy_score(all_scenario_targets, all_scenario_preds))
    scen_macro_f1 = float(f1_score(all_scenario_targets, all_scenario_preds, average="macro", zero_division=0))
    scen_weighted_f1 = float(f1_score(all_scenario_targets, all_scenario_preds, average="weighted", zero_division=0))

    needs_micro_f1 = float(f1_score(np.array(all_needs_targets), np.array(all_needs_preds), average="micro", zero_division=0))
    needs_macro_f1 = float(f1_score(np.array(all_needs_targets), np.array(all_needs_preds), average="macro", zero_division=0))

    avg_gate = float(np.mean(all_gate_scores)) if all_gate_scores else 0.5

    return {
        "val_loss": total_val_loss / max(1, len(dataloader)),
        "scenario_accuracy": scen_acc,
        "scenario_macro_f1": scen_macro_f1,
        "scenario_weighted_f1": scen_weighted_f1,
        "needs_micro_f1": needs_micro_f1,
        "needs_macro_f1": needs_macro_f1,
        "average_gate_score": avg_gate,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multimodal Crisis Intelligence Model")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_DATA_CSV), help="Path to dataset CSV")
    parser.add_argument("--text-model", type=str, default="vinai/bertweet-base", help="Hugging Face text model name")
    parser.add_argument("--vision-model", type=str, default="convnext_tiny", help="Timm vision model name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--fusion-dim", type=int, default=512, help="Hidden dimension of fusion unit")
    parser.add_argument("--max-length", type=int, default=96, help="Maximum token length")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize models without pretrained weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_training(args)
