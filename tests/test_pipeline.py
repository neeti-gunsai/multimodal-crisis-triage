"""Unit & Integration Tests for the Multimodal Crisis Intelligence Pipeline.

Works with both pytest and python's built-in unittest module.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = CURRENT_DIR / "data" / "multimodal_crisis_dataset.csv"
IMAGES_DIR = CURRENT_DIR / "data" / "images"

sys.path.insert(0, str(CURRENT_DIR))

from dataset import CrisisMultimodalDataset, SCENARIO_TO_IDX, RESOURCE_NEED_COLS
from inference import CrisisPredictor


class TestMultimodalPipeline(unittest.TestCase):

    def test_dataset_csv_and_images_exist(self):
        """Verify that dataset CSV is present and all image files exist on disk."""
        self.assertTrue(CSV_PATH.exists(), f"Dataset CSV missing at {CSV_PATH}")
        
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        self.assertGreater(len(df), 0, "Dataset is empty")

        # Check required columns
        expected_cols = ["id", "text", "image_path", "scenario"] + RESOURCE_NEED_COLS + ["event_id", "split"]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column: {col}")

        # Verify each image exists
        for _, row in df.iterrows():
            img_full_path = CURRENT_DIR / "data" / str(row["image_path"])
            self.assertTrue(img_full_path.exists(), f"Missing image file: {img_full_path}")

    def test_dataset_splits_distribution(self):
        """Verify train, validation, and test splits have samples."""
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        splits = set(df["split"].unique())
        self.assertIn("train", splits)
        self.assertIn("val", splits)
        self.assertIn("test", splits)

    def test_predictor_multimodal_inference(self):
        """Verify inference pipeline when given both text and an image."""
        predictor = CrisisPredictor()
        sample_img = Image.new("RGB", (224, 224), color=(30, 90, 140))
        sample_text = "Tapi river overflowed into low lying areas. 50 families need emergency rescue boats!"

        result = predictor.predict(text=sample_text, image=sample_img)

        self.assertIn("predicted_scenario", result)
        self.assertIn(result["predicted_scenario"], ["CYCLONE", "URBAN_FLOODING", "INDUSTRIAL_FIRE", "NOT_DISASTER"])
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertIn("resource_needs", result)
        self.assertEqual(len(result["resource_needs"]), len(RESOURCE_NEED_COLS))
        self.assertIn("gated_attribution", result)

    def test_predictor_text_only_fallback(self):
        """Verify inference gracefully succeeds when image is omitted."""
        predictor = CrisisPredictor()
        sample_text = "Factory boiler explosion in chemical unit, multiple casualties reported."

        result = predictor.predict(text=sample_text, image=None)

        self.assertIn("predicted_scenario", result)
        self.assertIn(result["modality_mode"], ["text_only", "multimodal"])
        self.assertIn("confidence", result)

    def test_predictor_vision_only_fallback(self):
        """Verify inference gracefully succeeds when text is omitted."""
        predictor = CrisisPredictor()
        sample_img = Image.new("RGB", (224, 224), color=(180, 50, 20))

        result = predictor.predict(text=None, image=sample_img)

        self.assertIn("predicted_scenario", result)
        self.assertIn(result["modality_mode"], ["vision_only", "multimodal"])
        self.assertIn("confidence", result)


if __name__ == "__main__":
    unittest.main()
