"""Quick CLI demonstration for Multimodal Crisis Intelligence.

Run via:
    python run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from inference import CrisisPredictor

def run_cli_demo():
    print("=" * 70)
    print("🚨 RESQ MULTIMODAL CRISIS INTELLIGENCE - TERMINAL DEMO")
    print("=" * 70)
    
    print("\n[+] Loading CrisisPredictor engine...")
    predictor = CrisisPredictor()
    print("[✓] Model engine initialized successfully!\n")

    test_cases = [
        {
            "title": "CASE 1: Urban Flash Floods (Ahmedabad Underpass)",
            "text": "Akhbarnagar underpass completely submerged in 8 feet water. Car floating with 2 passengers trapped on roof!",
            "image": "data/images/t2_urban_flooding.jpg",
        },
        {
            "title": "CASE 2: Severe Coastal Cyclone (Kutch Mandvi)",
            "text": "Severe cyclone winds tearing down roofs near Mandvi port. 15 families stranded near harbour without dry shelter.",
            "image": "data/images/c3_cyclone.jpg",
        },
        {
            "title": "CASE 3: Industrial Chemical Blaze (Dahej GIDC)",
            "text": "Massive explosion in chemical processing unit. Several workers suffer severe burn injuries, need ambulances immediately!",
            "image": "data/images/i2_industrial_fire.jpg",
        },
        {
            "title": "CASE 4: False Alarm / Social Metaphor (Everyday Post)",
            "text": "I am literally drowning under a flood of mid-semester assignments and exams this week!",
            "image": "data/images/n1_not_disaster.jpg",
        },
    ]

    for case in test_cases:
        print("-" * 70)
        print(f"📌 {case['title']}")
        print(f"📝 Text: \"{case['text']}\"")
        img_path = CURRENT_DIR / case["image"]
        img_obj = Image.open(img_path) if img_path.exists() else None
        print(f"🖼️ Image: {case['image']} ({'Found' if img_obj else 'Missing'})")
        
        result = predictor.predict(text=case["text"], image=img_obj)

        print(f"\n👉 Predicted Scenario: {result['predicted_scenario']} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"⚖️ Modality Weight: Text {result['gated_attribution']['text_weight']} | Vision {result['gated_attribution']['vision_weight']}")
        print("🆘 Urgent Needs Detected:")
        needs_found = False
        for need, info in result["resource_needs"].items():
            if info["detected"]:
                print(f"   • [ALERT] {need.replace('_', ' ').title()}: {info['probability']*100:.1f}%")
                needs_found = True
        if not needs_found:
            print("   • None (No emergency resources required)")
        print()

    print("=" * 70)
    print("✅ All demo cases evaluated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_cli_demo()
