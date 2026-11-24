"""Streamlit Interactive Demonstration App for Multimodal Crisis Intelligence.

Run via:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import streamlit as st

from inference import CrisisPredictor

# Page Setup
st.set_page_config(
    page_title="Multimodal Crisis Intelligence & Triage",
    page_icon="🚨",
    layout="wide",
)

CURRENT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = CURRENT_DIR / "data" / "images"

# Load Predictor (Cached in memory)
@st.cache_resource
def get_predictor():
    return CrisisPredictor()

predictor = get_predictor()

# App Header
st.title("🚨 Multimodal Crisis Intelligence & Emergency Triage")
st.markdown("""
**An End-to-End Multimodal Deep Learning System** fusing social text reports (**BERTweet**) and aerial/ground photos (**ConvNeXt-Tiny**) 
via a **Gated Multimodal Unit (GMU)** to predict disaster scenarios and dispatch-critical humanitarian resource needs in real time.
""")

st.divider()

# Preset Example Test Cases
PRESETS = {
    "Select a pre-loaded emergency preset...": {"text": "", "img_file": None},
    "🌧️ Urban Flash Flooding (Ahmedabad Underpass)": {
        "text": "Akhbarnagar underpass completely submerged in 8 feet water. Car floating with 2 passengers trapped on roof!",
        "img_file": "t2_urban_flooding.jpg",
    },
    "🌀 Severe Cyclone (Kutch / Saurashtra Coast)": {
        "text": "Severe cyclone winds tearing down roofs near Mandvi port. 15 families stranded near harbour without dry shelter.",
        "img_file": "c3_cyclone.jpg",
    },
    "🔥 Industrial Chemical Blaze (Dahej GIDC)": {
        "text": "Massive explosion in chemical processing unit. Several workers suffer severe burn injuries, need ambulances immediately!",
        "img_file": "i2_industrial_fire.jpg",
    },
    "📚 Social Metaphor (False Alarm Negative)": {
        "text": "I am literally drowning under a flood of mid-semester assignments and exams this week!",
        "img_file": "n1_not_disaster.jpg",
    },
}

# Layout: 2 Columns
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("1. Input Emergency Report")

    selected_preset = st.selectbox("Quick-Test Presets:", list(PRESETS.keys()))
    preset_data = PRESETS[selected_preset]

    default_text = preset_data["text"]
    user_text = st.text_area("Emergency Text Description / Social Post:", value=default_text, height=120)

    uploaded_file = st.file_uploader("Upload Incident Photo (JPG/PNG):", type=["jpg", "png", "jpeg"])

    image_to_process = None
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file).convert("RGB")
        st.image(image_to_process, caption="Uploaded Incident Photo", use_container_width=True)
    elif preset_data["img_file"]:
        preset_img_path = IMAGES_DIR / preset_data["img_file"]
        if preset_img_path.exists():
            image_to_process = Image.open(preset_img_path).convert("RGB")
            st.image(image_to_process, caption=f"Preset Image: {preset_data['img_file']}", use_container_width=True)

    needs_threshold = st.slider(
        "Candidate Decision Threshold for Humanitarian Needs:",
        min_value=0.30,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Probabilities above this threshold trigger advisory dispatch flags.",
    )

    analyze_button = st.button("🚀 Analyze Incident (Multimodal Inference)", type="primary", use_container_width=True)

with right_col:
    st.subheader("2. AI Triage & Modality Attribution")

    if analyze_button:
        if not user_text and not image_to_process:
            st.error("Please provide at least a text description or an image.")
        else:
            with st.spinner("Executing Gated Multimodal Forward Pass..."):
                result = predictor.predict(
                    text=user_text,
                    image=image_to_process,
                    needs_threshold=needs_threshold,
                )

            # Scenario Prediction Card
            scen = result["predicted_scenario"]
            conf = result["confidence"]

            if scen == "NOT_DISASTER":
                st.success(f"### Predicted Scenario: {scen} ({conf*100:.1f}% Confidence)")
            elif scen == "URBAN_FLOODING":
                st.warning(f"### Predicted Scenario: 🌊 {scen} ({conf*100:.1f}% Confidence)")
            elif scen == "CYCLONE":
                st.info(f"### Predicted Scenario: 🌀 {scen} ({conf*100:.1f}% Confidence)")
            else:
                st.error(f"### Predicted Scenario: 🔥 {scen} ({conf*100:.1f}% Confidence)")

            st.caption(f"**Modality Mode:** `{result['modality_mode'].upper()}`")

            # Gated Attribution Bar
            st.markdown("#### ⚖️ Gated Modality Attribution (GMU)")
            text_wt = float(result["gated_attribution"]["text_weight"].replace("%", ""))
            vision_wt = float(result["gated_attribution"]["vision_weight"].replace("%", ""))

            st.write(f"Text Contribution: **{text_wt}%** | Vision Contribution: **{vision_wt}%**")
            st.progress(text_wt / 100.0)
            st.caption("The Gated Unit dynamically weights text vs. image features to resolve ambiguous posts.")

            st.divider()

            # Humanitarian Needs Grid
            st.markdown("#### 🆘 Detected Urgent Humanitarian Needs")
            needs = result["resource_needs"]

            n_cols = st.columns(2)
            for idx, (need_name, need_info) in enumerate(needs.items()):
                col = n_cols[idx % 2]
                detected = need_info["detected"]
                prob = need_info["probability"]
                icon = "🚨" if detected else "⚪"
                status_color = "red" if detected else "gray"
                col.markdown(f"{icon} **{need_name.replace('_', ' ').title()}**")
                col.caption(f"Probability: {prob*100:.1f}% (Flagged: `{detected}`)")

            st.divider()

            # Scenario Probabilities
            st.markdown("#### 📊 Scenario Probability Distribution")
            for s_name, s_prob in result["scenario_probabilities"].items():
                st.write(f"**{s_name}:** {s_prob*100:.1f}%")
                st.progress(s_prob)

            # Human-in-the-loop Guardrail
            st.info(f"🛡️ **Safety Guardrail:** {result['advisory_disclaimer']}")
    else:
        st.info("👈 Select a preset or upload an image and click 'Analyze Incident' to view real-time triage.")

st.divider()
st.caption("ResQ AI/ML Multimodal Research Module | Event-Held-Out Evaluation Baseline")
