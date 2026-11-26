"""Streamlit web app for the brain MRI classifier.

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from brainmri import config
from brainmri.gradcam import GradCAM, overlay
from brainmri.predict import Predictor

SAMPLES_DIR = Path(__file__).parent / "samples"
LOW_CONFIDENCE = 0.70

st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠",
                   layout="wide")


@st.cache_resource
def load_predictor():
    predictor = Predictor()
    return predictor, GradCAM(predictor.model)


def load_metrics():
    path = config.REPORT_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def sidebar():
    st.sidebar.header("About the model")
    st.sidebar.markdown(
        "- **Backbone:** EfficientNet-B0, pretrained on ImageNet\n"
        "- **Training:** fine-tuned on about 4,800 labelled brain MRI slices\n"
        "- **Classes:** glioma, meningioma, pituitary tumor, no tumor\n"
        "- **Explanation:** Grad-CAM heatmap of the regions behind the prediction"
    )
    metrics = load_metrics()
    if metrics:
        st.sidebar.subheader("Held-out test set")
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col2.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        st.sidebar.caption(f"{metrics['test_images']:,} images the model never saw "
                           "during training.")
        cm_path = config.REPORT_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.sidebar.image(str(cm_path))


def choose_image():
    uploaded = st.file_uploader("Upload a brain MRI slice",
                                type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        return Image.open(uploaded), uploaded.name

    samples = sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
    if samples:
        names = ["(none)"] + [p.name for p in samples]
        choice = st.selectbox("…or try a sample scan from the test set", names)
        if choice != "(none)":
            return Image.open(SAMPLES_DIR / choice), choice
    return None, None


def main():
    st.title("🧠 Brain MRI Tumor Classifier")
    st.write("Upload a brain MRI slice to classify it as glioma, meningioma, "
             "pituitary tumor or no tumor, and see which regions the model "
             "focused on.")
    st.warning("Educational project only. It is not a medical device and must "
               "not be used for diagnosis.", icon="⚠️")
    sidebar()

    if not config.CHECKPOINT.exists():
        st.error("No trained model found. Run `python -m brainmri.train` first.")
        return

    image, name = choose_image()
    if image is None:
        return

    predictor, gradcam = load_predictor()
    with st.spinner("Analysing scan…"):
        result = predictor.predict(image)
        heatmap = gradcam(predictor.to_tensor(image), result.class_index)
        explained = overlay(image, heatmap)

    left, right = st.columns(2)
    left.image(image.convert("RGB"), caption=f"Input: {name}",
               width="stretch")
    right.image(explained, caption="Grad-CAM: warmer colours had more influence",
                width="stretch")

    st.subheader(f"Prediction: {result.label}")
    st.progress(result.confidence, text=f"Confidence {result.confidence:.1%}")
    if result.confidence < LOW_CONFIDENCE:
        st.info("The model is unsure about this scan. Treat the result with "
                "extra caution.")

    probs = pd.DataFrame({"Class": list(result.probabilities),
                          "Probability": list(result.probabilities.values())})
    bars = alt.Chart(probs).mark_bar().encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        y=alt.Y("Class:N", sort=None, title=None),
        tooltip=["Class", alt.Tooltip("Probability:Q", format=".1%")],
    )
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text("Probability:Q", format=".1%"))
    st.altair_chart((bars + labels).properties(height=180), width="stretch")


main()
