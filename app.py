import streamlit as st
from ultralytics import YOLO
import pandas as pd
import json
from PIL import Image
from huggingface_hub import hf_hub_download
import numpy as np

# ---- CONFIG ----
HF_REPO = "dhayal1/bird-detection-cluttered-scenes"

st.set_page_config(page_title="Bird Detection: Cluttered Scenes", page_icon="🦅", layout="centered")

# ---- LOAD MODEL + RESULTS (cached) ----
@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=HF_REPO, filename="bird_detector_best.pt")
    return YOLO(weights_path)

@st.cache_data
def load_results():
    results_path = hf_hub_download(repo_id=HF_REPO, filename="detection_robustness_results.json")
    with open(results_path) as f:
        return json.load(f)

model = load_model()
robustness_results = load_results()

# ---- UI ----
st.title("🦅 Bird Detection in Cluttered Scenes")
st.markdown(
    "A YOLOv8n object detector fine-tuned on **CUB-200-2011** bounding boxes — "
    "benchmarked not just on single, clearly visible birds, but on **synthetically "
    "composed multi-bird, overlapping scenes** to measure how detection quality holds up "
    "when birds partially occlude each other, matching the real challenge of field photography."
)

tab1, tab2 = st.tabs(["🔍 Try It", "📊 Robustness Benchmark"])

with tab1:
    uploaded_file = st.file_uploader("Upload a photo (single or multiple birds)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Detecting..."):
            results = model.predict(np.array(image), conf=0.25, verbose=False)
            result = results[0]
            annotated = result.plot()  # returns BGR numpy array with boxes drawn
            annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB

        st.image(annotated_rgb, caption=f"Detected {len(result.boxes)} bird(s)", use_container_width=True)

        if len(result.boxes) > 0:
            st.subheader("Detections")
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                st.write(f"**Bird {i+1}** — confidence: {conf*100:.1f}%")
        else:
            st.warning("No birds detected. Try a clearer or less cluttered photo.")

        st.caption(
            "⚠️ Trained to detect birds as a single class (not species). "
            "Works best on photos with reasonably visible birds — heavy occlusion "
            "or very small/distant birds may be missed, consistent with the "
            "robustness benchmark below."
        )
    else:
        st.info("Upload a photo to see bird detections.")

with tab2:
    st.subheader("Detection accuracy: single bird vs. cluttered multi-bird scenes")
    st.markdown(
        "Real bird photos often contain multiple birds, overlapping or partially "
        "hiding each other — very different from one clean bird centered in frame. "
        "This model was evaluated on the CUB-200-2011 test set under two conditions: "
        "the original single-bird images, and **300 synthetically composed scenes** "
        "combining 2–3 real bird crops (with their true bounding boxes) onto real "
        "backgrounds, with intentional overlap."
    )

    results_df = pd.DataFrame(robustness_results)
    results_df_display = results_df.copy()
    for col in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        results_df_display[col] = (results_df_display[col] * 100).round(1).astype(str) + '%'
    results_df_display.columns = ['Condition', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
    st.dataframe(results_df_display, use_container_width=True, hide_index=True)

    st.bar_chart(results_df.set_index('condition')[['mAP50', 'recall']] * 100)

    clean = results_df[results_df['condition'] == 'clean'].iloc[0]
    cluttered = results_df[results_df['condition'] == 'cluttered_multi_bird'].iloc[0]
    map_drop = (clean['mAP50'] - cluttered['mAP50']) * 100
    recall_drop = (clean['recall'] - cluttered['recall']) * 100

    st.warning(
        f"**Key finding:** mAP@0.5 drops **{map_drop:.1f} points** ({clean['mAP50']*100:.1f}% → "
        f"{cluttered['mAP50']*100:.1f}%) under cluttered multi-bird conditions. Recall takes the "
        f"larger hit — dropping **{recall_drop:.1f} points** ({clean['recall']*100:.1f}% → "
        f"{cluttered['recall']*100:.1f}%) — while precision degrades less. This suggests the model "
        f"isn't confusing birds for non-birds; it's **missing birds** (likely smaller, overlapping ones) "
        f"in crowded scenes, rather than producing false positives."
    )

    st.caption(
        "Note: cluttered scenes are synthetically composed (real bird crops pasted onto real "
        "backgrounds with known ground-truth boxes), not naturally photographed multi-bird scenes. "
        "This is a standard technique for controlled robustness benchmarking, but it's an "
        "approximation of field conditions."
    )

st.markdown("---")
st.caption("Built by Dhayal R · [GitHub](https://github.com/Dhayalramesh/bird-detection-cluttered-scenes)")
