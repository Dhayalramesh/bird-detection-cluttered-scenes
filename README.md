# 🦅 Bird Detection in Cluttered Scenes

A YOLOv8n object detector fine-tuned on **CUB-200-2011** bounding boxes — benchmarked not just on single, clearly visible birds, but on **synthetically composed multi-bird, overlapping scenes** to measure how detection quality holds up when birds partially occlude each other, matching the real challenge of field photography.

**Live demo:** [bird-detection-cluttered-scenes-g3ndcy5bpgfe4rpadj7s57.streamlit.app](https://bird-detection-cluttered-scenes-g3ndcy5bpgfe4rpadj7s57.streamlit.app)

---

## Why this project

Most bird detection demos show a single, clearly visible bird — easy conditions that don't reflect real field photography. Real bird photos often contain multiple birds at different angles, partially hiding behind each other, or clustered close together. This project measures exactly how much detection quality degrades under those conditions, using both a controlled benchmark and real-world test cases.

## What it does

- Fine-tunes a **YOLOv8n** object detector (single class: "bird") on CUB-200-2011's official bounding box annotations
- Evaluates the trained model on two conditions:
  - **Clean** — the original CUB-200-2011 test images (one bird per image)
  - **Cluttered/multi-bird** — 300 synthetically composed scenes combining 2–3 real bird crops (with true bounding boxes) onto real backgrounds, with intentional overlap
- Reports mAP@0.5, mAP@0.5:0.95, precision, and recall per condition
- Deploys as an interactive Streamlit app: upload any photo and see live detection boxes drawn, plus a benchmark dashboard

## Results

Evaluated on CUB-200-2011's test set (5,794 clean images) and 300 synthetic cluttered scenes (705 total bird instances, avg. 2.35 birds/scene):

| Condition | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| Clean (single bird) | 99.4% | 86.0% | 98.6% | 98.9% |
| Cluttered/multi-bird | 77.5% | 50.9% | 86.0% | 68.9% |

**Key finding:** mAP@0.5 drops **21.9 points** (99.4% → 77.5%) under cluttered multi-bird conditions. Recall takes the larger hit — dropping **30.0 points** (98.9% → 68.9%) — while precision degrades less (98.6% → 86.0%). This suggests the model isn't confusing birds for non-birds; it's **missing birds** (likely smaller, overlapping ones) in crowded scenes, rather than producing false positives.

### Real-world confirmation

Beyond the synthetic benchmark, testing the deployed model on a real photograph of ~13–14 wading birds (a tight cluster of ducks plus a separate group of spoonbills and an egret) reproduced this failure mode even more starkly: the model detected only **1 bird** total — a single loosely-fitted box (0.72 confidence) covering an entire cluster of ~5–6 ducks, and missed the spoonbills/egret group entirely.

This isn't a bug — it's the model's benchmark limitation showing up in a harder, denser real scene than the synthetic test set used. CUB-200-2011's training images are almost entirely one clearly visible bird per photo, so the model never learned to separate multiple tightly-packed, overlapping instances of the same class. The synthetic benchmark (2–3 birds per scene) already predicted this direction of failure; this real example confirms it at a more extreme density.

## Methodology

- **Dataset:** [CUB-200-2011](https://data.caltech.edu/records/65de6-vp158) (Caltech-UCSD Birds), official train/test split (5,994 train / 5,794 test), using the dataset's original bounding box annotations
- **Model:** YOLOv8n (Ultralytics), fine-tuned for 15 epochs at 640px, single class ("bird")
- **Cluttered scene construction:** real bird crops (using their true bounding boxes) resized and pasted onto real photo backgrounds, with randomized placement and intentional overlap, to simulate multi-bird field scenes while preserving ground-truth accuracy
- **Evaluation:** standard object detection metrics (mAP@0.5, mAP@0.5:0.95, precision, recall) computed via Ultralytics' validation pipeline on both conditions using the same trained model weights

**Honest limitation:** cluttered scenes are synthetically composed (real bird crops on real backgrounds with known ground-truth boxes), not naturally photographed multi-bird scenes. This is a standard, controlled technique for robustness benchmarking, but it's an approximation — real dense clusters (as shown in the wading-birds test above) can be even harder than the synthetic benchmark suggests, since real overlap patterns and bird-on-bird occlusion are more extreme and structured than random paste placement.

## Tech stack

Python, PyTorch, Ultralytics YOLOv8, OpenCV, pandas, Streamlit, Hugging Face Hub (model hosting)

## Project structure
bird-detection-cluttered-scenes/
├── bird_detection_cluttered_scenes.ipynb # full training + evaluation pipeline (Colab)
├── app.py # Streamlit app (live detection + benchmark dashboard)
├── requirements.txt # Python dependencies
├── packages.txt # system-level dependencies (libGL for OpenCV)
└── README.md

Trained model weights and benchmark results are hosted on [Hugging Face Hub](https://huggingface.co/dhayal1/bird-detection-cluttered-scenes) and pulled by the app at runtime.

## Run it yourself

**Training:** open `bird_detection_cluttered_scenes.ipynb` in Google Colab (free T4 GPU), run all cells top to bottom. Downloads CUB-200-2011 automatically.

**App (locally):**
```bash
pip install -r requirements.txt
streamlit run app.py
```
Note: requires system packages `libgl1` and `libglib2.0-0` for OpenCV (see `packages.txt`; on Debian/Ubuntu run `sudo apt-get install libgl1 libglib2.0-0`).

## Author

Dhayal R — [GitHub](https://github.com/Dhayalramesh) · [LinkedIn](https://linkedin.com/in/dhayalsr)
