import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2
import streamlit as st
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Emotion Detector", page_icon="🙂", layout="centered")
st.title("🙂 Emotion Detector (Fast, ONNX)")
st.caption("Upload a JPG/PNG → detect face(s) → predict emotion for each face (CPU).")

EMOTIONS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


# -------------------------
# Utilities
# -------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def downscale_for_detection(bgr: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    """Downscale big images to speed up face detection; return scaled image + scale factor."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


@dataclass
class FaceResult:
    box: Tuple[int, int, int, int]  # x,y,w,h
    emotion: str
    probs: np.ndarray


# -------------------------
# Cached resources
# -------------------------
@st.cache_resource
def load_face_detector() -> cv2.CascadeClassifier:
    # Lightweight, offline face detector
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade. Check OpenCV installation.")
    return detector


@st.cache_resource
def load_onnx_session():
    """
    Streamlit Cloud friendly:
    - Downloads ONNX model once (cached by HF hub + Streamlit cache)
    - Uses CPUExecutionProvider
    """
    model_path = hf_hub_download(
        repo_id="onnxmodelzoo/emotion-ferplus-12-int8",
        filename="emotion-ferplus-12-int8.onnx",
    )

    so = ort.SessionOptions()
    # Conservative threading for shared CPUs
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_type = inp.type  # e.g., tensor(uint8) or tensor(float)

    # Many FER+ ONNX models use shape [N, 1, 64, 64]
    input_shape = inp.shape
    return sess, input_name, input_type, input_shape


face_detector = load_face_detector()
sess, input_name, input_type, input_shape = load_onnx_session()


def detect_faces(bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return list(faces)  # list of (x, y, w, h)


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Prepare a face crop for the FER+ model.
    - Convert to grayscale
    - Resize to 64x64
    - Return tensor with shape [1, 1, 64, 64]
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    # Model expects NCHW
    x = gray[None, None, :, :]  # [1,1,64,64]
    if "uint8" in str(input_type).lower():
        x = x.astype(np.uint8)
    else:
        # Conservative normalization; FER+ models vary, but this is commonly fine.
        x = (x.astype(np.float32) / 255.0)
    return x


def predict_emotion(face_bgr: np.ndarray) -> tuple[str, np.ndarray]:
    x = preprocess_face(face_bgr)
    out = sess.run(None, {input_name: x})[0]
    # out usually [1, 8]
    scores = np.array(out).reshape(-1)
    probs = softmax(scores)
    best_idx = int(np.argmax(probs))
    return EMOTIONS[best_idx], probs


def run_on_image(bgr: np.ndarray, padding: float = 0.15) -> tuple[np.ndarray, List[FaceResult]]:
    scaled, scale = downscale_for_detection(bgr)
    faces = detect_faces(scaled)

    results: List[FaceResult] = []
    vis = scaled.copy()

    for (x, y, w, h) in faces:
        # Pad the face box a little for robustness
        pad = int(min(w, h) * padding)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(scaled.shape[1], x + w + pad)
        y1 = min(scaled.shape[0], y + h + pad)

        face_crop = scaled[y0:y1, x0:x1]
        if face_crop.size == 0:
            continue

        emotion, probs = predict_emotion(face_crop)
        results.append(FaceResult((x0, y0, x1 - x0, y1 - y0), emotion, probs))

        # Draw
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        label = f"{emotion} ({probs.max():.2f})"
        cv2.putText(
            vis,
            label,
            (x0, max(20, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # If we scaled down, scale visualization back to original size for display
    if scale != 1.0:
        vis = cv2.resize(vis, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Rescale boxes too
        scaled_results = []
        for r in results:
            x, y, w, h = r.box
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            scaled_results.append(FaceResult((x, y, w, h), r.emotion, r.probs))
        results = scaled_results

    return vis, results


# -------------------------
# App
# -------------------------
with st.expander("Model info", expanded=False):
    st.write("ONNX model: `onnxmodelzoo/emotion-ferplus-12-int8` (Hugging Face Hub)")
    st.write(f"Input shape (reported): `{input_shape}`  •  input type: `{input_type}`")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    padding = st.slider("Face padding", 0.0, 0.40, 0.15, 0.01)
with col2:
    show_probs = st.checkbox("Show full probability table", value=False)

if not uploaded:
    st.info("Upload a JPG/PNG to start.")
    st.stop()

pil_img = Image.open(uploaded)
bgr = pil_to_bgr(pil_img)

with st.spinner("Detecting faces and running emotion model..."):
    t0 = time.time()
    vis_bgr, results = run_on_image(bgr, padding=padding)
    dt = time.time() - t0

st.image(bgr_to_pil(vis_bgr), caption=f"Done in {dt:.2f}s • faces: {len(results)}", use_container_width=True)

if len(results) == 0:
    st.warning("No faces detected. Try a clearer frontal face image.")
    st.stop()

st.subheader("Results")
for i, r in enumerate(results, start=1):
    st.markdown(f"**Face {i}: {r.emotion}**  (confidence: `{r.probs.max():.3f}`)")
    if show_probs:
        # Display as a tidy table
        rows = [{"emotion": e, "prob": float(p)} for e, p in zip(EMOTIONS, r.probs)]
        st.dataframe(rows, use_container_width=True)
