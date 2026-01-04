import time
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download


# =========================
# App UI
# =========================
st.set_page_config(page_title="Emotion Detector", page_icon="🙂", layout="centered")
st.title("🙂 Emotion Detector (Fast)")
st.caption("Upload a JPG/PNG → detect face(s) → predict emotion for each face.")


# =========================
# Model labels (FER+ / 8 classes)
# =========================
EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]


# =========================
# Helpers
# =========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def downscale_for_detection(bgr: np.ndarray, max_side: int = 1200):
    """Downscale big images to speed up face detection; return scaled image + scale factors."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    scale = max_side / float(m)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# =========================
# Cached resources
# =========================
@st.cache_resource
def load_face_detector():
    # Lightweight, offline face detector
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


@st.cache_resource
def load_onnx_session():
    """
    Streamlit Cloud friendly:
    - Downloads ONNX model once (cached by HF hub + Streamlit cache)
    - Uses CPUExecutionProvider
    - Conservative threads to avoid contention on shared CPUs
    """
    model_path = hf_hub_download(
        repo_id="onnxmodelzoo/emotion-ferplus-12-int8",
        filename="emotion-ferplus-12-int8.onnx",
    )

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1

    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_type = sess.get_inputs()[0].type  # e.g., tensor(uint8) or tensor(float)
    return sess, input_name, input_type


face_detector = load_face_detector()
sess, input_name, input_type = load_onnx_session()


def detect_faces(bgr: np.ndarray):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces  # list of (x, y, w, h)


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Model expects: (1, 1, 64, 64) grayscale.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = resized.reshape(1, 1, 64, 64)

    # Match the model’s expected dtype
    if "uint8" in input_type:
        x = x.astype(np.uint8)
    else:
        x = x.astype(np.float32)
    return x


def predict_emotion(face_bgr: np.ndarray):
    x = preprocess_face(face_bgr)
    scores = sess.run(None, {input_name: x})[0].reshape(-1)  # (8,)
    probs = softmax(scores)
    best_idx = int(np.argmax(probs))
    return best_idx, probs


# =========================
# Upload + run
# =========================
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload a JPG/PNG to start.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
bgr_full = pil_to_bgr(img)

st.image(img, caption="Uploaded image", use_container_width=True)

# Speed optimization: downscale before detection
bgr_det, scale = downscale_for_detection(bgr_full, max_side=1200)

t0 = time.time()
faces_det = detect_faces(bgr_det)

if len(faces_det) == 0:
    st.warning("No face detected. Try a closer, front-facing photo with better lighting.")
    st.stop()

# Convert detected face boxes back to full-res coordinates
faces = []
inv_scale = 1.0 / scale
for (x, y, w, h) in faces_det:
    fx = int(x * inv_scale)
    fy = int(y * inv_scale)
    fw = int(w * inv_scale)
    fh = int(h * inv_scale)
    faces.append((fx, fy, fw, fh))

st.success(f"Detected {len(faces)} face(s).")

# Draw bounding boxes on full-res image for display
boxed = bgr_full.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(boxed, (x, y), (x + w, y + h), (0, 255, 0), 2)

st.image(bgr_to_pil(boxed), caption="Detected faces", use_container_width=True)

st.subheader("Predictions")

for i, (x, y, w, h) in enumerate(faces, start=1):
    face = bgr_full[y:y+h, x:x+w]

    t1 = time.time()
    best_idx, probs = predict_emotion(face)
    ms = (time.time() - t1) * 1000.0

    best_label = EMOTIONS[best_idx]
    best_prob = float(probs[best_idx])

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(bgr_to_pil(face), caption=f"Face #{i}", use_container_width=True)
    with c2:
        st.success(f"Face #{i}: **{best_label}** ({best_prob*100:.1f}%) — {ms:.1f} ms")
        order = np.argsort(-probs)
        for j in order:
            st.progress(float(probs[j]), text=f"{EMOTIONS[int(j)]}: {probs[j]*100:.1f}%")

total_ms = (time.time() - t0) * 1000.0
st.caption(f"Total processing time: {total_ms:.1f} ms (includes face detection + all faces inference)")

st.divider()
st.caption("Developed by Dr. Jishan Ahmed")
