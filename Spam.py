import math
from typing import Dict, Tuple

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")
st.title("📧 Spam Detector (Transformers)")
st.caption("Paste a message → get a spam probability and a clear decision using a threshold you control.")

MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"


@st.cache_resource
def load_model():
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    # logits: [1, num_labels]
    return torch.softmax(logits, dim=-1)


def get_label_mapping(model) -> Dict[int, str]:
    # Robustly pull id2label; fall back to LABEL_0/LABEL_1
    id2label = getattr(model.config, "id2label", None) or {}
    if not id2label:
        return {0: "ham", 1: "spam"}
    # Keys sometimes strings in HF configs
    out = {}
    for k, v in id2label.items():
        out[int(k)] = str(v)
    return out


def infer_spam(text: str) -> Tuple[float, Dict[str, float]]:
    tokenizer, model = load_model()
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits  # [1, num_labels]
        probs = _softmax(logits).squeeze(0)  # [num_labels]

    mapping = get_label_mapping(model)  # {0: label0, 1: label1}
    # Normalize labels (LABEL_0 → ham, LABEL_1 → spam if ambiguous)
    labels = {i: mapping.get(i, f"LABEL_{i}") for i in range(probs.numel())}

    # Heuristic: if any label contains "spam", that's spam; else assume index 1 is spam
    spam_idx = None
    for i, lab in labels.items():
        if "spam" in lab.lower():
            spam_idx = i
            break
    if spam_idx is None:
        spam_idx = 1 if probs.numel() > 1 else 0

    spam_prob = float(probs[spam_idx].item())

    per_label = {labels[i]: float(probs[i].item()) for i in range(probs.numel())}
    return spam_prob, per_label


with st.expander("Model info", expanded=False):
    st.write(f"Model: `{MODEL_ID}`")
    st.write("Runs on CPU. First run may take longer due to model download.")

default_example = "Congratulations! You have won a $1,000 gift card. Click the link to claim now."
text = st.text_area("Message text", value=default_example, height=160)

threshold = st.slider("Spam threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

colA, colB = st.columns([1, 1])
with colA:
    run = st.button("Run detection", type="primary")
with colB:
    st.caption("Tip: if it flags too much as spam, raise the threshold (e.g., 0.7).")

if run:
    if not text.strip():
        st.warning("Please paste a message.")
        st.stop()

    with st.spinner("Running model..."):
        spam_prob, per_label = infer_spam(text.strip())

    is_spam = spam_prob >= threshold
    decision = "SPAM 🚫" if is_spam else "NOT SPAM ✅"

    st.subheader("Decision")
    st.metric("Result", decision, delta=f"spam prob: {spam_prob:.3f}")

    st.subheader("Probabilities")
    st.progress(min(max(spam_prob, 0.0), 1.0))
    st.write(f"Spam probability: **{spam_prob:.3f}** (threshold: `{threshold:.2f}`)")

    st.dataframe(
        [{"label": k, "prob": v} for k, v in sorted(per_label.items(), key=lambda x: -x[1])],
        use_container_width=True,
    )

    st.subheader("How to interpret")
    st.write(
        "The model outputs probabilities for each label. "
        "You choose a threshold: if `spam_prob ≥ threshold`, we call it spam."
    )
