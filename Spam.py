import re
import streamlit as st
from transformers import pipeline

# Recommended for emails (Enron). For SMS model, replace with:
# MODEL_NAME = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
MODEL_NAME = "mrm8488/bert-tiny-finetuned-enron-spam-detection"


def clean_text(text: str) -> str:
    """Light cleanup for pasted emails (remove html tags + normalize whitespace)."""
    text = re.sub(r"<[^>]+>", " ", text)        # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return text


@st.cache_resource
def load_classifier(model_name: str):
    # return_all_scores gives us both classes; truncation helps on long emails
    return pipeline(
        task="text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        truncation=True,
    )


def to_score_map(model_output):
    """
    model_output format (return_all_scores=True):
      [[{'label': 'LABEL_0', 'score': ...}, {'label': 'LABEL_1', 'score': ...}]]
    """
    scores = model_output[0]
    score_map = {d["label"]: float(d["score"]) for d in scores}
    return score_map


st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="centered")
st.title("🛡️ Spam Detector (Email/SMS)")
st.caption(f"Model: `{MODEL_NAME}`")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Spam threshold", 0.00, 1.00, 0.50, 0.01)

    # Because some models output LABEL_0/LABEL_1 without meaning,
    # let the user explicitly decide which label is SPAM.
    spam_label = st.selectbox("Which label means SPAM?", ["LABEL_1", "LABEL_0"], index=0)

    st.markdown("---")
    st.write("Tip: Raise threshold to reduce false positives (good emails marked spam).")

email_text = st.text_area(
    "Paste your email (or message) text:",
    height=240,
    placeholder="Paste the email body here…",
)

col1, col2 = st.columns([1, 1])
with col1:
    detect = st.button("Detect", type="primary")
with col2:
    st.button("Clear", on_click=lambda: st.session_state.update({"_clear": True}))

if detect:
    if not email_text.strip():
        st.warning("Please paste some text first.")
        st.stop()

    clf = load_classifier(MODEL_NAME)
    x = clean_text(email_text)

    output = clf(x)
    score_map = to_score_map(output)

    # Determine spam probability from chosen spam_label
    spam_prob = score_map.get(spam_label, 0.0)

    is_spam = spam_prob >= threshold

    # Friendly result header
    if is_spam:
        st.error("Result: **SPAM**", icon="🚫")
    else:
        st.success("Result: **NOT SPAM**", icon="✅")

    st.write(f"**Spam probability:** `{spam_prob:.3f}` (threshold = `{threshold:.2f}`)")
    st.progress(min(max(spam_prob, 0.0), 1.0))

    # Show both label scores in a readable way
    st.subheader("Details")
    # Sort by score descending for readability
    sorted_items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    for label, score in sorted_items:
        tag = " ← used as SPAM" if label == spam_label else ""
        st.write(f"- `{label}`: `{score:.3f}`{tag}")

    with st.expander("Raw model output"):
        st.json(output)

st.markdown("---")
st.caption("Note: Long emails may be truncated by the model; results are best-effort.")

st.divider()
st.caption("Developed by Dr. Jishan Ahmed")