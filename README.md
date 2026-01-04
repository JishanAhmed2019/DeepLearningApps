# Applied Deep Learning with Streamlit  
### Interactive AI Demonstrations for Education

This repository contains a growing collection of **interactive Streamlit applications**
demonstrating how **deep learning models** can be applied, deployed, and interpreted
in real-world scenarios.

The primary goal is **education**: helping students understand how modern pretrained
models (vision and language) move from theory to **working applications**.

---

## 🚀 Deployed Applications

### 1️⃣ Emotion Detection (Computer Vision)

**Description**  
Detects human facial emotions from static images (`.jpg` / `.png`) using a fast,
CPU-efficient deep learning model.

**What students learn**
- Image preprocessing  
- Face detection  
- Deep learning inference  
- Probabilistic outputs vs hard decisions  
- Model efficiency and deployment constraints  

**Model**
- Quantized ONNX FER+ emotion recognition model (8 emotion classes)

**Input**
- Uploaded image containing one or more faces

**Output**
- Emotion prediction per detected face  
- Probability distribution over emotion classes  

---

### 2️⃣ Spam Detection (Natural Language Processing)

**Description**  
Classifies whether an email or message is **SPAM** or **NOT SPAM** using a pretrained
Transformer model.

**What students learn**
- Transformer-based text classification  
- Tokenization and truncation  
- Model confidence vs decision thresholds  
- False positives vs false negatives  
- Human-in-the-loop decision making  

**Model**
- `mrm8488/bert-tiny-finetuned-enron-spam-detection`  
- Fine-tuned on the Enron email dataset  

**Input**
- Email or message text (copy/paste)

**Output**
- Spam / Not Spam classification  
- Spam probability score  
- User-adjustable decision threshold  

---

## 🧠 Educational Objectives

These applications are designed to illustrate:

- How **pretrained deep learning models** are used in practice  
- Why raw model scores need interpretation  
- The role of **thresholds** in classification systems  
- Tradeoffs between accuracy, speed, and resources  
- Responsible use and limitations of AI systems  
- How to deploy ML models as interactive tools  

---

## 🛠️ Technologies Used

- **Python**  
- **Streamlit**  
- **Hugging Face Transformers**  
- **ONNX Runtime**  
- **OpenCV**  
- **NumPy**  
- **PyTorch** (via Transformers)  

---

## 📂 Repository Structure

```text
.
├── app_emotion.py        # Image-based emotion detection app
├── app_spam.py           # Text-based spam detection app
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

## ▶️ Running the Apps Locally

### 1️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the applications

**Emotion Detector**
```bash
streamlit run app_emotion.py
```

**Spam Detector**
```bash
streamlit run app_spam.py
```

---

## ☁️ Deployment

These applications are deployed using **Streamlit Community Cloud**.

Each app can be deployed independently by selecting the corresponding
entry-point file (`app_emotion.py` or `app_spam.py`).

---

## 🔮 Planned / Future Applications

This repository is intentionally structured to grow.
Future additions may include:

- 🔲 Sentiment Analysis (reviews, social media)  
- 🔲 Fake News Detection  
- 🔲 Topic Classification  
- 🔲 Image Classification (general objects)  
- 🔲 Time Series Forecasting  
- 🔲 Recommendation Systems  
- 🔲 LLM-based Q&A / RAG demonstrations  
- 🔲 Model comparison dashboards (speed vs accuracy)  

(Placeholders reserved for future Streamlit applications.)

---

## ⚠️ Disclaimer

These applications are **educational demonstrations only**.

Model predictions may be:
- inaccurate  
- biased  
- sensitive to input quality  

They should **not** be used for medical, legal, financial, or other high-stakes decisions.

---

## 👨‍🏫 Author

**Developed by Dr. Jishan Ahmed**  
Assistant Professor of Data Science  
Weber State University, Utah
