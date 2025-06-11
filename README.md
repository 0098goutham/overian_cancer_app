# 🧬 Ovarian Cancer Classifier using Vision Transformer (ViT)

This project is a web-based ovarian cancer classification system that uses a fine-tuned Vision Transformer (ViT) model to detect the type of ovarian cancer from histopathology images. It also estimates CA-125 biomarker levels to indicate the severity.

---

## 🌐 Live Demo

👉 [Click here to try the app](https://ovarian-cancer-classifier.streamlit.app)

---

## 💡 Features

- 📁 Upload histopathology images
- 🔬 Predict ovarian cancer subtype (MC, EC, HGSC, LGSC, CC)
- 🧪 Estimate CA-125 levels (U/mL)
- 📊 Display severity category based on predicted CA-125
- 🧑‍⚕️ Enter patient name for personalized report
- 🌐 Fully browser-based via Streamlit Cloud

---

## 🚀 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Streamlit
- Google Drive (for model hosting)
- GitHub + Streamlit Cloud (for deployment)

---

## 🧠 Model Overview

The deep learning model used here is a custom multi-task Vision Transformer (ViT):

- **Classification Head** → Predicts 1 of 5 cancer types
- **Regression Head** → Estimates CA-125 levels
- Model size: ~300MB, hosted on Google Drive and downloaded at runtime using `gdown`

---

## 📁 Folder Structure
  ├── app.py # Streamlit app logic <br>
  ├── requirements.txt # All dependencies

## 🔧 Setup Instructions

### 1. Clone the repository
git clone https://github.com/your-username/ovarian-cancer-classifier.git
cd ovarian-cancer-classifier
### 2. Install dependencies
pip install -r requirements.txt
### 3. Run the app locally
streamlit run app.py

## Google Drive Integration
MODEL_ID = "1IF8oIcp1KCoC7w_-dju42eqpR4ZKksmB"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# License
> ⚠️ **License**: This project is proprietary. All rights reserved © 2025 Goutham.


