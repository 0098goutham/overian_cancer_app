# app.py

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import gdown  # New import

# Streamlit page config
st.set_page_config(page_title="Histopathology ViT Classifier", layout="centered")

# Labels
labels = ["MC", "EC", "HGSC", "LGSC", "CC"]

# CA-125 classifier
def classify_ca125(ca125_pred):
    if ca125_pred < 35:
        return "<35 U/mL", "Normal or Mild"
    elif 35 <= ca125_pred <= 150:
        return "35–150 U/mL", "Mild to Moderate"
    elif 20 <= ca125_pred <= 300:
        return "20–300 U/mL", "Mild to Moderate"
    elif 50 <= ca125_pred <= 300:
        return "50–300 U/mL", "Moderate"
    elif ca125_pred > 100:
        return "100–5000+ U/mL", "High to Extremely High"
    return "Ambiguous", "Unclassified"

# Custom ViT
class MultiTaskViT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.vit = base_model
        self.class_head = nn.Linear(768, 5)
        self.reg_head = nn.Linear(768, 1)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        cls_token = outputs.hidden_states[-1][:, 0, :]
        class_logits = self.class_head(cls_token)
        reg_output = self.reg_head(cls_token)
        return class_logits, reg_output

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Google Drive model setup
MODEL_ID = "1IF8oIcp1KCoC7w_-dju42eqpR4ZKksmB"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "histo_ViT_1.pt"


def download_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model_from_gdrive()
    base_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = MultiTaskViT(base_model)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Sidebar
st.sidebar.title("Patient Information")
patient_name = st.sidebar.text_input("Enter Patient Name")
uploaded_file = st.sidebar.file_uploader("Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file and patient_name:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        # For display
        img_np = img_tensor.squeeze().numpy().transpose(1, 2, 0)
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        model = load_model()
        with torch.no_grad():
            class_logits, reg_output = model(img_tensor)

            predicted_idx = torch.argmax(class_logits, dim=1).item()
            predicted_label = labels[predicted_idx]
            ca125_pred = np.exp(reg_output.squeeze().numpy())
            range_str, class_str = classify_ca125(ca125_pred)

            prediction_text = (
                f"👤 **Patient Name**: {patient_name}\n\n"
                f"🧬 **Cancer Type**: {predicted_label} (Pos: {predicted_idx})\n"
                f"🧪 **CA-125 Level**: {ca125_pred:.2f} U/mL\n"
                f"📊 **Range**: {range_str}\n"
                f"🚨 **Severity**: {class_str}"
            )

            st.image(img_np, caption="Uploaded Histopathology Image", use_container_width=True)
            st.markdown(prediction_text)

    except Exception as e:
        st.error(f"Error processing image: {e}")

elif uploaded_file and not patient_name:
    st.warning("Please enter the patient's name.")
elif patient_name and not uploaded_file:
    st.warning("Please upload a histopathology image.")
else:
    st.info("Please enter patient name and upload an image to get prediction.")
