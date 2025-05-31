import os
os.environ["USE_TF"] = "0"

import torch
from transformers import BertTokenizer, BertConfig
from model_def import BertForClassification  # Make sure this matches your notebook's class
import os

MODEL_PATH = "./trained_models/classification_models_text_comments"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
config = BertConfig.from_pretrained(MODEL_PATH)
model = BertForClassification.from_pretrained(MODEL_PATH, config=config)
model.eval()

def predict(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=200, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()

    return "Rumor" if predicted_class == 1 else "Not Rumor"

