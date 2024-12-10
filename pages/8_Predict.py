from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import streamlit as st

# Load the pre-trained model and tokenizer
# model_name = '/mnt/c/Users/ASUS/Documents/Projects/data-sci/results/checkpoint-11500'
model_name = 'checkpoint-11500'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
model.to(device)

# Function to tokenize input texts
def tokenize_function(texts, max_length=512):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

# Sample texts for inference

text = st.text_input('Predict field from text')
text = text.strip()
new_texts = [text,]
new_encodings = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt').to(device)

# Convert predictions to labels
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

inverse_label_map = {v: k for k, v in label_map.items()}
display_labels = [inverse_label_map[i] for i in range(len(label_map))]
with torch.no_grad():
    outputs = model(**new_encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)
    pred_probs = torch.softmax(outputs.logits, dim=-1)
    for i, prediction in enumerate(predictions):
        print(f"Text: {new_texts[i]}, Predicted Label: {inverse_label_map[prediction.item()]}")
    # for label, prob in zip(display_labels, pred_probs[i]):
    #     print(f"Label: {label}, {prob*100:.2f}%")
    


predicted_labels = [display_labels[prediction] for prediction in predictions]
print(predicted_labels)
if text.strip() != '':
    st.write(predicted_labels[0])
