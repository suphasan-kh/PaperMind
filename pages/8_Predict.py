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
    
subject = ['AGRI - Agricultural and Biological Sciences', 'ARTS - Arts and Humanities', 'BIOC - Biochemistry, Genetics and Molecular Biology', 'BUSI - Business, Management and Accounting', 'CENG - Chemical Engineering', 'CHEM - Chemistry', 'COMP - Computer Science', 'DECI - Decision Sciences', 'DENT - Dentistry', 'EART - Earth and Planetary Sciences', 'ECON - Economics, Econometrics and Finance', 'ENER - Energy', 'ENGI - Engineering', 'ENVI - Environmental Science', 'HEAL - Health Professions', 'IMMU - Immunology and Microbiology', 'MATE - Materials Science', 'MATH - Mathematics', 'MEDI - Medicine', 'NEUR - Neuroscience', 'NURS - Nursing', 'PHAR - Pharmacology, Toxicology and Pharmaceutics', 'PHYS - Physics and Astronomy', 'PSYC - Psychology', 'SOCI - Social Sciences', 'VETE - Veterinary', 'MULT - Multidisciplinary']

predicted_labels = [display_labels[prediction] for prediction in predictions]

output = predicted_labels[0]
# if text.strip() != '':
#     for s in subject:
#         if text == s[0:4]:
#             st.write(predicted_labels[0])
#             break
if text.strip() != '':
    st.write(output)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data_preprocessed_1.csv')
df['texts'] = df['title'].fillna('') + " " + df['abstract'].fillna('') + " " + df['authkeywords'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['texts'])

def recommend_papers(input_text, tfidf_matrix, data, top_k=5):
    # Transform the input text using the same TF-IDF vectorizer
    input_vector = tfidf_vectorizer.transform([input_text])
    
    # Compute cosine similarity between the input text and all papers
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # Get the indices of the top_k most similar papers
    similar_indices = cosine_similarities.argsort()[-top_k:][::-1]
    
    # Retrieve the most similar papers
    similar_papers = data.iloc[similar_indices]
    
    return similar_papers

# Example usage
input_text = st.text_input('Get Suggestion',value='')
input_text = input_text.strip()
recommended_papers = recommend_papers(input_text, tfidf_matrix, df)

if input_text.strip() != '':
    st.write(recommended_papers[['title', 'abstract']])
