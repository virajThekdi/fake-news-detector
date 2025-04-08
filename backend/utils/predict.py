import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

import fitz  # PyMuPDF
# Get the directory of predict.py (which is inside utils/)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go two levels up: from utils -> backend -> news-detection (root)
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

# Correct path to model and vectorizer under the root directory
model_path = os.path.join(root_dir, 'model', 'fake_news_model.pkl')
vectorizer_path = os.path.join(root_dir, 'model', 'vectorizer.pkl')

# ✅ Load model and vectorizer using joblib
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_fake_news(text):
    x_input = vectorizer.transform([text])
    prediction = model.predict(x_input)
    return 'Fake' if prediction[0] == 0 else 'Real'

print("✅ Model path:", model_path)
print("✅ Vectorizer path:", vectorizer_path)

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text