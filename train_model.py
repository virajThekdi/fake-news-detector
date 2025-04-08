import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Label the data
true_df['label'] = 1
fake_df['label'] = 0

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Features and labels
X = df['text']
y = df['label']

# Text to vector
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save model & vectorizer
joblib.dump(model, "model/fake_news_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
