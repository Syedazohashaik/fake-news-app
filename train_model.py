import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load datasets - replace with correct local paths or URLs
df_fake = pd.read_csv("data/Fake.csv")
df_real = pd.read_csv("data/True.csv")

# Add labels
df_fake["label"] = 0
df_real["label"] = 1

# Check data loaded correctly
print("Fake dataset shape:", df_fake.shape)
print("Real dataset shape:", df_real.shape)

# Combine datasets
df = pd.concat([df_fake, df_real])

print("Combined dataset shape:", df.shape)
print("Label distribution:\n", df['label'].value_counts())

# Features and labels
X = df["text"]
y = df["label"]

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = tfidf_vectorizer.fit_transform(X)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, stratify=y, random_state=42
)

# Train Passive Aggressive Classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train, y_train)

# Predict on test set
y_pred = pac.predict(X_test)

# Evaluate accuracy
score = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {score * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print("Confusion Matrix:\n", cm)

# Save model and vectorizer
with open("model.pkl", "wb") as model_file:
    pickle.dump(pac, model_file)
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf_vectorizer, vec_file)

print("Model and vectorizer saved successfully.")
