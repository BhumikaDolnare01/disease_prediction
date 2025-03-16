import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the rare disease dataset
file_path = r"C:\Users\ajitr\OneDrive\Desktop\geminai\rare_disease_dataset.json"  # Update this path if needed
with open(file_path, "r") as f:
    data = json.load(f)

# Extract symptoms (X) and disease names (y)
symptoms = [" ".join(d["symptoms"]) for d in data]  # Join symptoms as a single string
labels = [d["disease_name"] for d in data]  # Disease names as labels

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(symptoms)
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print(f"Predictions: {predictions[:10]}")  # Print first 10 predictions
print(f"True labels: {y_test[:10]}")  # Print first 10 true labels
  
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")