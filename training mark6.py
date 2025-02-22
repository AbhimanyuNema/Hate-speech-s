import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv('labeled_data.csv')

# Preprocess the text data
corpus = df['tweet']
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(corpus)
y = df[['hate_speech', 'offensive_language', 'neither']]  # Use multiple columns as target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model for each category
models = {}
for category in y.columns:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train[category])
    models[category] = model

# Evaluate the models
train_accuracy = {}
test_accuracy = {}
for category, model in models.items():
    train_accuracy[category] = model.score(X_train, y_train[category])
    test_accuracy[category] = model.score(X_test, y_test[category])
    print(f"Training Accuracy ({category}):", train_accuracy[category])
    print(f"Testing Accuracy ({category}):", test_accuracy[category])

# Save the trained models
for category, model in models.items():
    with open(f'{category}_mark6.pkl', 'wb') as f:
        pickle.dump(model, f)

# Save the TfidfVectorizer
with open('cMark6.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
