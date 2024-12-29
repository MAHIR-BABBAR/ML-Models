"""This is a project to classify positive and negative reviews.
Here i used some pre built python functions to analayse the data available at the below link
https://www.kaggle.com/datasets/anandshaw2001/chatgpt-users-reviews
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# You can change the data set according to your need
data = pd.read_csv('')

data = data.dropna(subset=["Review"]).reset_index(drop=True)

data = data[data["Ratings"].isin([1, 2, 4, 5])]
data["Sentiment"] = np.where(data["Ratings"] >= 4, 1, 0)

X = data["Review"]
y = data["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28, stratify=y)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(random_state=28, max_iter=5000)
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:f}%")
print("\nClassification Report:\n")
print(classification_rep)

#Extra feature added to check the predictions
def predict_sentiment(review_text):
    review_tfidf = vectorizer.transform([review_text])
    prediction = model.predict(review_tfidf)
    return "Positive" if prediction[0] == 1 else "Negative"


example_review = "Very nice"
predicted_sentiment = predict_sentiment(example_review)
print(f"Review: {example_review}\nPredicted Sentiment: {predicted_sentiment}")


