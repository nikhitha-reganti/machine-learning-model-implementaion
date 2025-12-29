# CODTECH INTERNSHIP TASK - 4
# Machine Learning Model Implementation
# Spam Email Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "message": [
        "Win a free iPhone now",
        "Congratulations you won a lottery",
        "Call me tomorrow",
        "Let's meet for lunch",
        "Free entry in a contest",
        "Your bank account is credited",
        "Earn money from home",
        "Project meeting at 10am",
        "Get cheap loans instantly",
        "Are you coming today"
    ],
    "label": [1,1,0,0,1,0,1,0,1,0]
}

df = pd.DataFrame(data)

X = df["message"]
y = df["label"]

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

def predict_spam(message):
    msg_vector = vectorizer.transform([message])
    return "Spam" if model.predict(msg_vector)[0] == 1 else "Not Spam"

print("\nCustom Testing:")
print("Win free cash now →", predict_spam("Win free cash now"))
print("See you tomorrow →", predict_spam("See you tomorrow"))
