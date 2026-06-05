import pandas as pd
import re
import nltk
import joblib
from pathlib import Path
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Code/spam_email_dataset.csv", encoding = 'latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'text']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Delete Duplication
df = df.drop_duplicates().reset_index(drop = True)

print("Label Distribution:\n", df['label'].value_counts())

# Preprocessing
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Split FIRST
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# TF-IDF AFTER split
vectorizer = TfidfVectorizer(max_features = 3000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Model 1: Logistic Regression
lr = LogisticRegression(max_iter = 1000, class_weight = 'balanced')

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
lr_train_pred = lr.predict(X_train)

print("\n===== Logistic Regression =====")
print("Train Accuracy:", accuracy_score(y_train, lr_train_pred))
print("Test Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# Model 2: Naive Bayes
nb = MultinomialNB()

nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
nb_train_pred = nb.predict(X_train) 

print("\n===== Naive Bayes =====")
print("Train Accuracy:", accuracy_score(y_train, nb_train_pred))
print("Test Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# Model 3: SVM
svm = SVC(class_weight = 'balanced')

svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_train_pred = svm.predict(X_train)

print("\n===== SVM =====")
print("Train Accuracy:", accuracy_score(y_train, svm_train_pred))
print("Test Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

# Model 4: Decision Tree
dt = DecisionTreeClassifier(class_weight = 'balanced')

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_train_pred = dt.predict(X_train)

print("\n===== Decision Tree =====")
print("Train Accuracy:", accuracy_score(y_train, dt_train_pred))
print("Test Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))

# Model 5: Random Forest
rf = RandomForestClassifier(class_weight = 'balanced')

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_train_pred = rf.predict(X_train)

print("\n===== Random Forest =====")
print("Train Accuracy:", accuracy_score(y_train, rf_train_pred))
print("Test Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# 6. Save Models
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(lr, "lr.pkl")
joblib.dump(nb, "nb.pkl")
joblib.dump(svm, "svm.pkl")
joblib.dump(dt, "dt.pkl")
joblib.dump(rf, "rf.pkl")

print("\nModels saved successfully!")

# Prediction Function
def predict_new(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    results = {
        "Logistic Regression": lr.predict(vector)[0],
        "Naive Bayes": nb.predict(vector)[0],
        "SVM": svm.predict(vector)[0],
        "Decision Tree": dt.predict(vector)[0],
        "Random Forest": rf.predict(vector)[0],
    }

    return {k: ("Spam" if v == 1 else "Not Spam") for k, v in results.items()}
