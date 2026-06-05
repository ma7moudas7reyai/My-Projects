# NLP Project - Spam Detection

This project builds a spam detection system for email or message text using classic NLP preprocessing and machine learning models.

## What It Does

- Cleans text by lowercasing, removing links, numbers, special characters, and English stopwords.
- Converts cleaned messages into TF-IDF features.
- Trains and compares five classifiers:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine
  - Decision Tree
  - Random Forest
- Prints accuracy, classification reports, and confusion matrices.
- Saves trained models with `joblib`.
- Includes a Tkinter desktop GUI that lets the user enter a message and view each model's prediction plus a final voting result.

## Main Files

- `Code/Spam Detection.py` - training, evaluation, and model export.
- `Code/Spam Detection GUI.py` - desktop interface for testing messages.
- `Code/spam_email_dataset.csv` - dataset used for training.
- `Code/*.pkl` - saved vectorizer and trained models.

## Tech Stack

Python, pandas, NLTK, scikit-learn, joblib, Tkinter.

## How To Run

Install the required Python packages:

```bash
pip install pandas nltk scikit-learn joblib
```

Train and evaluate the models:

```bash
python "Code/Spam Detection.py"
```

Run the GUI:

```bash
python "Code/Spam Detection GUI.py"
```
