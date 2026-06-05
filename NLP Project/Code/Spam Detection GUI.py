import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent

# Theme (Clean Dark UI)
BG = "#0d1117"
CARD = "#161b22"
ACCENT = "#58a6ff"
TEXT = "#c9d1d9"
SUCCESS = "#3fb950"
ERROR = "#f85149"

# Load Stopwords
try:
    stopwords.words('english')
except:
    nltk.download('stopwords', quiet = True)

stop_words = set(stopwords.words('english'))

# Load Models
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

models = {
    "Logistic Regression": joblib.load(BASE_DIR / "lr.pkl"),
    "Naive Bayes": joblib.load(BASE_DIR / "nb.pkl"),
    "SVM": joblib.load(BASE_DIR / "svm.pkl"),
    "Decision Tree": joblib.load(BASE_DIR / "dt.pkl"),
    "Random Forest": joblib.load(BASE_DIR / "rf.pkl"),
}

# Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Prediction
def predict():
    user_text = input_box.get("1.0", tk.END).strip()

    if not user_text:
        messagebox.showwarning("Warning", "Please enter a message")
        return

    cleaned = clean_text(user_text)
    vector = vectorizer.transform([cleaned])

    results_text.delete("1.0", tk.END)

    votes = []

    for name, model in models.items():
        pred = model.predict(vector)[0]
        votes.append(pred)

        label = "SPAM" if pred == 1 else "NOT SPAM"
        symbol = "❌" if pred == 1 else "✅"

        results_text.insert(tk.END, f"{name}\n", "title")
        results_text.insert(tk.END, f"{label} {symbol}\n\n", "result")

    final = 1 if sum(votes) > len(votes) / 2 else 0
    final_label = "FINAL RESULT: SPAM ❌" if final else "FINAL RESULT: NOT SPAM ✅"

    results_text.insert(tk.END, "-------------------------\n", "divider")
    results_text.insert(tk.END, final_label, "final")


# GUI
root = tk.Tk()
root.title("Spam Detection AI")
root.geometry("650x600")
root.configure(bg = BG)

# Header
tk.Label(root, text = "Spam Detection", bg = BG, fg = ACCENT, font = ("Segoe UI", 20, "bold")).pack(pady = 15)

# Input Frame
input_frame = tk.Frame(root, bg = CARD)
input_frame.pack(padx = 20, pady = 10, fill = "x")

input_box = tk.Text(input_frame, height = 4, bg = CARD, fg = TEXT, insertbackground = "white", font = ("Segoe UI", 11), bd = 0, padx = 10, pady = 10)
input_box.pack(fill = "both")

# Button
btn = tk.Button(root, text = "Analyze", command = predict, bg = ACCENT, fg = "black", font = ("Segoe UI", 11, "bold"), bd = 0, padx = 15, pady = 5)
btn.pack(pady = 10)

# Results Box
results_frame = tk.Frame(root, bg = CARD)
results_frame.pack(padx = 20, pady = 10, fill = "both", expand = True)

results_text = tk.Text(results_frame, bg = CARD, fg = TEXT, font = ("Consolas", 11), bd = 0, padx = 10, pady = 10)
results_text.pack(fill = "both", expand = True)

# Styling text tags
results_text.tag_config("title", foreground = ACCENT, font = ("Segoe UI", 11, "bold"))
results_text.tag_config("result", foreground = TEXT)
results_text.tag_config("final", foreground = "white", font = ("Segoe UI", 12, "bold"))
results_text.tag_config("divider", foreground = "#555")

# Run
root.mainloop()