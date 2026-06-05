# import tkinter as tk
# from tkinter import messagebox
# import joblib
# import re
# import nltk
# from pathlib import Path
# from nltk.corpus import stopwords

# BASE_DIR = Path(__file__).resolve().parent

# # Load Stopwords
# try:
#     stopwords.words('english')
# except LookupError:
#     nltk.download('stopwords', quiet=True)
# stop_words = set(stopwords.words('english'))

# # Load Models
# vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")
# lr_model = joblib.load(BASE_DIR / "lr.pkl")
# nb_model = joblib.load(BASE_DIR / "nb.pkl")
# svm_model = joblib.load(BASE_DIR / "svm.pkl")
# dt_model = joblib.load(BASE_DIR / "dt.pkl")
# rf_model = joblib.load(BASE_DIR / "rf.pkl")

# # Cleaning Function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r'[^a-zA-Z]', ' ', text)

#     words = text.split()
#     words = [w for w in words if w not in stop_words]

#     return " ".join(words)

# # Prediction Function
# def predict():
#     text = input_box.get("1.0", tk.END).strip()

#     if text == "":
#         messagebox.showwarning("Empty Input", "Please enter a message first.")
#         return

#     cleaned = clean_text(text)
#     vector = vectorizer.transform([cleaned]).toarray()

#     results = [
#         lr_model.predict(vector)[0],
#         nb_model.predict(vector)[0],
#         svm_model.predict(vector)[0],
#         dt_model.predict(vector)[0],
#         rf_model.predict(vector)[0]
#     ]

#     labels = ["Spam" if r == 1 else "Not Spam" for r in results]

#     for i in range(5):
#         output_boxes[i].delete(0, tk.END)
#         output_boxes[i].insert(0, labels[i])

# # Paste Function 
# def paste_text(event = None):
#     try:
#         text = root.clipboard_get()
#         input_box.insert(tk.INSERT, text)
#     except:
#         pass

# # GUI Design
# root = tk.Tk()
# root.title("Spam Detection System")
# root.geometry("500x550")

# # Input Label
# tk.Label(root, text="Enter Message:", font=("Arial", 12)).pack()

# # Input Box
# input_box = tk.Text(root, height = 5, width = 50)
# input_box.pack()

# # Enable Ctrl+V
# input_box.bind("<Control-v>", paste_text)
# input_box.bind("<Control-V>", paste_text)

# # Right Click Menu
# menu = tk.Menu(root, tearoff = 0)
# menu.add_command(label = "Paste", command = paste_text)

# def show_menu(event):
#     menu.tk_popup(event.x_root, event.y_root)

# input_box.bind("<Button-3>", show_menu)

# # Button
# tk.Button(root, text = "Process", command = predict, bg = "lightblue").pack(pady = 10)

# # Output Boxes
# output_boxes = []

# models_names = [
#     "Logistic Regression",
#     "Naive Bayes",
#     "SVM",
#     "Decision Tree",
#     "Random Forest"
# ]

# for i in range(5):
#     tk.Label(root, text=models_names[i]).pack()
#     entry = tk.Entry(root, width=50)
#     entry.pack(pady=2)
#     output_boxes.append(entry)

# # Run App
# root.mainloop()

import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords

BASE_DIR = Path(__file__).resolve().parent

# =========================
# Load Stopwords
# =========================
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# =========================
# Load Models
# =========================
vectorizer = joblib.load(BASE_DIR / "vectorizer.pkl")

models = {
    "Logistic Regression": joblib.load(BASE_DIR / "lr.pkl"),
    "Naive Bayes": joblib.load(BASE_DIR / "nb.pkl"),
    "SVM": joblib.load(BASE_DIR / "svm.pkl"),
    "Decision Tree": joblib.load(BASE_DIR / "dt.pkl"),
    "Random Forest": joblib.load(BASE_DIR / "rf.pkl"),
}

# =========================
# Cleaning Function
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# =========================
# Prediction Function
# =========================
def predict():
    text = input_box.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning("Empty Input", "Please enter a message first.")
        return

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])   # ✔ FIX

    results = []
    votes = []

    for name, model in models.items():
        pred = model.predict(vector)[0]
        votes.append(pred)

        label = "Spam ❌" if pred == 1 else "Not Spam ✅"
        results.append((name, label, pred))

    # عرض النتائج
    for i, (name, label, pred) in enumerate(results):
        output_boxes[i].delete(0, tk.END)
        output_boxes[i].insert(0, label)

        if pred == 1:
            output_boxes[i].config(bg="#ff4d4d", fg="white")
        else:
            output_boxes[i].config(bg="#4CAF50", fg="white")

    # Final Decision (Voting)
    final = 1 if sum(votes) > len(votes)/2 else 0

    final_label = "FINAL → Spam ❌" if final == 1 else "FINAL → Not Spam ✅"

    final_box.delete(0, tk.END)
    final_box.insert(0, final_label)

    if final == 1:
        final_box.config(bg="black", fg="yellow")
    else:
        final_box.config(bg="black", fg="yellow")


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("Spam Detection System")
root.geometry("520x600")

# Title
tk.Label(root, text="Spam Detection System", font=("Arial", 16, "bold")).pack(pady=10)

# Input
tk.Label(root, text="Enter Message:", font=("Arial", 12)).pack()
input_box = tk.Text(root, height=5, width=55)
input_box.pack()

# Paste Support
def paste_text(event=None):
    try:
        text = root.clipboard_get()
        input_box.insert(tk.INSERT, text)
    except:
        pass

input_box.bind("<Control-v>", paste_text)
input_box.bind("<Control-V>", paste_text)

# Right Click Menu
menu = tk.Menu(root, tearoff=0)
menu.add_command(label="Paste", command=paste_text)

def show_menu(event):
    menu.tk_popup(event.x_root, event.y_root)

input_box.bind("<Button-3>", show_menu)

# Button
tk.Button(root, text="Process", command=predict, bg="lightblue").pack(pady=10)

# Output
output_boxes = []

for name in models.keys():
    tk.Label(root, text=name).pack()
    entry = tk.Entry(root, width=55)
    entry.pack(pady=2)
    output_boxes.append(entry)

# Final Result
tk.Label(root, text="Final Decision", font=("Arial", 12, "bold")).pack(pady=10)

final_box = tk.Entry(root, width=55)
final_box.pack(pady=5)

# Run
root.mainloop()