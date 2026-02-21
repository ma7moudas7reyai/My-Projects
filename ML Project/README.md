# 📱 Mobile Price Prediction (Machine Learning Project)

## 📌 Project Overview
This project aims to build a machine learning model that predicts mobile phone launch prices in the USA based on technical specifications such as RAM and Company Name.

The project includes full data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and performance evaluation.

---

## 🎯 Objective
To develop a regression model capable of predicting the launched price of mobile phones in the USA using structured tabular data.

---

## 📊 Dataset Description
The dataset contains mobile phone information including:

- Company Name
- RAM (GB)
- Launched Price (USA)

The dataset required extensive cleaning due to:
- Non-numeric characters (e.g., currency symbols)
- Missing values
- Duplicate records
- Outliers

---

## 🧹 Data Preprocessing

### 1️⃣ Data Cleaning
- Removed non-numeric characters using regular expressions
- Converted price and RAM columns to numeric format
- Removed missing values
- Removed duplicate rows

### 2️⃣ Outlier Detection
- Used IQR method (Interquartile Range)
- Removed extreme values from RAM and Price columns

---

## 📈 Exploratory Data Analysis (EDA)

- Created scatter plots (RAM vs. Launched Price)
- Colored by Company Name
- Observed positive correlation between RAM and price
- Identified pricing patterns among different brands

---

## 🤖 Modeling Approach

### 🔹 Target Variable:
Launched Price (USA)

### 🔹 Algorithms Used:
- Random Forest Regressor

### 🔹 Feature Engineering:
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical features

### 🔹 Hyperparameter Tuning:
- GridSearchCV (5-fold Cross Validation)
- Tuned:
  - n_estimators
  - max_depth
  - min_samples_split

---

## 📏 Model Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

The Random Forest model achieved strong predictive performance after tuning.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Project Structure

- Data Cleaning
- EDA
- Feature Engineering
- Model Training
- Hyperparameter Optimization
- Model Evaluation

---

## 📚 What I Learned

- End-to-end ML pipeline development
- Data cleaning techniques
- Outlier detection methods
- Feature encoding & scaling
- Hyperparameter tuning using GridSearchCV
- Model evaluation and interpretation

---

## 👥 Team Members
- Mahmoud Ashrey
- Moaz Mohamed
- Sohila Abdelnasser
- Alyaa Hesham
- Jumana Hazim
- Fatma Elzahraa

Under supervision of Dr. Mayar Aly.

---

## 🚀 Future Improvements
- Add more features (Storage, Battery, Camera specs)
- Try advanced models (XGBoost, Gradient Boosting)
- Perform feature importance analysis
- Deploy model using Flask or Streamlit
