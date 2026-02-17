import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Step 1: Data Cleaning and Preprocessing
print("Load Dataset: \n")
trainingData = pd.read_csv("C:\\Users\\zas\\OneDrive\\سطح المكتب\\University\\2- Semester 2\\Machine Learning\\Project\\Project Machine\\Linear Regression - Sheet1.csv", encoding = 'latin1')
df = pd.DataFrame(trainingData)                                              # Convert the data into a DataFrame
print("Dataset loaded successfully.")
print("Preview of the data:\n", df)                                          # Preview of the data 
print("______________________________________________________________________________________________________________")

# Handle missing values and duplicates
print("Checking for missing values: \n")
print(df.isnull())                                                           # Check for missing values
print("Missing values per column before:\n", df.isnull().sum())             # Count missing values per column
print("Count of missing values:", df.isnull().sum().sum())                   # Total count of missing values
print("We don't have missing values to remove it.")
print("______________________________________________________________________________________________________________")

# Check for duplicate rows
print("Checking for duplicate rows: ")
print(df.duplicated())                                                       # Check for duplicate rows                                 
print("Number of duplicate rows:", df.duplicated().sum())                    # Check for No. of duplicate rows
print("We don't have duplicate rows to remove it.")
print("______________________________________________________________________________________________________________")

# Summary statistics
print("Summary statistics of the dataset: ")
print(df.describe())
print("______________________________________________________________________________________________________________")

# Data types and memory usage
print("Data types and memory usage: ")
df.info()
print("______________________________________________________________________________________________________________")

# Remove rows with "Not available" or NaN in the dataset
for column in df:
    df = df[df[column].notnull()]
print("Rows with 'Not available' or NaN removed.")
print("______________________________________________________________________________________________________________")

# # Remove outliers based only on 'y' column

column = 'Y'

Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1                     : {Q1}")
print(f"Q3                     : {Q3}")
print(f"IQR                    : {IQR}")
print(f"Lower Bound            : {lower_bound}")
print(f"Upper Bound            : {upper_bound}")

outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
print(f"Number of outliers in {column}: {outliers.shape[0]}")
print("We don't have outliers to remove it.")
print("______________________________________________________________________________________________________________") 

# Step 2: Exploratory Data Analysis

# Histograms for the dataset
plt.figure(figsize=(8, 4))
sns.histplot(df['Y'], kde=True)  
plt.title('Histogram of Y')
plt.xlabel('Y values')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\nHistograms have been created.")
print("______________________________________________________________________________________________________________")

# Z-score method to detect outliers based on the 'Y' column
print("Outlier detection using Z-score method for 'Y' column:")

valid_values   = df['Y'].dropna()                       # Drop missing values
z_scores       = np.abs(stats.zscore(valid_values))     # Calculate Z-scores
outliers       = z_scores > 3                           # Identify outliers
outlier_values = df.loc[valid_values.index[outliers]]   # Extract outlier values

print("Outliers in 'Y' column:", outlier_values[['Y']])
# print(outlier_values[['Y']])  
print("\n" + "-" * 100)
print("Outliers have been detected using Z-score method.")
print("______________________________________________________________________________________________________________")

# Step 3 : Modeling and Prediction

print("\nTraining model for: Y column")

target_column = 'Y'

X = df.drop(columns = [target_column]) 
y = df[target_column]  

X = pd.get_dummies(X, drop_first = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

param_grid_rf = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [10, 20], 
    'min_samples_split': [2, 5]
}

rf             = RandomForestRegressor(random_state = 42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid_search_rf.fit(X_train, y_train)

best_rf   = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print(f"Random Forest - RMSE    : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"MAE                     : {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"R²                      : {r2_score(y_test, y_pred_rf):.2f}")
print( "Best RF Params          :", grid_search_rf.best_params_)
print("\n" + "-" * 100)

lr        = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(  f"Linear Regression - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(  f"MAE                     : {mean_absolute_error(y_test, y_pred_lr):.2f}")
print(  f"R²                      : {r2_score(y_test, y_pred_lr):.2f}")
print("\n" + "-" * 100)

param_grid_svr = {
    'C'     : [1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma' : ['scale', 'auto']
}

svr = SVR()
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)

best_svr = grid_search_svr.best_estimator_
y_pred_svr = best_svr.predict(X_test)

print(  f"SVR - RMSE              : {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}")
print(  f"MAE                     : {mean_absolute_error(y_test, y_pred_svr):.2f}")
print(  f"R²                      : {r2_score(y_test, y_pred_svr):.2f}")
print(   "Best SVR Params         :", grid_search_svr.best_params_)

print("\nModel training and evaluation completed.")
print("Congratulations! The models have been successfully completed.")
print("______________________________________________________________________________________________________________")
