import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns
from scipy                   import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.svm             import SVR
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Data Cleaning and Preprocessing
print("Load Dataset: \n")
trainingData = pd.read_csv("D:\\University\\2- Semester 2\\Machine Learning\\Project\\Machine Project\\Mobiles Dataset (2025).csv", encoding = 'latin1')
df           = pd.DataFrame(trainingData)                                    # Convert the data into a DataFrame
print("Dataset loaded successfully.")
print("Preview of the data:\n", df)                                          # Preview of the data 
print("______________________________________________________________________________________________________________")

# Remove symbols and convert to numeric (excluding certain columns)
exclude_cols = ['Company Name']                                              # Columns to exclude from cleaning
for col in df.columns:
    if col not in exclude_cols and df[col].dtype == 'object':                # Check if the column is not in the exclusion list and is of type object
        df[col] = df[col].str.replace('[^0-9.]', '', regex = True)           # Remove non-numeric characters
        df[col] = pd.to_numeric(df[col], errors = 'coerce')                  # Convert to numeric, coercing errors to NaN

# Handle missing values and duplicates
print("Checking for missing values:\n")
print(df.isnull().sum())
print(f"Total missing values: {df.isnull().sum().sum()}")
df = df.dropna().reset_index(drop = True)
print("Missing values removed and index reset.")
print("______________________________________________________________________________________________________________")

# Check for duplicate rows
print("Checking for duplicate rows:\n")
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df = df.drop_duplicates().reset_index(drop = True)
print("Duplicates removed and index reset.")
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
for column in ['RAM', 'Launched Price (USA)']:
    df = df[df[column].notnull()]

df = df.reset_index(drop = True)
print("Specific columns cleaned from missing values.")
print("______________________________________________________________________________________________________________")

# Remove outliers using IQR method
for column in ['RAM', 'Launched Price (USA)']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

df = df.reset_index(drop = True)
print("Outliers removed using IQR method.")
print("______________________________________________________________________________________________________________")

# Step 2: Exploratory Data Analysis (EDA)

# Filter data for valid RAM values
valid_rams = [4, 6, 8, 12]  
filtered_df = df[df['RAM'].isin(valid_rams)].reset_index(drop = True)

# Plot RAM vs. Launched Price (USA)
plt.figure     (figsize = (8, 5))
sns.scatterplot(
    x       = 'RAM',
    y       = 'Launched Price (USA)',
    data    = filtered_df,
    hue     = 'Company Name',
    palette = 'viridis',
    s       = 100,
    alpha   = 0.8
)

plt.title ('RAM vs. Launched Price (USA)', fontsize = 16)
plt.xlabel('RAM (GB)', fontsize = 14)
plt.ylabel('Launched Price (USA)', fontsize = 14)
plt.grid  (True, linestyle = '--', linewidth = 0.5)
plt.xticks(valid_rams)
plt.legend(
    title          = 'Company Name',
    bbox_to_anchor = (1.05, 1),
    loc            = 'upper left',
    fontsize       = 10
)
plt.tight_layout()
plt.show()

print("Scatter Plot has been created for RAM vs. Launched Price (USA).")
print("______________________________________________________________________________________________________________")

# Step 3 : Modeling and Prediction for Launched Price (USA)

# Since we're only working with the USA price column, we only need to train the model for the USA
target_country = 'USA'
print(f"\nTraining model for: {target_country}")

target_column = f'Launched Price ({target_country})'

# Drop the target column and columns related to other countries
X = df.drop(columns = [target_column])
y = df[target_column]

X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Standardize the data (scaling)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Random Forest Regressor Model
param_grid_rf = {
    'n_estimators'     : [100, 200],
    'max_depth'        : [10, 20],
    'min_samples_split': [2, 5]
}

# Initialize the Random Forest model
rf = RandomForestRegressor(random_state = 42)

# Perform Grid Search for hyperparameter tuning
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
grid_search_rf.fit(X_train, y_train)

# Best model after tuning
best_rf = grid_search_rf.best_estimator_

# Predict using the best Random Forest model
y_pred_rf = best_rf.predict(X_test)

# Evaluate the model's performance
print(  f"Random Forest - RMSE    : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(  f"MAE                     : {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(  f"R²                      : {r2_score(y_test, y_pred_rf):.2f}")
print(   "Best RF Params          :", grid_search_rf.best_params_)

print("\nModel training and evaluation completed.")
print("The Random Forest model has been successfully completed.")
print("______________________________________________________________________________________________________________")

print("Congratulations! The project has been successfully completed.")
