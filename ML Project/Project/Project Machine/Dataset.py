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
trainingData = pd.read_csv("C:\\Users\\zas\\OneDrive\\سطح المكتب\\University\\2- Semester 2\\Machine Learning\\Project\\Project Machine\\Mobiles Dataset (2025).csv", encoding = 'latin1')
df = pd.DataFrame(trainingData)                                              # Convert the data into a DataFrame
print("Dataset loaded successfully.")
print("Preview of the data:\n", df)                                          # Preview of the data 
print("______________________________________________________________________________________________________________")

# Remove symbols and convert to numeric (excluding certain columns)
exclude_cols = ['Company Name', 'Model Name', 'Processor']                   # Columns to exclude from cleaning
for col in df.columns:
    if col not in exclude_cols and df[col].dtype == 'object':                # Check if the column is not in the exclusion list and is of type object
        df[col] = df[col].str.replace('[^0-9.]', '', regex=True)             # Remove non-numeric characters
        df[col] = pd.to_numeric(df[col], errors = 'coerce')                  # Convert to numeric, coercing errors to NaN

# Handle missing values and duplicates
print("Checking for missing values: \n")
print(df.isnull())                                                           # Check for missing values
print("Missing values per column before: \n", df.isnull().sum())             # Count missing values per column
print("Count of missing values:", df.isnull().sum().sum())                   # Total count of missing values
df = df.dropna()                                                             # Drop rows with missing values
print("Missing values removed.")
print("Missing values per column after dropping: ", df.isnull().sum().sum()) # Check for missing values again
print("______________________________________________________________________________________________________________")

# Check for duplicate rows
print("Checking for duplicate rows: ")
print(df.duplicated())                                                       # Check for duplicate rows                                 
print("Number of duplicate rows:", df.duplicated().sum())                    # Check for No. of duplicate rows
df = df.drop_duplicates().reset_index(drop = True)                           # Drop duplicate rows and reset index
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

# Clean 'Company Name' column: remove whitespace and standardize brand names
df['Company Name'] = df['Company Name'].str.strip()                          # Remove leading/trailing whitespace
df['Company Name'] = df['Company Name'].replace({                            # Standardize brand names
                        'apple':   'Apple',
                        'samsung': 'Samsung',
                        'oneplus': 'OnePlus',
                        'vivo':    'Vivo',   
                        'iqoo':    'iQOO',
                        'oppo':    'Oppo',
                        'realme':  'Realme',
                        'xiaomi':  'Xiaomi',
                        'lenovo':  'Lenovo',
                        'motorola':'Motorola',
                        'huawei':  'Huawei',
                        'nokia':   'Nokia',
                        'sony':    'Sony',
                        'google':  'Google',
                        'tecno':   'Tecno',
                        'infinix': 'Infinix',
                        'honor':   'Honor',
                        'poco':    'Poco',
})
print("Company names standardized.")
print("______________________________________________________________________________________________________________")

# Clean and convert price columns to numeric values
price_columns = [
    'Launched Price (Pakistan)',
    'Launched Price (India)',
    'Launched Price (China)',
    'Launched Price (USA)',
    'Launched Price (Dubai)'
]
countries = [
    'Pakistan',
    'India', 
    'China', 
    'USA', 
    'Dubai'
]
features = [
    'RAM', 
    'Front Camera', 
    'Back Camera', 
    'Mobile Weight', 
    'Screen Size', 
    'Battery Capacity'
]

# Remove rows with "Not available" or NaN in any of the price columns
for column in price_columns:
    df = df[df[column].notnull()]
print("Price columns cleaned.")
print("______________________________________________________________________________________________________________")

# Display the cleaned price columns
print("Cleaned price columns: ")
pd.set_option('display.max_columns', None)                                   # Show all columns
print(df[price_columns])                                                     # Display the cleaned price columns
print("______________________________________________________________________________________________________________")

# Remove outliers using IQR method
all_numeric_cols = [
    'Mobile Weight',
    'RAM',
    'Front Camera',
    'Back Camera',
    'Battery Capacity',
    'Screen Size',
    'Launched Price (Pakistan)',
    'Launched Price (India)',
    'Launched Price (China)',
    'Launched Price (USA)',
    'Launched Price (Dubai)'
]

for column in all_numeric_cols:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    print(f"Number of outliers in {column}: {df.shape[0]}")

print("Outliers removed.")
print("______________________________________________________________________________________________________________")

# Display the final cleaned all numeric columns after removing outliers
print("Final cleaned price columns after removing outliers: ")
print(df[all_numeric_cols])
print("______________________________________________________________________________________________________________")



# Step 2: Exploratory Data Analysis (EDA)

# Scatter plot for RAM vs. Launched Price in Different Countries
validRams   = [4, 6, 8, 12]                                                   # Valid RAM values  
filtered_df = df[df['RAM'].isin(validRams)]                                  # Filter the DataFrame based on valid RAM values

# Plot RAM vs. Launched Price in Different Countries
for idx, (col, country) in enumerate(zip(price_columns, countries)):
    country_column = f'Launched Price ({country})'
    country_df     = filtered_df[filtered_df[country_column] > 0]

    plt.figure(figsize = (8, 4))

    sns.scatterplot(
        x       = 'RAM',
        y       = f'Launched Price ({country})',
        data    = filtered_df,
        hue     = 'Company Name',
        palette = 'viridis',
        s       = 100,
        alpha   = 0.7
    )

    plt.title (f'RAM vs Price in {country}', fontsize = 14)
    plt.xlabel('RAM (GB)',                   fontsize = 12)
    plt.ylabel(f'Price in ({country})',      fontsize = 12)
    plt.grid  (True, linestyle = '--', linewidth = 0.5)
    plt.xticks(validRams)
    plt.legend(
        title          = 'Company Name', 
        loc            = 'upper left',  
        fontsize       = 10, 
        bbox_to_anchor = (1.05, 1)  
    )
    plt.tight_layout()
    plt.show()
print("Scatter Plot has been created for RAM vs. Launched Price in Different Countries.")
print("______________________________________________________________________________________________________________")


# Box plot for Launched Price in Pakistan

for idx, (col, country) in enumerate(zip(price_columns, countries)):
    country_column = f'Launched Price ({country})'
    country_df     = filtered_df[filtered_df[country_column] > 0]

    plt.figure (figsize = (8, 4))
    sns.boxplot(x = country_df[country_column])
    plt.title  (f'Box Plot of Launched Price in {country}')
    plt.xlabel (f'Launched Price {country}')
    plt.grid   (True, linestyle = '--', linewidth = 0.5)
    plt.show()

print("Box Plot has been created for Launched Price in Pakistan.")
print("______________________________________________________________________________________________________________")

# Z-score method to detect outliers
print("Outlier detection using Z-score method:")
for country in countries:
    country_column = f'Launched Price ({country})'
    valid_values   = df[country_column].dropna()

    # Calculate Z-scores
    z_scores       = np.abs(stats.zscore(valid_values)) 
    outliers       = z_scores > 3
    outlier_values = df.loc[valid_values.index[outliers]]
    print(f"Outliers in {country}:")
    print(outlier_values[['Model Name', 'Company Name', 'RAM', country_column]])
    print("\n" + "-"*100)

print("Outliers have been detected using Z-score method.")
print("______________________________________________________________________________________________________________")

# Correlation Heatmap
correlation_matrix = df.corr(numeric_only = True) 

print(correlation_matrix)

plt.figure(figsize = (12, 8))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
plt.title('Correlation Heatmap Between Numeric Features')
plt.xticks(rotation = 45)
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

print("\nCorrelation Heatmap has been created.")
print("______________________________________________________________________________________________________________")

for col in price_columns:
    print(f"Value Counts for {col}:")
    print(df[col].value_counts())
    print("\n" + "-" * 100)
print("______________________________________________________________________________________________________________")

for col in price_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde = True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    plt.show()

print(df['Company Name'].value_counts(normalize = True))

print("\nHistograms have been created.")
print("______________________________________________________________________________________________________________")

# Step 3 :  Modeling and Prediction

countries = ['Pakistan', 'India', 'China', 'USA', 'Dubai']

for target_country in countries:
    print(f"\nTraining model for: {target_country}")

    
    target_column = f'Launched Price ({target_country})'

    price_columns_to_remove = [col for col in price_columns if col != target_column]
    X = df.drop(columns = price_columns_to_remove + [target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    param_grid_rf = {
        'n_estimators':      [100, 200],
        'max_depth':         [10, 20], 
        'min_samples_split': [2, 5]
    }

    rf             = RandomForestRegressor(random_state = 42)
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
    grid_search_rf.fit(X_train, y_train)

    best_rf   = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    print(  f"Random Forest - RMSE    : {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}, "
            f"MAE                     : {mean_absolute_error(y_test, y_pred_rf):.2f}, "
            f"R²                      : {r2_score(y_test, y_pred_rf):.2f}")
    print(   "Best RF Params          :", grid_search_rf.best_params_)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print(  f"Linear Regression - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}, "
            f"MAE                     : {mean_absolute_error(y_test, y_pred_lr):.2f}, "
            f"R²                      : {r2_score(y_test, y_pred_lr):.2f}")

    param_grid_svr = {
        'C'     : [1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma' : ['scale', 'auto']
    }

    svr = SVR()
    grid_search_svr = GridSearchCV(svr, param_grid_svr, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
    grid_search_svr.fit(X_train, y_train)

    best_svr   = grid_search_svr.best_estimator_
    y_pred_svr = best_svr.predict(X_test)

    print(  f"SVR - RMSE              : {np.sqrt(mean_squared_error(y_test, y_pred_svr)):.2f}, "
            f"MAE                     : {mean_absolute_error(y_test, y_pred_svr):.2f}, "
            f"R²                      : {r2_score(y_test, y_pred_svr):.2f}")
    print(   "Best SVR Params         :", grid_search_svr.best_params_)

print("\nModel training and evaluation completed for all countries.")
print("Congratulations! The models have been successfully complated.")
print("______________________________________________________________________________________________________________") 













# # Split the data into training and testing sets
# for country in countries:
#     print(f"\nTraining model for: {country}")

#     price_col = f'Launched Price ({country})'
#     if price_col not in df.columns:
#         print(f"[❌] Skipping {country}: column {price_col} not found.")
#         continue

#     X = df.drop(columns=[col for col in df.columns if 'Launched Price' in col])
#     y = df[price_col]

#     data = pd.concat([X, y], axis=1).dropna(subset=[price_col])
#     X = data.drop(columns=[price_col])
#     y = data[price_col]

#     X = pd.get_dummies(X, drop_first=True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     param_grid_rf = {
#         'n_estimators': [100],
#         'max_depth': [20],
#         'min_samples_split': [2]
#     }

#     rf = RandomForestRegressor(random_state=42)
#     grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
#     grid_search_rf.fit(X_train, y_train)

#     best_rf = grid_search_rf.best_estimator_
#     y_pred = best_rf.predict(X_test)

#     print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
#     print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
#     print(f"R²: {r2_score(y_test, y_pred):.2f}")
    
#     X = df.drop(f'Launched Price (Pakistan)', axis=1)
#     y = df.drop([f'Launched Price ({c})' for c in countries], axis=1)
#     X = pd.get_dummies(X, drop_first = True)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

# param_grid_rf = {
#     'n_estimators':      [50, 100, 200],
#     'max_depth':         [10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }


# rf = RandomForestRegressor(random_state = 42)
# grid_search_rf = GridSearchCV(estimator = rf, param_grid = param_grid_rf, cv = 5, n_jobs = -1, scoring = 'neg_mean_squared_error')
# grid_search_rf.fit(X_train, y_train)
# print(f"Best parameters for RandomForest: {grid_search_rf.best_params_}")

# best_rf = grid_search_rf.best_estimator_
# y_pred_rf = best_rf.predict(X_test)

# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)

# print(f"Random Forest Regressor - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.2f}")


# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# mae_lr = mean_absolute_error(y_test, y_pred_lr)
# r2_lr = r2_score(y_test, y_pred_lr)

# print(f"Linear Regression - RMSE: {rmse_lr:.2f}, MAE: {mae_lr:.2f}, R²: {r2_lr:.2f}")

# param_grid_svr = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'poly', 'rbf'],
#     'gamma': ['scale', 'auto']
# }

# svr = SVR()
# grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
# grid_search_svr.fit(X_train, y_train)
# print(f"Best parameters for SVR: {grid_search_svr.best_params_}")

# best_svr = grid_search_svr.best_estimator_
# y_pred_svr = best_svr.predict(X_test)

# rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
# mae_svr = mean_absolute_error(y_test, y_pred_svr)
# r2_svr = r2_score(y_test, y_pred_svr)

# print(f"Support Vector Regression - RMSE: {rmse_svr:.2f}, MAE: {mae_svr:.2f}, R²: {r2_svr:.2f}")

# print("______________________________________________________________________________________________________________")





































# rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)




# print(f"Random Forest Regressor - RMSE: {rmse_rf:.2f}, MAE: {mae_rf:.2f}, R²: {r2_rf:.2f}")

# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
# rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
# r2_lr = r2_score(y_test, y_pred_lr)

# best_svr = grid_search_svr.best_estimator_
# y_pred_svr = best_svr.predict(X_test)
# rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
# r2_svr = r2_score(y_test, y_pred_svr)
# print("hello")
# print(f"Linear Regression - RMSE: {rmse_lr:.2f}, R²: {r2_lr:.2f}")
# print(f"Support Vector Regression - RMSE: {rmse_svr:.2f}, R²: {r2_svr:.2f}")

# for name, model in models.items():
#       model.fit(X_train, y_train)  
#       y_pred = model.predict(X_test) 

#       mse = mean_squared_error(y_test, y_pred)
#       rmse = mse ** 0.5
#       r2 = r2_score(y_test, y_pred)
#       print(f"{name}:")
#       print(f"   RMSE: {rmse:.2f}")
#       print(f"   R²: {r2:.2f}")
#       print("-" * 50)



# models = {
#       "Linear Regression": LinearRegression(),
#       "Random Forest Regression": RandomForestRegressor(),
#       "Support Vector Regression": SVR()
# }





# plt.title ('RAM vs Launched Price (Pakistan)', fontsize = 16)
# plt.xlabel('RAM (GB)', fontsize = 14)
# plt.ylabel('Launched Price in Pakistan (PKR)', fontsize = 14)
# plt.xticks(validRams, fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.grid  (True, linestyle = '--', linewidth = 0.5)
# plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left', fontsize = 9)
# plt.tight_layout()
# plt.show()




# currency_prefix = {
#     'Launched Price (Pakistan)': 'PKR',
#     'Launched Price (India)':    'INR',
#     'Launched Price (China)':    'CNY',
#     'Launched Price (USA)':      'USD',
#     'Launched Price (Dubai)':    'AED'
# }



# for column in price_columns:
#       df[column] = df[column].str.replace(currency_prefix[column], '', regex=False)
#       df[column] = df[column].str.replace(',', '', regex=False)
#       df[column] = pd.to_numeric(df[column], errors='coerce')





# ==============================
# Point 2: Box Plot: Price Distribution
# ==============================

# df['Z-score'] = stats.zscore(
#       df['Launched Price (Pakistan)'],
#       df['Launched Price (India)'],
#       df['Launched Price (China)'],
#       df['Launched Price (USA)'],
#       df['Launched Price (Dubai)'])

# outliers = df[df['Z-score'].abs() > 3]
# print("               Outliers detected using Z-score method: ")
# print(outliers)



# # Scatter plot for each country
# sns.scatterplot(x = 'Launched Price (Pakistan)', y = 'RAM', data=df, label='Pakistan', alpha=0.7)
# sns.scatterplot(x = 'Launched Price (India)',    y = 'RAM', data=df, label='India',    alpha=0.7)
# sns.scatterplot(x = 'Launched Price (China)',    y = 'RAM', data=df, label='China',    alpha=0.7)
# sns.scatterplot(x = 'Launched Price (USA)',      y = 'RAM', data=df, label='USA',      alpha=0.7)
# sns.scatterplot(x = 'Launched Price (Dubai)',    y = 'RAM', data=df, label='Dubai',    alpha=0.7)

# plt.title('Comparison of RAM vs. Launched Price Across Countries')
# plt.xlabel('Launched Price')
# plt.ylabel('RAM (GB)')
# plt.legend()
# plt.grid(True)
# plt.show()



# Draw box plots for each price column to visualize the distribution and identify outliers
# price_columns = [
#       'Launched Price (Pakistan)',
#       'Launched Price (India)',
#       'Launched Price (China)',
#       'Launched Price (USA)',
#       'Launched Price (Dubai)'
# ]

# plt.figure(figsize=(12, 5))
# sns.boxplot(data=df[price_columns])
# plt.title('Box Plot of Launched Prices in Different Countries')
# plt.xlabel('Country')
# plt.ylabel('Launched Price')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#       x='RAM', 
#       y='Launched Price (India)', 
#       data=df, 
#       color='darkblue', 
#       alpha=0.7
# )

# plt.title('RAM vs Launched Price (India)')
# plt.xlabel('RAM (GB)')
# plt.ylabel('Launched Price in India (INR)')
# plt.grid(True)
# plt.show()













# # STEP 2 : Exploratory Data Analysis (EDA)

# print("*** STEP 8: Plot Histogram of Prices ***")

# # Data visualization: Draw histograms for each price column with KDE (Kernel Density Estimate) overlay
# # Import necessary libraries for image classification
# plt.figure(figsize=(10, 6))

# # Draw histograms for each price column with KDE (Kernel Density Estimate) overlay
# sns.histplot(df['Launched Price (Pakistan)'], kde=True, bins=30, label = 'Pakistan', color='blue',   alpha=0.5)
# sns.histplot(df['Launched Price (India)'],    kde=True, bins=30, label = 'India',    color='green',  alpha=0.5)
# sns.histplot(df['Launched Price (China)'],    kde=True, bins=30, label = 'China',    color='red',    alpha=0.5)
# sns.histplot(df['Launched Price (USA)'],      kde=True, bins=30, label = 'USA',      color='orange', alpha=0.5)
# sns.histplot(df['Launched Price (Dubai)'],    kde=True, bins=30, label = 'Dubai',    color='purple', alpha=0.5)

# # Set the title and labels for the Histogram
# plt.title('Comparison of Launched Prices in Different Countries')
# plt.xlabel('Launched Price')
# plt.ylabel('RAM')

# plt.legend()
# plt.grid(True)
# plt.show()








# ram_values = sorted(filtered_df['RAM'].unique())
      # plt.xticks(ram_values)

      
      # plt.ticklabel_format(style='plain', axis='y')

      # plt.xlim(min(ram_values)-1, max(ram_values)+1)
      # plt.ylim(0, filtered_df[price_col].max()*1.1)

      # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)









# import pandas as pd
# import numpy as np

# # تحميل البيانات من ملف CSV
# trainingData = pd.read_csv("C:\\Users\\zas\\OneDrive\\سطح المكتب\\University\\2- Semester 2\\Machine Learning\\Project\\Dataset Classification Cats & Dogs\Mobiles Dataset (2025).csv", encoding = 'latin1')

# # تحويل البيانات إلى DataFrame
# df = pd.DataFrame(trainingData)
# print(df)                # هنا بيطبع البيانات كلها على شكل جدول صفوف واعمدة
# print("______________________________________________________________________________________________________________")


# print(df.isnull())       # هنا بيفحص الجدول هل فى قيم مفقودة او لا عن طريق لو الخلية فيها قيمة مفقودة يكتب True و False لو العكس
# print("______________________________________________________________________________________________________________")

# print(df.isnull().sum()) # هنا بيقولى عدد القيم المفقودة فى كل عمود
# print("______________________________________________________________________________________________________________")

# print(df.describe())     # هنا بيقولى معلومات عن البيانات اللى موجودة فى الجدول زى عدد الصفوف وعدد الاعمدة ونوع البيانات الموجودة
# print("______________________________________________________________________________________________________________")

# print(df.info())         # هنا بيقولى معلومات عن البيانات اللى موجودة فى الجدول زى عدد الصفوف وعدد الاعمدة ونوع البيانات الموجودة
# print("______________________________________________________________________________________________________________")

# #هنا بنشوف لو فى بيانات متكررة ونحذفها
# print(df.duplicated())                              # هنا بيفحص الجدول هل فيه بيانات مكررة او لا عن طريق لو الخلية فيها قيمة مكررة يكتب True و False لو العكس
# print(df.duplicated().sum())                        # هنا بيقولى عدد القيم المكررة فى كل عمود
# df = df.drop_duplicates().reset_index(drop = True)  # هنا بيحذف الصفوف المكررة ويعيد ترتيب الفهرس
# print("______________________________________________________________________________________________________________")

# df['Company Name'] = df['Company Name'].str.strip() # هنا بنشيل المسافات الزيادة من اسم الشركة
# df['Company Name'] = df['Company Name'].replace({   
#             'apple' : 'Apple',
#             'samsung' : 'Samsung',
#             'oneplus' : 'OnePlus',
#             'vivo' : 'Vivo',
#             'iqoo' : 'iQOO',
#             'oppo' : 'Oppo',
#             'realme' : 'Realme',
#             'xiaomi' : 'Xiaomi',
#             'lenovo' : 'Lenovo',
#             'motorola' : 'Motorola',
#             'huawei' : 'Huawei',
#             'nokia' : 'Nokia',
#             'sony' : 'Sony',
#             'google' : 'Google',
#             'tecno' : 'Tecno',
#             'infinix' : 'Infinix',
#             'honor' : 'Honor',
#             'poco' : 'Poco',
#       }
# )

# # هنا بيحذف النصوص الموجودة فى القيم ويخليها ارقام 
# df['Launched Price (Pakistan)'] = df['Launched Price (Pakistan)'].str.replace('PKR', '', regex=False)
# df['Launched Price (Pakistan)'] = df['Launched Price (Pakistan)'].str.replace(',', '', regex=False)
# df['Launched Price (Pakistan)'] = pd.to_numeric(df['Launched Price (Pakistan)'], errors='coerce')

# df['Launched Price (India)']    = df['Launched Price (India)'].str.replace('INR', '', regex=False)
# df['Launched Price (India)']    = df['Launched Price (India)'].str.replace(',', '', regex=False)
# df['Launched Price (India)']    = pd.to_numeric(df['Launched Price (India)'], errors='coerce')

# df['Launched Price (China)']    = df['Launched Price (China)'].str.replace('CNY', '', regex=False)
# df['Launched Price (China)']    = df['Launched Price (China)'].str.replace(',', '', regex=False)
# df['Launched Price (China)']    = pd.to_numeric(df['Launched Price (China)'], errors='coerce')

# df['Launched Price (USA)']      = df['Launched Price (USA)'].str.replace('USD', '', regex=False)
# df['Launched Price (USA)']      = df['Launched Price (USA)'].str.replace(',', '', regex=False)
# df['Launched Price (USA)']      = pd.to_numeric(df['Launched Price (USA)'], errors='coerce')

# df['Launched Price (Dubai)']    = df['Launched Price (Dubai)'].str.replace('AED', '', regex=False)
# df['Launched Price (Dubai)']    = df['Launched Price (Dubai)'].str.replace(',', '', regex=False)
# df['Launched Price (Dubai)']    = pd.to_numeric(df['Launched Price (Dubai)'], errors='coerce')

# # هنا بنشيل القيم اللى فيها Not available
# df = df[df['Launched Price (Pakistan)'] != 'Not available']
# df = df[df['Launched Price (India)'] != 'Not available']
# df = df[df['Launched Price (China)'] != 'Not available']
# df = df[df['Launched Price (USA)'] != 'Not available']
# df = df[df['Launched Price (Dubai)'] != 'Not available']

# print(df['Launched Price (Pakistan)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (India)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (China)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (USA)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (Dubai)'])
# print("______________________________________________________________________________________________________________")

# # هنا بنعمل حاجة اسمها IQR (Interquartile Range) عشان نشيل القيم اللى فيها Outliers
# Q1_Pakistan = df['Launched Price (Pakistan)'].quantile(0.25)
# Q3_Pakistan = df['Launched Price (Pakistan)'].quantile(0.75)
# IQR_Pakistan = Q3_Pakistan - Q1_Pakistan

# Q1_India = df['Launched Price (India)'].quantile(0.25)
# Q3_India = df['Launched Price (India)'].quantile(0.75)
# IQR_India = Q3_India - Q1_India

# Q1_China = df['Launched Price (China)'].quantile(0.25)
# Q3_China = df['Launched Price (China)'].quantile(0.75)
# IQR_China = Q3_China - Q1_China

# Q1_USA = df['Launched Price (USA)'].quantile(0.25)
# Q3_USA = df['Launched Price (USA)'].quantile(0.75)
# IQR_USA = Q3_USA - Q1_USA

# Q1_Dubai = df['Launched Price (Dubai)'].quantile(0.25)
# Q3_Dubai = df['Launched Price (Dubai)'].quantile(0.75)
# IQR_Dubai = Q3_Dubai - Q1_Dubai

# df = df[(df['Launched Price (Pakistan)'] >= Q1_Pakistan - 1.5 * IQR_Pakistan) & (df['Launched Price (Pakistan)'] <= Q3_Pakistan + 1.5 * IQR_Pakistan)]
# df = df[(df['Launched Price (India)'] >= Q1_India - 1.5 * IQR_India) & (df['Launched Price (India)'] <= Q3_India + 1.5 * IQR_India)]
# df = df[(df['Launched Price (China)'] >= Q1_China - 1.5 * IQR_China) & (df['Launched Price (China)'] <= Q3_China + 1.5 * IQR_China)]
# df = df[(df['Launched Price (USA)'] >= Q1_USA - 1.5 * IQR_USA) & (df['Launched Price (USA)'] <= Q3_USA + 1.5 * IQR_USA)]
# df = df[(df['Launched Price (Dubai)'] >= Q1_Dubai - 1.5 * IQR_Dubai) & (df['Launched Price (Dubai)'] <= Q3_Dubai + 1.5 * IQR_Dubai)]

# print(df['Launched Price (Pakistan)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (India)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (China)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (USA)'])
# print("______________________________________________________________________________________________________________")
# print(df['Launched Price (Dubai)'])
# print("______________________________________________________________________________________________________________")





















































# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# # Specify the path to the train and test folders
# train_dir = 'C:\\Users\\zas\\OneDrive\\سطح المكتب\\University\\2- Semester 2\\Machine Learning\\Project\\Dataset Classification Cats & Dogs\\Train'
# test_dir = 'C:\\Users\\zas\\OneDrive\\سطح المكتب\\University\\2- Semester 2\\Machine Learning\\Project\\Dataset Classification Cats & Dogs\\Test'

# # Create an ImageDataGenerator for training with data augmentation
# train_datagen = ImageDataGenerator( 
#                   rescale = 1./255,                     # Normalize pixel values to the range [0, 1]
#                   rotation_range = 40,                  # Randomly rotate images by up to 40 degrees
#                   width_shift_range = 0.2,              # Randomly shift images horizontally by 20% of the width
#                   height_shift_range = 0.2,             # Randomly shift images vertically by 20% of the height
#                   shear_range = 0.2,                    # Apply random shearing transformations
#                   zoom_range = 0.2,                     # Randomly zoom in/out by up to 20%
#                   horizontal_flip = True,               # Randomly flip images horizontally
#                   fill_mode = 'nearest'                 # Fill in missing pixels after transformations using nearest pixel values
# )


# # Create an ImageDataGenerator for testing (only rescaling, no augmentation)
# test_datagen = ImageDataGenerator( rescale=1./255 ) # Normalize test images without applying transformations


# # Load training images from directory and prepare them in batches
# train_generator = train_datagen.flow_from_directory( 
#                   train_dir,                          # Path to training images (with subfolders for each class)
#                   batch_size=32,                      # Number of images per batch
#                   class_mode='binary',                # Since it's a binary classification (cats vs. dogs)
#                   target_size=(150, 150)              # Resize all images to 150x150 pixels
# )


# # Load test images in the same way (without data augmentation)
# test_generator = test_datagen.flow_from_directory(
#                   test_dir,
#                   batch_size = 32,
#                   class_mode = 'binary',
#                   target_size = (150, 150)
# )


# model = Sequential(
#       [
#             Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),  # First convolutional layer with 32 filters
#             MaxPooling2D(2, 2),                                                    # Max pooling layer to reduce spatial dimensions

#             Conv2D(64, (3, 3), activation = 'relu'),                               # Second convolutional layer with 64 filters
#             MaxPooling2D(2, 2),                                                    # Another max pooling layer

#             Conv2D(128, (3, 3), activation = 'relu'),                              # Third convolutional layer with 128 filters
#             MaxPooling2D(2, 2),                                                    # Another max pooling layer

#             Flatten(),                                                             # Flatten the 3D output to 1D for the fully connected layer
#             Dense(512, activation = 'relu'),                                       # Fully connected layer with 512 neurons
#             Dropout(0.5),                                                          # Dropout layer to prevent overfitting (50% dropout rate)
#             Dense(1, activation = 'sigmoid')                                       # Output layer with 1 neuron (binary classification)
#       ]
# ) 



# Collecting the model's parameters and compiling the model 
# model.compile(
#       loss = 'binary_crossentropy',                   # Loss function for binary classification
#       optimizer = 'adam',                             # Adam optimizer for training
#       metrics = ['accuracy']                          # Metric to evaluate during training and testing
# )

# model.summary()                                       # Print the model summary to see the architecture and number of parameters


# labels = ['cats', 'dogs']                                               # Class labels for the dataset
# count = [ 
#       len(os.listdir(os.path.join(train_dir, 'cats'))),                 # Count the number of cat images in the training set
#       len(os.listdir(os.path.join(train_dir, 'dogs')))                  # Count the number of dog images in the training set
# ]


# sns.barplot(x = labels, y = count)                                      # Create a bar plot of the class distribution
# plt.title('Distribution of Cats and Dogs')                              # Set the title of the plot
# plt.xlabel('Animals')                                                   # Set the x-axis label
# plt.ylabel('Count')                                                     # Set the y-axis label
# plt.show()                                                              # Display the plot



# # Train the model using the training data generator and validate using the test data generator
# history = model.fit(
#       train_generator,                                                      # Training data generator
#       steps_per_epoch = 100,                                                # Number of batches per epoch
#       epochs = 10,                                                          # Number of epochs to train
#       validation_data = test_generator,                                     # Validation data generator
#       validation_steps = 50,                                                # Number of batches for validation
# )

# test_loss, test_accuracy = model.evaluate(test_generator, steps = 50)       # Evaluate the model on the test data
# print(f"Test accuracy: {test_accuracy}")                                    # Print the test loss

