# Pattern Recognition Project - Air Quality Classification

This project analyzes Delhi NCR air quality data and trains machine learning models to classify AQI categories from environmental features.

## What It Does

- Loads and explores an air quality dataset.
- Checks missing values and class distribution.
- Encodes categorical columns and scales numerical features.
- Uses selected pollution and weather features such as PM2.5, PM10, CO, NO2, temperature, visibility, wind speed, and ozone.
- Trains and compares:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluates models using accuracy, classification reports, and confusion matrices.
- Visualizes class distribution, feature correlation, model confusion matrices, and accuracy comparison.

## Main Files

- `Code/index.py` - main training, evaluation, and visualization script.
- `Code/delhi_ncr_aqi_dataset.csv` - dataset used by the script.
- `ipynb/Project_Pattern.ipynb` - notebook version of the project workflow.
- `Report/` and `Presentation/` - supporting project materials.

## Tech Stack

Python, pandas, matplotlib, seaborn, scikit-learn.

## How To Run

Install the required Python packages:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

Run the main script:

```bash
python "Code/index.py"
```
