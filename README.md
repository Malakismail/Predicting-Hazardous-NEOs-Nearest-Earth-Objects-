## Predicting Hazardous NEOs (Nearest Earth Objects)
This project leverages NASA's dataset (1910–2024) from Kaggle to predict whether a Near Earth Object (NEO) is hazardous. By following a structured workflow—data import, cleaning, exploratory analysis, preprocessing (including handling class imbalance), and model training— implement two classification models: Logistic Regression and Random Forest Classifier.

## Table of Contents
•	Project Overview

•	Data

•	Approach

o	Data Import & Cleaning

o	Exploratory Data Analysis (EDA)

o	Preprocessing & Handling Imbalance

o	Model Training & Evaluation

	Logistic Regression

	Random Forest Classifier

•	Key Findings & Insights

•	Dependencies

•	Contributing

•	Contact


## Project Overview
This repository focuses on predicting the hazard potential of NEOs by applying machine learning techniques. The goal is to classify NEOs as either hazardous or non-hazardous based on features such as absolute magnitude, estimated diameter, and other relevant attributes.

## Data
The dataset used in this project is sourced from **Kaggle** and covers the period from 1910 to 2024. It includes detailed information about each NEO, with the target variable being is_hazardous, which indicates if a NEO is potentially dangerous.
**Dataset Link:** https://www.kaggle.com/datasets/ivansher/nasa-nearest-earth-objects-1910-2024/data

## Approach
# Data Import & Cleaning
•	Loading the Data:
The dataset is loaded using pandas.read_csv.
•	Initial Exploration:
We inspect the dataset’s shape, preview the first few rows, and obtain basic information with .info() and .describe().
•	Handling Missing Values:
Missing numerical values in absolute_magnitude, estimated_diameter_min, and estimated_diameter_max are filled using the column mean.
•	Duplicate Check:
The code verifies and handles duplicate records if present.

# Exploratory Data Analysis (EDA)
•	Distribution Analysis:
A count plot visualizes the distribution of hazardous vs. non-hazardous NEOs.
•	Pairplot:
A pairplot (with a KDE on the diagonal) helps understand the relationships between numerical features and how they differ by the hazard label.
•	Correlation Heatmap:
A heatmap of numerical features is generated to observe the correlations, aiding in feature selection.
•	Boxplot Analysis:
A boxplot compares the absolute magnitude distribution between hazardous and non-hazardous NEOs.
•	Categorical Encoding:
Any categorical features are encoded using LabelEncoder.

# Preprocessing & Handling Imbalance
•	Feature Selection:
The target variable is_hazardous is separated from the feature set.
•	Handling Class Imbalance:
SMOTE (Synthetic Minority Oversampling Technique) is applied to balance the classes.
•	Data Splitting:
The balanced dataset is split into training and testing sets (80/20 split).
•	Normalization:
Numerical features are normalized using StandardScaler to ensure optimal model performance.

# Model Training & Evaluation
Logistic Regression
•	Training:
A Logistic Regression model (with a maximum of 1000 iterations) is trained on the scaled training data.
•	Evaluation Metrics:
o	Classification Report: Provides precision, recall, and F1-score for both classes.
o	Confusion Matrix: Displays true vs. false classifications.
o	ROC AUC Score: Achieved a score of 0.8395.
o	Accuracy: Approximately 80.66%.
•	ROC Curve:
The ROC curve is plotted to visualize the trade-off between the true positive rate and false positive rate.
Logistic Regression Results:

Classification Report:
               precision    recall  f1-score   support

       False       0.87      0.72      0.79     59182
        True       0.76      0.89      0.82     58833

    accuracy                           0.81    118015
   macro avg       0.82      0.81      0.81    118015
weighted avg       0.82      0.81      0.81    118015

Confusion Matrix:
[[42535 16647]
 [ 6180 52653]]

**ROC AUC Score: 0.8395**
**Accuracy: 80.66%**

Random Forest Classifier
•	Training:
A Random Forest model with 100 estimators is trained on the same training set.
•	Evaluation Metrics:
o	Classification Report: Near-perfect performance (precision, recall, and F1-score nearly 0.99).
o	Confusion Matrix: Significantly fewer misclassifications.
o	ROC AUC Score: Achieved an outstanding score of 0.9992.
o	Accuracy: Approximately 98.90%.
•	ROC Curve:
The ROC curve is plotted, highlighting the model's exceptional discrimination power.
•	Feature Importance:
A bar chart is generated to display the top 10 features contributing to the model’s predictions.
Random Forest Results:

Classification Report:
               precision    recall  f1-score   support

       False       0.99      0.98      0.99     59182
        True       0.98      0.99      0.99     58833

    accuracy                           0.99    118015
   macro avg       0.99      0.99      0.99    118015
weighted avg       0.99      0.99      0.99    118015

Confusion Matrix:
[[58272   910]
 [  386 58447]]

**ROC AUC Score: 0.9992**
**Accuracy: 98.90%**

## Key Findings & Insights
•	Data Quality:
Initial data cleaning, including handling missing values and duplicates, was critical for ensuring reliable results.
•	Class Imbalance:
Addressing the imbalanced dataset using SMOTE improved the model’s ability to generalize across both classes.
•	Model Performance:
o	Logistic Regression: Moderate performance with an **accuracy of ~80.66%** and **ROC AUC of 0.8395**.
o	Random Forest Classifier: Superior performance with an accuracy of **~98.90%** and **ROC AUC nearing 1.0**.
•	Feature Importance:
The Random Forest model's feature importance analysis provides valuable insights into which variables most influence the prediction of hazardous NEOs.
These results highlight the effectiveness of ensemble methods like Random Forest in capturing complex relationships within the data, making it a preferred choice for this classification task.

## Dependencies
•	Python 3.x
•	pandas
•	numpy
•	matplotlib
•	seaborn
•	scikit-learn
•	imbalanced-learn
•	warnings (built-in)

## Contributing
Contributions are welcome! Fork the repository, create a branch for your feature or bug fix, and submit a pull request.

## Contact
For inquiries or collaboration opportunities, please reach out to me:

**Name**: Malak Ismail  

**Email**: malakismail706@gmail.com 

**LinkedIn**: https://www.linkedin.com/in/malakismail0/

