# Predicting-Heart-Failure
Using random forests to predict heart failure from clinical records.

## Overview
I made this project to learn about decision trees and RandomForestRegressor. The dataset I am using and the idea for this challenge are both from Kaggle. The project uses Python along with libraries like Pandas and scikit-learn.

## Dependencies
- Python
- Pandas: For data manipulation and analysis.
- scikit-learn: For machine learning model building and evaluation.

## Dataset
The dataset, named `heart_failure_clinical_records.csv`, includes various attributes of patients who either died of heart failure or did not.

## Data Preprocessing Steps
1. **Data Reading**: Utilizing Pandas to read the data.
2. **Data Cleaning**: Dropping rows with missing values to enhance data quality.
3. **Data Splitting**: Dividing the dataset into training (95%) and validation (5%) sets.

## Features
The model considers the following features for rent prediction:
- age: age of the patient [years]
- anaemia
- creatinine_phosphokinase
- diabetes
- ejection_fraction
- high_blood_pressure
- platelets
- serum_creatinine
- serum_sodium
- sex: sex of the patient [1: Male, 0: Female]
- smoking
- time

## Model Building and Evaluation
1. **Model Creation**: A RandomForestRegressor model is instantiated.
2. **Model Training**: The model is trained on the training set.
3. **Prediction**: The model predicts death causes on the validation set.
4. **Evaluation**: The Mean Absolute Error (MAE) metric is used for model evaluation based on the accuracy of predicted DEATH_EVENT: output class [1: heart disease, 0: Normal].

## Usage Instructions
- Read and preprocess the data.
- Split the data into training and validation sets.
- Define features and the target variable (DEATH_EVENT).
- Train the RandomForestRegressor model.
- Predict and evaluate the model using the validation set.

## Output
- The model outputs predict if someone will die of heart disease.
- The model's performance is measured using the Mean Absolute Error (MAE).

## Conclusion
This project demonstrates a structured approach to predict if someone will die of heart disease or not using RandomForestRegressor.
