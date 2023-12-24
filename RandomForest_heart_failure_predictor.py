import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Reading medical records csv
medical_records_path='heart_failure_clinical_records.csv'
medical_records=pd.read_csv(medical_records_path)

#Drop all dataframe rows that are missing values to clean up data.
medical_records=medical_records.dropna(axis=0)

#Glimpse and summary of input data
print(medical_records.describe())
print(medical_records.head())

#Now, let us split off our training set from our validation set
train, test = train_test_split(medical_records, test_size=0.05)

#Pick our targets
trainY = train.DEATH_EVENT
validateY = test.DEATH_EVENT

#Pick our features
medical_features=['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']
trainX=train[medical_features]
validateX=test[medical_features]


print("\nCreating model...")
forest_model = RandomForestRegressor(random_state=1)
print("Model created!")


print("\nTraining model...")
forest_model.fit(trainX,trainY)

print("Model trained!")

print("\nModel generating predictions...")
predictions = forest_model.predict(validateX)
print("Predictions generated!")

print("\nThe MAE of the model is:", mean_absolute_error(validateY,predictions),"\n")