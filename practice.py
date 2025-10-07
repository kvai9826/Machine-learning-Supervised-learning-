from statistics import linear_regression

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, minmax_scale, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/karthikeyavaibhav/Desktop/Machine_learning/sample sets/employee_regression_practice_dataset.csv')

#-------Data Cleaning-----------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df['Experience_Years'] = df['Experience_Years'].fillna(df['Experience_Years'].mean())
df['Performance_Rating'] = df['Performance_Rating'].fillna(df['Performance_Rating'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

#-----Feature encoding-----------------------------------------------

encoding = OneHotEncoder(sparse_output=False)
encoded = encoding.fit_transform(df[['Department','City']])
encoded_df = pd.DataFrame(encoded, columns=encoding.get_feature_names_out(['Department','City']))

#--------Feature scaling----------------------------------------------

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['Experience_Years','Performance_Rating','Salary']])
scaled_df = pd.DataFrame(scaled, columns=['Experience_Years','Performance_Rating','Salary'])

#----------Combine (Feature scaling + Feature encoding)--------------------
combined = pd.concat([encoded_df, scaled_df], axis=1)


#----------Train_test_split--------------------------

X = combined.drop(['Salary'], axis=1) #Take all columns in concatenated dataframe except 'Salary' as input data
y = df['Salary'] # Takes salary as output data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#---------Polynomial featuring------------------------

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#---------Linear regression---------------------------

model = LinearRegression()
model.fit(X_poly, y_train)
y_pred = model.predict(X_test_poly)


#---------Model evaluation---------------------------


r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(r2)
print(rmse)
print(mse)


dept = input("Enter Department (HR / Tech / Finance / Marketing / Operations): ")
city = input("Enter City (Bangalore / Hyderabad / Pune / Delhi / Chennai): ")
exp = float(input("Enter Years of Experience: "))
rating = float(input("Enter Performance Rating (1–10): "))

# Convert input to DataFrame
new_data = pd.DataFrame({
    'Department': [dept],
    'City': [city],
    'Experience_Years': [exp],
    'Performance_Rating': [rating]
})

# Apply same encoding
encoded_new = encoding.transform(new_data[['Department', 'City']])
encoded_new_df = pd.DataFrame(encoded_new, columns=encoding.get_feature_names_out(['Department', 'City']))

# Apply same scaling
scaled_new = scaler.transform([[exp, rating, df['Salary'].mean()]])  # Dummy salary just to fit shape
scaled_new_df = pd.DataFrame(scaled_new, columns=['Experience_Years', 'Performance_Rating', 'Salary'])

# Combine all
combined_new = pd.concat([encoded_new_df, scaled_new_df[['Experience_Years', 'Performance_Rating']]], axis=1)

# Apply polynomial features
new_poly = poly.transform(combined_new)

# Predict
predicted_salary = model.predict(new_poly)

print(f"\n💰 Predicted Salary: ₹{predicted_salary[0]:,.2f}")