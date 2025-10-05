import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv('/Users/karthikeyavaibhav/Desktop/Machine_learning/sample sets/employee_ratings_experience_salary.csv')

#Data cleaning
#-----------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)

df['Experience_Years'] = df['Experience_Years'].fillna(df['Experience_Years'].mean())
df['Performance_Rating'] = df['Performance_Rating'].fillna(df['Performance_Rating'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df.drop_duplicates(inplace=True)

#-----------------------------------------------

#Since we do not have any categorical values we dont need any feature encoding such as :-
# 1) Label encoding
# 2) OneHot encoding
# 3) Pandas get_dummies

#---------------------------------------------------
# Assign values

X = df[['Experience_Years', 'Performance_Rating']]
y = df['Salary']

#Train_Test_Split

#---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#----------------------------------------------------
#Linear regression
#----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)


# Taking input
#----------------------------------------------------

# Take inputs
exp = float(input('Enter Experience Years: '))
rating = float(input('Enter Performance Rating: '))

# Predict
prediction = model.predict([[exp, rating]])

print(f"Predicted Salary for {exp} years experience and rating {rating}: {prediction[0]:.2f}")