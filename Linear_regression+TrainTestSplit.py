import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
df = {
    'Experience': [1,3,5,2,7,9,8,6,6.7,8.2],
    'Salary': [23299,27890,34788,45778,51098,57098,63098,65790,71098,79072]
}
df = pd.DataFrame(df)

# Features & Target
X = df[['Experience']]
y = df['Salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
New = float(input("What is our experience? "))
prediction = model.predict(pd.DataFrame({'Experience':[New]}))

print(f"Predicted Salary for {New} years: {prediction[0]:.2f}")



