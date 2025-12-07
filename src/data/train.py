import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import json

def main():
    df = pd.read_csv("data/clean_data.csv")

    X = df[['TotalPremium']]
    y = df['TotalClaims']

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    with open("metrics.json", "w") as f:
        json.dump({"mse": mse}, f)

if __name__ == "__main__":
    main()