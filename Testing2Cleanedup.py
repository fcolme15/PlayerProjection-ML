import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def main():
    #Load CSV files
    stats_files = [
        "player_stats_2018.csv", "player_stats_2019.csv",
        "player_stats_2020.csv", "player_stats_2021.csv",
        "player_stats_2022.csv", "player_stats_2023.csv",
    ]
    combinedStats = pd.concat([pd.read_csv(f) for f in stats_files], axis=0)
    stats2024 = pd.read_csv("player_stats_2024.csv")

    #Filter data just djMoore
    djMoore = combinedStats[combinedStats['player_display_name'] == 'D.J. Moore']
    djMooreNewStats = stats2024[stats2024['player_display_name'] == 'D.J. Moore']

    #features and targets
    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    djMooreFeatures = djMoore[features].astype(float)
    djMooreTarget = djMoore[target].astype(float)
    djMooreNew = djMooreNewStats[features].astype(float)

    #Split data for training and testing
    djMooreX_train, djMooreX_test, djMoorey_train, djMoorey_test = train_test_split(
        djMooreFeatures, djMooreTarget, test_size=0.1, random_state=42)

    #Logistic Regression (classification)
    print("Model 1: Logistic Regression (Classification)")
    logisticTarget = (djMooreTarget > 50).astype(int) 
    logisticModel = LogisticRegression()
    logisticModel.fit(djMooreFeatures, logisticTarget)
    print("Weights:", logisticModel.coef_)
    print("Intercept:", logisticModel.intercept_)

    #K-Nearest Neighbors Regression
    print("\nModel 2: K-Nearest Regression")
    knnr = KNeighborsRegressor(n_neighbors=5)
    knnr.fit(djMooreX_train, djMoorey_train)
    knnrPrediction = knnr.predict(djMooreNew)
    print("Predicted Receiving Yards:", knnrPrediction)

    #Decision Tree Regression
    print("\nModel 3: Decision Tree Regression")
    DTR = DecisionTreeRegressor(max_depth=4, random_state=123)
    DTR.fit(djMooreX_train, djMoorey_train)
    print("Training R^2 Score:", DTR.score(djMooreX_train, djMoorey_train))
    print("Predicted Receiving Yards:", DTR.predict(djMooreNew))

if __name__ == "__main__":
    main()
