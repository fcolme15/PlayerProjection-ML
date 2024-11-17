import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def allPlayers(player_name, combinedStats, stats2024, features, target):
    print(f"\n--- {player_name} ---")

    #Filter data for the player
    playerStats = combinedStats[combinedStats['player_display_name'] == player_name]
    playerNewStats = stats2024[stats2024['player_display_name'] == player_name]

    #Features and target
    playerFeatures = playerStats[features].astype(float)
    playerTarget = playerStats[target].astype(float)
    playerNew = playerNewStats[features].astype(float)

    #Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        playerFeatures, playerTarget, test_size=0.1, random_state=42
    )

    #Logistic Regression (Classification)
    print("\nModel 1: Logistic Regression (Classification)")
    logisticTarget = (playerTarget > 50).astype(int)  # Binary target
    logisticModel = LogisticRegression()
    logisticModel.fit(playerFeatures, logisticTarget)
    print("Weights:", logisticModel.coef_)
    print("Intercept:", logisticModel.intercept_)

    #K-Nearest Neighbors Regression
    print("\nModel 2: K-Nearest Regression")
    knnr = KNeighborsRegressor(n_neighbors=5)
    knnr.fit(X_train, y_train)
    knnrPrediction = knnr.predict(playerNew)
    print("Predicted Receiving Yards:", knnrPrediction)

    #Decision Tree Regression
    print("\nModel 3: Decision Tree Regression")
    DTR = DecisionTreeRegressor(max_depth=4, random_state=123)
    DTR.fit(X_train, y_train)
    print("Training R^2 Score:", DTR.score(X_train, y_train))
    print("Predicted Receiving Yards:", DTR.predict(playerNew))


def main():
   
    stats_files = [
        "player_stats_2018.csv", "player_stats_2019.csv",
        "player_stats_2020.csv", "player_stats_2021.csv",
        "player_stats_2022.csv", "player_stats_2023.csv",
    ]
    combinedStats = pd.concat([pd.read_csv(f) for f in stats_files], axis=0)
    stats2024 = pd.read_csv("player_stats_2024.csv")

    #players
    players = ['D.J. Moore', 'Tyler Lockett', 'Tyreek Hill']


    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    #print
    for player_name in players:
        allPlayers(player_name, combinedStats, stats2024, features, target)

if __name__ == "__main__":
    main()
