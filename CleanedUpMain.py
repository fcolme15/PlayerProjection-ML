import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    return predictions


#Function to analyze a specific player's data
def allPlayers(player_name, combinedStats, stats2024, features, target):
    print(f"\n--- {player_name} ---")

    #Filter and Combine
    playerStats = combinedStats[combinedStats['player_display_name'] == player_name]
    playerNewStats = stats2024[stats2024['player_display_name'] == player_name]

    if playerStats.empty:
        print(f"No historical data found for {player_name}. Skipping analysis.")
        return

    #Features and target
    playerFeatures = playerStats[features].astype(float)
    playerTarget = playerStats[target].astype(float)
    playerNew = playerNewStats[features].astype(float)

    #Check if features or target are empty
    if len(playerFeatures) == 0 or len(playerTarget) == 0:
        print(f"Insufficient data for training {player_name}. Skipping all models.")
        return

    #Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        playerFeatures, playerTarget, test_size=0.1, random_state=42
    )

    #Check if training data is empty
    if len(X_train) == 0 or len(y_train) == 0:
        print(f"Insufficient training data for {player_name}. Skipping all models.")
        return

    #Logistic Regression (Classification)
    print("\nModel 1: Logistic Regression (Classification)")
    logisticTarget = (playerTarget > 50).astype(int)  
    if len(np.unique(logisticTarget)) < 2:
        print(f"Skipping Logistic Regression: only one class in the target data for {player_name}.")
    else:
        logisticModel = LogisticRegression()
        logisticModel.fit(playerFeatures, logisticTarget)
        print("Weights:", logisticModel.coef_)
        print("Intercept:", logisticModel.intercept_)

    #K-Nearest Neighbors Regression
    print("\nModel 2: K-Nearest Regression")
    if len(playerNew) == 0:
        print(f"Skipping K-Nearest Regression: no 2024 data available for {player_name}.")
    else:
        knnr = KNeighborsRegressor(n_neighbors=5)
        knnr.fit(X_train, y_train)
        knnrPrediction = knnr.predict(playerNew)
        print("Predicted Receiving Yards:", knnrPrediction)

    #Decision Tree Regression
    print("\nModel 3: Decision Tree Regression")
    if len(X_train) == 0 or len(y_train) == 0:
        print(f"Skipping Decision Tree Regression: insufficient training data for {player_name}.")
    else:
        DTR = DecisionTreeRegressor(max_depth=4, random_state=123)
        DTR.fit(X_train, y_train)
        print("Training R^2 Score:", DTR.score(X_train, y_train))
        print("Predicted Receiving Yards:", DTR.predict(playerNew) if len(playerNew) > 0 else "No 2024 data available.")

    #Support Vector Regression (SVR)
    print("\nModel 4: Support Vector Regression (SVR)")
    if len(X_train) == 0 or len(y_train) == 0 or len(playerNew) == 0:
        print(f"Skipping Support Vector Regression: insufficient data for {player_name}.")
    else:
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(X_train, y_train)
        svrPrediction = svr.predict(playerNew)
        print("Predicted Receiving Yards:", svrPrediction)

# Main function
def main():
    stats_files = [
        "player_stats_2018.csv", "player_stats_2019.csv",
        "player_stats_2020.csv", "player_stats_2021.csv",
        "player_stats_2022.csv", "player_stats_2023.csv",
    ]
    combinedStats = pd.concat([pd.read_csv(f) for f in stats_files], axis=0)
    stats2024 = pd.read_csv("player_stats_2024.csv")

    #Enter Names
    print("Enter the names of three players to analyze (separated by commas):")
    input_players = input().split(",")
    players = [player.strip() for player in input_players]

    #Features and target for prediction
    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    #Output
    for player_name in players:
        allPlayers(player_name, combinedStats, stats2024, features, target)


#Run the main function
if __name__ == "__main__":
    main()

