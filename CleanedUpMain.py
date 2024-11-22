import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#Store results summary
model_summary = []

#evaluate models
def evaluate_model(model, X_test, y_test, model_name, player_name):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    #results summary
    model_summary.append({
        "Player": player_name,
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    #Print individual model performance
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
    knnr = KNeighborsRegressor(n_neighbors=5)
    knnr.fit(X_train, y_train)
    evaluate_model(knnr, X_test, y_test, "KNN Regression", player_name)

    #Decision Tree Regression
    print("\nModel 3: Decision Tree Regression")
    DTR = DecisionTreeRegressor(max_depth=4, random_state=123)
    DTR.fit(X_train, y_train)
    evaluate_model(DTR, X_test, y_test, "Decision Tree Regression", player_name)

    #Support Vector Regression (SVR)
    print("\nModel 4: Support Vector Regression (SVR)")
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    evaluate_model(svr, X_test, y_test, "SVR", player_name)


#Main function
def main():
    stats_files = [
        "player_stats_2018.csv", "player_stats_2019.csv",
        "player_stats_2020.csv", "player_stats_2021.csv",
        "player_stats_2022.csv", "player_stats_2023.csv",
    ]
    combinedStats = pd.concat([pd.read_csv(f) for f in stats_files], axis=0)
    stats2024 = pd.read_csv("player_stats_2024.csv")

    #Enter Names
    print("Enter the names of the players to analyze (separated by commas):")
    input_players = input().split(",")
    players = [player.strip() for player in input_players]

    #Features and target for prediction
    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    #Output for each player
    for player_name in players:
        allPlayers(player_name, combinedStats, stats2024, features, target)

    #Summary of all models
    if model_summary:
        print("\n--- Model Performance Summary ---")
        summary_df = pd.DataFrame(model_summary)
        print(summary_df)

        #save summary to CSV
        summary_df.to_csv("performanceSummary.csv", index=False)
        print("Summary saved to 'performanceSummary.csv'.")

        
        # print("\n--- Best Model for Each Player ---")
        # for player in summary_df['Player'].unique():
        #     player_df = summary_df[summary_df['Player'] == player]
        #     best_model = player_df.loc[player_df['R2'].idxmax()]  # Highest R²
        #     print(f"Player: {player}")
        #     print(f"Best Model: {best_model['Model']}")
        #     print(f"R² Score: {best_model['R2']:.2f}")
        #     print(f"RMSE: {best_model['RMSE']:.2f}")
        #     print("-" * 40)

        # #Determine overall best model
        # print("\n--- Overall Best Model ---")
        # best_overall = summary_df.loc[summary_df['R2'].idxmax()]
        # print(f"Player: {best_overall['Player']}")
        # print(f"Best Model: {best_overall['Model']}")
        # print(f"R² Score: {best_overall['R2']:.2f}")
        # print(f"RMSE: {best_overall['RMSE']:.2f}")

# Run the main function
if __name__ == "__main__":
    main()


#Run the main function
if __name__ == "__main__":
    main()

