#Basic libraries
import pandas as pd
import numpy as np

#Diagram/Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns  #Plotting library

#sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#data pre-processing library
import missingno as msno  #Library to visualize missing data


def load_data():
    try:
        #Load CSV files
        stats2018 = pd.read_csv("player_stats_2018.csv")
        stats2019 = pd.read_csv("player_stats_2019.csv")
        stats2020 = pd.read_csv("player_stats_2020.csv")
        stats2021 = pd.read_csv("player_stats_2021.csv")
        stats2022 = pd.read_csv("player_stats_2022.csv")
        stats2023 = pd.read_csv("player_stats_2023.csv")
        stats2024 = pd.read_csv("player_stats_2024.csv")  #For testing

        #Combine datasets
        combined_stats = pd.concat([stats2018, stats2019, stats2020, stats2021, stats2022, stats2023], axis=0)
        return combined_stats, stats2024
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        exit()


def preprocess_data(data):

    #Visualize missing data, a bit confused on the output
    msno.matrix(data)
    plt.show()
    
    #Drop rows with missing target values
    data = data.dropna(subset=['receiving_yards'])

    #Fill missing features with zeros
    data = data.fillna(0)
    return data


def evaluate_model(model, X_test, y_test, model_name):
 
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test, predictions)

    # Print the evaluation results
    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    return predictions


def main():
    #Load data
    combined_stats, stats2024 = load_data()

    #Filter player-specific data
    players = ['D.J. Moore', 'Tyler Lockett', 'Tyreek Hill']
    stats2024 = stats2024[stats2024['player_display_name'].isin(players)]

    #Preprocess data
    combined_stats = preprocess_data(combined_stats)

    #Features and target
    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    for player in players:
        print(f"\nAnalyzing data for {player}...")
        player_data = combined_stats[combined_stats['player_display_name'] == player]
        player_new_stats = stats2024[stats2024['player_display_name'] == player]

        if player_data.empty:
            print(f"No data available for {player}. Skipping...")
            continue

        X = player_data[features]
        y = player_data[target]
        X_new = player_new_stats[features]

        #Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        #Logistic Regression
        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(X_train, y_train)
        evaluate_model(logistic_model, X_test, y_test, "Logistic Regression")

        #K-Nearest Neighbors Regression
        knnr = KNeighborsRegressor(n_neighbors=5)
        knnr.fit(X_train, y_train)
        evaluate_model(knnr, X_test, y_test, "K-Nearest Neighbors Regression")

        #Predict future performance
        knnr_prediction = knnr.predict(X_new)
        print(f"Future predictions for {player}: {knnr_prediction}")

        #Decision Tree Regression
        dtr = DecisionTreeRegressor(max_depth=4, random_state=42)
        dtr.fit(X_train, y_train)
        evaluate_model(dtr, X_test, y_test, "Decision Tree Regression")

        #Visualize tree if desired
        from sklearn.tree import plot_tree
        plt.figure(figsize=(12, 8))
        plot_tree(dtr, filled=True, feature_names=features, rounded=True)
        plt.title(f"Decision Tree for {player}")
        plt.show()


if __name__ == "__main__":
    main()
