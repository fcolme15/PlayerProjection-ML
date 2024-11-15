import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Plotting library
import missingno as msno  # Library to visualize missing data
from sklearn.linear_model import LogisticRegression  # Logistic Regression model (for future use)

def main():

    # Load CSV files
    stats2018 = pd.read_csv("player_stats_2018.csv")
    stats2019 = pd.read_csv("player_stats_2019.csv")
    stats2020 = pd.read_csv("player_stats_2020.csv")
    stats2021 = pd.read_csv("player_stats_2021.csv")
    stats2022 = pd.read_csv("player_stats_2022.csv")
    stats2023 = pd.read_csv("player_stats_2023.csv")
    #For testing, but not used in learning
    stats2024 = pd.read_csv("player_stats_2024.csv") 

    # Combine datasets
    combinedStats = pd.concat([stats2018, stats2019, stats2020, stats2021, stats2022, stats2023], axis=0)

    djMoore = combinedStats[combinedStats['player_display_name'] == 'D.J. Moore']
    tyLocket = combinedStats[combinedStats['player_display_name'] == 'Tyler Lockett']
    tyHill = combinedStats[combinedStats['player_display_name'] == 'Tyreek Hill']
    
    features = ['receptions', 'targets', 'receiving_yards']
    target = 'receiving_tds'

    djMooreFeatures = djMoore[features]
    djMooreTarget = djMoore[target]
    tyLocketFeatures = tyLocket[features]
    tyLocketTarget = tyLocket[target]
    tyHillFeatures = tyHill[features]
    tyHillTarget = tyHill[target]

    print(djMooreFeatures)



if __name__ == "__main__":
    main()