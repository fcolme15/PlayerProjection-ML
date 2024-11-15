#Basic libraries
import pandas as pd
import numpy as np

#Diagram/Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns #Plotting library #pip install seaborn

#sklearn libraries ->pip install scikit-learn
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#Data pre-processing libraries
from sklearn.model_selection import train_test_split
import missingno as msno #Library to drop missing data #pip install missingno 

def main():

    #Importing data set:
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
    djMooreNewStats = stats2024[stats2024['player_display_name'] == 'D.J. Moore']
    tyLocket = combinedStats[combinedStats['player_display_name'] == 'Tyler Lockett']
    tyHill = combinedStats[combinedStats['player_display_name'] == 'Tyreek Hill']
    
    features = ['receptions', 'targets', 'receiving_tds']
    target = 'receiving_yards'

    djMooreFeatures = djMoore[features]
    djMooreTarget = djMoore[target]
    djMooreNew = djMooreNewStats[features]
    tyLocketFeatures = tyLocket[features]
    tyLocketTarget = tyLocket[target]
    tyHillFeatures = tyHill[features]
    tyHillTarget = tyHill[target]

    print(djMooreFeatures)

    #####NOT IMPLEMENTED#####
    #Setting aside data for cross validation 
    # Set aside 10% of instances for testing
    djMooreX_train, djMooreX_test, djMoorey_train, djMoorey_test = train_test_split(djMooreFeatures, djMooreTarget, test_size=0.1, random_state=42)

    # Split training again into 70% training and 20% validation
    djMooreX_train, djMooreX_val, djMoorey_train, djMoorey_val = train_test_split(djMooreX_train, djMoorey_train, test_size=0.2/(1-0.1), random_state=42)  


    #####Model 1, Logistic Regression:#####
    # Initialize a logistic regression model
    logisticModel = LogisticRegression(penalty=None)

    # Fit the model
    logisticModel.fit(djMooreFeatures, np.ravel(djMooreTarget))

    # Print data for model 1
    print("Model 1: Logistic Regression")
    print('w1:', logisticModel.coef_)
    print('w0:', logisticModel.intercept_)
    #####Model 1 END#####


    #####Model 2, K-means Regression:#####
    print("Model 2: K-Nearest Regression")
    # Initialize a logistic regression model giving it the number of k features
    knnr = KNeighborsRegressor(n_neighbors=5)
    # Fit the model
    knnrFit = knnr.fit(djMooreFeatures, djMooreTarget)
    #Predict a value giving it new X data
    knnrPrediction = knnrFit.predict(djMooreNew)
    print("Prediction", knnrPrediction)
    #Returns the nearest neighbors distances and indices for each intance in Xnew
    knnrNeighbors = knnrFit.kneighbors(djMooreNew)
    print("Nearest instances: ", knnrNeighbors)
    #####Model 2 END#####


    #####Model 3, Decision Trees for Regression:#####
    print("Model 3: Decision Trees for Regression")
    DTR = DecisionTreeRegressor(max_depth=4, ccp_alpha=0.9, random_state=123)
    #DTR = DecisionTreeRegressor(max_depth=2)
    #DTR = DecisionTreeRegressor(ccp_alpha=0.00)
    #DTR = DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=20)

    #Fit the model 
    DTR.fit(djMooreFeatures,djMooreTarget) 

    #Score the data
    DTR.score(djMooreFeatures, djMooreTarget) #Prints the r-squared value for the training set

    #Predict values given a new X
    DTR.predict(djMooreNew) #Predict values using new X-values
    #####Model 3 END#####


    # #Model 4, SVM-Support Vector Machines
    # print("Model 4: SVM")
    # eps = 0.1
    # #Linear SVM machine 
    # print("SVM: Linear")
    # svr_lin = SVR(kernel='linear', epsilon=eps)
    # #Fit the model
    # svr_lin.fit(np.reshape(djMooreFeatures,(-1,1)), np.ravel(djMooreTarget))
    # #Get the coefficient
    # svr_lin.coef_[0][0] 
    # #Get the intercept
    # svr_lin.intercept_[0]

    # svr_lin.predict(np.reshape(djMooreNew,(-1,1)))

    # #Polynomial SVM machine
    # print("SVM: Polynomial")
    # svr_poly = SVR(kernel='poly', epsilon=eps, C=0.2, gamma=0.8)
    # svr_poly.fit(np.reshape(djMooreFeatures,(-1,1)), np.ravel(djMooreTarget))
    # #Get the coefficient
    # svr_poly.coef_[0][0] 
    # #Get the intercept
    # svr_poly.intercept_[0]

    # svr_poly.predict(np.reshape(djMooreNew,(-1,1)))
    
    # #rbf SVM machine
    # print("SVM: rbf")
    # svr_rbf = SVR(kernel='rbf', epsilon=eps, C=0.2, gamma=0.8)
    # svr_rbf.fit(np.reshape(djMooreFeatures,(-1,1)), np.ravel(djMooreTarget))
    # #Get the coefficient
    # svr_rbf.coef_[0][0] 
    # #Get the intercept
    # svr_rbf.intercept_[0]

    # svr_rbf.predict(np.reshape(djMooreNew,(-1,1)))


if __name__ == "__main__":
    main()