import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #Plotting library #pip install seaborn

import missingno as msno #Library to drop missing data #pip install missingno 

#sklearn ->pip install scikit-learn
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


def main():

    #Importing data set:

    #Model 1, Logistic Regression:
    # Initialize a logistic regression model
    logisticModel = LogisticRegression(penalty=None)

    # Fit the model
    logisticModel.fit(X, np.ravel(y))

    # Print data for model 1
    print('w1:', logisticModel.coef_)
    print('w0:', logisticModel.intercept_)


    #Model 2, K-means Regression:
    # Initialize a logistic regression model giving it the number of k features
    knnr = KNeighborsRegressor(n_neighbors=5)
    # Fit the model
    knnrFit = knnr.fit(X,y)
    #Predict a value giving it new X data
    neighbors = knnrFit.predict(Xnew)
    #Returns the nearest neighbors distances and indices for each intance in Xnew
    knnrFit.kneighbors(Xnew)


    #Model 3, Decision Trees for Regression:
    DTR = DecisionTreeRegressor(max_depth=4, ccp_alpha=0.9, random_state=123)
    #DTR = DecisionTreeRegressor(max_depth=2)
    #DTR = DecisionTreeRegressor(ccp_alpha=0.00)
    #DTR = DecisionTreeRegressor(max_leaf_nodes=10, min_samples_leaf=20)

    #Fit the model 
    DTR.fit(X,y) 

    #Score the data
    DTR.score(X, y) #Prints the r-squared value for the training set

    #Predict values given a new X
    DTR.predict(X_new) #Predict values using new X-values


    #Model 4, SVM-Support Vector Machines
    #Linear SVM machine 
    svr_lin = SVR(kernel='linear', epsilon=eps)
    #Fit the model
    svr_lin.fit(np.reshape(X_train,(-1,1)), np.ravel(y_train))
    #Get the coefficient
    svr_lin.coef_[0][0] 
    #Get the intercept
    svr_lin.intercept_[0]

    svr_lin.predict(np.reshape(X_new,(-1,1)))

    #Polynomial SVM machine
    svr_poly = SVR(kernel='poly', epsilon=eps, C=0.2, gamma=0.8)
    svr_poly.fit(np.reshape(X_train,(-1,1)), np.ravel(y_train))
     #Get the coefficient
    svr_poly.coef_[0][0] 
    #Get the intercept
    svr_poly.intercept_[0]

    svr_poly.predict(np.reshape(X_new,(-1,1)))
    
    #rbf SVM machine
    svr_rbf = SVR(kernel='rbf', epsilon=eps, C=0.2, gamma=0.8)
    svr_rbf.fit(np.reshape(X_train,(-1,1)), np.ravel(y_train))
    #Get the coefficient
    svr_rbf.coef_[0][0] 
    #Get the intercept
    svr_rbf.intercept_[0]

    svr_rbf.predict(np.reshape(X_new,(-1,1)))





if __name__ == "__main__":
    main()