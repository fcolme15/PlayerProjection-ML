import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #Plotting library #pip install seaborn

import missingno as msno #Library to drop missing data #pip install missingno 

#sklearn ->pip install scikit-learn
from sklearn.linear_model import LogisticRegression 


def main():

    #Model 1, Logistic Regression:
    # Initialize a logistic regression model
    logisticModel = LogisticRegression(penalty=None)

    # Fit the model
    logisticModel.fit(X, np.ravel(y))

    # Print the fitted model
    print('w1:', logisticModel.coef_)
    print('w0:', logisticModel.intercept_)


    #Model 2, K-means Regression:


    #Model 2, K-means Regression:




if __name__ == "__main__":
    main()