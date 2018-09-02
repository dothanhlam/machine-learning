from __future__ import division, print_function, unicode_literals
from sklearn import datasets, linear_model

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def predict(w_0, w_1, height):
    return w_1*weight + w_0

def main():
    # height (cm)
    x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
    # weight (kg)
    y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
    
    one = np.ones((x.shape[0], 1))
    xbar = np.concatenate((one, x), axis = 1)
    
    # Calculating weights of the fitting line 
    a = np.dot(xbar.T, xbar)
    b = np.dot(xbar.T, y)
    w = np.dot(np.linalg.pinv(a), b)
    print('w = ', w)
    
    # fit the model by Linear Regression
    regr = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
    regr.fit(xbar, y)
    
    # Compare two results
    print( 'Solution found by scikit-learn  : ', regr.coef_ )

    # Preparing the fitting line 
    w_0 = w[0][0]
    w_1 = w[1][0]
    x0 = np.linspace(145, 185, 2)
    y0 = w_0 + w_1*x0

#     plt.plot(x, y, 'ro')
#     plt.plot(x0, y0)               # the fitting line
# 
#     plt.axis([140, 190, 45, 75])
#     plt.xlabel('Height (cm)')
#     plt.ylabel('Weight (kg)')
#     plt.show()
    
    
if __name__ == "__main__":
    main()