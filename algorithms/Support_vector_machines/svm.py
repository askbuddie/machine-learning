# Author : Nishant Banjade
# Support vector machine

import numpy as np

class SVM():

    def __init__(self, learning_rate=0.001, lambda_value = 0.01, iterations = 1000):
        self.learning_rate = learning_rate
        self.lambda_value = lambda_value
        self.iterations = iterations
        # initially assign weight and bais as None
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape      # X.shape = (row, column)
        y_temp = np.where(y<=0, -1, 1)       # y (-1 , 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.iterations):
            # for individual data points, we use nested loop 
            for idx, x_i in enumerate(X):
                condition = y_temp[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.learning_rate * (2*self.lambda_value * self.w)
                else:
                    self.w -= self.learning_rate *(2*self.lambda_value * self.w - np.dot(x_i, y_temp[idx]))


    def predict(self, X):
       prediction = np.dot(X, self.w)-self.b
       return np.sign(prediction)

# Testing with datasets

from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples = 50, n_features=2, centers=2, cluster_std=1.05, random_state=40)

y = np.where(y==0, -1, 1)

svm = SVM()
svm.fit(X,y)
#prediction = svm.predict(X)

print(svm.w, svm.b)



# visualize the svm 

def visualize_svm():

    def hyperplane_value(x, w, b, offset):
        return (-w[0] * x+b+offset)/w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.scatter(X[:, 0], X[:, 1], marker = "o", c = y)

    x0_1 = np.amin(X[:,0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = hyperplane_value(x0_1, svm.w, svm.b,0)
    x1_2 = hyperplane_value(x0_2, svm.w, svm.b, 0)

    x1_1_m = hyperplane_value(x0_1, svm.w, svm.b, -1)
    x1_2_m = hyperplane_value(x0_2, svm.w, svm.b, -1)


    x1_1_p =hyperplane_value(x0_1, svm.w, svm.b, 1)
    x1_2_p = hyperplane_value(x0_2, svm.w, svm.b, 1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim(x1_min-3, x1_max+3)
    plt.show()
visualize_svm()















