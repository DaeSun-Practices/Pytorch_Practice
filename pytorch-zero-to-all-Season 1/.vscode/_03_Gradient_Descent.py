import numpy as np
import matplotlib.pyplot as plt
from random import *

# our model for the forward pass
def forward(x, w):
    return x * w


# Loss function
def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) * (y_pred - y)

def MSE(x_list, y_list, w):
    mse = 0
    for x, y in zip(x_list, y_list):
        mse +=loss(x, y, w)
    mse /= len(x_list)
    return mse

def gradient(x, y, w):
    return 2 * x * (x * w - y)

def gradient_descent (x_list, y_list, w):
    for x, y in zip(x_list, y_list):
        w -= 0.01 * gradient(x, y, w)
    
    return w


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = uniform(-1.0, 1.0)

print(w)
for i in range(1, 100):
    w = gradient_descent(x_data, y_data, w)
    print(str(i)+"th try: " + str(MSE(x_data, y_data, w)))
print(w)
