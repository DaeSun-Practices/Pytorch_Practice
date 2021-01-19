import numpy as np
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x,w):
    return x*w 

def loss (x, w, y):
    y_pred = forward(x,w)
    return (y_pred - y) * (y_pred - y)

def MSE_loss(x_list, y_list):
    mse_loss = []
    for w in np.arange(0.0, 4.01, 0.01):
        mse = 0
        for x, y in zip(x_data, y_data):
            mse += loss (x, w, y)
    mse /= 3
    mse_loss.append(mse)

    return mse_loss


w_list = np.arange(0.0, 4.01, 0.01)
mse_list =  MSE_loss(x_data, y_data)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()





