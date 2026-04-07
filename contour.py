import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import f

# Creating a grid of x,y vals with 100 proportionally split values from -5 to 5 
x_vals = np.linspace(-5,5,100)
y_vals = np.linspace(-5,5,100)
X, Y = np.meshgrid(x_vals,y_vals)


Z = f(X,Y)

def plot(history):
    plt.contour(X,Y,Z,levels=20)

    x = [p[0] for p in history]
    y = [p[1] for p in history]

    plt.plot(x,y,'r-o')
    plt.show()