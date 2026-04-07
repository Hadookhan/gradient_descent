import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gradient_descent import f

x_vals = np.linspace(-5,5,100)
y_vals = np.linspace(-5,5,100)
X, Y = np.meshgrid(x_vals,x_vals)

Z = f(X,Y)

def plot3d(history):

    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    # z points are results from function f of (x,y) points
    zs = [f(x, y) for x, y in zip(xs, ys)]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(xs,ys,zs,'r-o')

    ax.plot_surface(X,Y,Z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Gradient Descent Path')

    plt.show()
