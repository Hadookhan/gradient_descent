import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gradient_descent import f

# Creating a grid of x,y vals with 100 proportionally split values from -5 to 5 
x_vals = np.linspace(-5,5,100)
y_vals = np.linspace(-5,5,100)
X, Y = np.meshgrid(x_vals,y_vals)


Z = f(X,Y)

def plot(history, title):

    fig, ax = plt.subplots()
    ax.contour(X,Y,Z,levels=20)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Gradient Descent Path ({title})")

    line, = ax.plot([],[],'r-o')
    point, = ax.plot([],[],'bo')

    def update(frame):
        x = [p[0] for p in history[:frame+1]]
        y = [p[1] for p in history[:frame+1]]

        line.set_data(x,y)
        point.set_data([x[-1]],[y[-1]])

        return line,point

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=100,
        repeat=True
    )

    plt.show()

    return ani