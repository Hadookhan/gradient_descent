from gradient_descent import f, learn, momentum_learn
from contour import plot
from graph import plot3d
import random as rand

# Initial random starting points
# x, y = rand.randint(-5,5), rand.randint(-5,5)
x, y = 4, -3 # <- test with constant x,y

# Learning rate
# Will help function converge to the minima
alpha = 0.01

# Momentum coefficient
# Describes how much of the old velocity we remember and use on the new velocity 
beta = 0.9
    
def main():
    history = []

    # learn(alpha,x,y,history)
    momentum_learn(alpha,beta,x,y,history)

    for i in range(len(history)):
        cur_x = history[i][0]
        cur_y = history[i][1]
        print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={f(cur_x,cur_y)}")
    
    plot(history)
    plot3d(history)


if __name__ == '__main__':
    main()
