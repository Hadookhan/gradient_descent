from gradient_descent import f, nabla_f, learn
from contour import plot
import random as rand

# Initial random starting points
x, y = rand.randint(-5,5), rand.randint(-5,5)

# Learning rate
# Will help function converge to the minima
alpha = 0.01
    
def main():
    history = []

    learn(alpha,x,y,history)

    for i in range(len(history)):
        cur_x = history[i][0]
        cur_y = history[i][1]
        print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={f(cur_x,cur_y)}")
    
    plot(history)


if __name__ == '__main__':
    main()
