from gradient_descent import f, learn, momentum_learn, adam_learn, train_linear_regression
from contour import plot
from graph import plot3d
import random as rand

# Initial random starting points
# x, y = rand.randint(-5,5), rand.randint(-5,5)
x, y = 5, -4 # <- test with constant x,y

# Dummy x,y data
x_data = [1, 2, 3, 4, 5]
y_data = [3, 5, 7, 9, 11]

# Initial weight and bias values
w, b = 0.0, 0.0

# Learning rate
# Will help function converge to the minima
alpha = 0.01

# Momentum coefficient
# Describes how much of the old velocity we remember and use on the new velocity 
beta = 0.9

# Second momentum coefficient
# Used in the ADAM algorithm to determine step size depending on gradient size
beta2 = 0.999

# Epsilon
# Used as a very small quantity to ensure division is never by 0
epsilon = 1e-8

# I can manually switch modes here
mode = "momentum"

def main():
    history = []

    if mode == "gd":
        final_x, final_y = learn(alpha, x, y, history)

        for i in range(len(history)):
            cur_x, cur_y, cur_val = history[i]
            print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={cur_val}")

        print(f"\nFinal point: ({final_x}, {final_y})")
        print(f"Final value: {f(final_x, final_y)}")

        plot(history, "GD")
        plot3d(history)

    elif mode == "momentum":
        final_x, final_y = momentum_learn(alpha, beta, x, y, history)

        for i in range(len(history)):
            cur_x, cur_y, cur_val = history[i]
            print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={cur_val}")

        print(f"\nFinal point: ({final_x}, {final_y})")
        print(f"Final value: {f(final_x, final_y)}")

        plot(history, "Momentum GD")
        plot3d(history)

    elif mode == "linear":
        final_w, final_b = train_linear_regression(alpha, w, b, x_data, y_data, history)

        for i in range(len(history)):
            cur_w, cur_b, cur_loss = history[i]
            print(f"Step {i+1}: w={cur_w}, b={cur_b}, loss={cur_loss}")

        print(f"\nFinal parameters: w={final_w}, b={final_b}")
    
    elif mode == "adam":
        final_x, final_y = adam_learn(alpha, beta, beta2, epsilon, x, y, history)

        for i in range(len(history)):
            cur_x, cur_y, cur_val = history[i]
            print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={cur_val}")

        print(f"\nFinal point: ({final_x}, {final_y})")
        print(f"Final value: {f(final_x, final_y)}")

        plot(history, "ADAM")
        plot3d(history)

if __name__ == '__main__':
    main()
