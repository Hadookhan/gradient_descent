from gradient_descent import f, nabla_f, learn

# Initial random starting points
x, y = 12, 7
# Learning rate
# Will help function converge to the minima
alpha = 0.1
    
def main():
    history = []

    nabla_x, nabla_y = nabla_f(x,y)

    learn(alpha,x,y,nabla_x,nabla_y,history)

    for i in range(len(history)):
        cur_x = history[i][0]
        cur_y = history[i][1]
        print(f"Step {i+1}: x={cur_x}, y={cur_y}, f={f(cur_x,cur_y)}")


if __name__ == '__main__':
    main()
