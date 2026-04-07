# Function I will be minimising
def f(x,y):
    return x**2 + 10*y**2

# Taking the gradient of the function f
# This will be used to help calculate a reverse gradient
def nabla_f(x,y):
    return 2*x, 20*y

# Train x and y to converge to the minima point
# Then I update the gradients for the new x,y values
def learn(alpha,x,y,history):

    while True:
        nabla_x, nabla_y = nabla_f(x,y)

        x -= alpha * nabla_x
        y -= alpha * nabla_y

        val = f(x,y)

        history.append((x,y,val))
        # Breaks out of loop when the point is near enough to zero
        # This indicates when the gradient is 0 -> critical point
        if (abs(nabla_x) < 1e-6 and abs(nabla_y) < 1e-6):
            break
        if (abs(val) < 1e-6):
            break
        # If x or y value get too big, function is diverging
        if (abs(x) > 1e6 or abs(y) > 1e6):
            print("Diverging...")
            break
    return x,y

# Learning based off momentum (memory of previous velocity)
def momentum_learn(alpha,beta,x,y,history):
    # Initial velocity
    vx,vy = 0.0,0.0

    while True:
        nabla_x, nabla_y = nabla_f(x,y)

        vx = beta*vx - alpha*nabla_x
        vy = beta*vy - alpha*nabla_y

        x += vx
        y += vy

        val = f(x,y)
        history.append((x,y,val))

        if (abs(nabla_x) < 1e-6 and abs(nabla_y) < 1e-6):
            break
        if (abs(val) < 1e-6):
            break
        if (abs(x) > 1e6 or abs(y) > 1e6):
            print("Diverging...")
            break

    return x,y