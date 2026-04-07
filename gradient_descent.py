


# Function I will be minimising
def f(x,y):
    return x**2 + y**2 # This is a formula for a parabola

# Taking the gradient of the function f
# This will be used to help calculate a reverse gradient
def nabla_f(x,y):
    return 2*x, 2*y

# Train x and y to converge to the minima point
# Then I update the gradients for the new x,y values
def learn(alpha,x,y,nabla_x,nabla_y,history):

    while True:
        x -= alpha * nabla_x
        y -= alpha * nabla_y
        nabla_x, nabla_y = nabla_f(x,y)
        history.append((x,y))
        # Breaks out of loop when the point is near enough to zero
        # This indicates when the gradient is 0 -> critical point
        if abs(nabla_x) < 1e-6 and abs(nabla_y) < 1e-6:
            break
    return