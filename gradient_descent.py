# Function I will be minimising
def f(x,y):
    return x**2 + 10*y**2

# Taking the gradient of the function f
# This will be used to help calculate a reverse gradient
def nabla_f(x,y):
    return 2*x, 20*y

# Train x and y to converge to the minima point
# Then I update the gradients for the new x,y values
def learn(alpha,x,y,history,max_iters=10000):

    for _ in range(max_iters):
        nabla_x, nabla_y = nabla_f(x,y)

        x -= alpha * nabla_x
        y -= alpha * nabla_y

        val = f(x,y)

        history.append((x,y,val))

        # Gets gradients of new x,y values
        new_nabla_x, new_nabla_y = nabla_f(x, y)

        # Breaks out of loop when the point is near enough to zero
        # This indicates when the gradient is 0 -> critical point
        if (abs(new_nabla_x) < 1e-6 and abs(new_nabla_y) < 1e-6):
            break
        if (abs(val) < 1e-6):
            break
        # If x or y value get too big, function is diverging
        if (abs(x) > 1e6 or abs(y) > 1e6):
            print("Diverging...")
            break
    return x,y

# Learning based off momentum (memory of previous velocity)
def momentum_learn(alpha,beta,x,y,history,max_iters=10000):
    # Initial velocity
    vx,vy = 0.0,0.0

    for _ in range(max_iters):
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

# ADAM (Adaptive Moment Estimation)
# m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
# v_t = \beta_2 v_{t-1} + (1-\beta_2)g^2_t
# Correction step:
#\hat{m}_t=\frac{m_t}{1-\beta^t_1}
#\hat{v}_t=\frac{v_t}{1-\beta^t_2}
# Parameter update:
#x_{new} = x_{old}-\alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}

def adam_learn(alpha, beta1, beta2, epsilon, x, y, history, max_iters=10000):
    mx, my = 0.0,0.0    # First moment
    vx, vy = 0.0,0.0    # Second moment

    for t in range(1, max_iters+1):
        nabla_x, nabla_y = nabla_f(x,y)

        # First moment update
        mx= beta1*mx+(1-beta1)*nabla_x
        my= beta1*my+(1-beta1)*nabla_y

        # Second moment update
        vx= beta2*vx+(1-beta2)*(nabla_x**2)
        vy= beta2*vy+(1-beta2)*(nabla_y**2)

        # Correction step
        mx_hat = mx / (1 - beta1 ** t)
        my_hat = my / (1 - beta1 ** t)

        vx_hat = vx / (1 - beta2 ** t)
        vy_hat = vy / (1 - beta2 ** t)

        # Parameter update
        x -= alpha * mx_hat / ((vx_hat ** 0.5) + epsilon)
        y -= alpha * my_hat / ((vy_hat ** 0.5) + epsilon)

        val = f(x, y)
        history.append((x, y, val))

        new_nabla_x, new_nabla_y = nabla_f(x, y)

        if abs(new_nabla_x) < 1e-6 and abs(new_nabla_y) < 1e-6:
            break
        if abs(val) < 1e-6:
            break
        if abs(x) > 1e6 or abs(y) > 1e6:
            print("Diverging...")
            break

    return x, y



# MSE:
# weights -> \frac{1}{n}\sum_{i=1}^n(y_i-(wx_i+b))^2
# Gradients:
# dMSE/dw -> \frac{1}{n}\sum_{i=1}^n(y_i-(wx_i+b))(-x_i)
# dMSE/db -> \frac{1}{n}\sum_{i=1}^n(y_i-(wx_i+b))(-1)

def MSE(w, x_data, b, y_data):
    n=len(x_data)
    SE=0
    for i in range(n):
        SE+=(y_data[i]-(w*x_data[i]+b))**2

    return SE/n

def grad_w(w, x_data, b, y_data):
    n = len(x_data)
    total = 0

    for i in range(n):
        total += (y_data[i] - (w*x_data[i] + b)) * (-x_data[i])

    return (2/n) * total

def grad_b(w, x_data, b, y_data):
    n = len(x_data)
    total = 0

    for i in range(n):
        total += (y_data[i] - (w*x_data[i] + b)) * (-1)

    return (2/n) * total

def train_linear_regression(alpha, w, b, x_data, y_data, history, max_iters=10000):
    for _ in range(max_iters):
        dw = grad_w(w, x_data, b, y_data)
        db = grad_b(w, x_data, b, y_data)

        # Update weights and bias
        w -= alpha * dw
        b -= alpha * db

        loss = MSE(w, x_data, b, y_data)
        history.append((w,b,loss))

        if abs(dw) < 1e-6 and abs(db) < 1e-6:
            break
        if loss < 1e-6:
            break

    return w,b
