import numpy as np
def idk(w, b, x, y, alpha=0.1, num_iterations=1000):
    
    for _ in range(num_iterations):
        y_hat = feed_forward(w, b, x)
        m = 1
        w, b, j = backpropagation(m,w, b, x, y_hat, y, alpha)
    
    return w, b, j
    
    
def feed_forward(w, b, x):
    a = w * x + b
    y_hat = a
    return y_hat

def compute_cost(m, y_hat, y):
    j = 0
    for i in range(m):
        j = j + (y_hat - y) ** 2
    j = j / (2 * m)
    return j

def compute_dj_dw(m, x, y_hat, y):
    dj_dw = 0
    for i in range(m):
        dj_dw = dj_dw + (y_hat - y) * x
    dj_dw = dj_dw / m
    return dj_dw

def compute_dj_db(m, y_hat, y):
    dj_db = 0
    for i in range(m):
        dj_db = dj_db + (y_hat - y)
    dj_db = dj_db / m
    return dj_db

def backpropagation(m, current_w, current_b, x, y_hat, y, alpha):
    j = compute_cost(m, y_hat, y)
    dj_dw = compute_dj_dw(m, x, y_hat, y)
    dj_db = compute_dj_db(m, y_hat, y)
    w_new = current_w - alpha * (dj_dw)
    b_new = current_b - alpha * (dj_db)
    if w_new - current_w < 0.0000000000000001 and b_new - current_b < 0.0000000000000001:
        return w_new, b_new, j
    print(w_new, b_new, j)
    return w_new, b_new, j

w, b, j = idk(0, 0, 1, 5)