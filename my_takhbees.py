import numpy as np
def idk(w, b, x, y, alpha=0.1, num_iterations=1000):
    
    for _ in range(num_iterations):
        a = w * x + b
        y_hat = a
        m = 1
        
        j = 0
        for i in range(m):
            j = j + (y_hat - y) ** 2
        j = j / (2 * m)
        
        dj_dw = 0
        for i in range(m):
            dj_dw = dj_dw + (y_hat - y) * x
        dj_dw = dj_dw / m
        
        dj_db = 0
        for i in range(m):
            dj_db = dj_db + (y_hat - y)
        dj_db = dj_db / m
        w = w - alpha * (dj_dw)
        b = b - alpha * (dj_db)
        print(w, b, j)
    
    return w, b, j

w, b, j = idk(0, 0, 1, 5)