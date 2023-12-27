import numpy as np
import math
import matplotlib.pyplot as plt

fnum =2
if fnum == 1:
    f = lambda x, y: (1 - x)**2 + 100 * ((y - x**2)**2)
    delfx = lambda x, y: 2 * (x - 1) - 400 * x * (y - x**2)
    delfy = lambda x, y: 200 * (y - x**2)
    H11 = lambda x, y: 2 - 400 * y + 1200 * x**2
    H12 = lambda x, y: -400 * x
    H22 = lambda x, y: 200
    x0 = -1.5
    y0 = 2.5
    xstar = 1
    ystar = 1
elif fnum == 2:
    f = lambda x, y: (7 * x * y) / math.exp(x**2 + y**2)
    delfx = lambda x, y: (7 * y * (1 - 2 * x**2)) / math.exp(x**2 + y**2)
    delfy = lambda x, y: (7 * x * (1 - 2 * y**2)) / math.exp(x**2 + y**2)
    H11 = lambda x, y: (14 * x * y * (2 * x**2 - 3)) / math.exp(x**2 + y**2)
    H12 = lambda x, y: (7 * (1 - 2 * x**2 - 2 * y**2 + 4 * x**2 * y**2)) / math.exp(x**2 + y**2)
    H22 = lambda x, y: (14 * x * y * (2 * y**2 - 3)) / math.exp(x**2 + y**2)
    x0 = -1.5
    y0 = 2.5
    xstar = [-1, 1]
    xstar = [element / math.sqrt(2) for element in xstar]
    ystar = [1, -1]
    ystar = [element / math.sqrt(2) for element in ystar]
    
elif fnum == 3:
    pass
    
def RM(H11, H12, H22, x=[x0], y=[y0], _lambda=10, max_iter=10000, thresh=1e-6):
    delf = [1]
    for i in range(max_iter):
        temp1 = np.array([delfx(x[-1], y[-1]), delfy(x[-1], y[-1])])
        H = np.array([[H11(x[-1], y[-1]), H12(x[-1], y[-1])], 
                      [H12(x[-1], y[-1]), H22(x[-1], y[-1])]])
        temp2 = -np.linalg.inv(H + _lambda * np.eye(2)) @ temp1
        xtemp = x[-1] + temp2[0]
        ytemp = y[-1] + temp2[1]
        if f(xtemp, ytemp) < f(x[-1], y[-1]):
            _lambda *= 0.1
            x.append(xtemp)
            y.append(ytemp)
            delf.append(abs(f(x[-1], y[-1]) - f(x[-2], y[-2])))
        else:
            _lambda *= 10
            
        if len(delf) > 1 and abs(delf[-1] - delf[-2]) < thresh:
            break
    
    return x, y, i + 1, delf

x, y, _, _ = RM(H11, H12, H22)

# Plotting the trajectory
plt.figure(figsize=(8, 6))
plt.plot(x, y, '-o', color='blue')
plt.scatter(x[-1], y[-1], color='red', label='Final point')  # Marking the final point in red
plt.scatter(x[0], y[0], color='green', label='Start point')  # Marking the starting point in green
plt.xlabel('x')
plt.ylabel('y')
plt.title('Optimization Path')
plt.legend()
plt.grid(True)
plt.show()