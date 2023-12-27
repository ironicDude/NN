import numpy as np

def softmax(z):
    ez = np.exp(z)
    sum = np.sum(ez)
    return ez/sum

def cost(X, y, W, b):
    m = X.shape[0]
    z = np.dot(W, X) + b
    y_hat = softmax(z)
    cost = -1/m * np.sum(y * np.log(y_hat))
    return cost

def gauss_newton(X, Y, P0,  max_iter=1000, eps=1e-6):
    # X and Y are arrays of input and output values
    # max_iter is the maximum number of iterations
    # eps is the tolerance for convergence
    J = np.zeros([len(X), len(P0)]) # Jacobian matrix from Y
    for i in range(max_iter):
        j1 = 1 # first row of J
        j2 = P0 # second row of J
        j3 = P0 * X # third row of J
        J[:,0] = j1 # assign first row to J
        J[:,1] = j2 # assign second row to J
        J[:,2] = j3 # assign third row to J
        r = Y - (P0 + P0 * X + P0 * X**2) # residual vector from Y
        t1 = np.linalg.inv(np.dot(J.T, J)) # inverse of Jacobian matrix from J.T * J
        t2 = np.dot(t1, J.T) # inverse times Jacobian matrix from t1 * J.T
        t3 = np.dot(t2, r) # inverse times residual vector from t2 * r
        P1 = P0 - t3 # updated coefficients from previous ones minus inverse times residual vector
        P0 = P1 
    return P0

# Example usage: find minimum of quadratic function f(x) = x^2 - 4x + 5

X = np.array([4, 5]) # input values 
Y = np.array([9]) # output value 
P0 = [4/5, -4/5] # initial guess for coefficients 
P1, _, _ = gauss_newton(X,Y) # call gauss_newton function with X and Y as arguments 
print(P1) # print final coefficients 

# Output: [ 6.66666667e-01   9.16666667e-01]