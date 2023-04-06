import numpy as np

def circle_fitting(xi, yi):
    M = np.array([[np.sum(xi ** 2), np.sum(xi * yi), np.sum(xi)],
                  [np.sum(xi * yi), np.sum(yi ** 2), np.sum(yi)],
                  [np.sum(xi), np.sum(yi), 1*len(xi)]])
    Y = np.array([[-np.sum(xi ** 3 + xi * yi ** 2)],
                  [-np.sum(xi ** 2 * yi + yi ** 3)],
                  [-np.sum(xi ** 2 + yi ** 2)]])

    M_inv = np.linalg.inv(M) if not np.linalg.det(M) == 0 else 0
    X = np.dot(M_inv, Y)
    a = float(- X[0] / 2) if abs(float(- X[0] / 2)) > 1e-10 else 0.0
    b = float(- X[1] / 2) if abs(float(- X[1] / 2)) > 1e-10 else 0.0
    r = float(np.sqrt((a ** 2) + (b ** 2) - X[2]) if (a ** 2) + (b ** 2) - X[2] > 0 else 0)

    return np.array([a, b, r])

def calc_distance(argument):
    return np.sqrt((argument[0][0]-argument[1][0])**2+(argument[0][1]-argument[1][1])**2)

def line_fitting(xi, yi):
    covariance = np.cov(xi, yi)
    a = covariance[0][1]/covariance[0][0]
    b = np.average(yi)-a*np.average(xi)
    r = covariance[0][1]/np.sqrt(covariance[0][0]*covariance[1][1])
    
    return np.array([[xi[0],a*xi[0]+b],[xi[-1],a*xi[-1]+b]]),r

