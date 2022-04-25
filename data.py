import numpy as np
import random
import math

def solve_square_function(a, b, c):
    if (a == 0):
        if (b == 0):
            return [None]
        else:
            return [-c/b]
        return [None] 
    delta = b * b - 4 * a * c;
    if (delta > 0):
        x1 = (float)((-b + math.sqrt(delta)) / (2 * a));
        x2 = (float)((-b - math.sqrt(delta)) / (2 * a));
        return [x1,x2]
    elif (delta == 0):
        x1 = (-b / (2 * a))
        return [x1]
    else:
        return [None]
    

def circle(x):
    y = solve_square_function(1,-1, (x-0.5)**2)
    return y

def create_dummy_data(n=50, ir=4):
    num_mij = int(n/(1+ir))
    num_maj = int(n - num_mij)
    x_maj = np.linspace(0.0, 1.0,num=num_maj)
    x_rand = np.random.normal(0.0, 0.2, size=(num_maj, 2))
    y_maj = []
    for x in x_maj:
        y = circle(x)
        if len(y) > 1:
            y = random.choice(y)
        else:
            y = y[0]
        y_maj.append(y - 0.2)
    y_maj = np.array(y_maj) 
    data_maj = np.transpose(np.array([x_maj - 0.2, y_maj]), (1, 0)) + x_rand
    std, mean = data_maj.std(0), data_maj.mean(0)
    data_mij = np.random.normal(mean/2, std/2, size=(num_mij, 2))
    return data_maj, data_mij
