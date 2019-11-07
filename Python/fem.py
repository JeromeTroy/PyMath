import numpy as np
import matplotlib.pyplot as plt

def hatfun(x, index, xnodes):
    """
    Hat functions over an interval

    Input:
        x - x position
        index - current index
        xnodes - current x nodes
    Output:
        evaluation of hat function at x
    """

    # storage space
    out = np.zeros(np.shape(x))

    if index == 0:
        lower = 0
        upper = 1
    elif index == len(xnodes) - 1:
        lower = len(xnodes) - 2
        upper = len(xnodes) - 1
    else:
        lower = index - 1
        upper = index + 1

    for pos in range(len(x)):
        if x[pos] < xnodes[lower] or x[pos] > xnodes[upper]:
            out[pos] = 0
        elif x[pos] < xnodes[index]:
            out[pos] = (x[pos] - xnodes[lower]) / (xnodes[index] - xnodes[lower])
        else:
            out[pos] = (xnodes[upper] - x[pos]) / (xnodes[upper] - xnodes[index])
    return out

x = np.linspace(0, 1, 5)
print(x)
val = [0.25, 0.4]
val = np.array(val)
sol = hatfun(val, 1, x)

print(sol)
