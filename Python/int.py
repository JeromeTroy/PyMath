import numpy as np

def quad7(f, xspan):
    """
    Quadrature in 1D with Degree of Exactness 7

    Input:
        f - function to integrate
        xspan - span over which to integrate
    Output:
        integral - the integral of f over xspan
    """

    # gauss nodes for DOE 7
    lag_nodes = np.array([-np.sqrt(2/5), 0, np.sqrt(2/5)])

    true_nodes = (xspan[0] + xspan[1]) / 2 + (xspan[1] - xspan[0]) / 2 *lag_nodes
