#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 11:35:24 2019

@author: jerome
"""

import numpy as np
import matplotlib.pyplot as plt

import ode


def polyinterp(x_nodes, y_values, new_nodes):
    """
    Polynomial interpolation using the second barycentric form
    
    Input:
        x_nodes - x nodes of original data
        y_values - original function values
        new_nodes - points to interpolate at
    Output:
        new_yvals - interpolated y values
    """
    
    weights = np.ones(len(x_nodes))
    for j in range(len(x_nodes)):
        for k in range(len(x_nodes)):
            if k == j:
                continue
            else:
                weights[j] /= (x_nodes[j] - x_nodes[k])
    
    new_yvals = np.zeros(len(new_nodes))
    
    for l in range(len(new_nodes)):
        if new_nodes[l] in x_nodes:
            new_yvals[l] = y_values[x_nodes == new_nodes[l]]
        else:
            numerator = np.sum(weights * y_values / (new_nodes[l] - x_nodes))
            denominator = np.sum(weights / (new_nodes[l] - x_nodes))
            new_yvals[l] = numerator / denominator
            
    return new_yvals

def compute_mvalues(x_nodes, y_values, ends="natural"):
    """
    Helper function for cubic splines - computes m values for computation
    
    Input:
        x_nodes - x nodes of original data
        y_values - y values of original data
        Optional - ends - default 'natural'
            if ends = [yp1, yp2], use those as the derivative values 
            at the ends
    Output:
        mvals - m values (function as weights)
    """
    
    n = len(x_nodes)
    
    # get spacings
    hvals = x_nodes[1:] - x_nodes[:-1]
    
    delta = np.zeros(n)
    delta[1:-1] = (y_values[2:] - y_values[1:-1]) / hvals[1:] - \
                    (y_values[1:-1] - y_values[:-2]) / hvals[0:-1]
    
    
        
    diag = np.ones(n)
    diag[1:-1] = 2 * (hvals[:-1] + hvals[1:])
    upperdiag = np.copy(hvals)
    upperdiag[0] = 0
    lowerdiag = np.copy(hvals)
    lowerdiag[-1] = 0
    
    A = np.diagflat(diag) + np.diagflat(upperdiag, 1) + np.diagflat(lowerdiag, -1)
    
    if ends == "periodic":
        A[0, -1] = 1
        A[-1, 0] = 1
    
    if type(ends) == type(""):
        ends = ends.lower()
        if ends not in ["natural", "periodic"]:
            print("Warning, unknown ends specifier, interpreting as 'natural'")
            ends = "natural"
    elif type(ends) not in [type([]), type(np.array([]))]:
        print("Warning unknown type for ends specifier, interpreting as 'natural")
        ends = 'natural'
    else:
        [yp1, yp2] = ends
        
        delta[0] = yp1 - (y_values[1] - y_values[0]) / hvals[0]
        delta[-1] = yp2 - (y_values[-1] - y_values[-2]) / hvals[-1]
        A[0, :2] = [-2 * hvals[0], -hvals[0]]
        A[-1, -2:] = [hvals[-1], 2 * hvals[-1]]
            
    A /= 6

    mvals = np.linalg.solve(A, delta)
    
    return mvals
    
def compute_spline_coefs(mvals, hvals, y_values, ends=None):
    """
    Helper for splines - compute standard coefficients
    
    Input:
        mvals - computed m weights for spline
        hvals - step sizes
        y_values - known y points of data
        Optional - ends - defaults to None - no change
                if is a list, uses these as the new 
                yprime(x0) and yprime(xn)
    Output:
        coef1 
        coef2 
        coef3
        These are the three coefficients in the spline is
        coef1 * (x - xi)^3 + coef2 * (x - xi)^2 + coef3 * (x - xi)
    """
    coef1 = (mvals[1:] - mvals[:-1]) / (6 * hvals)
    coef2 = 0.5 * mvals[1:]
    coef3 = -coef1 * np.power(hvals, 2) + coef2 * hvals + \
            (y_values[1:] - y_values[:-1]) / hvals
    #if type(ends) in [type([]), type(np.array([]))]:
    #    print("here")
    #    coef3[-1] = ends[-1]
            
    return [coef1, coef2, coef3]

def cubespline(x_nodes, y_values, new_nodes, ends="natural"):
    """
    Build cubic spline interpolant
    
    Input:
        x_nodes - x nodes of original data
        y_values - y values of original data
        new_nodes - new nodes to get data values
        Optional - ends - string "natural, periodic"
            indicates type of spline
            Natural spline - zero second derivative at ends
            Periodic spline - second derivative is periodic
            true_vals - true derivative values
            [yp1, yp2] - specificed derivative values at ends
    Output: 
        new_values - new y values at new_nodes
    """
    
    if ends == "true_vals":
        D = ode.diffmat(x_nodes, num_nodes=5)
        yprime_true = np.matmul(D, y_values)
        ends = [yprime_true[0], yprime_true[-1]]
        
        return cubespline(x_nodes, y_values, new_nodes, ends=ends)
    
    mvals = compute_mvalues(x_nodes, y_values, ends=ends)
    
    hvals = x_nodes[1:] - x_nodes[:-1]
    
    # build piecewise function
    [coef1, coef2, coef3] = compute_spline_coefs(mvals, hvals, y_values, ends=ends)
    
    new_values = np.zeros(len(new_nodes))
    xsel = x_nodes[1:]
    ysel = y_values[1:]
    
    for j in range(len(new_nodes)):
        cur = new_nodes[j]
        index = (cur >= x_nodes[:-1]).astype(int) + (cur < x_nodes[1:]).astype(int) > 1
        
        if (sum(index.astype(int))) == 0:
            index = -1
        
        new_values[j] = coef1[index] * np.power(cur - xsel[index], 3) + \
                        coef2[index] * np.power(cur - xsel[index], 2) + \
                        coef3[index] * (cur - xsel[index]) + ysel[index]
        
    #for j in range(len(new_nodes)):
    #    new_values[j] = np.piecewise(new_nodes[j], cond_list, fun_list)
    
    return new_values
        
def splineder(x_nodes, y_values, new_nodes, ends="natural"):
    """
    Compute the derivative of a function using its cubic spline
    
    Input:
        x_nodes - x nodes of data
        y_values - corresponding y values
        new_nodes - new x coordinates
        optional - ends - type of spline, default to 'natural'
    Output:
        yprime - apprxoimated derivative
    """
    
    if ends == "true_vals":
        D = ode.diffmat(x_nodes, num_nodes=5)
        yprime_true = np.matmul(D, y_values)
        ends = [yprime_true[0], yprime_true[-1]]
        
        return splineder(x_nodes, y_values, new_nodes, ends=ends)
    
    mvals = compute_mvalues(x_nodes, y_values, ends=ends)
    hvals = x_nodes[1:] - x_nodes[:-1]
    [coef1, coef2, coef3] = compute_spline_coefs(mvals, hvals, y_values, ends=ends)
    
    # changes for second derivative
    coef1 *= 3
    coef2 *= 2
    
    yprime = np.zeros(len(new_nodes))
    xsel = x_nodes[1:]
    ysel = y_values[1:]
    
    for j in range(len(new_nodes)):
        cur = new_nodes[j]
        index = (cur >= x_nodes[:-1]).astype(int) + (cur < x_nodes[1:]).astype(int) > 1
        
        if (sum(index.astype(int))) == 0:
            index = -1
        
        yprime[j] = coef1[index] * np.power(cur - xsel[index], 2) + \
                        coef2[index] * (cur - xsel[index]) + \
                        coef3[index]
    return yprime   

def splineder2(x_nodes, y_values, new_nodes, ends="natural"):
    """
    Compute the second derivative of a function using its cubic spline
    
    Input:
        x_nodes - x nodes of data
        y_values - corresponding y values
        new_nodes - new x coordinates
        optional - ends - tyep of spline, default to 'natural'
    Output:
        yprime2 - approximated second derivative
    """
    
    if ends == "true_vals":
        D = ode.diffmat(x_nodes, num_nodes=5)
        yprime_true = np.matmul(D, y_values)
        ends = [yprime_true[0], yprime_true[-1]]
        
        return splineder2(x_nodes, y_values, new_nodes, ends=ends)
    
    mvals = compute_mvalues(x_nodes, y_values, ends=ends)
    hvals = x_nodes[1:] - x_nodes[:-1]
    [coef1, coef2, coef3] = compute_spline_coefs(mvals, hvals, y_values, ends=ends)
    
    # changes for second derivative
    coef1 *= 6
    coef2 *= 2
    
    yprime2 = np.zeros(len(new_nodes))
    xsel = x_nodes[1:]
    ysel = y_values[1:]
    
    for j in range(len(new_nodes)):
        cur = new_nodes[j]
        index = (cur >= x_nodes[:-1]).astype(int) + (cur < x_nodes[1:]).astype(int) > 1
        
        if (sum(index.astype(int))) == 0:
            index = -1
        
        yprime2[j] = coef1[index] * (cur - xsel[index]) + coef2[index]

    return yprime2 



n = 11
x_eq = 4 * np.linspace(-1, 1, n)
theta = np.linspace(0, np.pi, n)
x_cheb = 4 * -np.cos(theta)
#x = np.linspace(-1, 1, 10)
fun = lambda x: np.exp(-np.power(x, 2))

y_eq = fun(x_eq)
y_cheb = fun(x_cheb)

x_new = 4 * np.linspace(-1, 1, 1000)

y_eq_new = polyinterp(x_eq, y_eq, x_new)
y_cheb_new = polyinterp(x_cheb, y_cheb, x_new)

y_true = fun(x_new)
eq_error = np.abs(y_eq_new - y_true)
cheb_error = np.abs(y_cheb_new - y_true)

plt.figure()
plt.plot(x_eq, y_eq, 'o-', label='eq')
plt.plot(x_cheb, y_cheb, 'd-', label='cheb')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.figure()
plt.plot(x_new, y_eq_new, label="eq spline")
plt.plot(x_new, y_cheb_new, label="cheb spline")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

plt.figure()
#plt.plot(x_new, eq_error, label="eq spline")
plt.plot(x_new, cheb_error, label="cheb spline")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

theta = np.linspace(0, np.pi, 201)
x_trial = 4 * -np.cos(theta)

y_trial = polyinterp(x_cheb, y_cheb, x_trial)

x_final = np.linspace(-4, 4, 10000)
y_final = cubespline(x_trial, y_trial, x_final)

y_final = cubespline(x_eq, y_eq, x_final)
plt.figure()
plt.plot(x_cheb, y_cheb, 'o-')
plt.plot(x_final, y_final)
plt.show()

y_true = fun(x_final)

error = np.abs(y_true - y_final)

plt.figure()
plt.plot(x_final, error)
plt.show()