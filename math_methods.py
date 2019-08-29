#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:02:07 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def forward_euler(fun, init, tspan, num_nodes):
    """
    Basic forward euler method
    
    Input: 
        fun - callable ode function
        init - initial condition
        tspan - [t0, tf]
        num_nodes - integer - number of nodes to use
    Output:
        time - time nodes used
        vals - y values at time nodes
    """
    
    time = np.linspace(tspan[0], tspan[1], num_nodes)
    dt = (tspan[1] - tspan[0]) / num_nodes
    vals = np.zeros(np.shape(time))
    vals[0] = init
    for i in range(num_nodes - 1):
        vals[i + 1] = vals[i] + dt * fun(time[i], vals[i])
    
    return [time, vals]

def ode12(fun, init, tspan, tol=1e-5):
    """
    Adaptive ode solver using forward euler and RK2
    
    Input:
        fun - ode function
        init - initial condition
        tspan - [t0, tf]
        tol - allowed tolerance value
    Output:
        time - time nodes used
        vals - y values at nodes
    """
    
    cur_time = tspan[0]
    final_time = tspan[1]
    
    times = [cur_time]
    vals = [init]
    
    step = (tspan[1] - tspan[0]) / 100
    
    flag = False
    
    while cur_time <= final_time:
        stage1 = fun(cur_time, vals[-1])
        stage2 = fun(cur_time + 0.5 * step, vals[-1] + 0.5 * step * stage1)
        
        lte = np.abs(stage2 - stage1)
        
        if lte > tol:
            step /= 2
            flag = True
            continue
        
        elif lte < 0.2 * tol and not flag:
            step *= 2
            continue
        
        else:
            new_val = vals[-1] + step * stage2
            vals.append(new_val)
            cur_time += step
            times.append(cur_time)
            flag = False
    
    if cur_time < final_time:
        step = final_time - cur_time
        stage1 = fun(cur_time, vals[-1])
        stage2 = fun(cur_time + 0.5 * step, vals[-1] + 0.5 * step * stage1)
        final_val = vals[-1] + step * stage2
        vals.append(final_vals)
        times.append(final_time)
    
    return [times, vals]

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

def deriv(x, y):
    """
    Compute the first derivative (second order accuracy)
    
    Input:
        x - x nodes
        y - y nodes
    Output:
        dydx - derivative approximation at x nodes
    """
    
    h = x[1:] - x[:-1]
    dydx = np.zeros(np.shape(y))
    
    backward = (y[1:-1] - y[:-2]) / (x[1:-1] - x[:-2])
    forward = (y[1:-1] - y[2:]) / (x[1:-1] - x[2:])
    weights = 2 * (x[1:-1] - x[:-2]) / (x[2:] - x[:-2])
    dydx[1:-1] = 0.5 * (weights * forward + (2 - weights) * backward)
    
    frontweight = (2 * x[0] - x[1] - x[2]) / (x[0] - x[2])
    frontback = (y[0] - y[1]) / (x[0] - x[1])
    frontfor = (y[1] - y[2]) / (x[1] - x[2])
    dydx[0] = frontweight * frontback + (1 - frontweight) * frontfor
    
    backweight = (2*x[-1] - x[-2] - x[-3]) / (x[-1] - x[-3])
    backback = (y[-1] - y[-2]) / (x[-1] - x[-2])
    backfor = (y[-2] - y[-3]) / (x[-2] - x[-3])
    dydx[-1] = backweight * backback + (1 - backweight) * backfor
    
    return dydx
    
    
def deriv2(x, y):
    """
    Compute the second derivative (first order accurate)
    
    Input:
        x - x nodes 
        y - y nodes 
    Output:
        d2ydx2 - second derivative approximation
    """
    
    d2ydx2 = np.zeros(np.shape(y))
    acenter = 2 * (x[1:-1] - x[:-2]) / (x[2:] - x[:-2])
    ccenter = 2 * (x[2:] - x[1:-1]) / (x[2:] - x[:-2])
    bcenter = -2
    
    d2ydx2[1:-1] = (acenter * y[2:] + bcenter * y[1:-1] + ccenter * y[:-2]) / \
        ((x[2:] - x[1:-1]) * (x[1:-1] - x[:-2]))
    
    afor = 2
    bfor = 2 * (x[2] - x[0]) / (x[1] - x[2])
    cfor = -2 * (x[1] - x[0]) / (x[1] - x[2])
    d2ydx2[0] = (afor * y[0] + bfor * y[1] + cfor * y[2]) / (
            (x[1] - x[0]) * (x[2] - x[0]))
    abac = 2
    bbac = 2 * (x[-3] - x[-1]) / (x[-2] - x[-3])
    cbac = -2 * (x[-2] - x[-1]) / (x[-2] - x[-3])
    d2ydx2[-1] = (abac * y[-1] + bbac * y[-2] + cbac * y[-3]) / (
            (x[-2] - x[-1]) * (x[-3] - x[-1]))
    
    return d2ydx2

def newton(fun, x0, tol=1e-5, maxiter=100):
    """
    Newtons method for finding function zero
    
    Input:
        fun - callable function to solve = 0
        x0 - initial guess
    Optional:
        tol - tolerance allowed, default value 1e-5
        maxiter - maximum allowed iterations, default value 100
    Output:
        xf - final guess value
        x - all guessed values
    """
    
    x = [x0]
    error = tol + 1
    iterat = 0
    while error > tol and iterat < maxiter:
        deriv = (fun(x[-1] + tol) - fun(x[-1] - tol)) / (2 * tol)
        xf = x[-1] - fun(x[-1]) / deriv
        
        error = np.abs(fun(xf))
        iterat += 1
        x.append(xf)
    
    if iterat == maxiter:
        print("Warning, maximum number of iterations reached")
    
    x = np.array(x)
    return [xf, x]

def secant(fun, x0, x1=None, tol=1e-5, maxiter=100):
    """
    Secant method for finding function zero
    
    Input:
        fun - callable function in question
        x0 - first initial guess
    Optional:
        x1 - second initial guess
            if not given, computed via newton iteration
        tol - tolerance value, default 1e-5
        maxiter - maximum number of iterations allowed, default 100
    Output:
        xf - final guess value
        x - all nodes used
    """
    
    if x1 == None:
        deriv = (fun(x0 + tol) - fun(x0 - tol)) / (2 * tol)
        x1 = x0 - fun(x0) / deriv
        return secant(fun, x0, x1=x1, tol=tol, maxiter=maxiter)
    else:
        x = [x0, x1]
        error = tol + 1
        iterat = 0
        while error > tol and iterat < maxiter:
            xf = x[-1] - (fun(x[-1]) * (x[-1] - x[-2])) / (
                    fun(x[-1]) - fun(x[-2]))
            error = np.abs(fun(xf))
            iterat += 1
            x.append(xf)
        x = np.array(x)
        
        if iterat == maxiter:
            print("Warning, maximum number of iterations reached")
        return [xf, x]
        
def gradient_descent(fun, x0, step, tol=1e-5, maxiter=100):
    """
    Standard gradient descent
    
    Input:
        fun - callable function to be minimized
        x0 - initial guess
        step - step size
    Optional:
        tol - allowed tolerance, default 1e-5
        maxiter - maximum number of iterations, default 100
    Output:
        xf - final position of minimum
        x - list of used values
    """
    
    x = [x0]
    error = tol + 1
    iterat = 0
    
    while error > tol and iterat < maxiter:
        deriv = (fun(x[-1] + tol) - fun(x[-1] - tol)) / (2 * tol)
        xf = x[-1] - step * deriv
        if fun(xf) >= fun(x[-1]):
            step /= 2
            continue
        
        error = np.abs(xf - x[-1])
        iterat += 1
        x.append(xf)
    
    if iterat == maxiter:
        print("Warning, maximum number of iterations reached")
    x = np.array(x)
    return [xf, x]

def backward_euler(fun, init, tspan, num_nodes):
    """
    Backward euler ODE solver
    
    Input:
        fun - callable function
        tspan - [t0, tf]
        init - initial condition
        num_nodes - number of time nodes 
    Output:
        time - time nodes used
        vals - y values at each node
    """
    
    time = np.linspace(tspan[0], tspan[1], num_nodes)
    h = (tspan[1] - tspan[0]) / num_nodes
    vals = np.zeros(num_nodes)
    vals[0] = init
    
    for j in range(num_nodes - 1):
        tempfun = lambda u: u - h * fun(time[j+1], u) - vals[j]
        [vals[j+1], tmp] = newton(tempfun, vals[j])
    
    return [time, vals]
      
def crank_nicolson(fun, init, tspan, num_nodes):
    """
    Crank-Nicolson method for ODE
    
    Input:
        fun - callable function
        init - initial condition
        tspan - [t0, tf]
        num_nodes - number of time nodes to use
    Output:
        time - time nodes used
        vals - function values 
    """
    
    time = np.linspace(tspan[0], tspan[1], num_nodes)
    h = (tspan[1] - tspan[0]) / num_nodes
    vals = np.zeros(num_nodes)
    vals[0] = init
    
    for j in range(num_nodes - 1):
        tempfun = lambda u: u - 0.5 * h * fun(time[j+1], u) - (
                vals[j] + 0.5 * h * fun(time[j], vals[j]))
        [vals[j+1], tmp] = newton(tempfun, vals[j])
    
    return [time, vals]

def ode12s(fun, init, tspan, tol=1e-5):
    """
    Adaptive ODE solver for stiff problems using forward euler and 
    crank-nicolson
    
    Input:
        fun - callable function
        init - initial condition
        tspan - [t0, tf]
    Optional:
        tol - tolerance value, default 1e-5
    Output:
        time - time nodes used
        vals - function values
    """
    
    cur_time = tspan[0]
    final_time = tspan[1]
    
    time = [cur_time]
    vals = [init]
    
    step = (tspan[1] - tspan[0]) / 100
    flag = False
    
    while cur_time < final_time:
        #temp_fun_be = lambda u: u - step * fun(time[-1] + step, u) - vals[-1]        
        #[be_step, tmp] = newton(temp_fun_be, vals[-1])
        
        temp_fun_cn = lambda u: u - 0.5 * step * fun(time[-1] + step, u) - (
                vals[-1] + 0.5 * step * fun(time[-1], vals[-1]))
        [cn_step, tmp] = newton(temp_fun_cn, vals[-1])
        
        #lte = np.abs(cn_step - be_step) / step
        
        lte = 0.5 * np.abs(fun(time[-1], vals[-1]) - fun(time[-1] + step, cn_step))
        
        if lte > tol:
            step /= 2
            flag = True
            continue
        
        elif lte < 0.7 * tol and not flag:
            step *= 2
            continue
        
        else:
            new_val = cn_step
            vals.append(new_val)
            cur_time += step
            time.append(cur_time)
            flag = False
        
    if cur_time != final_time or np.isinf(time[-1]):

        if np.isinf(time[-1]) or cur_time > final_time:
            time = time[:-1]
            vals = vals[:-1]
            
        step = final_time - time[-1]
        temp_fun = lambda u: u - 0.5 * step * fun(cur_time + step, u) - (
                vals[-1] + 0.5 * step * fun(cur_time, vals[-1]))
        [final_val, tmp] = newton(temp_fun, vals[-1])
        vals.append(final_val)
        time.append(final_time)
        
    time = np.array(time)
    vals = np.array(vals)
    
    return [time, vals]
        

def diffmat(x, num_nodes=9, der_wanted=1):
    """
    Build a differentiation matrix
    
    Input:
        x - nodes at which function values are given
        num_nodes - number of nodes to use in approximation, default 9
        der_wanted - order derivative wanted, default 1
    Output:
        Dx - differentiation matrix
    """
    
    N = len(x)
    xi = np.linspace(-1, 1, N)

    Dx = np.zeros([N, N])
    
    scale = 2 / (np.max(x) - np.min(x))
    
    # verify centered differences
    if num_nodes % 2 == 0:
        num_nodes += 1
    
    if num_nodes > N:
        num_nodes = N
    
    
    for index in range(N):
        
        # determine locations to use for approximation
        if num_nodes < N:
            if index - int((num_nodes - 1) / 2) < 0:
                index_range = [0, num_nodes]
            elif N - index - int((num_nodes - 1) / 2) - 1 <= 0:
                index_range = [N-num_nodes-1, N - 1]
            else:
                index_range = [index - int((num_nodes - 1) / 2), 
                               index + int((num_nodes - 1) / 2) + 1]
        else:
            index_range = [0, N-1]
                
        weights = compute_weights(xi, index, index_range, der_wanted)
        Dx[index, index_range[0]:index_range[1]] = weights
    
    Dx *= np.power(scale, der_wanted) * np.math.factorial(der_wanted)
    return Dx
    
def compute_weights(x, cur_index, index_range, derivative_wanted):
    """
    Compute weights for function values to approximate a derivative
    
    Input:
        x - x nodes in use
        cur_index - the index at which we want the derivative
        allowed_indices - the indices used in the approximation
        derivative_wanted - integer - the derivative order desired
    Output:
        weights - function value weights
    """
    
    n = index_range[1] - index_range[0]
    V = np.zeros([n, n])
    b = np.zeros(n)
    b[derivative_wanted] = 1
    
    diffs = x[index_range[0]:index_range[1]] - x[cur_index]
    for i in range(n):
        V[i, :] = np.power(diffs, i)
        
    weights = la.solve(V, b)
    
    return weights

