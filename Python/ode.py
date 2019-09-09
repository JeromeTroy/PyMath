#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:02:07 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as ode

from nonlinsolve import newton
from nonlinsolve import secant
from nonlinsolve import gradient_descent

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
    
    init = np.array(init)
    m = len(init)
    
    time = np.linspace(tspan[0], tspan[1], num_nodes)
    dt = (tspan[1] - tspan[0]) / num_nodes
    
    vals = np.zeros([m, num_nodes])
    vals[:, 0] = init
    for i in range(num_nodes - 1):
        vals[:, i + 1] = vals[:, i] + dt * fun(time[i], vals[:, i])
    
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
    init = np.array(init)
    vals = [init]
    
    step = (tspan[1] - tspan[0]) / 100
    
    flag = False
    
    while cur_time <= final_time:
        stage1 = fun(cur_time, vals[-1])
        stage2 = fun(cur_time + 0.5 * step, vals[-1] + 0.5 * step * stage1)
        

        lte = np.linalg.norm(stage2 - stage1)
        
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
    
    vals = np.array(vals)
    times = np.array(times)
    return [times, vals]



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



def odefun(t, x):
    
    beta = 2
    gamma = 1
    f = np.sin(2 * np.pi * t)
    
    der = np.zeros(np.shape(x))
    der[0] = x[1]
    
    der[1] = f - beta * x[1] - gamma * x[0]
    
    return der


y0 = np.array([0, 1])
tspan = [0, 10]

sol = ode.solve_ivp(odefun, tspan, y0)

[my_time, my_sol] = ode12(odefun, y0, tspan, tol=1e-2)
time = sol.t
y = sol.y

my_y = my_sol[:, 0]

plt.figure()

plt.plot(time, y[0,:], label="scipy")
plt.plot(my_time, my_y, label="mine")

plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()

plt.show()