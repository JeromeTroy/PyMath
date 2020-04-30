#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 16:02:07 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la
from scipy.special import factorial
import matplotlib.pyplot as plt
import scipy.integrate as ode

import linalg as mla



EPS = 1e-16


# EXPLICIT METHODS

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


    
# IMPLICIT METHODS

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


# building differentiation matrices

# computing weights
def get_weights(delta, der=1, order=2):
    """
    Compute weights for function values to approximate derivatives
    
    Input:
        delta - vector of finite differences
    Optional:
        der - derivative wanted, integer
        order - order of accuracy wanted, integer
    Output:
        w - weights for function values, vector
    """
    
    # build linear system
    mat_size = len(delta)
    A = np.tile(delta, [mat_size, 1])
    powers = np.tile(np.arange(0, mat_size), [mat_size, 1]).T 
    denom = factorial(powers)
    A = np.divide(np.power(A, powers), denom)
    
    # right hand side
    e = np.zeros(mat_size)
    e[der] = 1
    
    # compute weights
    w = np.linalg.solve(A, e)
    return w

# determining weighting locations
def select_nodes(length, node, num_nodes, quiet=False):
    """
    Select nodes based on a specified node and width
    
    Input:
        length - integer, length of vector to index
        node - integer, current index
        num_nodes - integer - radius of nodes needed (to left and right)
    Optional:
        quiet - boolean, indicates whether to display output and progress
    Output:
        lower, upper - integers, indices to start and end at
    """
    
    # initial guesses
    lower = node - num_nodes
    upper = node + num_nodes + 1
    ###################################
    #print("lower, upper: ", lower, upper)
    ####################################
    # unshifted
    flag = False
    if lower < 0:
        # went too far to left, shift to right
        if not quiet:
            print("Shifting right")
        extras = abs(lower)
        lower = 0
        upper += extras
        # shifted once
        flag = True

    if upper > length:
        # too far to right, shift left 
        if not flag:
            # have not already shifted right
            if not quiet:
                print("Shifting left")
            extras = upper - length
            lower -= extras
            upper = length
            flag = True
            if lower < 0:
                # too many nodes required
                if not quiet:
                    print("Warning: too many nodes requested, defaulting to whole interval")
                lower = 0
        else:
            # shifted right before, too many nodes requested
            if not quiet:
                print("Warning: too many nodes requested, defaulting to whole interval")
            lower = 0
            upper = length
    ##############################################
    #print("updated: lower, upper: ", lower, upper)
    ##############################################
    return lower, upper

# build the differentiation matrix
def diffmat(x, der=1, order=2, quiet=True):
    """
    Build a differentiation matrix
    
    Input:
        x - nodes on which function is defined
    Optional:
        der - derivative wanted, integer, defaults to 1
        order - order of accuracy, integer, defaults to 2
        quiet - boolean, indicates if output is expressed, defaults to False
    Output:
        D - differentiation matrix 
    """
    
    # matrix of finite differences
    Delta_mat = np.tile(x, [len(x), 1]).T
    Delta_mat -= Delta_mat.T 

    # allocate space for D
    D = np.zeros(np.shape(Delta_mat))
    
    # number of nodes necessary for requirements
    if der % 2 == 1:
        num_nodes = int((der + order - 1) / 2)
    else:
        num_nodes = int((der + order) / 2) - 1
    
    # iterate over each node
    for node in range(len(x)):
        # select necessary nodes
        lower, upper = select_nodes(len(x), node, num_nodes, quiet=quiet)
        # relavant finite differences
        delta = Delta_mat[lower : upper, node]
        ################################
        #print(delta)
        ################################
        # finite difference weights
        w = get_weights(delta, der=der, order=order)
        
        # input into D
        D[node, lower : upper] = w
    
    return D
            
                


def build_vandermonde(vals, power):
    """
    Build a vandermonde matrix
    
    Input:
        vals - values to use
        power - maximum power for system
    Output:
        V - m x n matrix - m = len(vals), n = power + 1
    """
    
    n = power + 1
    m = len(vals)
    V = np.tile(vals, [n, 1]).T
    powers = np.arange(0, n)
    powers_mat = np.tile(powers, [m, 1])
    
    V = np.power(V, powers_mat)
    return V.T
    
def rk3(fun, init, tspan, num_nodes):
    """
    3rd order Runge Kutta method (explicit)
    
    Input:
        fun - ode fun (callable)
        init - initial condition
        tspan - [t0, tf]
        num_nodes - integer - number of time nodes
    Output:
        time - time nodes 
        vals - y values at time nodes
    """
    
    init = np.array(init)
    m = len(init)
    
    time = np.linspace(tspan[0], tspan[1], num_nodes+1)
    dt = (tspan[1] - tspan[0]) / num_nodes
    
    vals = np.zeros([m, num_nodes+1])
    vals[:, 0] = init
    
    for i in range(num_nodes):
        stage1 = fun(time[i], vals[:,i])
        stage2 = fun(time[i+1], vals[:,i] + dt*stage1)
        stage3 = fun(time[i] + 0.5*dt, vals[:,i] + 0.25*dt*(stage1 + stage2))
        
        vals[:,i+1] = vals[:,i] + dt / 6 * (stage1 + stage2 + 4*stage3)
    
    return [time, vals]


def grid_refine(xspan, error_fun, tol=1e-6, n_init=10, max_iter=10):
    """
    Refine a grid
    
    Input:
        xspan - [a, b]
        error_fun - callable, parameters x, dx, gives the error of the discretization
    Optional:
        tol - tolerance value for error, defaults to 10^-6
        n_init - initial number of x points, defaults to 10
        max_iter - maximum number of allowed iterations, defaults to 10
    Output:
        x - refined x positions
    """
    
    # loop preliminaries
    it = 0
    x = np.linspace(xspan[0], xspan[1], n_init)
    dx = x[1] - x[0]
    
    # is it good enough?
    err = error_fun(x, dx)
    refine_locs = err > tol
    
    refine = np.sum(refine_locs) > 0
    
    # refinement loop
    while refine:
        # refine positions
        x = refine_x(x, refine_locs)
        
        dx /= 2
        # update error and where needs to be refined
        err = error_fun(x, dx)
        refine_locs = err > tol
        refine = np.sum(refine_locs) > 0
        
        # do not exceed max_iter iterations
        it += 1
        if it > max_iter:
            print("Warning, maximum number of refinements reached")
            break
    print("total iterations: ", it)
    return x
        
def refine_x(x_curr, refine_locs):
    """
    Refine the x positions at the designated locations
    
    Input:
        x_curr - numpy array of current x positions
        refine_locs - list of locations to refine
    Output:
        x_new - refined x positions
    """
    
    x_list = list(x_curr)
    indices = np.arange(0, len(x_list))
    count = 0
    
    for ind in indices[refine_locs]:
        if ind == 0 or ind == len(x_list) - 1:
            continue
        else:
            left_pos = 0.5 * (x_curr[ind] + x_curr[ind - 1])
            x_list.insert(ind + count, left_pos)
            count += 1
    return np.array(x_list)


def slp(p, q, f, bcs, xspan=[0, 1], order=2, adapt=True, n=8, tol=1e-6, max_refine=10):
    """
    Sturm - Liouville problem solver

    Input:

    Optional:

    Output:
        x - x nodes used, n x 1
        u - u(x), n x 1
    """

    # determine x points

    err_fun = lambda x, dx: slp_error(p, q, f, x, bcs, order=order)

    refine = True
    # refine x 
    if adapt:
        x = grid_refine(xspan, err_fun, tol=tol, n_init=n, max_iter=max_refine)
    else:
        while refine:
            x = np.linspace(xspan[0], xspan[1], n)
            error = (xspan[1] - xspan[0]) / (n + 1) * np.max(err_fun(x, dx))
            refine = np.sum(error > tol) > 1
            n *= 2

    # solve 
    u = solve_slp(p, q, f, x, bcs, order=order)

    return x, u

def solve_slp(p, q, f, x, bcs, order=2):
    """
    Solve a sturm-liouville problem numerically

    Input:
        p, q, f - Dx(p Dx u) + qu = f 
        x - x positions
        bcs - boundary conditions [alpha, beta, gamma] 0, 1 by row
        order - orider of accuracy
    Output:
        u - u(x), n x 1 vector
    """

    # build SL problem
    p_vec = p(x)
    pavg = np.mean(p_vec)
    if np.linalg.norm(pavg - p_vec) < EPS:
        # p is identically 1
        Dxx = diffmat(x, der_wanted=2, order_wanted=order)
        Q = np.diagflat(q(x))
        A = pavg * Dxx + Q
    else:
        # p is not 1
        Dx = diffmat(x, der_wanted=1, order_wanted=order)

        P = np.diagflat(p(x))
        Q = np.diagflat(q(x))
        A = Dx @ P @ Dx + Q

    Dx = diffmat(x, der_wanted=1, order_wanted=order)

    # augment with boundary conditions
    A[0, :] = bcs[0, 1] * Dx[0, :] 
    A[0, 0] += bcs[0, 0]
    A[-1, :] = bcs[1, 1] * Dx[-1, :]
    A[-1, -1] += bcs[1, 0]

    # right hand side
    f_vec = f(x)
    f_vec[0] = bcs[0, 2]
    f_vec[-1] = bcs[1, 2]


    # solve
    u = mla.linalg_solve(A, f_vec)

    return u


def slp_error(p, q, f, x, bcs, order=2):
    """
    Build error for sturm-liouville problem

    Input:
        p, q, f - Dx(p Dx u) + qu = f
        x - x nodes
    Optional:
        order - order of accuracy, defaults to 2
    Output:
        error - error values at each x node
    """

    dx = np.zeros(np.shape(x))
    dx[1:] = x[1:] - x[:-1]
    dx[0] = dx[1]

    p_vec = p(x)
    pavg = np.mean(p_vec)

    # error from discretization of operator
    if np.linalg.norm(p_vec - pavg) < EPS:
        # p is identically 1
        D = diffmat(x, der_wanted=2+order, order_wanted=order)
        error = pavg * np.power(dx, order) * (D @ f(x))
    else:
        # p is not one
        D1 = diffmat(x, der_wanted=order-1, order_wanted=order)
        D2 = diffmat(x, der_wanted=order, order_wanted=order)
        fp = f(x) / p(x) 
        fp[np.isnan(fp)] = 0
        fp[np.isinf(fp)] = 0
        pn = 1 / p(x)
        pn[np.isnan(pn)] = 0
        pn[np.isinf(pn)] = 0
        error = np.power(dx, order) * (order * D1 @ fp + np.cumsum(f(x) * dx) * D2 @ pn)
        
    pn = 1 / p(x)
    pn[np.isnan(pn)] = 0
    pn[np.isinf(pn)] = 0
    # error from boundary terms
    Db = diffmat(x, der_wanted=order, order_wanted=order)
    tmp = np.power(dx, order) * Db @ pn
    error[0] = bcs[0, 1] * tmp[0]
    error[-1] = bcs[1, 1] * tmp[-1]
    
    

    return np.abs(error)
    
def verlet(fun, init_pos, init_vel, tspan, num_nodes):
    """
    Verlet's method for solving u'' = f(t, u)
    
    Input:
        fun - rhs
        init - [u(t = 0), v(t = 0)], where u is of interest, v is "velocity"
        tspan - [tstart, tstop]
        num_nodes - number of desired time nodes
    Ouput:
        time - time nodes used
        U - u(t)
    """
    
    # allocate space 
    try:
        U = np.zeros([len(init_pos), num_nodes])
    except:
        U = np.zeros([1, num_nodes])
    
    # time nodes 
    time = np.linspace(tspan[0], tspan[1], num_nodes)
    dt = time[1] - time[0]
    
    U[:, 0] = init_pos
    
    # get starting velocity, acceleration
    vel = init_vel
    acc = fun(time[0], U[:, 0])
    acc_new = acc[:]
    
    # time stepping velocity verlet
    for i in range(num_nodes-1):
        # update position
        U[:, i+1] = U[:, i] + vel * dt + 0.5 * acc * dt ** 2
        
        # update acceleration
        acc_new = fun(time[i+1], U[:, i+1])
        
        # update velocity
        vel += 0.5 * (acc + acc_new) * dt
        acc = acc_new[:]
        
    return time, U
