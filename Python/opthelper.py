#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:10:00 2019

@author: jerome

Methods for unconstrained function minimization

Helper functions for minimization are included.

These include line search tools, projectors for constraints, and others
"""

import numpy as np
import numpy.linalg as la

import linalg as mla
import nonlinsolve as nls



# line search methods

def line_search_backtrack(fun, loc, search, alpha0=1, alphamin=1e-16):
    """
    Line search via backtracking for optimization problems
    
    Input:
        fun - callable scalar function to be optimized
        loc - current x location (n x 1) vector
        search - current search direction (n x 1) vector
        alpha0 - starting alpha value, defaults to 1
        alphamin - minimum value of alpha allowed, defaults to machine eps
    Output:
        alpha - computed step size
    """
    
    alpha = alpha0
    cur_val = fun(loc)
    new_val = fun(loc + alpha * search)
    while new_val > cur_val and alpha > alphamin:
        alpha /= 2
        new_val = fun(loc + alpha * search)
    
    return alpha

def line_search_wolfe(fun, loc, search, jac=None, alpha0=1, alphamin=1e-16):
    """
    Line search using Wolfe condition
    
    Input:
        fun - callable function to be optimized
        loc - current x position (n x 1)
        search - current search direction (n x 1)
        jac - jacobian function (defaults to none - finite difference approx)
        alpha0 - starting alpha value - defaults to 1
        alphamin - minimum allowed alpha - defaults to machine eps
    Output:
        alpha - computed optimal step size
    """
    
    # build function of alpha
    if jac is not None:
        alpha_fun = lambda a: np.dot(jac(loc + a * search), search)
    else:
        jac_fun = lambda x: nls.fdjac(fun, x)
        alpha_fun = lambda a: np.dot(jac_fun(loc + a*search), search)
    
    # bracket alpha
    alphamin = 0
    f_min = alpha_fun(alphamin)
    alphamax = 0.1
    f_max = alpha_fun(alphamax)
    while f_min < 0:
        alphamin = alphamax
        alphamax *= 2
        f_min = f_max
        f_max = alpha_fun(alphamax)
    
    # interval
    alpha_zone = [alphamin, alphamax]
    
    # build cubic hermite interpolants
    delta = 1e-4 * alphamin
    f_min_prime = (f_min - alpha_fun(alphamin + delta)) / (-delta)
    f_max_prime = (f_max - alpha_fun(alphamax - delta)) / delta
    
    
    # compute minimum explicitly
    
    
    
    return alpha



# constraints
    
# linear
    
def project_search(search, constr, flags):
    """
    Project search direction as to respect constraints
    
    Input:
        search - search direction vector
        constr - constraint matrix - 
                    A | b
                    where A is the matrix multiplying the unknowns x_i
                    and b is the right hand side so that 
                    
                    Ax > = b
                    (>, =, >=) variously
        flags - indicates which rows of A are equality constraints, 
                active constraints, and inactive constraints
                0 - eq, 1 - inactive, 2 - active
    Output:
        new_search - new search direction which has been projected