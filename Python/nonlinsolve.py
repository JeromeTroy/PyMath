#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 10:51:00 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la

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