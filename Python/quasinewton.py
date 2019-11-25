#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 11:43:00 2019

@author: jerome

Quasi-Newton methods for function minimization
"""

import numpy as np 
import numpy.linalg as la 

import linalg as mla
import nonlinsolve as nls 


def levenberg(fun, x0, y=None, jac=None, tol=1e-7, maxiter=100, record=False):
    """
    Levenberg solver for least squares and minimization
    Also useful for rootfinding (reformat as rootfinding problem

    Input:
        fun - callable objective function
        x0 - initial guess
        y - optional - function being approximated in least squares
        jac - optional - jacobian matrix function
        tol - optional - tolerance in minimum location
        maxiter - optional - maximum allowed number of iterations
        record - optional - boolean of whether to record previous iterations
    Output:
        x - approximate location of minimum
    """

    x = np.array(x0)
    
    if record:
        result = nls.Minimizer()
        result.set_algorithm("Levenberg")
        result.add_iter(x)
    
    # get working sizes
    m = mla.get_length(fun(x0))
    n = mla.get_length(x0)

    if y is None:
        y = np.zeros(np.shape(fun(x0)))


    # setup for looping
    lam = 100    
    iterat = 0
    error = np.inf
    
    # build jacobian matrix
    if jac is not None:
        J = jac(x)
    else:
        J = nls.fdjac(fun, x)
        
    # residual
    r = y - fun(x)
    
    # minimization loop
    while iterat < maxiter and error > tol:

        # build and solve linear system
        A = mla.matmul(J.T, J) + lam * np.eye(n)
        b = mla.matmul(J.T, r)
        delta = mla.linalg_solve(A, b)        
        
        # update guess
        xnew = x + delta

        # check if the solution is better
        rnew = y - fun(xnew)
        if la.norm(rnew) < la.norm(r):
            # accept
            x = xnew
            r = rnew
            iterat += 1
            lam /= 10       # closer to newton
            
            if record:
                result.add_iter(x)
            
            # update jacobian and error
            if jac is not None:
                J = jac(x)
            else:
                J = nls.fdjac(fun, x)
                
            error = la.norm(delta, 1)
        else:
            # reject, maintain position
            lam *= 4        # closer to gradient descent

    if iter == maxiter:
        print("Warning, maximum number of iterations reached")

    if not record:
    	result = x

    return result

