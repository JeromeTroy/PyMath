#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 10:51:00 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la
import linalg as mla

def newton(fun, x0, dfdx=None, tol=1e-7, maxiter=100):
    """
    Newtons method for finding function zero

    Input:
        fun - callable function to solve = 0
        x0 - initial guess
        dfdx - derivative, optional, callable, same signature as fun
    Optional:
        tol - tolerance allowed, default value 1e-5
        maxiter - maximum allowed iterations, default value 100
    Output:
        xf - final guess value
        x - all guessed values
    """

    x = [x0]
    try:
        n = len(x0)
        is_scalar = (n == 1)
    except TypeError:
        is_scalar = True

    error = [np.inf]
    delta_x = np.inf
    iterat = 0
    while error[-1] > tol and iterat < maxiter and delta_x > tol:
        if dfdx is not None:
            jac = dfdx(x[-1])
        elif is_scalar:
            deriv = (fun(x[-1] + tol) - fun(x[-1] - tol)) / (2 * tol)
            xf = x[-1] - fun(x[-1]) / deriv
            error.append(np.abs(fun(xf)))
            delta_x = np.abs(x[-1] - xf)
        else:
            jac = fdjac(fun, x[-1])
            delta = mla.linalg_solve(jac, -fun(x[-1]))
            xf = x[-1] + delta
            error.append(np.max(np.abs(fun(xf))))
            delta_x = np.max(np.abs(delta))

        iterat += 1
        x.append(xf)

    if iterat == maxiter:
        print("Warning, maximum number of iterations reached")

    error = np.array(error)
    return [xf, x, error]

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

def fdjac(f, x0, delta=1e-4):
    """
    Build jacobian matrix for function

    Input:
        f - callable function
        x0 - point about which to compute jacobian
        delta - optional - step size for computing derivative
    Output:
        J - jacobian matrix
    """
    n = len(x0)
    try:
        m = len(f(x0))
    except TypeError:
        m = 1
    J = np.zeros([m,n])

    ej = np.zeros(np.shape(x0))
    for j in range(n):
        ej[j] = 1
        J[:,j] = ( f(x0 + delta*ej) - f(x0 - delta*ej) ) / (2 * delta)
        ej[j] = 0

    return J

def levenberg(fun, x0, y=None, jac=None, tol=1e-7, maxiter=100):
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
    Output:
        x - approximate location of minimum
    """

    if y is None:
        y = np.zeros(np.shape(fun(x0)))

    # determine working sizes
    try:
        m = len(fun(x0))
    except TypeError:
        m = 1
    try:
        n = len(x0)
    except TypeError:
        n = 1

    # setup for looping
    lam = 100
    iter = 0

    # clean x
    x = np.matrix(x0)
    [l, k] = np.shape(x)
    if k > 1:
        x = np.transpose(x)
    error = np.inf
    # build jacobian matrix
    if jac is not None:
        J = jac(x)
    else:
        J = fdjac(fun, x)
    # right hand side
    r = y - fun(x)

    # minimization loop
    while iter < maxiter and error > tol:

        # build and solve linear system
        A = np.matmul(np.transpose(J), J) + lam * np.eye(n)
        r = np.matrix(r)
        b = np.matmul(np.transpose(J), r)
        delta = mla.linalg_solve(A, b)

        # update guess
        xnew = x + delta

        # check if the solution is better
        rnew = y - fun(xnew)
        if la.norm(rnew) < la.norm(r):
            # accept
            x = xnew
            r = rnew
            iter += 1
            lam /= 10       # closer to newton
            # update jacobian
            J = fdjac(fun, x)
            error = la.norm(delta,1)
        else:
            # reject, maintain position
            lam *= 4        # closer to gradient descent
    if iter == maxiter:
        print("Warning, maximum number of iterations reached")
    print(iter)
    return x


f = lambda x: np.power(la.norm(x),2)
x0 = np.array([1,1])
x0 = np.matrix(x0)
x0 = np.transpose(x0)

[xf, xvals, errors] = newton(f, x0, tol=1e-16)
print(xf)
print(len(xvals))

xl = levenberg(f, x0, tol=1e-16)
print(xl)
