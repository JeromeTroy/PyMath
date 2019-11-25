#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 10:51:00 2019

@author: jerome
"""

import numpy as np
import numpy.linalg as la

import linalg as mla

class Minimizer():
	"""
	Return class for minimization and rootfinding algorithms
	"""

	def __init__(self):
		self._x = None
		self._xprev = []
		self._n = -1
		self._algorithm = ""

	def get_optimizer(self):
		return self._x

	def get_previous_attemps(self):
		return np.array(self._xprev)

	def get_number_iterations(self):
		return self._n

	def get_which_algorithm(self):
		return self._algorithm

	def set_algorithm(self, alg):
		self._algorithm = alg

	def add_iter(self, x):
		self._n += 1
		self._x = np.array(x)
		self._xprev.append(np.array(x))



def newton(fun, x0, dfdx=None, tol=1e-7, maxiter=100, record=False):
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

    x = x0
    if record:
    	result = Minimizer()
    	result.add_iter(x)
    	result.set_algorithm("newton")

    n = mla.get_length(x)

    error = np.inf
    delta_x = np.inf
    iterat = 0

    while error > tol and iterat < maxiter and delta_x > tol:

        # compute jacobian
        if dfdx is not None:
            jac = dfdx(x)
        else:
        	jac = fdjac(fun, x)
    	
        delta = mla.linalg_solve(jac, -fun(x))
        xf = x + delta

        error = np.max(np.abs(fun(xf)))

        delta_x = np.max(np.abs(delta))

        iterat += 1
        x = xf

        if record:
            result.add_iter(x)

    if iterat == maxiter:
        print("Warning, maximum number of iterations reached")

    if not record:
        result = x

    return result

def secant_scalar(fun, x0, x1=None, tol=1e-5, maxiter=100, record=False):
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
        # initialization
        deriv = (fun(x0 + tol) - fun(x0 - tol)) / (2 * tol)
        x1 = x0 - fun(x0) / deriv
        return secant(fun, x0, x1=x1, tol=tol, maxiter=maxiter)

    else:
        if record:
            result = Minimizer()
            result.set_algorithm("secant")
            result.add_iter(x0)
            result.add_iter(x1)

        xprev = x0
        x = x1
        error = np.inf
        iterat = 0

        while error > tol and iterat < maxiter:

            xnew = x - (fun(x) * (x - xprev)) / (fun(x) - fun(xprev))
            error = np.abs(fun(xnew))
            iterat += 1
            
            xprev = x
            x = xnew

            if record:
                result.add_iter(x)

        if not record:
            result = np.array(x)

        if iterat == maxiter:
            print("Warning, maximum number of iterations reached")
        
        return result

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
    Needs to be updated
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
    n = mla.get_length(x0)
    m = mla.get_length(f(x0))
    
    J = np.zeros([m,n])

    ej = np.zeros(np.shape(x0))

    if n > 1:
        for j in range(n):
            ej[j] = 1
            J[:,j] = ( f(x0 + delta*ej) - f(x0 - delta*ej) ) / (2 * delta)
            ej[j] = 0
    else:
        J = (f(x0 + delta) - f(x0 - delta)) / (2 * delta)

    if m == 1 or n == 1:
        J = J.flatten()
    return J




f_scalar = lambda x: np.power(x, 2) - x
dfdx_s = lambda x: 2 * x - 1
x0s = 10

A = np.array([[-2, 1], [3, -4]])
c = np.array([-3, 1])
f_vector = lambda x: np.dot(x, np.squeeze(np.asarray(mla.matmul(A, x)))) - np.dot(c, x)
jac = lambda x: mla.matmul(A, x) - c
x0v = np.array([-2, 0])

