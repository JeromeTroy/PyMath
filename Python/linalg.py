#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 06 10:51:00 2019

@author: jerome

The goal of this package is to build a simple linear equation solver for 
equations of the form Ax = b.

The hope is to be versitile, easy to understand and modify, and 
work well.  This way should errors present themselves in usage,
finding and fixing the problem should be easy.  

This is not intended to be a heavy-duty solver, more a simple and robust
package.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def backward_substitute(R, b):
    """
    Backward substitution for an upper triangular system
    
    Input:
        R - upper triangular matrix, m x m
        b - solution matrix m x 1
    Output:
        x - solution to Rx = b, m x 1
    
    Note:
        we suppose here that R is invertible, i.e. no diagonal 
        entries are zero.
        If there are zeros on the diagonal, then we must satisfy the 
        Fredholm alternative, i.e. forall v in the null space of R,
        we must have that v is orthogonal to b.
    """
    
    [m, n] = np.shape(R)
    if m != n:
        print("Error, rectangular solution not yet implemented")
        return None
    
    x = np.zeros([m, 1])
    
    mult_diag = np.prod(np.diag(R))
    if mult_diag == 0:
        print("Error, singular solution not yet implemented")
        return None
    
    for i in range(m-1, -1, -1):
        if i == m-1:
            x[i] = b[i] / R[i, i]
        else:
            after = R[i, i+1:]
            x[i] = (b[i] - np.dot(after, x[i+1:])) / R[i, i]
    return x
        
def jacobi_method(R, b, x0=None, tol=1e-5, maxiter=500):
    """
    Jacobi iterative method for solving a system
    
    Input:
        R - upper triangular matrix (for now) m x m
        b - solution m x 1
    Output:
        x - solution to Rx = b, m x 1
    
    Note: this method currently assumes R is invertible
    """
    
    D = np.diagflat(np.diag(R))
    Dn = np.diagflat(1.0 / np.diag(R))
    print(Dn)
    C = R - D
    
    if type(x0) == None:
        xprev = b
    else:
        xprev = b    
    
    ite = 0
    error = np.inf
    
    while ite < maxiter and error > tol:
        xnew = np.matmul(Dn, b - np.matmul(C, xprev))
        
        error = np.linalg.norm(np.matmul(R, xnew) - b)
        #error = np.linalg.norm(xnew - xprev)
        xprev = xnew
        ite += 1
    
    if ite == maxiter:
        print("Warning, maximum number of iterations reached")
        
    print("iterations: ", ite)
    return xnew


def householder_triangulation_solve(A, b):
    """
    Implicit QR via householder triangulation
    
    Input:
        A - matrix to factorize m x n
        b - RHS for Ax = b, m x 1
    Output:
        x - solution to Ax = b, n x 1
        
    Note: this is error prone, will need to be revised
    """
    
    [m, n] = np.shape(A)
    
    R = np.copy(A)
    beta = np.copy(b)
    R = R.astype(np.float32)
    beta = beta.astype(np.float32)
    
    for k in range(n):
        
        # select area to modify
        z = R[k:, k]
        
        # build unit vector e1
        e1 = np.zeros(np.shape(z))
        e1[0] = 1
        
        # build v vector
        vec = z[0] / np.abs(z[0]) * np.linalg.norm(z) * e1 + z
        vec /= np.linalg.norm(vec)      # normalize
        v = np.matrix(vec)
        v = v.T
        
        # update R and beta
        R[k:, k:] = R[k:, k:] - 2 * np.matmul(v, np.matmul(np.transpose(v), R[k:, k:]))
        if k < n-1:
            R[k+1:, k] = 0
        
        beta[k:] -= 2.0 * np.float(np.dot(v.T, beta[k:])) * v
    
    # loop leaves last row unaltered
    R[-1, -1] *= -1
    beta[-1] *= -1
    
    # build solution
    x = backward_substitute(R, beta)
    
    return x


def linsolve(A, b):
    """
    Solve the linear system Ax = b
    
    Input:
        A - m x n
        b - m x 1 
    Output:
        x - n x 1
    """
    
    x = householder_triangulation_solve(A, b)
    
    return x


A = np.random.randint(low=-5, high=5, size=[5,5])
print(A)

b = np.random.randint(low=-3, high=3, size=[5, 1])
print(b)

[Q, R] = np.linalg.qr(A)
x_control = np.linalg.solve(R, np.matmul(Q.T, b))
print(x_control)

x_mine = linsolve(A, b)
print(x_mine)

control_error = np.matmul(A, x_control) - b
my_error = np.matmul(A, x_mine) - b

delta = x_mine - x_control

print('control error')
print(np.linalg.norm(control_error))
print('my error')
print(np.linalg.norm(my_error))

print('delta')
print(np.linalg.norm(delta))
