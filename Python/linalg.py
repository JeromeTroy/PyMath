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

def get_length(vec):
    """
    Wrapper for getting length of vector that may have length 1

    Input:
        vec
    Output:
        n - length of the vector
    """
    try:
        n = len(vec)
    except TypeError:
        # scalar
        n = 1
    return n

def matmul(m1, m2):
    """
    Matrix multiplication wrapper to deal with vector * scalar
    
    Input:
        m1, m2 - numpy arrays
    Output:
        m - resulting matrix
    """

    # numpy matrix multiplication
    try:
        m = m1 @ m2
    except:
        m = m1 * m2

    return m


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

    x = np.zeros(m)

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
    Needs to be updated
    """

    D = np.diagflat(np.diag(R))
    Dn = np.diagflat(1.0 / np.diag(R))
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


def qr_solve(A, b):
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


def build_vandermonde(x, max_power):
    """
    Build a vandermonde matrix

    Input:
        x - x values for vandermonde - m nodes
        max_power - maximum power of x in vandermonde - n-1
    Output:
        V - vander monde matrix - m x n
    """

    # vector lengths
    m = len(x)
    n = max_power + 1

    # build powers of each row
    powers = np.arange(0, max_power+1)

    # allocate space and set positions in vandermonde system
    V = np.tile(x, [n, 1])

    # perform power operations
    powers = np.tile(powers, [m, 1])
    V = np.power(V, powers.T)

    return V


def linalg_solve(A, b):
    """
    Wrapper/dispatcher for solving a linear system

    Input:
        A - matrix to invert
        b - right hand side vector
    Output:
        x - solution to Ax = b
    """
    l = get_length(b)
    try:
        [m, n] = np.shape(A)
    except ValueError:
        Atmp = np.matrix(A)
        [m, n] = np.shape(Atmp)
    if l != m:
        print("Error, right hand side of wrong shape")
        return None

    else:   # may be able to solve, begin dispatcher
        if n == 1 and m == 1:
            # scalar x scalar = scalar
            x = b / A
        elif n == 1 and m != 1:
            # vector x scalar = vector
            x = la.norm(b) / la.norm(A)

        else:
            # matrix x vector = vector

            if m == n:
                # square system, apply standard solver, won't complain
                x = la.solve(A, b)

            elif m > n:
                # more equations than unknown, use qr
                [Q, R] = la.qr(A)
                z = matmul(Q.T, b)
                x = backward_substitute(R, z)

            else:
                # m < n more unknowns than equations, use SVD
                try:
                    [U, S, V] = la.svd(A)
                except la.LinAlgError:
                    A = np.reshape(A, [1, len(A)])
                    [U, S, V] = la.svd(A)
                # here S is a list of the singular values
                z = matmul(U.T, b)
                y = np.zeros(n)
                y[:m] = z / S           # pointwise division 
                x = matmul(V.T , y)

        return x
