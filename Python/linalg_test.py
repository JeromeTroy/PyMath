#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:54:54 2020

@author: jerome
"""

# lengths and shapes of scalars and vectors
from linalg import *

a = 1
arr = np.array(1)

print(shape(a))
print(shape(arr))

lst = [0, 1]
arr = np.array(lst)
print(shape(lst))
print(shape(arr))

print(len(shape(lst)))

# matrix multiplication
scalar = 2
vector = np.array([1, 1])
matrix = np.array([[2, 3], [0, 1]])

vector2 = np.array([1, 0, 0])
matrix2 = np.random.rand(3,3)

v = matmul(matrix, vector)
print(v)

u = matmul(vector, matrix)
print(u)

try:
    w = matmul(matrix, vector2)
except ValueError:
    print("Size error successfully caught")

matmul(matrix, scalar)


x = solve(matrix, vector)
x1 = solve(matrix, matrix)
x2 = solve(vector, vector)
x3 = solve(matrix, scalar)

print(x3)