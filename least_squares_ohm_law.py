#!/usr/bin/env python
# coding: utf-8

# The goal is to determine the resistance R considering Ohm's law V=RI, by using the method of least squares.
# We will be fitting a linear model, as y=Rx.
# We have the following data for I and V:
# 
# | Current (A) | Voltage (V) |
# |-------------|-------------|
# | 0.2         | 1.23        |
# | 0.3         | 1.38        |
# | 0.4         | 2.06        |
# | 0.5         | 2.47        |
# | 0.6         | 3.17        |


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Store the voltage and current data as column vectors.
I = np.mat([0.2, 0.3, 0.4, 0.5, 0.6]).T
V = np.mat([1.23, 1.38, 2.06, 2.47, 3.17]).T


# Plot the measurements
plt.scatter(np.asarray(I), np.asarray(V))
plt.figure(1)
plt.xlabel('Current (A)')
plt.ylabel('Voltage (V)')
plt.grid(True)
plt.show()


# Estimating the slope parameter, which is the resistance R (y=Rx), using the least squares formulation
# R^=((H.T*H)^-1)*H.T*y

# Define the H matrix
H = np.mat([1,1,1,1,1]).T
print(H)
# Estimate the resistance parameter.
# The estimated value will not match the true resistance value exactly, since we have only a limited number of noisy measurements.
R = np.linalg.inv(H.T*H) * H.T * (V/I)
# print(np.linalg.inv(H.T*H))
# print(H.T * (V/I))

print('The slope parameter (i.e., resistance) for the best-fit 2D line is:')
print(R)


# Plot results
I_line = np.arange(0, 0.8, 0.1)
print('I_line is =', I_line)
print(I_line.shape)
print(R)
print(R.shape)
V_line = R*I_line
print(V_line[0])
print(V_line[0].shape)
V_line = np.asarray(V_line)
print(V_line.shape)

plt.figure(2)
plt.scatter(np.asarray(I), np.asarray(V))
plt.plot(I_line, V_line[0], 'r')
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()