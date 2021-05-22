#!/usr/bin/env python
# coding: utf-8

# The goal is to determine the resistance R considering Ohm's law V=RI, by using the method of recursive least squares.
# We will be fitting a linear model with an offset, as y=Rx+b.
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
I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])


# Plot the measurements
plt.scatter(np.asarray(I), np.asarray(V))
plt.figure(1)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()

# ### Batch Estimator
# Before implementing recursive least squares, we review the parameter estimate given by the batch least squares method for comparison.
## Batch Solution
H = np.ones((5,2))
H[:, 0] = I
print("H.shape=",H.shape)
print("H=",H)
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The parameters of the line fit are ([R, b]):')
print(x_ls)
#Plot
I_line = np.arange(0, 0.8, 0.1)
V_line = x_ls[0]*I_line + x_ls[1]

plt.figure(2)
plt.scatter(np.asarray(I), np.asarray(V))
plt.plot(I_line, V_line, 'g')
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()


# We begin with a prior estimate of R = 4.
# We assume perfect knowledge of current I, while voltage V data are corrupted by additive, independent and identically distributed Gaussian noise of variance 0.0255 V^2.
# R_hat ~ N(4,10) , b_hat ~ N (0,0.2)

# Estimating the slope parameter, which is the resistance R (y=Rx+b), using the recursive least squares formulation
# Initialize the parameter and covariance estimates: x0_hat = E[x], P0 = E[(x-x0_hat)*(x-x0_hat).T]
# Then, for every measurement k:
# Calculate the correction gain: Kk = Pk-1*Hk.T*(Hk*Pk-1*Hk.T+Rk)^-1
# Update the parameter estimate: xk_hat = xk-1_hat+Kk*(yk-Hk*xk-1_hat)
# Update the covariance estimate: Pk = (I-Kk*Hk)*Pk-1

#Initialize the 2x2 covariance matrix
P_k = np.array([[10,0],[0,0.2]])
print("P_0 = ")
print(P_k)
#Initialize the parameter estimate x
x_k = np.array([4, 0]).T
print("x_0 = ", x_k)
#Our measurement variance
Var = 0.0225

#Pre allocate our solutions so we can save the estimate at every step
num_meas = I.shape[0]
x_hist = np.zeros((num_meas + 1,2))
P_hist = np.zeros((num_meas + 1,2,2))
# print("P_hist.shape=",P_hist.shape)
# print("x_hist.shape=",x_hist.shape)
# print("x_hist=",x_hist)

print("##################################")

x_hist[0] = x_k
P_hist[0] = P_k
# print("P_hist[0].shape=",P_hist[0].shape)
# print("P_hist[0]=",P_hist[0])
# print("x_hist[0].shape=",x_hist[0].shape)
# print("x_hist[0]=",x_hist[0])

print("##################################")


#Iterate over the measurements
for k in range(num_meas):
    print("k=",k, ", num_meas=", num_meas)
    
    #Construct the Jacobian H_k
    H_k = np.array([I[k], 1]).reshape(1, 2)
    print("H_k.shape=",H_k.shape)
    print("H_k=",H_k)
    
    R_k = np.array([Var])
    #print("R_k.shape=",R_k.shape)
    
  
    #Construct K_k - Gain Matrix
    #a 1×1 is already a diagonal matrix existing of one single real number which 
    #means it is its own determinant and if it isn’t zero it has an inverse: the inverse of this number.
    K_k = P_hist[k].dot(H_k.T).dot(inv(H_k.dot(P_hist[k]).dot(H_k.T) + R_k))
    np.reshape(K_k, (2, 1))
    print("K_k.shape=",K_k.shape)
    print("K_k=",K_k)
    
    
    #Update our estimate
    #np.reshape(x_hist[k], (2, 1))
    x_k = x_hist[k].reshape(2, 1) + K_k.dot(V[k] - H_k.dot(x_hist[k].reshape(2, 1)))
    print("x_k.shape=",x_k.shape)
    print("x_k=",x_k)
    

    #Update our uncertainty - Estimator Covariance
    P_k = (np.eye(2, dtype=int) - K_k.dot(H_k)).dot(P_hist[k])
    print("P_k.shape=",P_k.shape)
    print("P_k=",P_k)
    
    
    #Keep track of our history
    P_hist[k+1] = P_k.reshape(2, 2)
    x_hist[k+1] = x_k.reshape(1, 2)
    
print('The parameters of the line fit are ([R, b]):')
print(x_k)



# Plot results
I_line = np.arange(0, 0.8, 0.1)
plt.figure(3)
plt.scatter(np.asarray(I), np.asarray(V))
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.plot(I_line, V_line, label='Batch Least-Squares Solution')
for k in range(num_meas):
    V_line = x_hist[k,0]*I_line + x_hist[k,1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.grid(True)
plt.legend()
plt.show()