import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# cgs
def cgs(A):
  """
    Q,R = cgs(A)
    Apply classical Gram-Schmidt to mxn rectangular/square matrix. 

    Parameters
    -------
    A: mxn rectangular/square matrix   

    Returns
    -------
    Q: mxn square matrix
    R: nxn upper triangular matrix

  """
  # ADD YOUR CODES
  m,n= A.shape # get the number of rows of A
  # get the number of columns of A

  R= np.zeros((n,n)) # create a zero matrix of nxn
  Q= np.ones((m,n)) # copy A (deep copy)
  for k  in range(n):
    w = A[:,k]
    
    for j in range(k-1):
      R[j,k]=np.dot(np.transpose(Q[:,j]),w)
    for j in range(k-1):
      w = w - np.dot(R[j,k],Q[:,j])
    R[k,k]= np.linalg.norm(w, ord=2)
    #print(w/R[k,k])
    Q[:,k] = w/ R[k,k]
    #print (R[:,k])
  return Q,R
  

# Implement BACK SUBS
def backsubs(U, b):

  """
  x = backsubs(U, b)
  Apply back substitution for the square upper triangular system Ux=b. 

  Parameters
  -------
    U: nxn square upper triangular array
    b: n array
    

  Returns
  -------
    x: n array
  """

  n= U.shape[1]
  x= np.zeros((n,))
  b_copy= np.copy(b)

  if U[n-1,n-1]==0.0:
    if b[n-1] != 0.0:
      print("System has no solution.")
  
  else:
    x[n-1]= b_copy[n-1]/U[n-1,n-1]
  for i in range(n-2,-1,-1):
    if U[i,i]==0.0:
      if b[i]!= 0.0:
        print("System has no solution.")
    else:
      for j in range(i,n):
        b_copy[i] -=U[i,j]*x[j]
      x[i]= b_copy[i]/U[i,i]
  return x

# Add ones
def add_ones(X):

  # ADD YOUR CODES
  m,n = X.shape
  New_X = np.ones((m,n+1))
  New_X[:m,1:n+1]= X
  return New_X

X= add_ones(X)


def split_data(X,Y, train_size):
  # ADD YOUR CODES
  # shuffle the data before splitting it
  train_size =int(train_size*len(Y))
  m,n = X.shape
  X_Y = np.zeros((m,n+1))
  Y=Y.reshape(len(Y),1)
  X_Y=np.concatenate((X, Y), axis=1)
  np.random.shuffle(X_Y)
  x_train = X_Y[:train_size,:-1]
  y_train  = X_Y[:train_size,-1]
  x_test = X_Y[train_size:,:-1]
  y_test = X_Y[train_size:,-1]
  return  x_train, y_train, x_test, y_test


X_train, Y_train, X_test, Y_test = split_data(X,y,0.8)


def mse(y, y_pred):
    # ADD YOUR CODES
    error = np.subtract(y,y_pred)**2
    return np.mean(error)

def normalEquation(X,y):
    # ADD YOUR CODES
  Theta = np.linalg.inv(X.T@X)@X.T@y
  return Theta


