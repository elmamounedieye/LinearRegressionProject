import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

# Get Data: Do not touch it.
def get_data():
  data_url = "http://lib.stat.cmu.edu/datasets/boston"
  raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
  X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
  y = raw_df.values[1::2, 2]
  return X,y

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
  