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

  