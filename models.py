import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


class LinearRegression:

  def __init__(self, arg):
      # ADD YOUR CODES
      self.arg = arg
      self.theta = None

  def fit(self,x,y):
      # ADD YOUR CODES
      if self.arg == "cgs":
        Q,R =cgs(x)
        y = Q.T@y
        self.theta = backsubs(R,y)
      elif self.arg == "NormalEq":
        self.theta = normalEquation(x,y)
    
  def predict(self,x):
      #ADD YOUR CODE\
      y_pred = x@self.theta
      return y_pred
