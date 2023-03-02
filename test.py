import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets



# Make a prediction on X_test
y_pred = model.predict(X_test)

# Compute the MSE (Evaluate both, regression and classification)
mse(Y_test,y_pred)