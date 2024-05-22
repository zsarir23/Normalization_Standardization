import numpy as np
from math import sqrt


class ft_MinMaxscalar:
    def __init__(self):
        self.min = None
        self.max = None
    
    def fit(self, X):
        """Compute the min and max for scaling"""
        self.min = np.min(X)
        self.max = np.max(X)
    
    def transform(self, X):
        """Scale the data using the computed min and max"""
        if self.min is None or self.max is None:
            raise Exception("The scaler has not been fitted yet.")
        return (X - self.min) / (self.max - self.min)
    
    def fit_transform(self, X):
        """Fit to the data, then transform it"""
        self.fit(X)
        return self.transform(X)
