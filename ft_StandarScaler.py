import numpy as np
from math import sqrt

def mean(mylist: list):

	if len(mylist) == 0:
		raise Exception('Empty data')
	else:
		return sum(mylist)/len(mylist)


def std(mylist: list):
	if len(mylist) == 0:
		raise Exception('Empty data')
	else:
		n = len(mylist)
		_mean = mean(mylist)
		var = sum([(x - _mean)**2 for x in mylist])/n
		return sqrt(var)

class ft_StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Compute the mean and standard deviation for scaling"""
        self.mean = mean(X)
        self.std = std(X)
    
    def transform(self, X):
        """Scale the data using the computed mean and standard deviation"""
        if self.mean is None or self.std is None:
            raise Exception("The scaler has not been fitted yet.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit to the data, then transform it"""
        self.fit(X)
        return self.transform(X)
