import numpy as np

from sklearn import cross_validation
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD


np.random.seed(1337)

X = np.load('/Users/colinni/evAl-chess/X.npy')
Y = np.load('/Users/colinni/evAl-chess/Y.npy')

X, Y = map(np.random.permutation, (X, Y))

for i in range(100):
    print(Y[i])

X = X[np.where(np.abs(Y) < 5.0)]
Y = Y[np.where(np.abs(Y) < 5.0)]

for i in range(100):
    print(Y[i])
