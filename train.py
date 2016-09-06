import numpy as np

from sklearn import cross_validation, preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

np.random.seed(1337)

X = np.load('/Users/colinni/evAl-chess/X.npy')
Y = np.load('/Users/colinni/evAl-chess/Y.npy')

# Discard samples where the evaluation is out of [-10, +10].
X = X[np.where(np.abs(Y) < 10.0)]
Y = Y[np.where(np.abs(Y) < 10.0)]

permuted = np.random.permutation(len(Y))
X, Y = X[permuted], Y[permuted]
X, Y = map(preprocessing.scale, (X, Y))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    X, Y,
    test_size=0.3
)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(X_train)
print(Y_train)

model = Sequential(
    [
        Dense(200, input_shape=(389,)),
        Activation('relu'),
        Dense(50),
        Activation('relu'),
        Dense(10),
        Activation('relu'),
        Dense(1),
        Activation('relu')
    ]
)

model.summary()
model.compile(loss='mean_squared_error', optimizer='sgd')

history = model.fit(
    X_train,
    Y_train,
    batch_size=32,
    nb_epoch=200,
    verbose=1
)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score', score)

predictions = model.predict(X_test)
print(len(predictions), Y_test.shape)
for i in range(30):
    print(predictions[i], Y_test[i])
