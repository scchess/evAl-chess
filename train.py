import numpy as np


from sklearn import cross_validation, preprocessing
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

np.random.seed(1337)

X = np.load('/Users/colinni/evAl-chess/X.npy')
Y = np.load('/Users/colinni/evAl-chess/Y.npy')

# Discard samples where the evaluation is out of [-10, +10].
X = X[np.where(np.abs(Y) < 1.5)]
Y = Y[np.where(np.abs(Y) < 1.5)]

Y = np.sqrt(np.abs(Y)) * (2 * (Y < 0) - 1)

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
# model = load_model('/Users/colinni/evAl-chess/saved_keras_model.h5')

while True:
    inp = input('Continue training for how many epochs (\'s\' to stop)? ')
    if inp == 's':
        break
    history = model.fit(
        X_train,
        Y_train,
        batch_size=128,
        nb_epoch=int(inp),
        verbose=1
    )

model.save('/Users/colinni/evAl-chess/saved_keras_model.h5')

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score', score)

predictions = model.predict(X_test)
print(len(predictions), Y_test.shape)
for i in range(30):
    print(np.round(predictions[i], 3), np.round(Y_test[i], 3))
