import numpy as np
import extract_features
import itertools

from sklearn import cross_validation, preprocessing
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout, Merge
from keras.optimizers import SGD

np.random.seed(1337)

X = np.load('/Users/colinni/evAl-chess/X.npy')
Y = np.load('/Users/colinni/evAl-chess/Y.npy')

# Discard samples where the evaluation is out of [-10, +10].
X = X[np.where(np.abs(Y) < 35.0)]
Y = Y[np.where(np.abs(Y) < 35.0)]
# X = X[np.where(np.abs(Y) > 0.1)]
# Y = Y[np.where(np.abs(Y) > 0.1)]
# opening_positions = np.where(X[:, 5] >= 7) # X = X[opening_positions, :][0]
print(X.shape)
print(Y.shape)


Y = np.sqrt(np.abs(Y)) * (2 * (Y > 0) - 1)

permuted = np.random.permutation(len(Y))
X, Y = X[permuted], Y[permuted]


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(
    X, Y,
    test_size=0.3
)

scaler_X, scaler_Y = preprocessing.StandardScaler(), preprocessing.StandardScaler()
# Scale based only on training data.
X_train, Y_train = scaler_X.fit_transform(X_train), scaler_Y.fit_transform(np.reshape(Y_train, (len(Y_train), 1)))

X_test, Y_test = scaler_X.transform(X_test), scaler_Y.transform(np.reshape(Y_test, (len(Y_test), 1)))

def check_test_perf(model, X_test, Y_test):
    score = model.evaluate(X_test, Y_test, verbose=1)
    print('Test score', score)

    predictions = scaler_Y.inverse_transform(model.predict(X_test)) * abs(scaler_Y.inverse_transform(model.predict(X_test)))
    Y_test = scaler_Y.inverse_transform(Y_test) * np.abs(scaler_Y.inverse_transform(Y_test))
    print(len(predictions), Y_test.shape)
    for i in range(64):
        print(round(float(predictions[i]), 3), round(float(Y_test[i]), 3), round(float(Y_test[i]) - float(predictions[i]), 3), sep='\t')

if __name__ == '__main__':

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    print(X_train)
    print(Y_train)


    X_train, X_test = extract_features.split_features(X_train), extract_features.split_features(X_test)

    # model = Sequential(
    #     [
    #         Merge(
    #             [
    #                 Sequential(
    #                     [
    #                         Dense(
    #                             output_dim=8,
    #                             input_dim=17
    #                         )
    #                     ]
    #                 ),
    #                 Sequential(
    #                     [
    #                         Dense(
    #                             output_dim=128,
    #                             input_dim=244
    #                         )
    #                     ]
    #                 ),
    #                 Sequential(
    #                     [
    #                         Dense(
    #                             output_dim=64,
    #                             input_dim=128
    #                         )
    #                     ]
    #                 ),
    #             ],
    #             mode='concat'
    #         ),
    #         Dropout(0.05),
    #         Activation('relu'),
    #         Dense(output_dim=64),
    #         Dropout(0.05),
    #         Activation('relu'),
    #         Dense(output_dim=1),
    #         Activation('relu')
    #     ]
    # )
    #
    # model.summary()
    # model.compile(loss='mean_squared_error', optimizer='sgd')

    model = load_model('/Users/colinni/evAl-chess/saved_keras_model_merged_first.h5')

    while True:
        inp = input('Continue training for how many epochs (\'s\' to stop and save, t to check test performance)? ')
        if inp == 's':
            break
        if inp == 't':
            check_test_perf(model, X_test, Y_test)
            continue
        bs = input('Batch size? ')
        history = model.fit(
            X_train,
            Y_train,
            batch_size=int(bs),
            nb_epoch=int(inp),
            verbose=1
        )

    model.save('/Users/colinni/evAl-chess/saved_keras_model_merged_first.h5')
    check_test_perf(model, X_test, Y_test)
