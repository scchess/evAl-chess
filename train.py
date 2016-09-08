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

print('X, Y shape.')
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
scaler_X.fit(X_train)
scaler_Y.fit(np.reshape(Y_train, (len(Y_train), 1)))


if __name__ == '__main__':


    def material_count(config):
        return 1 * config[0] + 3 * config[1] + 3 * config[2] + 5 * config[3] + 9 * config[4]

    def get_material_imbalanced_positions(X, Y, material_diff):
        material_diffs = np.array([abs(material_count(sample[5:11]) - material_count(sample[11:17])) for sample in X])
        material_diff_positions = np.where(material_diffs >= material_diff)
        return X[material_diff_positions, :][0], Y[material_diff_positions]

    def get_range_positions(X, Y, lower, upper):
        X = X[np.where(lower <= np.abs(Y))]
        Y = Y[np.where(lower <= np.abs(Y))]
        X = X[np.where(np.abs(Y) <= upper)]
        Y = Y[np.where(np.abs(Y) <= upper)]
        return X, Y

    def get_openings(X, Y):
        opening_positions = np.where(X[:, 5] >= 7)
        return X[opening_positions, :][0], Y[opening_positions]

    def __evaluate_model(test_data):
        X_test, Y_test = test_data
        print(X_test.shape, Y_test.shape)
        X_test = extract_features.split_features(X_test)
        X_test, Y_test = scaler_X.transform(X_test), scaler_Y.transform(Y_test)
        return model.evaluate(X_test, Y_test)


    def check_test_perf(model, X_test, Y_test):
        print(X_test.shape, Y_test.shape)
        print('Test score on all positions', __evaluate_model((X_test, Y_test)))
        print('Test score on material imabalanced positions', __evaluate_model(get_material_imbalanced_positions(X_test, Y_test, 2.5)))
        print('Test score on ground truth range 3-15', __evaluate_model(get_range_positions(X_test, Y_test, 3, 15)))
        print('Test score on ground truth range 1-3', __evaluate_model(get_range_positions(X_test, Y_test, 1, 3)))
        print('Test score on ground truth range 0.1-1', __evaluate_model(get_range_positions(X_test, Y_test, 0.1, 1)))
        print('Test score on openings', __evaluate_model(get_openings(X_test, Y_test)))


    # X_train, Y_train = get_material_imbalanced_positions(X_train, Y_train, 2.5)
    X_train, Y_train = get_range_positions(X_train, Y_train, 0.05, 15)

    X_train = scaler_X.transform(X_train)
    Y_train = scaler_Y.transform(np.reshape(Y_train, (len(Y_train), 1)))
    X_train = extract_features.split_features(X_train)



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
