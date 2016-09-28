'''
Creates the data with which we use to train our model.

Kaggle offers a database of 25,000 FIDE chess games, each analyzed by
Stockfish for 1 second per move; this yields on the order of 2 million
positions (assuming 40 moves -- 80 plies -- per game) and their
'ground-truth' evalutions.

The `create_data()` method iterates through the games in PGN
format -- the default -- and for each position extracts the low-level
features using `extract_features.py` and pairs them with Stockfish's
valuation. It then stores this data as a load-able numpy array for
later training.
'''

import extract_features
import chess.pgn
import re
import time
import pprint
import numpy as np
import time

def create_data(n_samples, verbose=False):
    file_game_pgns, file_stockfish_evals = (
        open(
            '../evAl-chess/data.pgn',
            encoding='utf-8-sig',
            errors='surrogateescape'
        ),
        open('../evAl-chess/stockfish.csv')
    )
    # Discard the first line; it contains headers.
    file_stockfish_evals.readline()

    # The accumulated data samples.
    data_X, data_Y = [],[]

    # Iterate through every game in the archive.
    curr_game = chess.pgn.read_game(file_game_pgns)
    n_curr_sample = 0
    # (`chess.pgn.read_game()` returns None when it reaches the EOF.)
    while curr_game is not None and n_curr_sample < n_samples:

        print('\rcurr game |', n_games, end='')

        # The evaluations of each position of each game.
        stockfish_evals = (
            # The evaluations are given in centi-pawns. Convert to the
            # more standard pawn scale.
            float(stockfish_eval) / 100.0
            # Stockfish gives 'NA' for forced mates.
            if stockfish_eval != 'NA'
            else None
            # The lines each begin with a number and comma
            # (e.g., '451,') which aren't part of the evaluations.
            # Discard by splitting the string by the comma, taking the
            # second part, and splitting once again to get the
            # individual numbers.
            for stockfish_eval in (
                file_stockfish_evals.readline()
                .split(',')[1]
                .split()
            )
        )

        # Iterate through every move played using the `chess.Game`
        # class.
        curr_game_node = (
            curr_game.root().variation(0)
            if not curr_game.root().variation(0).is_end()
            else None
        )
        # Setting `curr_game_node` to `None` as a flag is sloppy, but
        # the `chess.Game` class doesn't have a better way of detecting
        # 0-move games, which the database does contain.
        while curr_game_node is not None and n_curr_sample < n_samples:
            features, stockfish_eval = (
                extract_features.get_features(
                    game_node.board(),
                    verbose=verbose
                ),
                next(stockfish_evals)
            )
            # Stockfish gives 'NA' for forced mates, which we earlier
            # set to `None`.
            if stockfish_eval is not None:
                data_X.append(features)
                data_Y.append(stockfish_eval)
                n_curr_sample += 1

            # Set curr_game_node to the next position in the game. If
            # it's the end of the game, set it to None as a flag.
            curr_game_node = (
                curr_game_node.variation(0)
                if not curr_game_node.is_end()
                else None
            )

        # Get the next game in the pgn file.
        curr_game = chess.pgn.read_game(file_game_pgns)

    # Convert `data_X` and `data_Y` into numpy arrays and store them
    # in numpy's npy format. To load, `np.load(path)`.
    np.save('../evAl-chess/X.npy', np.array(data_X).astype(float))
    np.save('../evAl-chess/Y.npy', np.array(data_Y))
