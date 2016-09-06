'''
Creates data from an archive of engine games.

Each data sample consists of around 300 low-level features of a chess position
and an engine's evaluation of the position.

Loads the games from the 2015 TCEC. Extracts the low-level features for
each position in each game and the evaluation given by the engine. Writes
this data into a text file.
'''

import extract_features
import chess.pgn
import re
import time
import pprint
import numpy as np
import time

def get_features(game_node):
    features = extract_features.get_features(
        game_node.board(),
        verbose=True
    )
    return features


def create_data():
    # (I thank you with all my heart, python-chess documentation.)
    data_games = open(
        '/Users/colinni/evAl-chess/data.pgn',
        encoding='utf-8-sig',
        errors='surrogateescape'
    )
    stockfish_evals = open('/Users/colinni/evAl-chess/stockfish.csv')

    data_X, data_Y = [],[]
    stockfish_evals.readline()

    # Iterate through every game in the archive.
    curr_game = chess.pgn.read_game(data_games)
    n_games = 0

    # chess.pgn.read_game returns None when it reaches the EOF.
    while curr_game is not None:
        print('\rcurr game |', n_games, end='')
        # The lines begin with a number and comma (e.g., '451,') which aren't
        # part of the evaluations. Discard by splitting the string by the
        # comma, taking the second part, and splitting once again to get the
        # individual numbers.
        evals = (
            float(_eval) / 100.0
            # Stockfish gives 'NA' when it can't evaluate the position.
            if _eval != 'NA'
            else None
            for _eval in stockfish_evals.readline().split(',')[1].split()
        )
        # Iterate through every move played.
        curr_game_node = (
            curr_game.root().variation(0)
            if not curr_game.root().variation(0).is_end()
            else None
        )
        while curr_game_node is not None:
            features, _eval = get_features(curr_game_node), next(evals)
            if _eval is not None:
                data_X.append(features)
                data_Y.append(_eval)

            # Set curr_game_node to the next position in the game. If it's the
            # end of the game, set it to None.
            curr_game_node = (
                curr_game_node.variation(0)
                if not curr_game_node.is_end()
                else None
            )

        # Get the next game in the pgn file.
        curr_game = chess.pgn.read_game(data_games)
        n_games += 1
        if n_games == 5:
            break
    print()

    # np.save('/Users/colinni/evAl-chess/X.npy', np.array(data_X).astype(float))
    # np.save('/Users/colinni/evAl-chess/Y.npy', np.array(data_Y))

create_data()
