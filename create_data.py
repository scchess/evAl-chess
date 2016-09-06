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

def get_features(game_node):
    features = extract_features.get_features(
        game_node.board(),
        verbose=False
    )
    return features


# def get_engine_eval(game_node):
#     '''
#     Returns what the engine evaluated the position in `game_node` to be;
#     returns None if the move played was taken from the book.
#
#     The returned evaluation is by the engine that played the move that
#     immediately led to `game_node`; the TCEC updates the engines' evaluations
#     only after moves they play.
#     '''
#
#     # TCEC's pgn files log engine information such as time remaining
#     # and depth searched for each move played. `python-chess`, when parsing
#     # the pgn files, considers this information to be the `comment` of each
#     # move.
#     comment = game_node.comment
#
#     # Search for floats in `comment`. No other piece of information logged
#     # by the TCEC comes in the form of a float, so this yields the evaluation.
#     engine_eval = re.findall('\d+\.\d+', comment)
#     if len(engine_eval) == 0:
#         # This move wasn't calculated because it was taken from the book.
#         return None
#     else:
#         assert len(engine_eval) == 1
#         return float(engine_eval[0])

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
        print(n_games)
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
        curr_game_node = curr_game.root().variation(0)
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
        if n_games == 10:
            break

    np.save('/Users/colinni/evAl-chess/X.npy', np.array(data_X).astype(float))
    np.save('/Users/colinni/evAl-chess/Y.npy', np.array(data_Y))

create_data()


# def create_data():
#     # (I thank you with all my heart, python-chess documentation.)
#     tcec_2015_archive = open(
#         '/Users/colinni/evAl-chess/tcec_2015_archive.pgn',
#         encoding='utf-8-sig',
#         errors='surrogateescape'
#     )
#
#     data_file = open('/Users/colinni/evAl-chess/data.txt', 'w')
#
#     data_X, data_Y = [],[]
#
#     # Iterate through every game in the archive.
#     curr_game = chess.pgn.read_game(tcec_2015_archive)
#
#     # chess.pgn.read_game returns None when it reaches the EOF.
#     while curr_game is not None:
#
#         engine_white, engine_black = (
#             curr_game.headers[color]
#             for color in ('White', 'Black')
#         )
#
#         if 'fish' in engine_white:
#             stock = engine_white
#         elif 'fish' in engine_black:
#             stock = engine_black
#         else:
#             curr_game = chess.pgn.read_game(tcec_2015_archive)
#             continue
#
#         # Iterate through every move played.
#         curr_game_node = curr_game.root()
#         while curr_game_node is not None:
#             if (
#                 (stock == engine_white and not curr_game_node.board().turn)
#                 or (stock == engine_black and curr_game_node.board().turn)
#             ):
#                 features, engine_eval = get_features_and_eval(curr_game_node)
#                 if engine_eval is not None:
#                     data_X.append(features)
#                     data_Y.append(engine_eval)
#             # Set curr_game_node to the next position in the game. If it's the
#             # end of the game, set it to None.
#             curr_game_node = (
#                 curr_game_node.variation(0)
#                 if not curr_game_node.is_end()
#                 else None
#             )
#
#         # Get the next game in the pgn file.
#         curr_game = chess.pgn.read_game(tcec_2015_archive)
#
#     np.save('/Users/colinni/evAl-chess/X.npy', np.array(data_X))
#     np.save('/Users/colinni/evAl-chess/Y.npy', np.array(data_Y))
