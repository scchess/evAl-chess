'''
Work in-progress.
'''

from keras.models import load_model
import chess
import train
import extract_features
import numpy as np

model = load_model(
    '/Users/colinni/evAl-chess/saved_keras_model.h5'
)

def engine_evaluate(position):
    '''
    The zero-search engine's evaluation of `position`; a higher number
    means that the engine evaluates that the position favors white.
    '''
    x_unscaled = np.array([extract_features.get_features(position)]).astype(float)
    x_scaled = train.scaler_X.transform(x_unscaled)
    y_scaled = scaled_evaluation = model.predict(extract_features.split_features(x_scaled))
    y_unscaled = unscaled_evaluation = train.scaler_Y.inverse_transform(y_scaled)
    return unscaled_evaluation[0][0]


def get_engine_analysis(position):
    '''
    The result of the zero-search engine's mini-max search for each possible
    move. Of course for an engine that doesn't search, it's basically just
    the evaluation of the positions at depth 1.

    (Funnily enough, the engine can't differentiate between legal and
    illegal positions. So we're going to give it only legal moves to decide
    from. Heh, oops -- I thought I forgot to add something to the training
    data...)
    '''
    engine_analysis = {}
    for move in position.legal_moves:
        # Play the move; evaluate the position; then 'unplay' the move.
        position.push(move)
        eval_position = engine_evaluate(position)
        engine_analysis[move] = eval_position
        position.pop()

    return engine_analysis


def get_engine_move(position, playing_color):
    return (
        max(
            get_engine_analysis(position).items(),
            key=lambda item : item[1]
        )
        if playing_color == chess.WHITE
        else min(
            get_engine_analysis(position).items(),
            key=lambda item : item[1]
        )
    )

def alpha_beta(position, depth, alpha=-500, beta=+500, color=chess.WHITE):
    if depth == 0:
        _eval = engine_evaluate(position)
        # print(position, _eval)
        return _eval, None

    branches = []
    for move in position.legal_moves:
        child_position = position.copy()
        child_position.push(move)
        branches.append((move, child_position))

    branches_sorted = tuple(
        sorted(
            branches,
            key=(
                lambda item :
                    - 2 * item[1].is_check()
                    - 1 * (position.piece_type_at(item[0].to_square) is not None)
            )
        )
    )

    if color == chess.WHITE:
        best_eval, best_move = -500, None
        for move, child_position in branches_sorted:
            child_eval, _ = alpha_beta(child_position, depth - 1, alpha, beta, chess.BLACK)
            if child_eval > best_eval:
                best_eval = child_eval
                best_move = move
            alpha = max(alpha, child_eval)
            if beta <= alpha:
                break
        return best_eval, best_move
    elif color == chess.BLACK:
        best_eval, best_move = +500, None
        for move, child_position in branches_sorted:
            child_eval, _ = alpha_beta(child_position, depth - 1, alpha, beta, chess.WHITE)
            if child_eval < best_eval:
                best_eval = child_eval
                best_move = move
            beta = min(beta, child_eval)
            if beta <= alpha:
                break
        return best_eval, best_move


def play_engine(verbose=False):

    position = chess.Board()
    while not position.is_game_over():
        # engine_move, _eval = get_engine_move(position, chess.WHITE)
        engine_eval, engine_move = alpha_beta(position, 4)

        # if verbose:
        #     for move, _eval in get_engine_analysis(position).items():
        #         print(move, _eval)
        print('Engine played', engine_move)
        print('Engine evaluation', engine_eval)
        position.push(engine_move)
        print(position.__unicode__())
        print()
        player_move = input('Your move: ')
        position.push_san(player_move)

play_engine(True)

# position = chess.Board()
#
# moves = ['Nf3', 'd6', 'Ne5', 'dxe5', 'd4', 'e4', 'Qd3', 'exd3', 'exd3']
# for move in moves:
#     position.push_san(move)
#     print(position)
#     print(engine_evaluate(position))
#     print()
