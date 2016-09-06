'''
Given a position, extracts basic low-level features to be used for training.
Currently using the same features Matthew Lai used in Giraffe.

Features used:
1. The side to move.
2. Castling rights.
3. The count of each type of piece.
4. The location of each piece and value of the lowest-valued attacker and
   defender of the piece.
5. The distance each sliding piece, {bishop, rook, queen}, can legally
   move in each direction.
6. The value of the lowest-valued attacker and defender of each square.

See the docstrings in each method for more information; this module uses one
method for each of the 6 groups of features listed above.

Usage:
    Call get_features() on a `python-chess` `Board` object. It returns
    a list of `int`s with length on the order of 300.

        import extract_features
        import chess

        board = chess.Board()
        features = extract_features.get_features(board)
'''

import chess
from chess import Board
import re
import numpy as np
import time

# Add some crucial constants to the `chess` module.
chess.PIECES = (
    'P', 'N', 'B', 'R', 'Q', 'K',
    'p', 'n', 'b' ,'r', 'q', 'k'
)
chess.PIECE_CAPACITY = {
    'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
    'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
}
chess.MISSING_PIECE_SQUARE = -1

runtimes = {
    i : 0.0
    for i in range(10)
}

def get_features(position, verbose=False):
    '''
    Returns a list of low-level features of `position` to be used for
    training. (See individual docstrings for more information.)

    Warning: assigns new members to `position`.

    Parameters:
        `position` : `chess.Board` object
            The position from which to extract features.
        `verbose` : bool
            Prints the features by group if True.

    Returns:
        list of `int`s, length on the order of 300
            The selected features of `position`.
    '''
    prev_time = time.time()
    _init_square_data(position)
    runtimes[0] += time.time() - prev_time
    # features = (
    #     []
    #     + _side_to_move(position, verbose)
    #     + _castling_rights(position, verbose)
    #     + _material_configuration(position, verbose)
    #     + _piece_lists(position, verbose)
    #     + _sliding_pieces_mobility(position, verbose)
    #     + _attack_and_defend_maps(position, verbose)
    # )

    features = []

    prev_time = time.time()
    features += _side_to_move(position, verbose)
    runtimes[1] += time.time() - prev_time

    prev_time = time.time()
    features += _castling_rights(position, verbose)
    runtimes[2] += time.time() - prev_time

    prev_time = time.time()
    features += _material_configuration(position, verbose)
    runtimes[3] += time.time() - prev_time

    prev_time = time.time()
    features += _piece_lists(position, verbose)
    runtimes[4] += time.time() - prev_time

    prev_time = time.time()
    features += _sliding_pieces_mobility(position, verbose)
    runtimes[5] += time.time() - prev_time

    prev_time = time.time()
    features += _attack_and_defend_maps(position, verbose)
    runtimes[6] += time.time() - prev_time


    assert all(type(feature) in [int, bool] for feature in features)
    return features

def __init_min_attackers(piece_squares):

    # 0 if no piece. 1 if white piece. 2 if black piece.
    piece_exists = np.zeros((8, 8))
    for piece in chess.PIECES:
        for square in piece_squares[piece]:
            piece_exists[__to_coord(square)] = True


    attackers_of_white = np.zeros((8, 8))
    attackers_of_black = np.zeros((8, 8))

    def in_range(i, j):
        return (0 <= i < 8) and (0 <= j < 8)

    def assign(arr, i, j, val):
        if not in_range(i, j):
            return False
        elif piece_exists[i, j] == 1 or piece_exists[i, j] == 2:
            arr[i, j] = val
            return False
        else:
            arr[i, j] = val
            return True

    def assign_while(arr, i, di, j, dj, val):
        while assign(arr, i + di, j + dj, val):
            i += di
            j += dj

    for kingi, kingj in (__to_coord(square) for square in piece_squares['K']):
        for di, dj in zip((+1, +1, +0, -1, -1, -1, +0, +1), (+0, +1, +1, +1, +0, -1, -1, -1)):
            assign(attackers_of_white, kingi + di, kingj + dj, 6)

    for queeni, queenj in (__to_coord(square) for square in piece_squares['Q']):
        for di, dj in zip((+1, +1, +0, -1, -1, -1, +0, +1), (+0, +1, +1, +1, +0, -1, -1, -1)):
            assign_while(attackers_of_white, queeni, di, queenj, dj, 5)

    for rooki, rookj in (__to_coord(square) for square in piece_squares['R']):
        for di, dj in zip((+1, +0, -1, +0), (+0, +1, +0, -1)):
            assign_while(attackers_of_white, rooki, di, rookj, dj, 4)

    for bishopi, bishopj in (__to_coord(square) for square in piece_squares['B']):
        for di, dj in zip((+1, -1, -1, +1), (+1, +1, -1, -1)):
            assign_while(attackers_of_white, bishopi, di, bishopj, dj, 3)

    for knighti, knightj in (__to_coord(square) for square in piece_squares['N']):
        for di, dj in zip((+2, +1, -1, -2, -2, -1, +1, +2), (+1, +2, +2, +1, -1, -2, -2, -1)):
            assign(attackers_of_white, knighti + di, knightj + dj, 2)

    for pawni, pawnj in (__to_coord(square) for square in piece_squares['P']):
        for di, dj in zip((-1, -1), (-1, +1)):
            assign(attackers_of_white, pawni + di, pawnj + dj, 1)

    for kingi, kingj in (__to_coord(square) for square in piece_squares['k']):
        for di, dj in zip((+1, +1, +0, -1, -1, -1, +0, +1), (+0, +1, +1, +1, +0, -1, -1, -1)):
            assign(attackers_of_black, kingi + di, kingj + dj, 6)

    for queeni, queenj in (__to_coord(square) for square in piece_squares['q']):
        for di, dj in zip((+1, +1, +0, -1, -1, -1, +0, +1), (+0, +1, +1, +1, +0, -1, -1, -1)):
            assign_while(attackers_of_black, queeni, di, queenj, dj, 5)

    for rooki, rookj in (__to_coord(square) for square in piece_squares['r']):
        for di, dj in zip((+1, +0, -1, +0), (+0, +1, +0, -1)):
            assign_while(attackers_of_black, rooki, di, rookj, dj, 4)

    for bishopi, bishopj in (__to_coord(square) for square in piece_squares['b']):
        for di, dj in zip((+1, -1, -1, +1), (+1, +1, -1, -1)):
            assign_while(attackers_of_black, bishopi, di, bishopj, dj, 3)

    for knighti, knightj in (__to_coord(square) for square in piece_squares['n']):
        for di, dj in zip((+2, +1, -1, -2, -2, -1, +1, +2), (+1, +2, +2, +1, -1, -2, -2, -1)):
            assign(attackers_of_black, knighti + di, knightj + dj, 2)

    for pawni, pawnj in (__to_coord(square) for square in piece_squares['p']):
        for di, dj in zip((+1, +1), (-1, +1)):
            assign(attackers_of_black, pawni + di, pawnj + dj, 1)

    return [(j, i) for i, j in zip(attackers_of_white.flatten().astype(int).tolist(), attackers_of_black.flatten().astype(int).tolist())]



def _init_square_data(position):
    '''
    Calculates some basic information of the position and stores it in
    `position`.

    `position.piece_squares`:

        Each possible piece -- 8 pawns, 3 knights, 2 queens, etc. -- and
        its square. If the piece isn't on the board, the square is set to
        `chess.MISSING_PIECE_SQUARE`. The length is constant regardless of
        the position because the number of possible pieces is constant.

        Pieces are grouped together in the same order as `chess.PIECES` --
        'P', 'N', 'B' ... 'p', 'n', 'b' ... -- but their squares are randomly
        permuted. As a result, the first 8 pieces are guaranteed to be 'P'
        but their squares random.

    `position.min_attacker_of`:

        The value of the lowest-valued attacker of each square for each
        color. `position.min_attacker_of[square][chess.BLACK]` is the value
        of the lowest-valued black piece that attacks `square`.
    '''
    # The squares of the pieces on the board.
    piece_squares = { piece : [] for piece in chess.PIECES }
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)

    position.min_attacker_of = __init_min_attackers(piece_squares)

    # Add the missing pieces and their squares, `chess.MISSING_PIECE_SQUARE`.
    for piece in chess.PIECES:
        piece_squares[piece] += (
            [chess.MISSING_PIECE_SQUARE]
            * (chess.PIECE_CAPACITY[piece] - len(piece_squares[piece]))
        )
    # Set to `position.piece_squares` with the pieces ordered correctly and
    # the squares of each piece permuted.

    position.piece_squares = [
        (piece, square)
        for piece in chess.PIECES
        for square in np.random.permutation(piece_squares[piece]).tolist()
    ]


    # Use lists instead of square: dicts, you dumbass! Squares are in range(64).

def _side_to_move(position, verbose=False):
    '''
    True if it's White turn to move.

    Number of features contributed: 1.
    '''
    side_to_move = [position.turn]
    if verbose:
        print()
        print('Side to move')
        print('------------------------------------')
        print([position.turn])
        print()
    return side_to_move


def _castling_rights(position, verbose=False):
    '''
    True if {White, Black} has castling rights on the {kingside, queenside}.

    Number of features contributed: 4.
    '''
    castling_rights = [
        position.has_kingside_castling_rights(chess.WHITE),
        position.has_kingside_castling_rights(chess.BLACK),
        position.has_queenside_castling_rights(chess.WHITE),
        position.has_queenside_castling_rights(chess.BLACK)
    ]

    if verbose:
        print('Castling rights')
        print('------------------------------------')
        rights = (right for right in castling_rights)
        print('White kingside:', next(rights))
        print('Black kingside:', next(rights))
        print('White queenside:', next(rights))
        print('Black queenside:', next(rights))
        print()

    return castling_rights


def _material_configuration(position, verbose=False):
    '''
    The number of each piece on the board.

    Number of features contributed: 12. Six types of pieces for each side.
    '''

    # The first field of the FEN representation of the position. To count
    # the number of each piece in the position, simply count the frequency
    # of its corresponding letter in `board_fen`.
    board_fen = position.board_fen()

    material_config = [
        board_fen.count(piece)
        for piece in chess.PIECES
    ]

    if verbose:
        print('Material Config')
        print('------------------------------------')
        print_m_config = (m for m in material_config)
        for piece in chess.PIECES:
            print('Piece:', piece, '\tCount:', next(print_m_config))
        print()

    return material_config


def __to_coord(square):
    '''
    The row-major coordinate of square. Used by `_piece_lists()`.
    '''
    return (8 - square // 8 - 1, square % 8)


def _piece_lists(position, verbose=False):
    '''
    For each possible piece (*):
    1. Its row-major, zero-indexed coordinate. By default, (-1, -1).
    2. Whether the piece is on the board.
    3. The values of the minimum-valued {attacker, defender} of the piece
       stored in a tuple. By default, (-1, -1).

    Number of features contributed: 180. Sixteen slots for the original
    pieces plus two extra for a knight and queen per side. Each slot takes
    5 features, two for the coordinate, one for the bool, 2 for the
    attacker and defender.

    (*) Yes, it's technically possible to have, say, 10 knights per side. But
    in practice having an extra queen and knight slot is enough unless
    you're somehow in a position where you need to underpromote a pawn in the
    opening or middlegame.
    '''
    piece_lists = list(
        sum(
            [
                (-1, -1, False, -1, -1)
                if square == chess.MISSING_PIECE_SQUARE
                else (
                    __to_coord(square)
                    + (True, )
                    + (
                        position.min_attacker_of[square]
                        if position.piece_at(square).color == chess.WHITE
                        else tuple(reversed(position.min_attacker_of[square]))
                    )
                )
                for piece, square in position.piece_squares
            ],
            tuple()
        )
    )

    if verbose:
        print('Piece lists')
        print('------------------------------------')
        print_piece_lists = (element for element in piece_lists)
        for piece in chess.PIECES:
            print('Piece:', piece)
            for num in range(chess.PIECE_CAPACITY[piece]):
                print('#', num, end=': ', sep='')
                for i in range(5):
                    print(next(print_piece_lists), end=' ')
                print()
        print()

    return piece_lists


def __direction(from_square, to_square):
    '''
    The direction traveled in going from `from_square` to `to_square`.
    The value v returned yields the direction in <cos(v * pi / 4),
    sin(v * pi / 4)>.
    '''
    from_coord, to_coord = __to_coord(from_square), __to_coord(to_square)
    dx, dy = (to_coord[0] - from_coord[0], to_coord[1] - from_coord[1])
    dx, dy = dx // max(abs(dx), abs(dy)), dy // max(abs(dx), abs(dy))
    return {
        (0, 1) : 0,
        (-1, 1) : 1,
        (-1, 0) : 2,
        (-1, -1) : 3,
        (0, -1) : 4,
        (1, -1) : 5,
        (1, 0) : 6,
        (1, 1) : 7
    }[(dx, dy)]


def _sliding_pieces_mobility(position, verbose=False):
    '''
    How far each {white, black} {bishop, rook, queen} can slide in each
    legal direction.

    Number of features contributed: 64. One `int` for each direction for
    each possible piece. Four bishops and 4 rooks with 4 directions each
    yields 32; four queens with 8 directions each yields another 32.

    The code is difficult to understand; there aren't any good names for
    the variables.
    '''
    up_down, diag = (0, 2, 4, 6), (1, 3, 5, 7)
    movable_dirs = {
        'B' : diag,
        'R' : up_down,
        'Q' : up_down + diag,
        'b' : diag,
        'r' : up_down,
        'q' : up_down + diag
    }

    # Get all pseudo-legal moves in the position. `Board.pseudo_legal_moves`
    # yields the moves only for the side to play, so switch the turn and
    # query twice to get all of them.
    side_1_moves = [move for move in position.pseudo_legal_moves]
    position.turn = not position.turn
    side_2_moves = [move for move in position.pseudo_legal_moves]
    position.turn = not position.turn
    all_pseudo_legal_moves = side_1_moves + side_2_moves

    sliding_pieces = ('B', 'R', 'Q', 'b', 'r', 'q')

    # The number of moves each sliding piece can make in each direction.
    # For instance, if a white rook on `square` could legally move 2 right
    # and 3 down, legal_move_dirs[('R', square)] would equal [2, 0, 0, 3].
    legal_move_dirs = {
        (piece, square) : []
        for piece, square in position.piece_squares
        if square != chess.MISSING_PIECE_SQUARE and piece in sliding_pieces
    }
    for move in all_pseudo_legal_moves:
        piece = position.piece_at(move.from_square).symbol()
        if piece in sliding_pieces:
            legal_move_dirs[(piece, move.from_square)].append(
                __direction(move.from_square, move.to_square)
            )

    mobilities = []
    for piece, square in position.piece_squares:
        if piece in sliding_pieces:
            sliding_piece = piece
            if square == chess.MISSING_PIECE_SQUARE:
                mobilities += [-1] * len(movable_dirs[sliding_piece])
            else:
                mobilities += [
                    legal_move_dirs[(sliding_piece, square)]
                    .count(movable_dir)
                    for movable_dir in movable_dirs[sliding_piece]
                ]

    if verbose:
        print('Sliding pieces mobility')
        print('------------------------------------')
        print_mobls = (mobl for mobl in mobilities)
        for sliding_piece in sliding_pieces:
            print('Sliding piece:', sliding_piece)
            for num in range(chess.PIECE_CAPACITY[sliding_piece]):
                print('#' + str(num), end=': ')
                for movable_dir in movable_dirs[sliding_piece]:
                    print(next(print_mobls), end=' ')
                print()
        print()

    return mobilities


def _attack_and_defend_maps(position, verbose=False):
    '''
    The value of the lowest-valued attack and defender of each square; by
    default, 0.

    For the following 4x4 board, R B . .
                                 . . b .
                                 q k . .
                                 . . . P,
    the attack map would be  5 3 5 3
                             5 5 6 0
                             6 3 6 3
                             3 5 6 0,
    and the defend map  4 4 0 0
                        3 0 3 0
                        4 0 1 3
                        0 0 0 0.

    Number of features contributed: 128. Sixty-four integers for each the
    attack and defend maps.
    '''

    attack_and_defend_maps = [
        position.min_attacker_of[square][color]
        for color in (chess.BLACK, chess.WHITE)
        for square in chess.SQUARES
    ]

    if not verbose:
        print('Attack and defend maps.')
        print(position)
        print('------------------------------------')
        print('White attackers.')
        print(np.reshape(attack_and_defend_maps, (16, 8))[:8])
        print()
        print('White defenders.')
        print(np.reshape(attack_and_defend_maps, (16, 8))[8:])
        print()

    return attack_and_defend_maps
