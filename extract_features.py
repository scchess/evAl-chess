'''
Given a position represented as a `chess.Board` object, extracts basic
low-level features to be used for training.

Currently uses the same features Matthew Lai used in Giraffe as
described in his paper 'Using Deep Reinforcement to Play Chess',
available at http://arxiv.org/pdf/1509.01549v2.pdf.

Features used:
1. The side to move.
2. Castling rights.
3. The count of each type of piece.
4. The location of each piece and value of the lowest-valued attacker
   and defender of the piece.
5. The distance each sliding piece, {bishop, rook, queen}, can legally
   move in each direction.
6. The value of the lowest-valued attacker and defender of each square.

See the docstrings in each method for more information; this module
uses one method for each of the 6 groups of features listed above.

Usage:
    Call get_features() on a `chess.Board` object; it returns a list
    of `int`s with length on the order of 300.

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
import string
import itertools

# Add some crucial constants to the `chess` module.
chess.WHITE_PIECES, chess.BLACK_PIECES = (
    ('P', 'N', 'B', 'R', 'Q', 'K'),
    ('p', 'n', 'b' ,'r', 'q', 'k')
)
chess.PIECES = chess.WHITE_PIECES + chess.BLACK_PIECES
chess.SLIDING_PIECES = (
    'B', 'R', 'Q', 'b', 'r', 'q'
)
chess.PIECE_CAPACITY = {
    'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
    'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
}
chess.MISSING_PIECE_SQUARE = -1
chess.PIECE_MOVEMENTS = {
    'R' : tuple(zip((+1, +0, -1, +0), (+0, +1, +0, -1))),
    'B' : tuple(zip((+1, -1, -1, +1), (+1, +1, -1, -1))),
    'N' : tuple(
        zip(
            (+2, +1, -1, -2, -2, -1, +1, +2),
            (+1, +2, +2, +1, -1, -2, -2, -1)
        )
    ),
    'P' : tuple(zip((-1, -1), (-1, +1))),
    'p' : tuple(zip((+1, +1), (-1, +1)))
}
chess.PIECE_MOVEMENTS['Q'] = chess.PIECE_MOVEMENTS['K'] = (
    chess.PIECE_MOVEMENTS['B'] + chess.PIECE_MOVEMENTS['R']
)
for piece in ('k', 'q', 'r', 'b', 'n'):
    chess.PIECE_MOVEMENTS[piece] = chess.PIECE_MOVEMENTS[piece.upper()]

# Some constants for use in this module.
FEATURE_MODALITY_SIZES = (0, 17, 244, 128)
FEATURE_MODALITY_SPLIT_POINTS = tuple(
    itertools.accumulate(FEATURE_MODALITY_SIZES)
)

def get_features(position, verbose=False):
    '''
    Returns a list of low-level features of `position` to be used for
    training. (See individual docstrings for more information.)

    Warning: assigns new members to `position`.

    Parameters:
        `position` : a `chess.Board` instance
            The position from which to extract features.
        `verbose` : bool
            Prints the features by group if True.

    Returns:
        list of `int`s, length on the order of 300
            The selected features of `position`.
    '''
    _init_square_data(position)
    features = (
        []
        + _side_to_move(position, verbose)
        + _castling_rights(position, verbose)
        + _material_configuration(position, verbose)
        + _piece_lists(position, verbose)
        + _sliding_pieces_mobility(position, verbose)
        + _attack_and_defend_maps(position, verbose)
    )
    return features

def split_features(features):
    '''
    Split `features` into groups of features with the same modality.

    Parameters:
        `features` : list of `int`'s
            A list of features returned by `get_features()`.

    Returns:
        list of lists of `int`'s
            Each list represents a modality. (A list because numpy
            doesn't like tuples.)
    '''
    return [
        features[ : , start : end]
        for start, end in zip(
            FEATURE_MODALITY_SPLIT_POINTS[ : -1],
            FEATURE_MODALITY_SPLIT_POINTS[1 : ]
        )
    ]

def __init_attackers_and_scope(position, piece_squares):
    '''
    Manually calculates the value of the lowest-valued attacker and
    defender of each square and the scope of each sliding piece in
    `position`. Stores this information as:

        `position.min_attacker_of`:
            The value of the lowest-valued attacker of each square for
            each color. `position.min_attacker_of[square][chess.BLACK]`
            is the value of the lowest-valued black piece that attacks
            `square`.

        `position.sliding_piece_scopes`:
            How far each sliding piece can slide in each direction
            before either hitting a piece or the edge of the board.
            If it hits a piece of the opposite color, the square
            the piece is on counts as a square onto which it can slide.

    Using `chess.Board.attackers` to calculate this stuff takes too
    long -- something like 5e-2 seconds per position, which would mean
    20-plus hours to run through the 2-million positions offered by
    Kaggle (see `create_data.py`). This method improves the time to
    around 3e-3 seconds per position; still, that would take around 2
    hours for the 2-million positions. For now, we use just 200,000
    positions.

    TODO: Use Stockfish's code. (Major pain.)
    '''
    # The color of each piece on each square -- -1 if the square is
    # empty.
    piece_colors = np.full(shape=(8, 8), fill_value=-1, dtype=int)
    for piece in piece_squares:
        for square in piece_squares[piece]:
            piece_colors[__to_coord(square)] = (
                chess.WHITE if piece in chess.WHITE_PIECES
                else chess.BLACK
            )

    # The value of the lowest-valued attacker of each square.
    min_white_attacker_of, min_black_attacker_of = (
        np.zeros((8, 8)),
        np.zeros((8, 8))
    )

    def in_range(i, j):
        '''
        Whether the row-major coordinate `(i, j)` exists.
        '''
        return (0 <= i < 8) and (0 <= j < 8)

    def assign(arr, i, j, val):
        '''
        Returns:
            3-d tuple
                The first element is a bool that is `True` if the
                square exists; the second is a bool that is `True`
                if the square has a piece on it; the third is the color
                of the piece if the second is `True` and `None`
                otherwise.
        '''
        if not in_range(i, j):
            return False, False, None
        elif piece_colors[i, j] != -1:
            arr[i, j] = val
            return True, True, piece_colors[i, j]
        else:
            arr[i, j] = val
            return True, False, None

    def assign_while(arr, piece_color, i, di, j, dj, val):
        '''
        Simulates a sliding piece moving.

        Starts at `(i, j)` and iterates `(di, dj)` until it hits a
        piece or the edge of the board. Assigns its value to each
        square it visited.

        Returns:
            `int`
                The number of times it assigned a square -- the scope
                of the piece at `(i, j)`.
        '''
        continue_assigning, scope = True, 0
        while continue_assigning:
            exists, had_piece, other_piece_color = assign(
                arr, i + di, j + dj, val
            )
            continue_assigning = exists and not had_piece
            scope += (
                (exists and not had_piece)
                or (had_piece and not other_piece_color == piece_color)
            )
            i += di
            j += dj
        return scope

    # The relative value of each piece.
    relative_vals = {
        'P' : 1, 'N' : 2, 'B' : 3, 'R' : 4, 'Q' : 5, 'K' : 6,
        'p' : 1, 'n' : 2, 'b' : 3, 'r' : 4, 'q' : 5, 'k' : 6
    }

    # How far each sliding piece can move in each direction.
    position.sliding_piece_scopes = {
        (sliding_piece, square) : []
        for sliding_piece in chess.SLIDING_PIECES
        for square in piece_squares[sliding_piece]
    }

    # Iterate through all legal moves of each piece, beginning with the
    # highest value, assigning the piece's value to the `attackers`
    # arrays. The result is the value of the lowest-valued attacker
    # for each square.
    for piece in reversed(chess.PIECES):
        piece_color = (
            chess.WHITE
            if piece in chess.WHITE_PIECES
            else chess.BLACK
        )
        # Which array to which to assign.
        arr = (
            min_white_attacker_of
            if piece_color == chess.WHITE
            else min_black_attacker_of
        )
        # If it's a sliding piece, assign its value in each direction
        # while it can continue moving in the direction.
        if piece in chess.SLIDING_PIECES:
            for square in piece_squares[piece]:
                i, j = __to_coord(square)
                for di, dj in chess.PIECE_MOVEMENTS[piece]:
                    scope = assign_while(
                        arr, piece_color,
                        i, di,
                        j, dj,
                        relative_vals[piece]
                    )
                    position.sliding_piece_scopes[(piece, square)].append(
                        scope
                    )
        # If it's not a sliding piece, simply iterate through each of
        # its movements and assign its value.
        else:
            for i, j in (
                __to_coord(square)
                for square in piece_squares[piece]
            ):
                for di, dj in chess.PIECE_MOVEMENTS[piece]:
                    assign(arr, i + di, j + dj, relative_vals[piece])

    position.min_attacker_of = [
        (j, i)
        for i, j in zip(
            min_white_attacker_of.flatten().astype(int).tolist(),
            min_black_attacker_of.flatten().astype(int).tolist()
        )
    ]


def _init_square_data(position):
    '''
    Calculates some basic information of the position and stores it in
    `position`.

    `position.piece_squares`:

        Each possible piece -- 8 pawns, 3 knights, 2 queens, etc. --
        and its square. If the piece isn't on the board, the square is
        set to `chess.MISSING_PIECE_SQUARE`. The length is constant
        regardless of the position because the number of possible
        pieces is constant.

        Pieces are grouped together in the same order as `chess.PIECES`
        -- 'P', 'N', 'B' ... 'p', 'n', 'b' ... -- but their squares are
        randomly permuted. As a result, the first 8 pieces are
        guaranteed to be 'P' but their squares random.
    '''
    # The squares of the pieces on the board.
    piece_squares = { piece : [] for piece in chess.PIECES }
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)

    # Pass `piece_squares` before adding the missing pieces' squares.
    # This is a bit ugly, I know.
    __init_attackers_and_scope(position, piece_squares)

    # Add the missing pieces and their squares,
    # `chess.MISSING_PIECE_SQUARE`.
    for piece in chess.PIECES:
        piece_squares[piece] += (
            [chess.MISSING_PIECE_SQUARE]
            * (chess.PIECE_CAPACITY[piece] - len(piece_squares[piece]))
        )

    # Set to `position.piece_squares` with the pieces ordered correctly
    # and the squares of each piece permuted.
    position.piece_squares = [
        (piece, square)
        for piece in chess.PIECES
        for square in np.random.permutation(piece_squares[piece]).tolist()
    ]


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
    True if {White, Black} has castling rights on the {kingside,
    queenside}.

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

    Number of features contributed: 12. Six types of pieces for each
    side.
    '''

    # The first field of the FEN representation of the position. To
    # count the number of each piece in the position, simply count the
    # frequency of its corresponding letter in `board_fen`.
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
    3. The values of the minimum-valued {attacker, defender} of the
       piece stored in a tuple. By default, (-1, -1).

    Number of features contributed: 180. Sixteen slots for the original
    pieces plus two extra for a knight and queen per side. Each slot
    takes 5 features, two for the coordinate, one for the bool, 2 for
    the attacker and defender.

    (*) Yes, it's technically possible to have, say, 10 knights per
    side. But in practice having an extra queen and knight slot is
    enough unless you're somehow in a position where you need to
    underpromote a pawn in the opening or middlegame.
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

    Number of features contributed: 64. One `int` for each direction
    for each possible piece. Four bishops and 4 rooks with 4 directions
    each yields 32; four queens with 8 directions each yields another
    32.
    '''
    mobilities = [
        scope_dir
        for piece, square in position.piece_squares
        if piece in chess.SLIDING_PIECES
        for scope_dir in (
            position.sliding_piece_scopes[(piece, square)]
            if square != chess.MISSING_PIECE_SQUARE
            else [-1] * len(chess.PIECE_MOVEMENTS[piece])
        )
    ]

    if verbose:
        print('Sliding pieces mobility')
        print('------------------------------------')
        print_mobls = (mobl for mobl in mobilities)
        for sliding_piece in chess.SLIDING_PIECES:
            print('Sliding piece:', sliding_piece)
            for num in range(chess.PIECE_CAPACITY[sliding_piece]):
                print('#' + str(num), end=': ')
                for movable_dir in chess.PIECE_MOVEMENTS[sliding_piece]:
                    print(next(print_mobls), end=' ')
                print()
        print()
        print(np.reshape(mobilities,(len(mobilities) //4, 4)))

    return mobilities


def _attack_and_defend_maps(position, verbose=False):
    '''
    The value of the lowest-valued attack and defender of each square;
    by default, 0.

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

    Number of features contributed: 128. Sixty-four integers for each
    the attack and defend maps.
    '''

    attack_and_defend_maps = [
        position.min_attacker_of[square][color]
        for color in (chess.BLACK, chess.WHITE)
        for square in chess.SQUARES
    ]

    if verbose:
        print('Attack and defend maps.')
        print('------------------------------------')
        print('White attackers.')
        print(np.reshape(attack_and_defend_maps, (16, 8))[:8])
        print()
        print('White defenders.')
        print(np.reshape(attack_and_defend_maps, (16, 8))[8:])
        print()

    return attack_and_defend_maps
