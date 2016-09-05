import chess
from chess import Board
import re
import numpy as np

# Add some crucial constants and functions to the chess module.

chess.PIECES = [
    'P', 'N', 'B', 'R', 'Q', 'K',
    'p', 'n', 'b' ,'r', 'q', 'k'
]

chess.PIECE_FREQS = {
    'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
    'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
}

def __all_pseudo_legal_moves(self):
    # `Board.pseudo_legal_moves` yields the moves only for the side to
    # play. Change the side to play and append the opposing sides' moves
    # to get all the pseudo-legal moves.
    all_pseudo_legal_moves = [move for move in self.pseudo_legal_moves]
    self.turn = not self.turn
    all_pseudo_legal_moves += [move for move in self.pseudo_legal_moves]
    self.turn = not self.turn
    return all_pseudo_legal_moves

Board.all_pseudo_legal_moves = __all_pseudo_legal_moves


def _side_to_move(position):
    '''
    True if it's White to move.

    Number of features contributed: 1.
    '''
    print('Side to move')
    print([position.turn])
    print()
    return [position.turn]


def _castling_rights(position):
    '''
    True if {White, Black} has castling rights on the {kingside, queenside}.

    Number of features contributed: 4.
    '''
    print('Castling rights')
    print([
        position.has_kingside_castling_rights(chess.WHITE),
        position.has_kingside_castling_rights(chess.BLACK),
        position.has_queenside_castling_rights(chess.WHITE),
        position.has_queenside_castling_rights(chess.BLACK)
    ])
    print()
    return [
        position.has_kingside_castling_rights(chess.WHITE),
        position.has_kingside_castling_rights(chess.BLACK),
        position.has_queenside_castling_rights(chess.WHITE),
        position.has_queenside_castling_rights(chess.BLACK)
    ]


def _material_configuration(position):
    '''
    The number of each piece on the board.

    Number of features contributed: 12. Six types of pieces for each side.
    '''
    # The first field of the FEN representation of the position. To count
    # the number of each piece in the position, simply count the frequency
    # of its corresponding letter in `board_fen`.
    board_fen = position.board_fen()
    print('Material Config')
    print([
        board_fen.count(piece)
        for piece in chess.PIECES
    ])
    print()
    return [
        board_fen.count(piece)
        for piece in chess.PIECES
    ]

def _to_coord(square):
    '''
    The row-major coordinate of square. Used by `_piece_lists()`.
    '''
    return (8 - square // 8 - 1, square % 8)


def _piece_lists(position):
    '''
    For each reasonably-possible piece (*):
    1. Its row-major, zero-indexed coordinate. By default, (-1, -1).
    2. Whether the piece is on the board.
    3. The values of the minimum-valued {attacker, defender} of the piece
       stored in a tuple. By default, (-1, -1).

    Number of features contributed: 180. Sixteen slots for the original
    pieces plus two extra for a knight and queen per side. Each slot takes
    5 features, two for the coordinate, one for the bool, 2 for the
    attacker and defender.

    (*) It's technically possible to have, say, 10 knights per side. But
    in practice having an extra queen and knight slot is enough unless
    you're somehow in a position where you need to underpromote a pawn in the
    opening or middlegame.
    '''
    ###########################################
    piece_on_board = (
        square != chess.MISSING_PIECE_SQUARE
        for square in position.piece_squares
    )
    a = [
        element
        for square in position.piece_squares
        for element in (
            (-1, -1, next(piece_on_board), -1, -1)
            if square == chess.MISSING_PIECE_SQUARE
            else (
                _to_coord(square)
                + (next(piece_on_board), )
                + (
                    position.min_attacker_of[square]
                    if position.piece_at(square).color == chess.WHITE
                    else tuple(reversed(position.min_attacker_of[square]))
                )
            )
        )
    ]
    for i in range(len(a) // 5):
        for j in range(5):
            print(a[i * 5 + j], end=' ')
        print()
    print()
    ###########################################

    piece_on_board = (
        square != chess.MISSING_PIECE_SQUARE
        for square in position.piece_squares
    )
    return [
        element
        for square in position.piece_squares
        for element in (
            (-1, -1, next(piece_on_board), -1, 1)
            if square == chess.MISSING_PIECE_SQUARE
            else (
                _to_coord(square)
                + (next(piece_on_board), )
                + position.min_attacker_of[square]
            )
        )
    ]

def _direction(from_square, to_square):
    from_coord, to_coord = _to_coord(from_square), _to_coord(to_square)
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

def _sliding_pieces_mobility(position):
    '''
    How far each {white, black} {bishop, rook, queen} can slide in each
    applicable direction d.
    '''
    up_down, diag = (0, 2, 4, 6), (1, 3, 5, 7)
    sliding_pieces = ('B', 'R', 'Q', 'b', 'r', 'q')
    movable_dirs = {
        'B' : diag,
        'R' : up_down,
        'Q' : up_down + diag,
        'b' : diag,
        'r' : up_down,
        'q' : up_down + diag
    }

    legal_move_dirs = { }
    for move in position.all_pseudo_legal_moves():
        piece = position.piece_at(move.from_square).symbol()
        if piece in sliding_pieces:
            if (piece, move.from_square) not in legal_move_dirs:
                legal_move_dirs[(piece, move.from_square)] = []
            legal_move_dirs[(piece, move.from_square)].append(
                _direction(move.from_square, move.to_square)
            )


    mobilities = []
    for sliding_piece in sliding_pieces:
        found = [val for (piece, from_square), val in legal_move_dirs.items() if piece == sliding_piece]
        n_found, n_missing = len(found), chess.PIECE_FREQS[sliding_piece] - len(found)
        for val in found:
            for movable_dir in movable_dirs[sliding_piece]:
                mobilities.append(val.count(movable_dir))
        mobilities += [-1] * len(movable_dirs[sliding_piece]) * n_missing


    print('Sliding pieces mobility')
    print(mobilities)
    print()

    return mobilities


def _attack_and_defend_maps(position):
    print('Attack and defend maps')
    print(np.reshape([
        position.min_attacker_of[square][color]
        for color in (chess.WHITE, chess.BLACK)
        for square in chess.SQUARES_180
    ], (16, 8)))
    print()
    return [
        position.min_attacker_of[square][color]
        for color in (chess.WHITE, chess.BLACK)
        for square in chess.SQUARES_180
    ]

def _init_square_data(position):

    piece_squares = { piece : [] for piece in chess.PIECES }
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)


    chess.MISSING_PIECE_SQUARE = -1
    for piece in chess.PIECES:
        piece_squares[piece] += (
            [chess.MISSING_PIECE_SQUARE]
            * (chess.PIECE_FREQS[piece] - len(piece_squares[piece]))
        )

    position.piece_squares = [
        square
        for piece in chess.PIECES
        for square in np.random.permutation(piece_squares[piece])
    ]

    position.min_attacker_of = {
        square : tuple(
            min(
                (
                    position.piece_type_at(square)
                    for square in position.attackers(color, square)
                ),
                default=0
            )
            for color in (chess.WHITE, chess.BLACK)
        )
        for square in chess.SQUARES
    }


def get_position_features(position):
    _init_square_data(position)
    features = (
        []
        + _side_to_move(position)
        + _castling_rights(position)
        + _material_configuration(position)
        + _piece_lists(position)
        + _sliding_pieces_mobility(position)
        + _attack_and_defend_maps(position)
    )
    return features
