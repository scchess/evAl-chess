import chess
import re
import numpy as np

chess.PIECES = [
    'P', 'N', 'B', 'R', 'Q', 'K',
    'p', 'n', 'b' ,'r', 'q', 'k'
]


    # Search for floats in `comment`. No other piece of information logged
    # by the TCEC comes in the form of a float, so this yields the evaluation.
    engine_eval = re.findall('\d+\.\d+', comment)
    if len(engine_eval) == 0:
        # This move wasn't calculated because it was taken from the book.
        return None
    else:
        assert len(engine_eval) == 1
        return engine_eval[0]


def _side_to_move(position):
    '''
    True if it's White to move.

    Number of features contributed: 1.
    '''
    return [position.turn]


def _castling_rights(position):
    '''
    True if {White, Black} has castling rights on the {kingside, queenside}.

    Number of features contributed: 4.
    '''
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

    # Warning: very ugly code.
    piece_on_board = (
        square != chess.MISSING_PIECE_SQUARE
        for square in position.piece_squares
    )

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


def _sliding_pieces_mobility(position):
    '''
    How far each {white, black} {bishop, rook, queen} can slide in each
    direction.
    '''
    # TODO: Refactor. Code taken from `_piece_lists()`.
    sliding_pieces = (
        'B', 'R', 'Q',
        'b', 'r', 'q'
    )
    sliding_piece_squares = { piece : [] for piece in sliding_pieces }

    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece is not None and piece.symbol() in sliding_pieces:
            sliding_piece_squares[piece.symbol()].append(square)
    print(sliding_piece_squares)

    # `chess.Board.pseudo_legal_moves` yields the moves only for the side to
    # play. Change the side to play and append the opposing sides' moves
    # to get all the pseudo-legal moves.
    all_pseudo_legal_moves = [move for move in position.pseudo_legal_moves]
    position.turn = not position.turn
    all_pseudo_legal_moves += [move for move in position.pseudo_legal_moves]
    position.turn = not position.turn

    # TODO: Optimize.
    return [
        [
            move.from_square in sliding_piece_squares[piece]
            for move in all_pseudo_legal_moves
        ]
        .count(True)
        for piece in sliding_pieces
    ]


def _attack_and_defend_maps(position):
    return [
        position.min_attacker_of[square][color]
        for color in (chess.WHITE, chess.BLACK)
        for square in chess.SQUARES
    ]

def _init_square_data(position):

    piece_squares = { piece : [] for piece in chess.PIECES }
    for square in np.random.permutation(chess.SQUARES):
        piece = position.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)

    piece_freqs = {
        'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
        'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
    }

    chess.MISSING_PIECE_SQUARE = -1
    for piece in chess.PIECES:
        piece_squares[piece] += (
            [chess.MISSING_PIECE_SQUARE]
            * (piece_freqs[piece] - len(piece_squares[piece]))
        )

    position.piece_squares = [
        square
        for piece in chess.PIECES
        for square in piece_squares[piece]
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
