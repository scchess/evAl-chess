import chess
import re
import numpy as np

def get_engine_eval(game_node):
    '''
    Returns what the engine evaluated the position in `game_node` to be;
    returns None if the move played was taken from the book.

    The returned evaluation is bh the engine that played the move that
    immediately led to `game_node`; the TCEC updates the engines' evaluations
    only after moves they play.
    '''

    # TCEC's pgn files log engine information such as time remaining
    # and depth searched for each move played. `python-chess`, when parsing
    # the pgn files, considers this information to be the `comment` of each
    # move.
    comment = game_node.comment

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
        for piece in [
            'P', 'N', 'B', 'R', 'Q', 'K',
            'p', 'n', 'b' ,'r', 'q', 'k'
        ]
    ]

def _to_coord(square):
    '''
    The row-major coordinate of square. Used by `_piece_lists()`.
    '''
    return (8 - square // 8 - 1, square % 8)


def _get_min_attacker(position, color, square):
    '''
    The value of the minimum-valued attacker of `square`; 0 if `square` is
    not attacked by any piece.
    '''
    return min(
        (
            position.piece_type_at(square)
            for square in position.attackers(color, square)
        ),
        default=0
    )

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

    pieces = [
        'P', 'N', 'B', 'R', 'Q', 'K',
        'p', 'n', 'b' ,'r', 'q', 'k'
    ]

    piece_freqs = {
        'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
        'p' : 8, 'n' : 3, 'b' : 2, 'r' : 2, 'q' : 2, 'k' : 1
    }

    piece_squares = { piece : [] for piece in pieces }
    piece_min_attacker_and_defender = {}

    SQUARES_COL_ORDERED = (
        np.reshape(chess.SQUARES, (8, 8))
        .transpose()
        .flatten()
        .tolist()
    )
    for square in SQUARES_COL_ORDERED:
        piece = position.piece_at(square)
        if piece is not None:
            piece_squares[piece.symbol()].append(square)
            piece_min_attacker_and_defender[square] = tuple(
                _get_min_attacker(position, color, square)
                for color in (not piece.color, piece.color)
            )

    square_of_missing_piece = (-1, -1)
    for piece in pieces:
        piece_squares[piece] += (
            [square_of_missing_piece]
            * (piece_freqs[piece] - len(piece_squares[piece]))
        )

    piece_on_board = (
        square != square_of_missing_piece
        for piece in pieces
        for square in piece_squares[piece]
    )

    return [
        element
        for piece in pieces
        for square in piece_squares[piece]
        for element in (
            (-1, -1, next(piece_on_board), -1, 1)
            if square == square_of_missing_piece
            else (
                _to_coord(square)
                + (next(piece_on_board), )
                + piece_min_attacker_and_defender[square]
            )
        )
    ]


def _sliding_pieces_mobility(position):
    '''
    How far each {white, black} {bishop, rook, queen} can slide in each
    direction.
    '''
    # TODO: Refactor. Code taken from `_piece_lists()`.
    sliding_pieces = ('B', 'R', 'Q', 'b', 'r', 'q')
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
        _get_min_attacker(position, color, square)
        for color in (chess.WHITE, chess.BLACK)
        for square in chess.SQUARES
    ]

def get_position_features(position):
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
