import chess
import re

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
    Returns True if it's White to move.
    '''
    return [position.turn]


def _castling_rights(position):
    '''
    Returns True if {White, Black} has castling rights on the
    {kingside, queenside}.
    '''
    return [
        position.has_kingside_castling_rights(chess.WHITE),
        position.has_kingside_castling_rights(chess.BLACK),
        position.has_queenside_castling_rights(chess.WHITE),
        position.has_queenside_castling_rights(chess.BLACK)
    ]


def _material_configuration(position):
    '''

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
    Returns the row-major coordinate of square. Used by `_piece_lists()`.
    '''
    return (square // 8, square % 8)


def _get_min_attacker(position, color, square):
    '''
    Returns the value of the minimum-valued attacker of `square`; returns 0
    if `square` is not attacked by any piece.
    '''
    return min(
        (
            position.piece_type_at(square)
            for square in position.attackers(color, square)
        ),
        default=0
    )

def _piece_lists(position):
    # There are things in life that you should just take for granted. Do
    # yourself a favor and save me the shame and skip over this method.
    piece_freqs = {
        'P' : 8, 'N' : 3, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
        'p' : 8, 'n' : 3, 'b' : 2 ,'r' : 2, 'q' : 2, 'k' : 1
    }

    piece_coords = { piece : [] for piece in piece_freqs }
    for square in chess.SQUARES_180:
        piece = position.piece_at(square)
        if piece is not None:
            min_attacker_and_defender = tuple(
                _get_min_attacker(position, color, square)
                for color in (not piece.color, piece.color)
            )
            piece_coords[piece.symbol()].append(_to_coord(square) + min_attacker_and_defender)

    return [
        element
        for piece in piece_freqs
        for coord in piece_coords[piece] + [(-1, -1)] * (piece_freqs[piece] - len(piece_coords[piece]))
        for element in coord + (coord != (-1, -1),)
    ]


def _sliding_pieces_mobility(position):
    return [[
        any(
            position.piece_type_at(move.from_square) == piece
            for piece in (chess.QUEEN, chess.BISHOP, chess.ROOK)
        )
        for move in position.pseudo_legal_moves
    ].count(True)]


def _attack_and_defend_maps(position):
    return [
        _get_min_attacker(position, color, square)
        for color in (chess.WHITE, chess.BLACK)
        for square in chess.SQUARES_180
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
