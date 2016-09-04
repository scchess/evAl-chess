import chess
import re

def get_engine_eval(game_node):
    '''
    Returns what the engine evaluated the position in `game_node` to be; None
    if the move played was taken from the book.

    The returned evaluation is by the engine that played the move that lead
    to `game_node`; the TCEC updates an engine's eval only after moves
    it plays -- every other move played.
    '''

    # TCEC's pgn files include engine information -- such as time remaining
    # and depth searched -- for each move played. `python-chess` considers
    # this information to be a `comment` of each move.
    comment = game_node.comment

    # Search for floats in `comment`. No other piece of information
    # comes in the form of a float, so this yields the evaluation.
    engine_eval = re.findall('\d+\.\d+', comment)
    if len(engine_eval) == 0:
        # This move wasn't calculated because it was taken from the book.
        return None
    else:
        assert len(engine_eval) == 1, 'Using `re.findall` won\'t work.'
        return engine_eval[0]


def _side_to_move(position):
    return [position.turn]


def _castling_rights(position):
    return [
        position.has_kingside_castling_rights(chess.WHITE),
        position.has_kingside_castling_rights(chess.BLACK),
        position.has_queenside_castling_rights(chess.WHITE),
        position.has_queenside_castling_rights(chess.BLACK)
    ]


def _material_configuration(position):
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


def _piece_lists(position):
    # There are things in life that you should just take for granted. Do
    # yourself a favor and save me the shame and skip over this method.
    pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b' ,'r', 'q', 'k']
    piece_freqs = {
        'P' : 8, 'N' : 2, 'B' : 2, 'R' : 2, 'Q' : 2, 'K' : 1,
        'p' : 8, 'n' : 2, 'b' : 2 ,'r' : 2, 'q' : 2, 'k' : 1
    }
    piece_coords = { piece : [] for piece in pieces }
    for i in range(8):
        for j in range(8):
            piece = position.piece_at(64 - 8 * (i + 1) + j)

            if piece is not None:
                piece_color = piece.color
                min_attacker_and_defender = tuple([
                    min(
                        (
                            position.piece_type_at(square)
                            for square in position.attackers(
                                color,
                                64 - 8 * (i + 1) + j
                            )
                        ),
                        default=0
                    )
                    for color in ((chess.BLACK, chess.WHITE) if piece_color == chess.WHITE else (chess.WHITE, chess.BLACK))
                ])
                piece_coords[piece.symbol()].append((i, j) + min_attacker_and_defender)
    return [
        element
        for piece in pieces
        for coord in piece_coords[piece] + [(-1, -1)] * (piece_freqs[piece] - len(piece_coords[piece]))
        for element in coord + (coord != (-1, -1),)
    ]


def _sliding_pieces_mobility(position):
    return []


def _attack_and_defend_maps(position):
    return [
        min(
            (
                position.piece_type_at(square)
                for square in position.attackers(
                    color,
                    64 - 8 * (i + 1) + j
                )
            ),
            default=0
        )
        for color in (chess.WHITE, chess.BLACK)
        for i in range(8)
        for j in range(8)
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
