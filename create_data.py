import extract_features
import chess.pgn
import re


# (I thank you with all my heart, python-chess documentation.)
tcec_2015_archive = open(
    '/Users/colinni/evAl-chess/tcec_2015_archive.pgn',
    encoding='utf-8-sig',
    errors='surrogateescape'
)


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


# Iterate through every game in the archive.
curr_game = chess.pgn.read_game(tcec_2015_archive)

# chess.pgn.read_game returns None when it reaches the EOF.
while curr_game is not None:

    # Iterate through every move played.
    curr_game_node = curr_game.root()
    while curr_game_node is not None:

        # The name of the engine that played the move leading to this node.
        engine_name = (
            curr_game.headers['White']
            if not curr_game_node.board().turn
            else curr_game.headers['Black']
        )
        # What that engine evaluated this position to be.
        engine_eval = get_engine_eval(curr_game_node)

        print(curr_game_node.board())
        print(engine_name, engine_eval)
        print(extract_features.get_features(curr_game_node.board()))
        print()

        # Set curr_game_node to the next position in the game. If it's the
        # end of the game, set it to None.
        curr_game_node = (
            curr_game_node.variation(0)
            if not curr_game_node.is_end()
            else None
        )

    # Get the next game in the pgn file.
    curr_game = chess.pgn.read_game(tcec_2015_archive)
    break
