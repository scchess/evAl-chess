import extract_features
import chess.pgn


# (I thank you with all my heart, python-chess documentation.)
tcec_2015_archive = open(
    '/Users/colinni/evAl-chess/tcec_2015_archive.pgn',
    encoding='utf-8-sig',
    errors='surrogateescape'
)

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
        engine_eval = extract_features.get_engine_eval(curr_game_node)
        print(engine_name, engine_eval)
        print(curr_game_node.board(), end='\n\n')

        print(extract_features.get_position_features(curr_game_node.board()))

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
