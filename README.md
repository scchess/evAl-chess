# evAl-chess
Garry Kasparov held his own against Deep Blue in their '96 chess match. But while Kasparov looked at on the order of 3 positions per second, his computer opponent looked at millions; engines make up for a low-level evaluation function by simply looking at more positions. The idea behind this project is to create a slower but higher-level evaluation function. Of course what follows is a chess engine that analyzes few positions; something like a human, but that's stretching the point.

# Similar projects
This idea isn't new: NeuroChess and Giraffe are two examples of such engines. They both purposely trade off evaluation speed for accuracy and depth in understanding. Again stretching the point, we could consider humans to be the prime example of a slow-but-deep engine; even the very best humans examine very few positions but still play quite well. A computer engine that searches a similar number of positions per second as a human would be a challenge to create (toungue very much in cheek), so perhaps 1,000 positions per second is a good goal -- that is, while playing at, say, a 2200 FIDE level.

Ideally, our function would take as input the FEN reprentation of a board and output `1`, `0`, or `-1` -- a win, a draw, or a loss for white. Of course this is realistically impossible. Instead, we extract the low-level features of a given position and use them as the input, and we train our function to output the pawn evaluation. The neural net is our function of choice and for similar projects as well.

This, the problem of creating a strong engine that evaluates few positions, boils down to a surprisingly typical data-mining / machine-learning problem. We want to train a function that takes some information about a board and outputs an accurate pawn evaluation.

# The features (the input)
We currently use the same features Matthew Lai used in Giraffe as described in his paper 'Using Deep Reinforcement to Play Chess', available at http://arxiv.org/pdf/1509.01549v2.pdf. It's important to note that it's theoretically possible to take for input the minimum information needed to model a postition -- a FEN string representation, for instance. But that would require magnitudes more of time and computation to train. Using features that emulate the relavant information in a position such as those of Giraffe's is simply a practical and obvious choice.

The features we use are as follows (also listed in `extract_features.py`):
  1. The side to move.
  2. Castling rights for each color for each side
  3. The count of each type of piece.
  4. The location of each piece and value of the lowest-valued attacker
     and defender of the piece.
  5. The distance each sliding piece, {bishop, rook, queen}, can legally
     move in each direction.
  6. The value of the lowest-valued attacker and defender of each square.

# Extracting the features
`evAl-chess` uses `PyChess` to model a game and its positions. `PyChess` contains methods to make a move from a position and stores data about the position such as the castling rights, the locations of each piece, the side to move, etc. `extract_features.py` uses this functionality to get most of the features we need; the rest we find manually which in fact takes up most of the code.

# The data
Kaggle offers a database of 25,000 FIDE chess games, each analyzed by Stockfish for 1 second per move; this yields on the order of 2 million positions (assuming 40 moves -- 80 plies -- per game) and their 'ground-truth' evalutions. We use this data to train our model.

# Progress
After around 2 hours of training, our current model gives a squared-error of 0.446 -- that is, between our function's output and Stockfish's ground-truth. On positions where the ground-truth evaluation is between 0.1 and 1, the squared-error is even lower at 0.167.

The problem is that we don't know whether this is accurate enough. Using `PyChess` to extract the features is slow, so we can't use our evaluation function in an engine to test its performance. The next step in this project is to compare our function's performance with that of Stockfish's by evaluating the same positions in the dataset using Stockfish's function. This is annoying to do, because Stockfish's code is so technical. (See https://github.com/mcostalba/Stockfish.)

If our function performs worse than or at a similar level to Stockfish's function, we need to get more data and/or train further. Otherwise if it performs better, we're done, and all we need to do is port `extract_features.py` to C++ to improve its performance so it can be used in an engine. (That's the goal, after all.)
