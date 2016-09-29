'''
Compares the performance of our evaluation function against
Stockfish's.

The Stockfish in `evAl-chess` contains an extra method which returns
the static evaluation for a position. This is something that isn't
available through the UCI protocol. (Trust me; I tried.)
'''

import subprocess
import chess

p = subprocess.Popen(
    ['/Users/colinni/evAl-chess/Stockfish_modified/src/stockfish'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)
p.stdout.readline() # Prints the authors when first started.

def _stockfish_static_eval (position):
    '''
    Stockfish's static evaluation of a position.

    Assumes that the Stockfish subprocess has already been opened;
    communicates with it through stdin and stdout.
    '''
    p.stdin.write(('position fen ' + position.fen() + '\n').encode('utf-8'))
    p.stdin.flush()
    p.stdin.write(b'eval\n')
    p.stdin.flush()
    _eval = p.stdout.readline()
    p.stdout.readline()
    return int(_eval.decode())
