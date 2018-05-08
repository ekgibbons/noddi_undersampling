import sys

sys.path.append("/home/mirl/egibbons/noddi")

from gfa_2d import train as train_gfa

directions = [128, 64, 32, 24, 16, 8]

for n_directions in directions:
    train_gfa.train(n_directions)
