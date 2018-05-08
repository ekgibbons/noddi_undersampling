import sys

sys.path.append("/home/mirl/egibbons/noddi")

from golkov_multi import train as train_gm

directions = [128, 64, 32, 24, 16, 8]

for n_directions in directions:
    train_gm.train(n_directions)
