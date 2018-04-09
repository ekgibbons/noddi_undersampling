import sys

sys.path.append("/home/mirl/egibbons/noddi")

from model_2d import train as train_2d

directions = [128, 64, 32, 16]

for n_directions in directions:
    train_2d.train(n_directions)
