import sys

sys.path.append("/home/mirl/egibbons/noddi")

from sampling_test import train as train_sample

seeds = [100, 225, 300, 325, 400, 425, 500, 525, 600]

for random_seed in seeds:
    train_sample.train(random_seed)
