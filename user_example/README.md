# Usage

This is a simple example of how to train and then test the network.  This uses the 2D network and trains it for the 24 direction case.  Training should take about a half hour or so using a single P40 card.

```
python train.py
python test_fig.py
```

And that should give you a figure with the reference standard (top row), the predicted images (middle row), and difference images (bottom row).