import numpy as np

def get_delta_histogram(predicted, truth, bins=100):

    delta = predicted - truth

    delta_bins, delta_hist_edges = np.histogram(delta_gfa,
                                                bins=bins)

    bin_locations = np.diff(delta_hist_edges)
    
    return delta_bins, bin_locations, delta
