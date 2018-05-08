from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from skimage import measure


def signed_rank(data1, data2, debug=False):

    diffs = data1 - data2
    diffs_abs = abs(diffs)
    diffs_sign = np.sign(diffs)

    ranks = stats.rankdata(diffs_abs)

    W = np.sum(ranks*diffs_sign)

    print("statistic: %i" % W)

    N_r = data1.shape[0]
    sigma = np.sqrt((N_r*(N_r + 1)*(2*N_r + 1))/6)

    if debug == True:
        for ii in range(N_r):
            print("data1[%i] = %f, data2[%i] = %f, diff: %f, signed rank: %i"
                  % (ii, data1[ii], ii, data2[ii],
                     diffs_abs[ii], diffs_sign[ii]*ranks[ii]))
    z = W/sigma
    p_value = stats.norm.sf(abs(z))*2
            
    return W, sigma, W/sigma, p_value

    
def volume_all(data, truth):

    data = data.astype(float)
    truth = truth.astype(float)

    nrmse = volume_nrmse(data,truth)
    ssim = volume_ssim(data,truth)
    psnr = volume_psnr(data,truth)
    
    return [ssim, psnr, nrmse]

def volume_ssim(data, truth):

    values = np.zeros(data.shape[2])

    jj = 0
    for ii in range(data.shape[2]):
        # if np.sum(data[:,:,ii] - truth[:,:,ii]) == 0:
        #     jj += 1
        #     continue

        values[ii] = measure.compare_ssim(data[:,:,ii],
                                          truth[:,:,ii])

    # return np.mean(values[:-1-jj]), np.std(values[:-1-jj])
    return values[:-1-jj]

def volume_psnr(data, truth):

    values = np.zeros(data.shape[2])

    jj = 0
    for ii in range(data.shape[2]):
        # if np.sum(data[:,:,ii] - truth[:,:,ii]) == 0:
        #     jj += 1
        #     continue

        values[ii] = measure.compare_psnr(data[:,:,ii],
                                          truth[:,:,ii],
                                          data_range=1.)

    # return np.mean(values[:-1-jj]), np.std(values[:-1-jj])
    return values[:-1-jj]


def volume_mse(data, truth):

    values = np.zeros(data.shape[2])

    jj = 0
    for ii in range(data.shape[2]):
        # if np.sum(data[:,:,ii] - truth[:,:,ii]) == 0:
        #     jj += 1
        #     continue

        values[ii] = measure.compare_mse(data[:,:,ii],
                                         truth[:,:,ii])

    # return np.mean(values[:-1-jj]), np.std(values[:-1-jj])
    return values[:-1-jj]


def volume_nrmse(data, truth):

    values = np.zeros(data.shape[2])

    jj = 0
    for ii in range(data.shape[2]):
        # if np.sum(data[:,:,ii] - truth[:,:,ii]) == 0:
        #     jj += 1
        #     continue
 
        values[ii] = measure.compare_nrmse(data[:,:,ii],
                                           truth[:,:,ii])

    # return np.mean(values[:-1-jj]), np.std(values[:-1-jj])
    return values[:-1-jj]

def main():
    data1 = np.random.randn(10)
    data2 = np.random.randn(10)

    print(signed_rank(data1,data2))

if __name__ == "__main__":
    main()
