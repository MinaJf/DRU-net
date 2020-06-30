import cv2
import numpy as np
import scipy.ndimage as nd

#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import create_pairwise_gaussian, create_pairwise_bilateral


# Normalization---------------------------------------
def min_max(x):
    _x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return _x

def zero_mean(x):
    _x = (x - np.mean(x)) / np.std(x)
    return _x

def median_mean(x):
    _x = (x - np.median(x)) / np.std(x)
    return _x
# ----------------------------------------------------

# Label-----------------------------------------------
def one_hot(x, n_class):
    if type(n_class) is int:
        class_interval = (np.max(x) - np.min(x)) / (n_class - 1)
        x_ = np.zeros(list(x.shape) + [n_class])
        for i in range(n_class):
            cur_baseline = np.min(x) + class_interval * (i - 0.5)
            cur_topline = np.min(x) + class_interval * (i + 0.5)
            x_[..., i][np.all([x >= cur_baseline, x < cur_topline], 0)] = 1
    elif type(n_class) in [list, tuple, np.ndarray]:
        x_ = np.zeros(list(x.shape) + [len(n_class)])
        rx = np.round(x)
        for i in range(len(n_class)):
            x_[..., i][rx == n_class[i]] = 1
    else:
        raise Exception('Wrong type of n_class.')
    return x_

def channel_check(x, n_channel):
    _x = x
    if x.shape[-1] != n_channel:
        assert n_channel == 1 and x.shape[-1] != 3, 'Expect channel {}, actual shape {}.'.format(n_channel, x.shape)
        _x = np.reshape(x, list(x.shape) + [1])
    return _x

# ----------------------------------------------------

# Image-----------------------------------------------
def rgb2gray(x):
    assert x.shape[-1] == 3, 'Not RGB image!'
    _x = np.dot(x[...,:3], [0.299, 0.587, 0.114])
    return _x

def resize2d(x, new_size):
    if new_size is None:
        return x
    _x = cv2.resize(x, new_size[::-1])
    return _x

def resize3d(x, new_size):
    if new_size is None:
        return x
    sx = x.shape[0]
    sy = x.shape[1]
    sz = x.shape[2]
    zoom = [new_size[0] / sx, new_size[1] / sy, new_size[2] / sz]
    _x = nd.zoom(x, zoom=zoom)
    assert np.all(_x.shape[0:3] == np.array(new_size)), 'Fail to resize 3d image: expect {}, got {}.'.format(new_size, _x.shape[0:3])
    return _x

def dencecrf(x, y, nlabels, w1, w2, alpha, beta, gamma, iteration=5):
    
    _x = np.array(x)
    _y = np.array(y, np.float32)

    assert len(_x.shape) == len(_y.shape), 'Shape of x and y should be (..., nchannels), (..., nlabels)'
    assert _y.shape[-1] == nlabels, 'Expect y.shape (...,{}), got {}'.format(nlabels, _y.shape)

    _y = _y.reshape((-1, nlabels))
    _y = np.transpose(_y, (1,0))

    d = dcrf.DenseCRF(np.prod(_x.shape[0:-1]), nlabels)
    d.setUnaryEnergy(_y.copy(order='C'))

    imgdim = len(_x.shape)-1

    gamma = (gamma,) * imgdim
    alpha = (alpha,) * imgdim
    beta = (beta,) * _x.shape[-1]

    featG = create_pairwise_gaussian(gamma, _x.shape[0:-1]) # g
    featB = create_pairwise_bilateral(alpha, beta, _x, imgdim) #a b

    d.addPairwiseEnergy(featG, w2) # w2
    d.addPairwiseEnergy(featB, w1) # w1

    out = d.inference(iteration)
    out = np.transpose(out, (1,0))
    out = np.reshape(out, y.shape)

    return out

# ----------------------------------------------------


