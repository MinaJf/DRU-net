import os
import numpy as np

from PIL import Image
from utils import util as U

def save_str(data_dict, filename, idx):
    with open(filename, 'a+') as f:
        f.write('{}: {}\n'.format(idx, U.dict_to_str(data_dict)))

def save_img(data_dict, filepath, idx):
    if data_dict is None or not data_dict:
        return
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    for k in data_dict:
        imgs = data_dict[k]
        imgs = np.concatenate(imgs, 0)
        Image.fromarray(imgs).save('{}/idx_{}_{}.png'.format(filepath, idx, k))

    