import os

import numpy as np
import nibabel as nib

from PIL import Image
    
    
def load_file(path, dtype=np.float32):
    # Since there are some errors for catching exception, now just use if
    path = path.strip()
    assert os.path.isfile(path), 'File {} not found!'.format(path)
    suffix = path.split('.')[-1]
    # Nifty load
    if suffix in ['gz', 'nii']:
        data = nib.load(path)
        return np.array(data.dataobj).astype(dtype)

    # text to ndarray
    if suffix == 'txt':
        data = np.genfromtxt(path, dtype)
        return data
        
    # PIL load
    try:
        data = Image.open(path)
    except:
        pass
    else:
        return np.array(data, dtype)
    

    raise IOError('Invalid data type: {}'%path)