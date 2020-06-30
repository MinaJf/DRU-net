from abc import ABCMeta,abstractmethod
from utils import process_methods as M

class Processor(metaclass=ABCMeta):
    """Interface for Processor
    """

    @abstractmethod
    def pre_process(self, data_dict):
        """Abstract method, need to be inherited for pre-processing.
        Parameters
        ----------
        data_dict: dict
            Dictionary for all of the input data.

        Returns
        -------
        dict
            Dictionary for all the data after processing.
            Size won't change if no resize operation.
        """
        
    
    @abstractmethod
    def post_process(self, data_dict):
        """Abstract method, need to be inherited for post-processing.
        Parameters
        ----------
        data_dict: dict
            Dictionary for all of the input data.

        Returns
        -------
        dict
            Dictionary for all the data after processing.
            Size won't change if no resize operation.
        """


class SimpleImageProcessor(Processor):
    """Class for process images.

    Parameters
    ----------
    pre: {key: [m1,m2,...]}: a dict of list of None or str or tuple, optional
        Use different processing methods for different data in data_dict according to the key.
        For each set of data, process the images in order of the methods in corresponding [mn] list.
        The available methods are as follows:
            None: no process.
            'min-max': min-max normalization, x = (x - min(x)) / (max(x) - min(x)).
            'zero-mean': zero-mean normalization, x = (x - mean(x)) / std(x).
            'mediam-mean': mediam-mean normalization, x = (x - mediam(x)) / std(x).
            'rgb2gray': Convert RGB image to Gray image.
            ('one_hot', n_class): One hot for labels, output size is [..., n_class].
            (resize2d, (x,y,c)): Resize image to (x,y,...).
            ('channelcheck', int): Check channel.
            (custom_method, **kwargs): This class allow user use custom method. User can 
                                       put method and its arguments in a list. The processor 
                                       will process the data according to this method.

    Attributes
    ----------
    _pre: {key: [m1,m2,...]}: a dict of list of None or str or tuple, optional
        Use different processing methods for different data in data_dict according to the key.
        For each set of data, process the images in order of the methods in corresponding [mn] list.
        The available methods are as follows:
            None: no process.
            'min-max': min-max normalization, x = (x - min(x)) / (max(x) - min(x)).
            'zero-mean': zero-mean normalization, x = (x - mean(x)) / std(x).
            'mediam-mean': mediam-mean normalization, x = (x - mediam(x)) / std(x).
            'rgb2gray': Convert RGB image to Gray image.
            ('one_hot', n_class): One hot for labels, output size is [..., n_class].
            ('resize2d', (x,y,c)): Resize image to (x,y,...).
            ('channelcheck', int): Check channel.
            (custom_method, **kwargs): This class allow user use custom method. User can 
                                       put method and its arguments in a list. The processor 
                                       will process the data according to this method.
    
    _post: TODO

    _mdict: dict
            A dictionary for methods. Key are names of method, values are the corresponding functions. 

    
    """
    def __init__(self, pre=None, post=None):
        self._pre = pre
        self._post = post
        self._mdict = {
                    'min-max': M.min_max,
                    'zero-mean': M.zero_mean,
                    'median-mean': M.median_mean,
                    'one-hot': M.one_hot,
                    'rgb2gray': M.rgb2gray,
                    'resize2d': M.resize2d,
                    'resize3d': M.resize3d,
                    'channelcheck': M.channel_check
                    #TODO new method add here
        }


    def pre_process(self, data_dict):
        """A basic pre processing pipline. TODO: Modify as required

        Parameters
        ----------
        data_dict: dict
            Dictionary for all of the input data.

        Returns
        -------
        dict
            Dictionary for all the data after processing.
            Size won't change if no resize operation.

        """
        if self._pre is None:
            return data_dict

        for key in self._pre:
            for augm in self._pre[key]:
                if key in data_dict:
                    item = data_dict[key]
                    new_item = self._process(item, augm)
                    data_dict.update({key: new_item})
        
        return data_dict

    def post_process(self, data_dict):
        """Post processing method
        TODO
        """
        if self._pre is None:
            return data_dict
        return data_dict

    def _process(self, data, augm):
        """Process data according to accepted augments.

        Parameters
        ----------
            data: ndarray
                Data need to be processed.
            augm: str or dict
                Method name or a list of method name and required augments (e.g. ['resize2d', (512,512)]).
                Custom method is allowed here. If it is, the augm should like [function, {'arg1': arg1, 'arg2': arg2, ...}]

        """
        if type(augm) is str:
            assert augm in self._mdict, 'Method "{}" not found!'.format(augm)
            return self._mdict[augm](data)
        if type(augm) is tuple:
            # custom process method
            if callable(augm[0]):
                return augm[0](data, **augm[1])
            assert augm[0] in self._mdict, 'Method "{}" not found!'.format(augm[0])
            return self._mdict[augm[0]](data, augm[1])
