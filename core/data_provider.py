import os
import glob
import tempfile
import numpy as np

from utils import data_loader as L
from utils import util as U


class DataProvider:
    """Callable data provider.

    Parameter
    ----------
    data: str, list or dict
        str: Path to search the file.
        list: A list of all filenames.
        dict: A dict for all data.
    data_suffix: list or tuple
        a list of suffix, the first value should be the suffix of input images.
    processor: obj: Processor, optional
        A processor for precess images.
    is_pre_load: bool, optional
        True: load all the files into memory. False: load files into memory only when others call it.
    is_shuffle: bool, optional
        Random shuffle data or not after taking out all data each round.

    Attributes
    ----------
    _data_suffix: str
        Suffix for input images and their corresponding key in the output.
    _other_suffixes: list of tuple, optional
        Suffix for other images and their corresponding keys in the output.
    _is_shuffle: bool
        Random shuffle data or not after taking out all data each round.
    _processor: obj: Processor
        A processor for precess images.
    _file_list: list:str
        A list of all filenames.
    _all_data: dict
        Dict of ndarray that corresponds to the data when preloaded data is selected.
    _cur_i: int
        Current index for cycle.

    """

    def __init__(self,
                 data,
                 data_suffix,
                 processor=None,
                 is_save_temp=False,
                 is_pre_load=False,
                 is_shuffle=False):
        assert len(data_suffix) > 0, 'Empty suffix!'
        self._org_suffix = data_suffix[0]
        self._other_suffix = data_suffix[1:]
        self._is_shuffle = is_shuffle
        self._is_save_temp = is_save_temp

        self._processor = processor

        self._file_list = None
        self._all_data = None
        if type(data) is str:
            self._file_list = glob.glob(data)
        elif type(data) is list or type(data) is np.ndarray:
            self._file_list = data
        elif type(data) is dict:
            self._all_data = data
        else:
            raise ValueError('Only accept one of (search_path, file_list, data_dict).')

        self._cur_i = 0
        if self._file_list is not None:
            if is_pre_load:
                self._all_data = self._load_data(len(self._file_list))
            elif is_save_temp:
                self._temp_dir = self._build_temp_folder()

    def __call__(self, n):
        """Require images.

        Parameters
        ----------
        n: int
            The number of images required.

        Returns
        -------
        dict
            A dictionary of ndarray data:
                {
                    'data_suffix':      ndarray,
                    'other_suffix 1':   ndarray,
                    'other_suffix 1':   ndarray,
                    ...
                }
            The shape of ndarray will be (n, x, y, ..., c).
                n is the number of data, which caller asked.
                x, y, ... is the size of data.
                c is the number of channels (for label is the numebr of classes).
        """
        data_dict = {}
        if self._all_data is not None:
            idx_list = np.array(range(self._cur_i, self._cur_i + n)) % len(self._file_list)
            for key in self._all_data:
                data_dict.update({key: self._all_data[key][idx_list]})
                self._next_idx(n)
        elif self._is_save_temp:
            data_dict.update(self._load_temp_file(n))
        else:
            data_dict.update(self._load_data(n))
        return data_dict

    @property
    def size(self):
        return len(self._file_list) if self._file_list is not None else len(self._all_data)

    def _load_data(self, n):
        """Load and process data one by one

        Parameters
        ----------
        n: int
            The number of images loaded.

        Returns
        -------
        dict
            A dictionary of ndarray data:
                {
                    'data_suffix':      ndarray,
                    'other_suffix 1':   ndarray,
                    'other_suffix 1':   ndarray,
                    ...
                }
            The shape of ndarray will be (n, x, y, ..., c).
                n is the number of data, which caller asked.
                x, y, ... is the size of data.
                c is the number of channels (for label is the numebr of classes).
        """
        data_dict = {}
        for _ in range(n):
            sub_data_dict = {}
            x_name = self._file_list[self._cur_i]
            sub_data_dict.update({self._org_suffix: L.load_file(x_name)})

            for o_suffix in self._other_suffix:
                o_name = x_name.replace(self._org_suffix, o_suffix)
                sub_data_dict.update({o_suffix: L.load_file(o_name)})
            # process
            if self._processor is not None:
                sub_data_dict = self._processor.pre_process(sub_data_dict)

            data_dict = U.dict_append(data_dict, sub_data_dict)
            self._next_idx()
        U.dict_list2arr(data_dict)
        return data_dict

    def _build_temp_folder(self):
        print('Build temp folder...')
        temp_dir = tempfile.TemporaryDirectory()
        new_filelist = []
        for i in range(len(self._file_list)):
            data_dict = self._load_data(1)
            temp_filepath = '{}/temp_dict_{}.npy'.format(temp_dir.name, i)
            new_filelist.append(temp_filepath)
            np.save(temp_filepath, data_dict)
        self._file_list = new_filelist
        print('Processed temp files were saved to \'{}\''.format(temp_dir))
        return temp_dir

    def _load_temp_file(self, n):
        assert self._temp_dir is not None, 'Temp dir is None'
        assert os.path.exists(self._temp_dir.name), 'Can\'t find temp directory \'{}\''.foramt(self._temp_dir.name)
        data_dict = {}
        for _ in range(n):
            temp_filename = self._file_list[self._cur_i]
            sub_data_dict = np.load(temp_filename, allow_pickle='TRUE').item()
            self._next_idx()
            data_dict = U.dict_concat(data_dict, sub_data_dict)
        return data_dict

    def _next_idx(self, n=1):
        """Cycle index.
        Parameters
        ----------
        n: int, optional
            The value needs to be added。
        """
        self._cur_i += n
        if self._cur_i >= len(self._file_list):
            self._cur_i = self._cur_i % len(self._file_list)
            if self._is_shuffle:
                shuffle_idx = np.random.permutation(len(self._file_list))
                if self._file_list is not None:
                    self._file_list = [self._file_list[i] for i in shuffle_idx]
                if self._all_data is not None:
                    for key in self._all_data:
                        self._all_data[key] = self._all_data[key][shuffle_idx]