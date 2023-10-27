import torch
import csv
import os
import pickle
from typing import Optional

from numpy.random import seed as set_seed

from .dataset.dataset import Dataset
from .dataset.paint_images import Painter
from .dataset.config import OBJECT_ATTRIBUTES
from .dataset.dataset_utils import get_mt_labels
from .wrappers import TorchDatasetWrapper, get_per_ch_mean_std

__all__ = ['EnvCSV', 'TorchDatasetCSV']

def check_paths(csv_path, cache_path):
    """Check if the provided paths for the cache and CSV files are valid."""

    if (csv_path is None) and (cache_path is None): # no path is provided
        raise ValueError('Either csv_path or cache_path must be specified.')
        
    elif (csv_path is not None) and (cache_path is not None): # both paths are provieded
        if not os.path.exists(cache_path): # cache file not found
            if not os.path.exists(csv_path): # csv file not found
                raise ValueError('Cache file not found and no csv file was found.')
            else:
                print("Cache file not found. Generating images from the provided CSV.")
        
    elif cache_path is not None:  # only cache path is provided
        if not os.path.exists(cache_path):
            raise ValueError('Cache file not found.')
            
    elif csv_path is not None:  # only csv path is provided
        if not os.path.exists(csv_path):
            raise ValueError('CSV file not found.')


class EnvCSV(Dataset):
    """Subclass of DiagVib dataset to generate images from customized CSV specifications.
    
    Args:
        mnist_preprocessed_path (str): Path to the processed MNIST dataset. If there is no such dataset, you can generate it by calling process_mnist.get_processed_mnist(mnist_dir).
        csv_path (str): Path to the CSV file containing the dataset specifications.
        t (Optional[str]): Type of dataset to be generated, corresponding to the key 'category' (e.g. 'train').
        seed (Optional[int]): Random seed for the dataset generation.
    
    """

    def __init__(self,
                mnist_preprocessed_path: str,
                csv_path: str,
                t: str = 'train',
                seed: Optional[int] = 123):
        
        set_seed(seed) # numpy bc thats how files are generated
        
        self.painter = Painter(mnist_preprocessed_path)

        # maybe I want to append smth to this list
        self.OBJECT_ATTRIBUTES_CSV = OBJECT_ATTRIBUTES
        self.OBJECT_ATTRIBUTES_CSV['environment'] = ['first', 'second'] # decide this later
        self.FACTORS_CSV = list(self.OBJECT_ATTRIBUTES_CSV.keys())
        self.length_factors = len(self.FACTORS_CSV)

        # Needed to avoid overriding methods
        self.spec = {} 
        self.task = 'tag' # so that the target is the environment
        self.spec['shape'] = [1, 128, 128] # MNIST expected shape

        self.images = []
        self.env = []
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader) # skip column names
            for row in reader:
                mode_spec = {}
                obj_spec = {}
                obj_spec['category'] = t
                for i in range(self.length_factors-1): # -1 because the last one is the environment
                    obj_spec[self.FACTORS_CSV[i]] = [self.OBJECT_ATTRIBUTES_CSV[self.FACTORS_CSV[i]][int(row[i])]] # get factor value from index
                mode_spec['tag'] = str(self.OBJECT_ATTRIBUTES_CSV[self.FACTORS_CSV[-1]][int(row[-1])]) # environment the last one
                mode_spec['objs'] = [obj_spec]

                image_specs, images, env_label = self.draw_mode(mode_spec, 1) # 1 image per mode
                self.images += images
                self.env += env_label

        self.permutation = list(range(len(self.images))) # Needed to avoid overriding methods

    def getitem(self, idx):
        return {
            'image': self.images[idx],
            'env': self.env[idx]
        }
    

class TorchDatasetCSV(TorchDatasetWrapper):
    """Wrapper class for the DiagVib dataset to generate images from customized CSV specifications.
    
    Args:
        mnist_preprocessed_path (str): Path to the processed MNIST dataset. If there is no such dataset, you can generate it by calling process_mnist.get_processed_mnist(mnist_dir).
        csv_path (Optional[str]): Path to the CSV file containing the dataset specifications, if no cache file is specified or found.
        cache_path (Optional[str]): Path to the cache file containing the dataset. If the cache file does not exist, the dataset will be generated from the CSV file and then stored in the specified cache path.
        t (Optional[str]): Type of dataset to be generated, corresponding to the key 'category' (e.g. 'train').
        seed (Optional[int]): Random seed for the dataset generation.
        normalization (Optional[str]): Normalization type. Defaults to 'z-score'.
        mean (Optional[float]): Mean value for the normalization.
        std (Optional[float]): Standard deviation value for the normalization.
    
    """
    def __init__(self,
                 mnist_preprocessed_path: str,
                 csv_path: Optional[str] = None,
                 cache_path: Optional[str] = None,
                 t: str = 'train',
                 seed: Optional[int] = 123,
                 normalization: Optional[str] = 'z-score', 
                 mean: Optional[float] = None, 
                 std: Optional[float] = None):
        
        check_paths(csv_path, cache_path) # check if the provided paths are valid
        
        # Cache path is prioritized over csv_path.
        # Load dataset object (uint8 images) from cache if available
        if (cache_path is not None) and (os.path.exists(cache_path)):
            with open(cache_path, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = EnvCSV(mnist_preprocessed_path, csv_path, t, seed)
            if cache_path is not None: # we want to store it as cache
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)
        
        self.normalization = normalization

        self.mean, self.std = mean, std
        self.min = 0.
        self.max = 255.
        if self.normalization == 'z-score' and (self.mean is None or self.std is None):
            self.mean, self.std = get_per_ch_mean_std(self.dataset.images)

    def __getitem__(self, item):
        sample = self.dataset.getitem(item)
        image, env = sample.values()
        image = self._normalize(self._to_T(image, torch.float))
        target = torch.tensor(get_mt_labels(('environment', env), OBJECT_ATTRIBUTES=self.dataset.OBJECT_ATTRIBUTES_CSV))
        return {'image': image, 'target': target}