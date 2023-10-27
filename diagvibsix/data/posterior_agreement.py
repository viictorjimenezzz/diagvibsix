import torch
import csv
from typing import Optional

from .dataset.dataset import Dataset
from .dataset.paint_images import Painter
from .dataset.config import OBJECT_ATTRIBUTES
from .dataset.dataset_utils import get_mt_labels
from .wrappers import TorchDatasetWrapper, get_per_ch_mean_std

class EnvCSV(Dataset):
    """Subclass of DiagVib dataset to generate images from customized CSV specifications."""

    def __init__(self,
                mnist_preprocessed_path: str,
                csv_path: str,
                t: str = 'train'):
        
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
    def __init__(self,
                 mnist_preprocessed_path: str,
                 csv_path: str,
                 t: str = 'train',
                 seed: Optional[int] = 123,
                 normalization: Optional[str] = 'z-score', 
                 mean: Optional[float] = None, 
                 std: Optional[float] = None):
        
        self.dataset = EnvCSV(mnist_preprocessed_path, csv_path, t)
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