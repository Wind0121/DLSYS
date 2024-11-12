import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        dict_list = []
        data_path_list = []
        if train:
            for i in range(1, 6):
                data_path_list.append(f'data_batch_{i}')
        else:
            data_path_list.append(f'test_batch')
        
        for path in data_path_list:
            dict_list.append(unpickle(os.path.join(base_folder, path)))
        
        X_list = [dic[b'data'] for dic in dict_list]
        y_list = [dic[b'labels'] for dic in dict_list]

        X = np.concatenate(X_list, axis=0)
        X = X / 255
        X = X.reshape((-1, 3, 32, 32))
        y = np.concatenate(y_list, axis=None)

        self.X = X
        self.y = y
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if self.transforms:
            image = np.array([self.transformer(img) for img in self.X[index]])
        else:
            image = self.X[index]
        label = self.y[index]
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
