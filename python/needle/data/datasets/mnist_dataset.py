from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as image_file:
        magic, n, row, col = struct.unpack(">4i", image_file.read(16))
        assert(magic == 2051)
        size = row * col
        X = np.vstack([np.array(struct.unpack(f"{size}B", image_file.read(size)), dtype=np.float32) for _ in range(n)])
        X -= np.min(X)
        X /= np.max(X)

    with gzip.open(label_filename, "rb") as label_file:
        magic, n = struct.unpack(">2i", label_file.read(8))
        assert(magic == 2049)
        Y = np.array(struct.unpack(f"{n}B", label_file.read(n)), dtype=np.uint8)
    
    return (X, Y)
    ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)).reshape((28 * 28, )) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape((28, 28, 1))).reshape((28 * 28, ))
        return imgs, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION