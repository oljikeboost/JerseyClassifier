import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import example_pb2


class Dataset(data.Dataset):
    def __init__(self, path_to_lmdb_dir, transform):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        with self._reader.begin() as txn:
            value = txn.get(self._keys[index])

        example = example_pb2.Example()
        example.ParseFromString(value)

        image = np.frombuffer(example.image, dtype=np.uint8)
        image = image.reshape([64, 64, 3])
        image = Image.fromarray(image)
        image = self._transform(image)

        length = example.length
        digits = example.digits

        return image, length, digits


class JerseyDataset(data.Dataset):
    def __init__(self, path_to_anno, transform, size=64):

        with open(path_to_anno) as f:
            self.data = f.readlines()

        self._length = len(self.data)
        self._transform = transform
        self.size = size

    def __len__(self):
        return self._length

    def __getitem__(self, index):

        img_path, label = self.data[index].strip().split(',')
        image = cv2.imread(img_path)

        image = cv2.resize(image, (self.size,self.size))
        image = Image.fromarray(image)
        image = self._transform(image)

        length = len(label)
        digits = [10, 10] # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label):
            digits[idx] = int(label[idx])

        return image, length, digits
