import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import os

subjects = 15
classes = 3
version = 1

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class EmotionDataset(InMemoryDataset):
    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, edge_index=None, transform=None, pre_transform=None):
        self.stage = stage
        self.subjects = subjects
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        self.edge_index = edge_index
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['./V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, self.stage, self.subjects, self.sub_i)]

    def process(self):
        data_list = []
        num_samples = np.shape(self.Y)[0]
        for sample_id in tqdm(range(num_samples)):
            x = self.X[sample_id, :, :]
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(self.Y[sample_id, :])
            data = Data(x=x, y=y)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def normalize(data):
    mee = np.mean(data, 0)
    data = data - mee
    stdd = np.std(data, 0)
    data = data / (stdd + 1e-7)
    return data

def build_dataset(subjects):

    load_flag = True
    for sub_i in range(subjects):
        path = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, 'Train', subjects, sub_i)
        processed_dir = './processed'
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        if not os.path.exists(path):
            if load_flag:
                mov_coefs, labels = get_data()
                used_coefs = mov_coefs
                load_flag = False
            index_list = list(range(subjects))
            del index_list[sub_i]
            test_index = sub_i
            train_index = index_list
            X = used_coefs[train_index, :].reshape(-1, 62, 265 * 5)
            Y = labels[train_index, :].reshape(-1)
            testX = used_coefs[test_index, :].reshape(-1, 62, 265 * 5)
            testY = labels[test_index, :].reshape(-1)
            _, Y = np.unique(Y, return_inverse=True)
            Y = to_categorical(Y, classes)
            _, testY = np.unique(testY, return_inverse=True)
            testY = to_categorical(testY, classes)

            train_dataset = EmotionDataset('Train', './', subjects, sub_i, X, Y)
            test_dataset = EmotionDataset('Test', './', subjects, sub_i, testX, testY)


def get_dataset(subjects, sub_i):
    target_sub = sub_i
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)
    target_dataset = EmotionDataset('Test', './', subjects, target_sub)
    return train_dataset, target_dataset
