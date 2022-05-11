import os
import zipfile
from subprocess import call

import numpy as np
import torch
from torch.utils.data import Dataset
import requests
from tqdm import tqdm


class HAR(Dataset):

    def __init__(self, dir, transform=None, train=True):
        self.root = dir
        self.root_dir = f'{dir}/HAR'
        dir = f'{dir}/HAR'

        if not os.path.exists(dir):
            self._download()
            self._zip()

        INPUT_SIGNAL_TYPES = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        if train:
            X_signals_paths = [
                f'{dir}/train/Inertial Signals/{signal}train.txt' for signal in INPUT_SIGNAL_TYPES
            ]
            label_path = f'{dir}/train/y_train.txt'
        else:
            X_signals_paths = [
                f'{dir}/test/Inertial Signals/{signal}test.txt' for signal in INPUT_SIGNAL_TYPES
            ]
            label_path = f'{dir}/test/y_test.txt'
        # inputs
        self.X_array = self.load_X(X_signals_paths)
        # targets
        self.label_array = self.load_label(label_path)

        self.tranform = transform

    def __getitem__(self, index):
        input = self.X_array[index]
        target = self.label_array[index]

        if self.tranform != None:
            input, target = self.tranform(input).data.squeeze(0), torch.tensor(target[0]).type(torch.long)

        return input, target

    def __len__(self):
        return len(self.X_array)

    def load_X(self, X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append([np.array(serie, dtype=np.float32) for serie in
                 [row.replace('  ', ' ').strip().split(' ') for row in file]])
            file.close()
        X_signals_array = np.array(X_signals)
        return np.transpose(X_signals_array, (1, 2, 0))

    # Load "y" (the neural network's training and testing outputs)
    def load_label(self, label_path):
        file = open(label_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()
        # Substract 1 to each output class for friendly 0-based indexing
        y_ = y_ - 1
        return y_

    def _download(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip'
        name = url.split("/")[-1]
        resp = requests.get(url, stream=True)
        content_size = int(resp.headers['Content-Length']) / 1024  # get file size
        # download to dist_dir
        path = f'{self.root}/{name}'
        with open(path, "wb") as file:
            for data in tqdm(iterable=resp.iter_content(1024), total=int(content_size), unit='kb', desc='downloading...'):
                file.write(data)
        print("finish download UCI HAR Dataset.zip\n\n")

    def _zip(self):
        zFile = zipfile.ZipFile(f'{self.root}/UCI HAR Dataset.zip', "r")
        for fileM in tqdm(zFile.namelist(), unit='file', desc='ziping...'):
            zFile.extract(fileM, path=f'{self.root}')
        zFile.close()
        os.rename(f'{self.root}/UCI HAR Dataset', f'{self.root}/HAR')
        print("finish zip HAR.zip\n\n")
        os.remove(f'{self.root}/UCI HAR Dataset.zip')