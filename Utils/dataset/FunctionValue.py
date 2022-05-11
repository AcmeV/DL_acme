import csv
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FunctionValue(Dataset):

    def __init__(self, dir, time_step=10, transform=None, train=True):

        df = pd.read_csv(dir)
        self.total_len = df.shape[0]
        self.time_step = time_step
        X = []
        Y = []
        for i in range(df.shape[0] - self.time_step):
            X.append(np.array(df.iloc[i:(i + self.time_step), ].values, dtype=np.float32))
            Y.append(np.array(df.iloc[(i + self.time_step), 0], dtype=np.float32))
        if train:
            self.X, self.Y = X[:int(0.8 * self.total_len)], Y[:int(0.8 * self.total_len)]
        else:
            self.X, self.Y = X[int(0.8 * self.total_len):], Y[int(0.8 * self.total_len):]
        self.tranform = transform

    def __getitem__(self, index):
        x = self.X[index]
        y = float(self.Y[index])
        if self.tranform != None:
            x, y = self.tranform(x).data.squeeze(0), \
                   torch.tensor(y).unsqueeze(0).type(torch.float32)
        return x, y

    def __len__(self):
        return len(self.X)

def gen_function_value(dir):
    file = open(dir, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Values'])
    file.flush()
    start, end = 0 * np.pi, 100 * np.pi
    xs = np.linspace(start, end, 10000, dtype=np.float32)
    for x in xs:
        y = np.sin(x)
        # y = pow(np.sin(3 * x), 2) - pow(np.cos(2 * x), 3) + np.cos(0.5 * x) - np.sin(2 * x)
        # y = pow(np.sin(3 * x), 2) - pow(np.cos(2 * x), 3) + np.cos(0.5 * x) - np.sin(2 * x) + random.uniform(-0.1, 0.1)
        # if y > 1.0 or y < -1.0:
        #     y = random.uniform(-0.9, 0.9)
        writer.writerow([y])
        file.flush()
    file.close()