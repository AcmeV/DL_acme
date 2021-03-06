import os
import sys
import shutil
import zipfile

import requests
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.root = root
        self.root_dir = f'{root}/TinyImageNet'
        self.transform = transform
        if not os.path.exists(f'{self.root_dir}'):
            self._download()
            self._zip()
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.val_dir = os.path.join(self.root_dir, 'val')

        if (self.train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(train=self.train)

        words_file = os.path.join(self.root_dir, 'words.txt')
        wnids_file = os.path.join(self.root_dir, 'wnids.txt')

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]


    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir ,d))]
        classes = sorted(classes)

        n_image = 0
        for root, dirs, files in os.walk(self.train_dir):
            for file in files:
                if file.endswith('.JPEG'):
                    n_image += 1
        self.len_dataset = n_image

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_img_dir = os.path.join(self.val_dir, 'images')

        if sys.version_info >= (3, 5):
            imgs = [d.name for d in os.scandir(val_img_dir) if d.is_dir()]
        else:
            imgs = [d for d in os.listdir(self.val_img_dir) if os.path.isdir(os.path.join(self.train_dir, d))]

        val_annotations_file_path = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()

        with open(val_annotations_file_path) as anno_file:
            lines = anno_file.readlines()
            for line in lines:
                words = line.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])
        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))

        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, train=True):
        self.imgs = []
        if train:
            img_root_dir = self.train_dir
            list_of_dirs = [tgt for tgt in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ['images']

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if(fname.endswith('.JPEG')):
                        path = os.path.join(root, fname)
                        if train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.imgs.append(item)

    def _download(self):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        name = url.split("/")[-1]
        resp = requests.get(url, stream=True)
        content_size = int(resp.headers['Content-Length']) / 1024  # ??????????????????????????????
        # ??????????????????
        path = f'{self.root}/{name}'
        with open(path, "wb") as file:
            for data in tqdm(iterable=resp.iter_content(1024), total=int(content_size), unit='kb', desc='downloading...'):
                file.write(data)
        print("finish download TinyImageNet.zip\n\n")

    def _zip(self):
        zFile = zipfile.ZipFile(f'{self.root}/tiny-imagenet-200.zip', "r")
        for fileM in tqdm(zFile.namelist(), unit='file', desc='ziping...'):
            zFile.extract(fileM, path=f'{self.root}')
        zFile.close()
        os.rename(f'{self.root}/tiny-imagenet-200', f'{self.root}/TinyImageNet')
        print("finish zip TinyImageNet.zip\n\n")
        os.remove(f'{self.root}/tiny-imagenet-200.zip')

    def return_label(self, idxs):
        return [self.class_to_label[self.tgt_idx_to_class[idx.item()]] for idx in idxs]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.imgs[idx]

        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

def handle_train(src_dir, dst_dir):
    '''
        transform 'tiny-imagenet train' to the form that can be
        handled by 'torchvision.datasets.ImageFolder'
    :param
        src_dir:
    :param
        dst_dir:
    :return:
    '''
    for clz in os.listdir(src_dir):
        img_new_path = f'{dst_dir}/{clz}'
        if not os.path.exists(img_new_path):
            os.mkdir(img_new_path)

        filelist = os.listdir(f'{src_dir}/{clz}/images')
        for file in filelist:
            src = f'{src_dir}/{clz}/images/{file}'
            dst = f'{img_new_path}/{file}'
            shutil.copy(src, dst)

def handle_val(src_dir, dst_dir):
    '''
        transform 'tiny-imagenet val' to the form that can be
        handled by 'torchvision.datasets.ImageFolder'
    :param
        src_dir:
    :param
        dst_dir:
    :return:
    '''

    val_img_to_class = {}
    set_of_classes = set()
    with open(f'{src_dir}/val_annotations.txt') as anno_file:
        lines = anno_file.readlines()
        for line in lines:
            words = line.split("\t")
            val_img_to_class[words[0]] = words[1]
            set_of_classes.add(words[1])

    for clz in set_of_classes:
        img_new_path = f'{dst_dir}/{clz}'
        if not os.path.exists(img_new_path):
            os.mkdir(img_new_path)

    filelist = os.listdir(f'{src_dir}/images')
    for file in filelist:
        src = f'{src_dir}/images/{file}'
        clz = val_img_to_class[file]
        dst = f'{dst_dir}/{clz}/{file}'
        shutil.copy(src, dst)