import os
import zipfile

import requests
from tqdm import tqdm


def download(url, root_dir):
    name = url.split("/")[-1]
    resp = requests.get(url, stream=True)
    content_size = int(resp.headers['Content-Length']) / 1024  # get file size
    # download to dist_dir
    path = f'{root_dir}/{name}'
    with open(path, "wb") as file:
        for data in tqdm(iterable=resp.iter_content(1024), total=int(content_size), unit='kb', desc='downloading...'):
            file.write(data)
    print(f"finish download {name}\n\n")
    return path


def zip(root_dir, zip_file_path, dist_file_path, rm_zip=True):
    zFile = zipfile.ZipFile(zip_file_path, "r")
    for fileM in tqdm(zFile.namelist(), unit='file', desc='ziping...'):
        zFile.extract(fileM, path=root_dir)
    zFile.close()
    source_file_path = '.'.join(zip_file_path.split('.')[0: -1])
    os.rename(source_file_path, dist_file_path)
    print("finish zip \n\n")
    if rm_zip:
        os.remove(zip_file_path)