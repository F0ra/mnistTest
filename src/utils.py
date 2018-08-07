import os
import cv2
import h5py
import json
import datetime
import numpy as np
from random import shuffle


def exists(path):
    return os.path.exists(path)


def make_txt_file(path, mode="testing"):
    """ makes shuffled list of files in txt format. Each row contain path to single
    png file and comma separated label.

        Args:
            path (str):path to folder containing testing or training folder with png files inside.
            mode (str):optional training or testing.

        Returns:
            str: relative path to txt file.
    """
    path = os.path.join(path + f"/{mode}/")

    print(f"creating {mode}.txt file from png dataset.", end="", flush=True)
    with open(f'{mode}.txt', "w") as f:
        total_files = []
        for digit in range(10):
            file_list = os.listdir(os.path.join(path + str(digit)))
            [total_files.append(f"{path}{digit}/" + file + f",{digit}") for file in file_list]

        shuffle(total_files)
        [f.write(file + "\n") for file in total_files]
    print("\ndone!")
    return f'./{mode}.txt'


def create_hdf5_dataset(path, name='testing', compress=False):
    """ makes HDF5 dataset from provided txt file(contains path to images and labels).
        HDF5 dataset includes three sub datasets:
            -image set with shape(n_samples,784),if not compressed each image
                squished from uint8 [0,256] to float32 [-1,1].
            -labels set with shape(total_images,).
            -metadata provides HDF5 dataset info.
        what is hdf5 and why to use it: http://docs.h5py.org/en/latest/quick.html

            Args:
                path (str): path to txt file.
                name (str): name of dataset to create.
                compress (bool): optional

            Returns:
                str: relative path to hdf5 file.
        """
    file_name = f'{"compressed_" if compress else ""}{name}.hdf5'

    with h5py.File(file_name, 'w') as f, open(path, "r") as txt_file:
        # ==== block for printing progress bar
        print(f'creating {file_name} ==>[', end="", flush=True)
        total_lines = txt_file.read().count('\n')
        txt_file.seek(0, 0)
        progress_step = total_lines // 25
        progress = 0
        # ====================================
        image_index = 0
        if compress:
            image_set = f.create_dataset('images', (1, 784), maxshape=(None, 784), dtype=np.uint8,
                                         compression="gzip", compression_opts=3)
            labels_set = f.create_dataset('labels', (1,), maxshape=(None,), dtype=np.uint8,
                                          compression="gzip", compression_opts=3)
            for line in txt_file:
                if progress == image_index:
                    progress += progress_step
                    print(end="#", flush=True)

                # assume if line contains comma it also contains label
                # ex: path_to_png/testing/2/1234.png,3
                # or path_to_png/testing/2/1234.png
                if "," in line:
                    image_path, image_label = line.split(',')
                else:
                    image_path = line
                    image_label = None
                image_set.resize((image_index + 1, 784))
                labels_set.resize((image_index + 1,))
                image_set[image_index] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).reshape(-1)
                labels_set[image_index] = int(image_label)
                image_index += 1
        else:
            image_set = f.create_dataset('images', (1, 784), maxshape=(None, 784), dtype=np.float32)
            labels_set = f.create_dataset('labels', (1,), maxshape=(None,), dtype=np.float32)

            for line in txt_file:
                if progress == image_index:
                    progress += progress_step
                    print(end="#", flush=True)

                if "," in line:
                    image_path, image_label = line.split(',')
                else:
                    image_path = line
                    image_label = None

                image_set.resize((image_index + 1, 784))
                labels_set.resize((image_index + 1,))
                image_set[image_index] = np.float32((cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) - 127.5) / 127.5).reshape(-1)
                labels_set[image_index] = np.float32(image_label)
                image_index += 1

        date_time = datetime.datetime.now()
        metadata = {'Date': date_time.strftime("%w/%d/%Y"),
                    'Time': date_time.strftime("%H:%M:%S"),
                    'User': 'Fora',
                    'OS': os.name,
                    'total_images': image_index,
                    'images dtype': f"{'uint8' if compress else 'float32'}",
                    'compressed': f"{'yes' if compress else 'no'}",
                    }
        f.create_dataset('metadata', data=json.dumps(metadata))
        print("]\ndone!")
        return f'./{file_name}'


def create_f_extracted_hdf5(extracted_features, labels, name='testing'):
    """ creates HDF5 feature extracted dataset.
            HDF5 dataset includes three sub datasets:
                -extracted features set with shape(total_samples,128).
                -labels set with shape(total_samples,).
                -metadata provides HDF5 dataset info.
            what is hdf5 and why to use it: http://docs.h5py.org/en/latest/quick.html

                Args:
                    extracted_features (np.array,int): extracted features.
                    labels (np.array,int): labels.
                    name (str): name for hdf5 dataset

                Returns:
                    str: relative path to hdf5 file.
    """
    assert type(extracted_features) == np.ndarray, 'extracted_features must be np.array type!'
    assert type(labels) == np.ndarray, 'labels must be np.array type!'
    assert extracted_features.shape[0] == labels.shape[0], \
        'number of sample in extracted_features and labels must be same!'

    n_samples = extracted_features.shape[0]
    file_name = f'f-extracted-{name}.hdf5'
    with h5py.File(file_name, 'w') as f:
        feature_set = f.create_dataset('images', (n_samples, 128), maxshape=(None, 128), dtype=np.uint8,
                                       compression="gzip", compression_opts=4)
        labels_set = f.create_dataset('labels', (n_samples,), maxshape=(None,), dtype=np.uint8,
                                      compression="gzip", compression_opts=4)

        feature_set[:] = extracted_features
        labels_set[:] = labels
    print(f'./{file_name} created!')
    return f'./{file_name}'


def print_hdf5_metadata(filename):
    with h5py.File(filename, 'r') as f:
        metadata = json.loads(f['metadata'][()])
        print(f"=== {filename} metadata ==============")
        for k in metadata:
            print(f'{k} => {metadata[k]}')
        print("======================================")


def next_batch(samples, labels=None, batch_size=1):
    """Batch generator for training/testing model

                Args:
                    samples (np.array):
                    labels (np.array): optional
                    batch_size (int): optional

                Returns:
                    [np.array,np.array,int]: if labels provided
                    [np.array,int]: else

    """
    if labels is not None:
        assert samples.shape[0] == labels.shape[0], "num of samples must be same in images and labels"

    n_samples = samples.shape[0]
    num_batches = n_samples // batch_size if n_samples % batch_size == 0 else n_samples // batch_size + 1
    batch_number = 0
    for batch in range(num_batches):
        start = 0 if batch == 0 else end
        end = (start + batch_size) if (start + batch_size) < n_samples else n_samples
        batch_number += 1
        if labels is None:
            yield samples[start:end], batch_number
        else:
            yield samples[start:end], labels[start:end], batch_number
