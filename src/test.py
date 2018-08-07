import os
import h5py
import pathlib
import numpy as np
from src.utils import exists
from src.utils import next_batch
from sklearn.externals import joblib


def test_svm(test_file, hdf5_path, saved_model, inference_file, batch_size=1000, use_extractor=False):

    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]
        n_samples = images.shape[0]
        images = (images - 127.5) / 127.5

        if use_extractor:
            assert exists(f'{saved_model}.pca'), 'extractor model does not exist'
            pca = joblib.load(f'{saved_model}.pca')
            images = pca.transform(images)

        progress_step_in_percentage = (batch_size/n_samples*100)
        current_progress = 0.

        svm_model = joblib.load(saved_model)
        predicted_labels = []

        for x_batch, _ in next_batch(images, batch_size=batch_size):
            print(f' Testing SVM model [{current_progress:{4}.{3}}%]\r', end="")
            predicted_labels = np.concatenate((predicted_labels, svm_model.predict(x_batch)))
            current_progress += progress_step_in_percentage

        print(f' Testing SVM model [{100}%] ', end="\n")
        save_prediction(test_file, inference_file, predicted_labels)


def save_prediction(test_file, inference_file, predicted_labels):
    path, file = os.path.split(inference_file)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    with open(test_file, "r") as test_f, open(inference_file, "w") as inference_f:
        index = 0
        for line in test_f:
            image_path = line.split(',')[0]
            inference_f.write(f'{image_path},{int(predicted_labels[index])}\n')
            index += 1

    print(f"model predictions saved @ {inference_file}")

