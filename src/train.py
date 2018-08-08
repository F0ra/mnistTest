import os
import h5py
import pathlib
import sklearn
from sklearn import svm
from sklearn.svm import LinearSVC
from src.utils import next_batch
from sklearn.externals import joblib
from sklearn.decomposition import PCA


def train_svm(hdf5_path, save_file, model, batch_size=1000, use_extractor=False):

    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]
        n_samples = images.shape[0]
        images = (images - 127.5) / 127.5  # map to [-1,1]

        if use_extractor:
            print('using pca for dim reduction...')
            pca = PCA(n_components=64, svd_solver='randomized').fit(images)
            images = pca.transform(images)  # map from [n_samples,784] to [n_samples,20]

        if model == 'non-linearSVM':
            print(' learning kernelSVM model. Could take a while...\r', end="")
            svm_model = svm.SVC()
            svm_model.fit(images, labels)
            print()

        elif model == 'linearSVM':
            svm_model = sklearn.linear_model.SGDClassifier(learning_rate='constant', eta0=0.1, shuffle=False,
                                                           max_iter=1000)
            classes = [i for i in range(10)]

            progress_step_in_percentage = (batch_size / n_samples * 100)

            for epoch in range(1):
                current_progress = 0.
                for x_batch, y_batch, _ in next_batch(images, labels, batch_size):
                    print(f' Learning {model} [{current_progress:{4}.{3}}%]\r', end="")
                    svm_model.partial_fit(x_batch, y_batch, classes=classes)
                    current_progress += progress_step_in_percentage
                print(f' Learning {model} [{100}%] ', end="\n")

        else:
            print('model does not implemented yet')
            return -1

        path, file = os.path.split(save_file)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(svm_model, save_file)
        if use_extractor:
            joblib.dump(pca, f'{save_file}.pca')
            print(f"model saved @ {save_file}")
            print(f"feature extractor saved @ {save_file}.pca")
        else:
            print(f"model saved @ {save_file}")
