import os
import h5py
import pathlib
import numpy as np
from src.utils import exists
from src.utils import next_batch
from sklearn.externals import joblib

import os


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


def test_conv_net(test_file, hdf5_path, inference_file): 
    from src.convNet import net
    import tensorflow as tf
    
    batch_size=1

    with h5py.File(hdf5_path, 'r') as f:
        images = f['images'][:]
        n_samples = images.shape[0]
        images = (images/255).astype(np.float32)

        progress_step_in_percentage = (1/n_samples*100)
        current_progress = 0.
        predicted_labels = []

        conv_net = net
        input_image = tf.placeholder(tf.float32, shape=[None, 784])
        keep_prob = tf.placeholder(tf.float32)
        prediction = conv_net(input_image=input_image, keep_prob=keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess: 
            # checkpoints restore -------------------
            ckpt = tf.train.get_checkpoint_state('./convNet-ckpt/')

            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)  # automatically updates the path to the latest checkpoint.
                print('conv Net restore from checkpoints')
            else:
                print('conv Net save file does not found!')
                return -1

            inCorrect={}
            for i in range(images.shape[0]):
                if i%10 == 0: print(f' Testing conv Net model [{current_progress:{4}.{3}}%]\r', end="")
                pred = prediction.eval(feed_dict={input_image: images[i:i+1], keep_prob:1.0})
                predicted_labels.append(pred)
                current_progress += progress_step_in_percentage

        print(f' Testing conv Net model [{100}%] ', end="\n")
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

