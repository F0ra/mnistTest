from src.utils import exists
from src.test import test_svm
from src.train import train_svm
from src.utils import make_txt_file
from argparse import ArgumentParser
from src.utils import create_hdf5_dataset

TEST_FILE = 'not_exists.txt'
BATCH_SIZE = 1000
TRAIN_FILE = 'not_exists.txt'
SAVED_MODEL = 'models/model.sav'
INFERENCE_FILE = 'inference/model_inference.txt'
PATH_TO_PNG_DATASET = './mnist_png'
PATH_TO_HDF5_TRAINING_DATASET = 'compressed_training.hdf5'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train', choices=['linearSVM', 'non-linearSVM'], default=False,
                        nargs='?', dest='train', help='select model from choices to train', metavar='TRAIN_MODE')

    parser.add_argument('--train-file', '-T', type=str, dest='train_file', help='txt file for training',
                        metavar='TXT_FILE', default=TRAIN_FILE)

    parser.add_argument('--save-file', '-S', type=str, dest='save_file', help='path to save model',
                        metavar='FILE_TO_SAVE_MODEL', default=SAVED_MODEL)

    parser.add_argument('--test', type=str, dest='saved_model', help='path to saved model',
                        metavar='SAVED_MODEL_FILE', nargs='?', default=False)

    parser.add_argument('--test-file', '--tf', type=str, dest='test_file', help='txt file for testing',
                        metavar='TXT_FILE', default=TEST_FILE)

    parser.add_argument('--inference-file', '--if', type=str, dest='inference_file', help='path to model inference file',
                        metavar='INFERENCE_FILE', default=INFERENCE_FILE)

    parser.add_argument('--use-extractor', '-U', dest='use_extractor',
                        help='use extractor for training/testing ?', action='store_true',)

    parser.add_argument('--keep-calm', '-C', dest='calm', action='store_true', help='keep calm?')

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    # train block ===>
    if options.train:
        if exists(options.train_file):
            hdf5_path = create_hdf5_dataset(path=options.train_file, name='training', compress=True)
            train_svm(hdf5_path=hdf5_path, save_file=options.save_file, model=options.train, batch_size=BATCH_SIZE,
                      use_extractor=options.use_extractor)
        elif not exists(options.train_file) and exists(PATH_TO_HDF5_TRAINING_DATASET):
            print('train-file does not exists! using default hdf5 training dataset.')
            hdf5_path = PATH_TO_HDF5_TRAINING_DATASET
            train_svm(hdf5_path=hdf5_path, save_file=options.save_file, model=options.train, batch_size=BATCH_SIZE,
                      use_extractor=options.use_extractor)
        else:
            print('train-file does not exists!')
            options.train_file = make_txt_file(PATH_TO_PNG_DATASET, mode="training")
            hdf5_path = create_hdf5_dataset(path=options.train_file, name='training', compress=True)
            train_svm(hdf5_path=hdf5_path, save_file=options.save_file, model=options.train, batch_size=BATCH_SIZE,
                      use_extractor=options.use_extractor)
    # <===

    # test block ===>
    if options.saved_model:
        if not exists(options.saved_model):
            raise TypeError('model-file does not exists!')

        elif not exists(options.test_file):
            print('test_file does not exists!')
            options.test_file = make_txt_file(PATH_TO_PNG_DATASET, mode="testing")
            hdf5_path = create_hdf5_dataset(path=options.test_file, compress=True)
            test_svm(test_file=options.test_file, hdf5_path=hdf5_path, saved_model=options.saved_model,
                     inference_file=options.inference_file, use_extractor=options.use_extractor)

        else:
            hdf5_path = create_hdf5_dataset(path=options.test_file, compress=True)
            test_svm(test_file=options.test_file, hdf5_path=hdf5_path, saved_model=options.saved_model,
                     inference_file=options.inference_file, use_extractor=options.use_extractor)

    if options.saved_model is None:
        options.saved_model = SAVED_MODEL
        if not exists(options.saved_model):
            raise TypeError('model-file does not exists!')
        elif not exists(options.test_file):
            print('test_file does not exists!')
            options.test_file = make_txt_file(PATH_TO_PNG_DATASET, mode="testing")
            hdf5_path = create_hdf5_dataset(path=options.test_file, compress=False)
            test_svm(test_file=options.test_file, hdf5_path=hdf5_path, saved_model=options.saved_model,
                     inference_file=options.inference_file, use_extractor=options.use_extractor)
        else:
            hdf5_path = create_hdf5_dataset(path=options.test_file, compress=False)
            test_svm(test_file=options.test_file, hdf5_path=hdf5_path, saved_model=options.saved_model,
                     inference_file=options.inference_file, use_extractor=options.use_extractor)
    # <===


if __name__ == '__main__':
    main()

