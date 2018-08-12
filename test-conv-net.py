from src.utils import exists
from src.test import test_conv_net
from src.utils import make_txt_file
from argparse import ArgumentParser
from src.utils import create_hdf5_dataset

TEST_FILE = 'not_exists.txt'
INFERENCE_FILE = 'inference/convNet_inference.txt'
PATH_TO_PNG_DATASET = './mnist_png'


def build_parser():
    parser = ArgumentParser()

    parser.add_argument('--test-file', '--tf', type=str, dest='test_file', help='txt file for testing',
                        metavar='TXT_FILE', default=TEST_FILE)

    parser.add_argument('--inference-file', '--if', type=str, dest='inference_file', help='path to model inference file',
                        metavar='INFERENCE_FILE', default=INFERENCE_FILE)

    parser.add_argument('--keep-calm', '-C', dest='calm', action='store_true', help='keep calm?')

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not exists(options.test_file):
        print('test_file does not exists!')
        options.test_file = make_txt_file(PATH_TO_PNG_DATASET, mode="testing")
        hdf5_path = create_hdf5_dataset(path=options.test_file, compress=True)
        test_conv_net(test_file=options.test_file, hdf5_path=hdf5_path, inference_file=options.inference_file)

    else:
        hdf5_path = create_hdf5_dataset(path=options.test_file, compress=True)
        test_conv_net(test_file=options.test_file, hdf5_path=hdf5_path, inference_file=options.inference_file)


if __name__ == '__main__':
    main()

