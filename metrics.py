import os
import sklearn
from src.utils import exists
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from src.plot_cm import plot_confusion_matrix
from src.plot_incorr import plot_incorrect

def parseint(string):
    return int(''.join([x for x in string if x.isdigit()]))


def metrics(test_file, inference_file, return_incorr=False):
    """ using sklearn.metrics lib to calculate accuracy score and Confusion matrix.

            Args:
                test_file (str):txt file with path to png samples and true labels.
                inference_file (str):txt file with path to png samples and predicted labels.

            Returns:
                float: accuracy_score.
                array: Confusion matrix.
        """
    with open(test_file) as f, open(inference_file) as ff:
        true_labels = []
        predicted_labels = []
        inCorrect={}
        i=0
        for line1, line2 in zip(f, ff):
            true = parseint(line1.split(",")[1])
            pred = parseint(line2.split(",")[1])
            path_to_img = line1.split(",")[0]
            true_labels.append(parseint(line1.split(",")[1]))
            predicted_labels.append(parseint(line2.split(",")[1]))
            if return_incorr and true != pred:
                inCorrect[i] = [path_to_img,true,pred]
            i += 1
    cm = confusion_matrix(true_labels, predicted_labels, labels=[i for i in range(10)], sample_weight=None)
    accuracy_score = sklearn.metrics.accuracy_score(true_labels, predicted_labels, normalize=True, sample_weight=None)

    return accuracy_score, cm, inCorrect


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--test-file', '-T', type=str, dest='test_file',
                        help='txt file with path to png samples and true labels',
                        metavar='TXT_FILE', required=True)

    parser.add_argument('--inference-file', '-I', type=str, dest='inference_file',
                        help='txt file with path to png samples and predicted labels',
                        metavar='INFERENCE_FILE', required=True)

    parser.add_argument('--plot-incorr', '-P', dest='plot_incorr', action='store_true', help='plot incorrect classified images')

    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not exists(options.test_file):
        print(f'{options.test_file} does not exist!')
        return -1
    elif not exists(options.inference_file):
        print(f'{options.inference_file} does not exist!')
        return -1

    if options.plot_incorr:
        accuracy_score, cm, inCorrect = metrics(test_file=options.test_file, inference_file=options.inference_file,return_incorr=True)
    else:
        accuracy_score, cm, _ = metrics(test_file=options.test_file, inference_file=options.inference_file)
    
    print(f'confusion_matrix: \n{cm}')
    print(f'accuracy_score: {accuracy_score}')

    target_names = [i for i in range(10)]
    png_path = f'{options.inference_file[0:-4]}.png'  # cut off .txt part and add .png in inference path
    png_path_incorr = f'{options.inference_file[0:-4]}_incorr.png'  # cut off .txt part and add _incorr.png in inference path 
    
    path, file = os.path.split(options.inference_file)
    plot_confusion_matrix(cm=cm, target_names=target_names, path=png_path, normalize=False,
                          title=f'{file} Confusion matrix')
    print(f'{file} Confusion matrix plot saved @ {png_path}')

        
    if options.plot_incorr:
        plot_incorrect(inCorrect,png_path_incorr)
        print(f'{file} Plot with incorrect images saved @ {png_path_incorr}')
    



if __name__ == '__main__':
    main()
