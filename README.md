# mnist Test
recognizing handwritten digits using **Support Vector Machine** 

## SVM :
- linear
- non-linear

## Feature extraction:
- **PCA** with dimention reduction [n_samples,784] ==> [n_samples,64]


## ex:
| mode | command line |
| ---- | ------ |
| training||
||main_script.py --train linearSVM --train-file training.txt --save-file models/linearSVM|
||main_script.py --train linearSVM  --save-file models/PCA-linearSVM --use-extractor|
||main_script.py --train non-linearSVM --save-file models/PCA-kernelSVM --use-extractor|
|testing||
||main_script.py --test models/linearSVM --test-file testing.txt --inference-file inference/linearSVM.txt|
||main_script.py --test models/PCA-linearSVM --test-file testing.txt --inference-file inference/PCA-linearSVM.txt --use-extractor|
||main_script.py --test models/PCA-kernelSVM --test-file testing.txt --inference-file inference/PCA-kernelSVM.txt --use-extractor|
|metrics||
||metrics.py --test-file testing.txt --inference-file inference/linearSVM.txt|
||metrics.py --test-file testing.txt --inference-file inference/PCA-linearSVM.txt|
||metrics.py --test-file testing.txt --inference-file inference/PCA-kernelSVM.txt|


## dependencies:
- cv2
- h5py
- numpy
- sklearn
- matplotlib

## install dep:
script requires python 3.6+ to run.
pip install opencv-python h5py numpy scipy scikit-learn matplotlib

## metrics example:
![Confusion matrix example](https://raw.githubusercontent.com/F0ra/mnistTest/master/inference/PCA-kernelSVM.png)

### Todos

 - add deep convNet 