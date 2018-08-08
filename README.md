# mnist Test
recognizing handwritten digits using **Support Vector Machine** 

## SVM :
- linear
- non-linear

## Feature extraction:
- **PCA** with dimention reduction [n_samples,784] ==> [n_samples,64]


## examples:
**pipeline: [training] ==> [testing] ==> [show metrics]**
## ex 1 
- [training] (train linearSVM on training.txt data set) ==>[models/linearSVM] 
- [testing] (test models/linearSVM on testing.txt) ==>[inference/linearSVM.txt]  
- [show metrics] (testing.txt, inference/linearSVM.txt) ==> [ inference/linearSVM.png]

| mode | command line |
| ---- | ------ |
| training|main_script.py --train linearSVM --train-file training.txt --save-file models/linearSVM |
|testing|main_script.py --test models/linearSVM --test-file testing.txt --inference-file inference/linearSVM.txt|
|metrics| metrics.py --test-file testing.txt --inference-file inference/linearSVM.txt|

## ex 2
- [training] (train  linearSVM on training.txt data set using feature extraction) ==>[models/linearSVM,models/linearSVM.pca] 
- [testing] (test models/linearSVM on testing.txt using feature extraction) ==>[inference/PCA-linearSVM.txt]  
- [show metrics] (testing.txt, inference/PCA-linearSVM.txt) ==> [ inference/PCA-linearSVM.png]

| mode | command line |
| ---- | ------ |
| training|main_script.py --train linearSVM  --save-file models/PCA-linearSVM --use-extractor|
|testing|main_script.py --test models/PCA-linearSVM --test-file testing.txt --inference-file inference/PCA-linearSVM.txt --use-extractor|
|metrics|metrics.py --test-file testing.txt --inference-file inference/PCA-linearSVM.txt|

## ex 3 
- [training] (train non-linearSVM on training.txt data set using feature extraction) ==>[models/PCA-kernelSVM] 
- [testing] (test models/PCA-kernelSVM on testing.txt using feature extraction) ==>[inference/PCA-kernelSVM.txt]  
- [show metrics] (testing.txt, inference/PCA-kernelSVM.txt) ==> [ inference/PCA-kernelSVM.png]

| mode | command line |
| ---- | ------ | 
| training|main_script.py --train non-linearSVM --save-file models/PCA-kernelSVM --use-extractor|
|testing|main_script.py --test models/PCA-kernelSVM --test-file testing.txt --inference-file inference/PCA-kernelSVM.txt --use-extractor|
|metrics|metrics.py --test-file testing.txt --inference-file inference/PCA-kernelSVM.txt|

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
