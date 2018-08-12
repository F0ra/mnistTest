import cv2
import matplotlib.pyplot as plt

def plot_incorrect(inCorrect:dict,path):
    fig = plt.figure(figsize=(15,8))
    
    total_incorrect = len(inCorrect)
    rows = total_incorrect//10 + 1
    cols = 10
    plt.suptitle(f"total incorrect: {total_incorrect}, in {path}")
    
    i=1
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=None)
    for keys in inCorrect:

        ax = fig.add_subplot(rows,cols,i)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        
        pred_label = int(inCorrect[keys][2])
        cor_label = inCorrect[keys][1]
        ax.set_title(f"pred: {pred_label}, corr: {cor_label} ", fontsize='small')
        image = cv2.imread(inCorrect[keys][0],cv2.IMREAD_GRAYSCALE)
        ax.imshow(image,cmap='gray')

        i+=1
    fig.savefig(path)
