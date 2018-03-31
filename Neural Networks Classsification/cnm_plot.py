
# coding: utf-8


import numpy as np
#import PyQt4
import matplotlib
#matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools



pdfFile = PdfPages("Confusion_matrix.pdf")

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig = plt.gcf()
    plt.show()
    plt.draw()
    pdfFile.savefig(fig)
    plt.clf()
    
    
def closeFile() :
    plt.close('all')
    pdfFile.close()

