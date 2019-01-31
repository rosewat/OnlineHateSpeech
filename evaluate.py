import pandas as pd
import numpy as np
import re
import json
import pickle
from keras.preprocessing import text, sequence
import math
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import itertools




def plot_cm(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
def plot_cm2(y_true,y_pred,labels,norm=True,conf_matrix=None,title=''):  
    """plots a colormap confusion matrix for predictions versus truth """
    #normalize the confusin matrix so imbalanced classes are represented
    if conf_matrix==None:
        cm=confusion_matrix(y_true,y_pred)
    else:
        cm=np.array(conf_matrix)
    norm_cm=np.zeros([cm.shape[0],cm.shape[1]])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            norm_cm[i][j]=cm[i][j]/sum(cm[i])
    cm2=cm
    if norm==True:
        cm=norm_cm
        
    x=range(0,3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax=ax.imshow(cm,cmap='GnBu')
    fig.colorbar(cax);

    #thresh = cm2.max() / 1.5 if norm else cm2.max() / 2
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(j, i, "{:,}".format(cm2[i, j]),
                 horizontalalignment="center",
                 color="black")# if cm2[i, j] > thresh else "black")

    plt.xticks(x,labels,rotation='vertical')
    plt.yticks(x,labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()
    pass

def f1_plot(hist):  
    plt.plot(hist.history['f1'],label='train f1')
    plt.plot(hist.history['val_f1'],label='val f1')

    plt.title('model training and val f1')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    return plt

def acc_plot(hist):  
    plt.plot(hist.history['acc'],label='train acc')
    plt.plot(hist.history['val_acc'],label='val acc')

    plt.title('model training and val acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    return plt


def loss_plot(hist):  
  
    plt.plot(hist.history['loss'],label='train loss')
    plt.plot(hist.history['val_loss'],label='val loss')

    plt.title('model training and val loss')
    plt.ylabel('f1')
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.show()
    return plt
