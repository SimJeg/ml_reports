__author__ = 'Simon Jegou'
__date__ = 'February 2018'

import numpy as np
import matplotlib.pylab as plt
from IPython import get_ipython
from functools import partial
import warnings
from sklearn.metrics import log_loss, confusion_matrix, roc_curve, roc_auc_score,\
f1_score, accuracy_score, average_precision_score, precision_recall_curve

def binary_classification_report(Y_true, Y_pred, show=True, bins=50):
    """
    Print different metrics for binary classification 
    and display 4 plots (interactive if in a notebook): 
    1. Confusion matrix
    2. Histogram of predictions
    3. ROC curve
    4. Precision Recall curve
    
    Example : 
        
    n = 10000
    p = np.random.rand(n)
    Y_pred = np.random.rand(n)**p
    Y_true = p < 0.5
    binary_classification_report(Y_true, Y_pred, show=True, bins=100)
        
    Parameters
    ----------
    Y_true : 1D binary array. Ground truth : array of 0 and 1
    Y_pred : 1D array. Predictions  : float in [0, 1]. If not in [0, 1], automatically rescaled
    show : bool. Whether to display plot or just print metrics.
    bins : int, number of bins in the histogram of predictions

    Returns
    -------
    None
    
    """

    # Assertions and warnings
    
    Y_true = np.array(Y_true).astype(float)    
    Y_pred = np.array(Y_pred).astype(float)
    
    assert(np.ndim(Y_true) == 1), 'Y_true must be a 1D array, found shape {}'.format(Y_true.shape)
    assert(np.ndim(Y_pred) == 1), 'Y_pred must be a 1D array, found shape {}'.format(Y_pred.shape)
    assert len(Y_pred) == len(Y_true),\
        'Y_pred and Y_true must have the same length, found {} / {}'.format(
        len(Y_pred), len(Y_true))
    assert set(Y_true) == {0, 1}, 'Y_true must be a binary vector'
    if (np.min(Y_pred)) < 0 or (np.max(Y_pred) > 1):
        warnings.warn('Y_pred have been rescaled in [0 1] because some values were out of this range')
        Y_pred = (Y_pred - Y_pred.min()) / (Y_pred.max() - Y_pred.min())

    # Compute metrics

    n = len(Y_true)
    P = int(np.sum(Y_true))
    N = n - P

    print('n = {} samples'.format(n))
    print('0/1 : {:.2f}% (N={}) / {:.2f}% (P={})'.format(
        100. * N / n, N, 100. * P / n, P))

    acc_list = [np.mean(Y_true == (Y_pred > t)) for t in np.arange(0, 1, 0.01)]
    best_threshold = np.argmax(acc_list)
    best_acc = 100 * np.max(acc_list)
    print('Best accuracy : {:.2f}% for threshold = {:.2f}%'.format(best_acc, best_threshold))
    print('Logloss : {:.4f}'.format(log_loss(Y_true, Y_pred)))

    if not show:
        Y_pred_01 = (Y_pred > 0.5).astype('int')
        print('Accuracy : {:.2f}%'.format(100 * accuracy_score(Y_true, Y_pred_01)))
        print('F1 score : {:.4f}'.format(f1_score(Y_true, Y_pred_01)))
        print('ROC AUC  : {:.2f}%'.format(100 * roc_auc_score(Y_true, Y_pred)))
        print('Mean Average precision : {:.4f}'.format(average_precision_score(Y_true, Y_pred)))

    else:
        def interactive_confusion_matrix(threshold):

            plt.figure(figsize=(22, 7))

            # Plot 1 : confusion matrix
            plt.subplot(141)

            Y_pred_01 = (Y_pred > threshold).astype('int')
            cm = confusion_matrix(Y_true, Y_pred_01)
            TN, FP, FN, TP = cm.flatten()
            TPR = float(TP) / P
            FPR = float(FP) / N
            
            if TP+FP > 0:
                prec = float(TP)/(TP+FP)
            else:
                prec = 1.
            
            plt.imshow(np.array([[1,4],[1,4]]), cmap='RdYlBu', vmin=0, vmax=5)
            plt.plot([-0.5, 1.5], [0.5,0.5], c='black', alpha=0.8)
            plt.plot([0.5,0.5], [-0.5, 1.5], c='black', alpha=0.8)
               
            plt.title('Confusion Matrix\nAccuracy : {:.2f}%\n F1 score : {:.4f}'.
                format(100 * accuracy_score(Y_true, Y_pred_01), f1_score(Y_true, Y_pred_01)))
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, [1, 0])
            plt.yticks(tick_marks, [1, 0])
            plt.xlabel('True label')
            plt.ylabel('Predicted label')

            plot_text = partial(
                plt.text,
                horizontalalignment='center',
                verticalalignment='center',
                size='12')
            
            plot_text(1, 1, 'TN : {}\nSpecificity : {:.2f}%'.format(TN, 100. * TN / N))
            plot_text(0, 1, 'FN : {}'.format(FN))
            plot_text(1, 0, 'FP : {}\nFPR : {:.2f}%'.format(FP, 100. * FPR))
            plot_text(0, 0, 'TP : {}\nSensitivity : {:.2f}%\nPrecision : {:.2f}%'.format(TP, 100. * TPR, 100.*prec))

            # Plot 2 : Histogram of predictions
            plt.subplot(142)
            plt.hist(Y_pred[Y_true == 0], bins, histtype='bar', label='0', color='#69A3CB', alpha=0.8)
            plt.hist(Y_pred[Y_true == 1], bins, histtype='bar', label='1', color='#F2613B', alpha=0.8)
            plt.axis([0, 1, 0, None])
            plt.plot([threshold, threshold], [0, plt.ylim()[1]], c='r', alpha=0.5)
            plt.title('Histogram of predictions')
            plt.legend()

            # Plot 3 : roc curve
            plt.subplot(143)
            fpr, tpr, _ = roc_curve(Y_true, Y_pred)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], c='gray', alpha=0.5)
            plt.scatter(FPR, TPR, c='r')
            plt.title('ROC curve \nROC AUC : {:.2f}%'.
                      format(100 * roc_auc_score(Y_true, Y_pred)))
            plt.xlabel('FPR = 1 - Specificity = FP / N')
            plt.ylabel('TPR = Sensitivity = TP / P')
            plt.grid()

            # Plot 4 : precision recall curve
            plt.subplot(144)
            precision, recall, _ = precision_recall_curve(Y_true, Y_pred)
            plt.plot(recall, precision)
            plt.scatter(TPR, prec, c='r')
            plt.title('Precision Recall curve \nMean Average Precision : {:.4f}'.
                format(average_precision_score(Y_true, Y_pred)))
            plt.xlabel('Recall = TPR = Sensitivity = TP / P')
            plt.ylabel('Precision = PPV = TP / (TP + FP)')
            plt.grid()
        plt.show()

        # Check if we are in a notebook
        if getattr(get_ipython(), 'kernel', None) is not None:
            from ipywidgets import interact, FloatSlider
            interact(interactive_confusion_matrix, 
                     threshold=FloatSlider(value=0.5, 
                                           min=0, 
                                           max=1, 
                                           step=0.01, 
                                           continuous_update=False))
        else:
            interactive_confusion_matrix()
