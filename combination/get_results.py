import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import class_likelihood_ratios
from sklearn.metrics import det_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import pandas as pd

seasons = ['201617', '201718', '201819']
models = ['Team_Model']

for model in models:
    for season in seasons:
        test = np.load('results/' + model +'/test_' + season + '.npy')
        pred = np.load('results/' + model +'/pred_' + season + '.npy')
        proba = np.load('results/' + model +'/proba_' + season + '.npy')

        fpr, tpr, _ = roc_curve(test, proba[:, 1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.savefig('results/' + model +'/reports/roc_' + season + '.png')

        with open('results/' + model +'/reports/report_' + season + '.txt', 'w') as f:
            print('Accuracy:', accuracy_score(test, pred), file=f)
            print('Average Precision:', average_precision_score(test, proba[:,1]), file=f)
            print('Confusion Matrix:\n', confusion_matrix(test, pred), file=f)
            print('Classification Report:', classification_report(test, pred), file=f)

        
        precision, recall, threshold = precision_recall_curve(test, proba[:, 1])
        prd = PrecisionRecallDisplay(precision, recall).plot()
        plt.savefig('results/' + model +'/reports/pr_' + season + '.png')

