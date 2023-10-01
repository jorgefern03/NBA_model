import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import class_likelihood_ratios
from sklearn.metrics import det_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
import pandas as pd

#seasons = ['201617', '201718', '201819']
seasons = ['201516', '201617', '201718']
folder = 'Team_Model/comp/'
models = ['Team_Model/comp/logreg', 'Team_Model/comp/bets']
names = ['Original', 'Modelo + Apuestas']

for model, name in zip(models, names):
    test = []
    pred = []
    proba = []
    for season in seasons:
        test += [np.load('results/' + model +'/test_' + season + '.npy')]
        pred += [np.load('results/' + model +'/pred_' + season + '.npy')]
        proba += [np.load('results/' + model +'/proba_' + season + '.npy')]
        
    test = np.concatenate(test)
    pred = np.concatenate(pred)
    proba = np.concatenate(proba)

    print(classification_report(test, pred))
    print('Accuracy:', accuracy_score(test, pred))
    print(f'F1: {f1_score(test, pred)}')
    print(f'Average precision score: {average_precision_score(test, pred)}')

    fpr, tpr, thresh = roc_curve(test, proba[:, 1])
    auc = roc_auc_score(test, proba[:, 1])
    plt.plot(fpr,tpr,label= name+", AUC="+str(round(auc, 4)))

plt.legend(loc=0)
plt.savefig('results/' + folder +'ogvsbets_roc.png')    
    

