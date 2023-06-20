from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import loguniform
import pandas as pd

def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    '''
    Realiza la validación cruzada con TimeSeriesSplit para cada modelo y muestra los resultados.
    :param X_train: set de entrenamiento
    :param y_train: valores objetivo para X_train
    :return: DataFrame con los resultados de la validación cruzada para cada modelo
    '''
    models = [
        ('LogReg', LogisticRegression(max_iter=1000000)),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVC linear', SVC(kernel='linear', max_iter=10000000)),
        ('SVC rbf', SVC(kernel='rbf', max_iter=10000000)),
        ('GNB', GaussianNB()),
    ]

    scoring = ['accuracy', 'precision_weighted',
               'recall_weighted', 'f1', 'roc_auc']

    target_names = ['W', 'L']
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train)
    for name, model in models:

        # Validación cruzada con TimeSeriesSplit separando por temporadas
        kfold = model_selection.TimeSeriesSplit(n_splits=5, test_size=1230)

        cv_results = model_selection.cross_validate(
            model, X_train, y_train.values.ravel(), cv=kfold, scoring=scoring, n_jobs=-1, return_estimator=True)

        print('\n', name, '\n')
        for mesure in cv_results.keys():
            if 'estimator' not in mesure:
                print(mesure, ': ', cv_results[mesure].mean())

    return 0


seasons = ['201314', '201415', '201516', '201617', '201718', '201819']

folder = 'csv/entry_data/combination_entry/'
matrices_og_c = []

for season in seasons:
    matrices_og_c +=  [pd.read_csv(folder + 'entry_data_' + season +'.csv', low_memory=False)]


matriz_train = pd.concat(matrices_og_c, ignore_index=True)

matriz_train.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                        'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True, errors='ignore')

win_train = matriz_train['H_WL']
matriz_train.drop(['H_WL'], axis=1, inplace=True)

matriz_train.drop(['H_OFF_RATING', 'H_DEF_RATING', 'A_OFF_RATING', 'A_DEF_RATING'], axis=1, inplace=True, errors='ignore')

run_exps(matriz_train, win_train)

tscv = model_selection.TimeSeriesSplit(n_splits=5, test_size=1230)
for train, test in tscv.split(matriz_train):
    print("%s %s" % (train, test))