from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from scipy.stats import loguniform
import numpy as np
import pandas as pd


def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Ejecuta los modelos de clasificación y devuelve un DataFrame con las predicciones
    :param X_train: set de entrenamiento
    :param y_train: vector objetivo de entrenamiento
    :param X_test: set de testeo
    :param y_test: vector objetivo de testeo
    :return: DataFrame con las predicciones de cada modelo
    '''

    dfs = []

    # Definimos los modelos a utilizar
    models = [
        ('LogReg', LogisticRegression(max_iter=1000000)),
        #('RF', RandomForestClassifier()),
        #('KNN', KNeighborsClassifier()),
        #('SVC linear', SVC(kernel='linear', max_iter=10000000)),
        #('SVC rbf', SVC(kernel='rbf', max_iter=10000000, probability=True)),
        #('GNB', GaussianNB()),
    ]

    accuaracy = {}

    # Estandarizamos los datos
    ss = MinMaxScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    for name, model in models:

        # Entrenamos el modelo
        clf = model.fit(X_train, y_train.values.ravel())
        
        #y_pred = clf.predict(X_test)

        # Calculamos la probabilidad de que la predicción de clase sea 1 (W)
        y_proba = clf.predict_proba(X_test)
        threshold = 0.5

        # Convertimos las probabilidades en predicciones de clase
        y_pred = (y_proba[:, 1] >= threshold).astype('int')
        y_pred = np.array(y_pred)

        print(name)

        accuaracy[name] = (y_test, y_pred)

        print(f'{len(y_test)}, Accuracy: {accuracy_score(y_test, y_pred)}')
        
    return accuaracy, y_proba


#seasons = ['201314', '201415', '201516']
#seasons = ['201415', '201516', '201617']
#seasons = ['201516', '201617', '201718']
seasons = ['201314', '201415']

# Temporada a predecir
predict_season = '201516'
model_folder = 'results/Team_Model/comp/'
folder = 'csv/entry_data/team_entry/'
matrices_og_c = []

for season in seasons:
    matrices_og_c +=  [pd.read_csv(folder + 'entry_data_' + season +'.csv', low_memory=False)]


matriz_pred_c = pd.read_csv(folder + 'entry_data_' + predict_season +'.csv', low_memory=False)

target_names = ['W', 'L']

accuracy = []
proba_tot = []

matriz_train = pd.concat(matrices_og_c, ignore_index=True)

# Eliminamos las columnas que no queremos utilizar
matriz_train.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                        'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True)

matriz_train.drop(['H_OFF_RATING', 'H_DEF_RATING', 'A_OFF_RATING', 'A_DEF_RATING'], axis=1, inplace=True)

win_train = matriz_train['H_WL']
matriz_train.drop(['H_WL'], axis=1, inplace=True)

ss = MinMaxScaler()
X_train = ss.fit_transform(matriz_train)

m_teams = matriz_train.iloc[:,-36:]

# Descomentar para utilizar RFE (seleccion de caracteristicas)
"""
#RandomForestClassifier()
#LogisticRegression(max_iter=100000)
#SVR(kernel='linear')

rfe_selector2 = RFE(estimator=LogisticRegression(max_iter=100000), step = 1, n_features_to_select=0.9)
rfe_selector2.fit(X_train[:,-36:], win_train)

m_teams = m_teams[m_teams.columns[rfe_selector2.get_support()]]
m_teams = m_teams[m_teams.columns]
"""
all_features = pd.concat([m_teams], axis=1)


# Dividimos el set de entrenamiento en 12 partes de 102 partidos cada una aproximadamente
for i in range(0, 12):
    retrain = 102
    step = i

    matriz_pred = matriz_pred_c.copy()
    matrices_og = matrices_og_c.copy()
    matrices_og += [matriz_pred.head(retrain*step)]
    matriz_pred = matriz_pred.tail(1230 - retrain*step)

    if i < 11:
        matriz_pred = matriz_pred.head(retrain)

    matriz_train = pd.concat(matrices_og, ignore_index=True)

    matriz_train.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                        'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True)

    matriz_pred.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                        'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True)
    matriz_pred.drop(['H_OFF_RATING', 'H_DEF_RATING', 'A_OFF_RATING', 'A_DEF_RATING'], axis=1, inplace=True)


    win_train = matriz_train['H_WL']
    win_pred = matriz_pred['H_WL']
    matriz_train.drop(['H_WL'], axis=1, inplace=True)
    matriz_pred.drop(['H_WL'], axis=1, inplace=True)

   
    matriz_train = matriz_train[all_features.columns]
    matriz_pred = matriz_pred[matriz_train.columns]

    now = step*retrain
    before = now - retrain*3
    after = now + retrain*3
    last = now - retrain*2
    
    season1 = 1230
    season2 = 1230*2
    season3 = 1230*3
    season4 = 1230*4
    
    # Cuando i es menor a 3, se utilizan todos los datos de entrenamiento
    if i < 3:
        final, proba = run_exps(matriz_train, win_train, matriz_pred, win_pred)

    # Cuando i es menor a 8, se utilizan los datos en un margen de tiempo alrededor del segmento de temporada a predecir    
    elif i <8:
              
        matriz_train = pd.concat([matriz_train.iloc[before:after,:], matriz_train.iloc[season1+before:season1+after,:], matriz_train.iloc[season2+before:season2+after,:], matriz_train.tail(last)], ignore_index=True)
        win_train = pd.concat([win_train.iloc[before:after], win_train.iloc[season1+before:season1+after], win_train.iloc[season2+before:season2+after], win_train.tail(last)], ignore_index=True)
        
        final, proba = run_exps(matriz_train, win_train, matriz_pred, win_pred)

    # Cuando i es menor a 11, se utiliza la parte final de las temporadas anteriores    
    elif i < 11:
        
        matriz_train = pd.concat([matriz_train.iloc[last:season1,:], matriz_train.iloc[season1+last:season2,:], matriz_train.iloc[season2+last:season3,:], matriz_train.tail(last)], ignore_index=True)
        win_train = pd.concat([win_train.iloc[last:season1], win_train.iloc[season1+last:season2], win_train.iloc[season2+last:season3], win_train.tail(last)], ignore_index=True)
        
        final, proba = run_exps(matriz_train, win_train, matriz_pred, win_pred)
    
    # En el ultimo caso, se usan todos los datos de entrenamiento
    else:
        final, proba = run_exps(matriz_train, win_train, matriz_pred, win_pred)
    
    accuracy += [final]
    proba_tot += [proba]




for model in accuracy[0].keys():
    y_test = []
    y_pred = []
    for split in accuracy:
        y_test += [split[model][0]]
        y_pred += [split[model][1]]

    y_test = np.concatenate(y_test)
    y_pred = np.concatenate(y_pred)

    print('\n', model, '\n')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1: {f1_score(y_test, y_pred)}')
    print(f'AUC-score: {roc_auc_score(y_test, y_pred)}')
    print(f'Average precision score: {average_precision_score(y_test, y_pred)}')

    #np.save(model_folder + 'svm/test_' + predict_season + '.npy', y_test)
    #np.save(model_folder + 'svm/pred_' + predict_season + '.npy', y_pred)
    np.save(model_folder + 'logreg/test_' + predict_season + '.npy', y_test)
    np.save(model_folder + 'logreg/pred_' + predict_season + '.npy', y_pred)

y_proba = np.concatenate(proba_tot)
#np.save(model_folder + 'svm/proba_' + predict_season + '.npy', y_proba)
np.save(model_folder + 'logreg/proba_' + predict_season + '.npy', y_proba)
