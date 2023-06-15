from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from scipy.stats import loguniform
import numpy as np
import pandas as pd


def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    '''
    Entrena el modelo con todos los datos de entrenamiento y realiza las predicciones sobre el conjunto de test.
    :param X_train: conjunto de entrenamiento
    :param y_train: conjunto con los valores objetivo para X_train
    :param X_test: conjunto de test
    :param y_test: conjunto con los valores objetivo para X_test
    :return: diccionario con los resultados de las predicciones para cada modelo
    '''

    dfs = []

    # Definimos todos los posible modelos a probar
    models = [
        ('LogReg', LogisticRegression(max_iter=1000000)),
        ('RF', RandomForestClassifier()),
        #('KNN', KNeighborsClassifier()),
        #('SVC linear', SVC(kernel='linear', max_iter=10000000)),
        #('SVC rbf', SVC(kernel='rbf', max_iter=10000000)),
        #('GNB', GaussianNB()),
    ]

    names = []

    accuaracy = {}

    ss = MinMaxScaler()

    # Escalamos los datos
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    for name, model in models:

        # Entrenamos el modelo
        clf = model.fit(X_train, y_train.values.ravel())

        # Realizamos las predicciones
        y_pred = clf.predict(X_test)

        y_pred = np.array(y_pred)

        print(name)

        #print(classification_report(y_test, y_pred, target_names=target_names))

        accuaracy[name] = (y_test, y_pred)
        names.append(name)
        
    return accuaracy

def teams_feature_selection(matriz_train, win_train, model, p_features):
    '''
    Realiza la selección de características para el modelo de equipos.
    :param matriz_train: matriz de entrenamiento
    :param win_train: vector con los valores objetivo para matriz_train
    :param model: modelo para realizar la selección de características
    :param p_features: porcentaje de características a seleccionar
    :return: columnas seleccionadas
    '''
    ss = MinMaxScaler()
    X_train = ss.fit_transform(matriz_train)
    m_teams = matriz_train.copy()

    rfe_selector2 = RFE(estimator=model, step = 1, n_features_to_select=p_features)
    rfe_selector2.fit(X_train, win_train)

    m_teams = m_teams[m_teams.columns[rfe_selector2.get_support()]]
    m_teams = m_teams[m_teams.columns]
    all_features = pd.concat([m_teams], axis=1)
    return all_features

def player_feature_selection(matriz_train, win_train, model, p_features):
    '''
    Realiza la selección de características para el modelo de jugaores.
    :param matriz_train: matriz de entrenamiento
    :param win_train: vector con los valores objetivo para matriz_train
    :param model: modelo para realizar la selección de características
    :param p_features: porcentaje de características a seleccionar
    :return: columnas seleccionadas
    '''
    
    ss = MinMaxScaler()
    X_train = ss.fit_transform(matriz_train)

    m_players = matriz_train.copy()

    rfe_selector1 = RFE(estimator=model, step = 50, n_features_to_select=p_features)
    rfe_selector1.fit(X_train, win_train)


    m_players = m_players[m_players.columns[rfe_selector1.get_support()]]

    all_features = pd.concat([m_players], axis=1)

    return all_features

def combination_feature_selection(matriz_train, win_train, model_player, model_teams, p_features_player, p_features_teams):
    '''
    Realiza la selección de características para el modelo combinado.
    :param matriz_train: matriz de entrenamiento
    :param win_train: vector con los valores objetivo para matriz_train
    :param model_player: modelo para realizar la selección de características de jugadores
    :param model_teams: modelo para realizar la selección de características de equipos
    :param p_features_player: porcentaje de características a seleccionar para jugadores
    :param p_features_teams: porcentaje de características a seleccionar para equipos
    :return: columnas seleccionadas
    '''
    ss = MinMaxScaler()
    X_train = ss.fit_transform(matriz_train)

    m_players = matriz_train.iloc[:,:-36]
    m_teams = matriz_train.iloc[:,-36:]


    rfe_selector1 = RFE(estimator=model_player, step = 50, n_features_to_select= p_features_player)
    rfe_selector1.fit(X_train[:,:-36], win_train)

    m_players = m_players[m_players.columns[rfe_selector1.get_support()]]

    if p_features_player != 1:
        rfe_selector2 = RFE(estimator=model_teams, step = 1, n_features_to_select=p_features_teams)
        rfe_selector2.fit(X_train[:,-36:], win_train)

        m_teams = m_teams[m_teams.columns[rfe_selector2.get_support()]]

    all_features = pd.concat([m_players, m_teams], axis=1)

    return all_features


def get_next_season(season):
    year1, year2 = season[:4], season[4:]
    year1, year2 = int(year1), int(year2)
    year1 += 1
    year2 += 1
    next_season = str(year1) + str(year2)
    return next_season


all_seasons = ['201314', '201415', '201516', '201617', '201718']
predict_seasons = ['201617', '201718', '201819']

model_folder = 'results/Team_Model/'
folder = 'csv/entry_data/team_entry/'
accuracy = []


for predict_season in predict_seasons:    
    matrices_og_c = []
    
    # Obtenemos las matrices de entrenamiento
    if predict_season in all_seasons:
        predict_index = all_seasons.index(predict_season)
        seasons = all_seasons[:predict_index]
    else:
        seasons = all_seasons
    
    # En este caso usamos 3 temporadas para entrenar
    seasons = seasons[-3:]
    print(seasons, predict_season)

    for season in seasons:
        matrices_og_c += [pd.read_csv(folder + 'entry_data_' +
                                    season + '.csv', low_memory=False)]


    matriz_pred_c = pd.read_csv(
        folder + 'entry_data_' + predict_season + '.csv', low_memory=False)

    target_names = ['W', 'L']

    proba_tot = []

    matriz_train = pd.concat(matrices_og_c, ignore_index=True)

    # Eliminamos las columnas que no son necesarias
    matriz_train.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                    'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True, errors='ignore')

    matriz_train.drop(['H_OFF_RATING', 'H_DEF_RATING', 'A_OFF_RATING',
                    'A_DEF_RATING'], axis=1, inplace=True, errors='ignore')

    win_train = matriz_train['H_WL']
    matriz_train.drop(['H_WL'], axis=1, inplace=True)

    matriz_pred_c.drop(['SEASON_YEAR', 'H_TEAM_ID', 'H_TEAM_NAME', 'GAME_ID', 'GAME_DATE',
                        'MATCHUP', 'A_TEAM_ID', 'A_TEAM_NAME', 'A_WL'], axis=1, inplace=True, errors='ignore')

    matriz_pred_c.drop(['H_OFF_RATING', 'H_DEF_RATING', 'A_OFF_RATING',
                    'A_DEF_RATING'], axis=1, inplace=True, errors='ignore')


    win_pred = matriz_pred_c['H_WL']
    matriz_pred_c.drop(['H_WL'], axis=1, inplace=True)

    # Descomentar para realizar la selección de características
    """
    #RandomForestClassifier()
    #LogisticRegression(max_iter=100000)
    #SVR(kernel="linear")
    #SVR(kernel="rbf")
    all_features = player_feature_selection(matriz_train, win_train, SVR(kernel="linear"), 0.5)
    all_features = combination_feature_selection(matriz_train, win_train, 
                                                 LogisticRegression(max_iter=100000), SVR(kernel="linear"), 
                                                 p_features_player=0.05, p_features_teams=0.9)

    all_features = teams_feature_selection(matriz_train, win_train, SVR(kernel='linear'), 0.9)

    matriz_train = matriz_train[all_features.columns]
    matriz_pred_c = matriz_pred_c[matriz_train.columns]
    """

    print(f'X train shape: {matriz_train.shape}')
    print(f'X test shape: {matriz_pred_c.shape}')

    # Recogemos los resultados de los modelos
    accuracy += [run_exps(matriz_train, win_train, matriz_pred_c, win_pred)]

for model in accuracy[0].keys():
    y_test = []
    y_pred = [] 
    for iter in accuracy:
        
        y_test += iter[model][0].tolist()
        y_pred += iter[model][1].tolist()


    print('\n', model, '\n')
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'F1: {f1_score(y_test, y_pred)}')
    print(f'AUC-score: {roc_auc_score(y_test, y_pred)}')
    print(f'Average precision score: {average_precision_score(y_test, y_pred)}')
    print(f'Classification report: \n {classification_report(y_test, y_pred)}')