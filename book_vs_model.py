import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

seasons = ['201516', '201617', '201718']
model = 'Team_Model'
book_names = ['Pinnacle Sports', '5Dimes', 'Bookmaker', 'BetOnline', 'Bovada', 'Heritage', 'Intertops', 'YouWager', 'JustBet']

for book in book_names:
    
    start = 1000
    
    for season in seasons:
        cont = 0
        y_pred = []
        y_test = []
        m_pred = []
        
        # Cargamos los datos
        test = np.load('combination/results/' + model +'/test_' + season + '.npy')
        pred = np.load('combination/results/' + model +'/pred_' + season + '.npy')
        proba = np.load('combination/results/' + model +'/proba_' + season + '.npy')
        bets = pd.read_csv('bets/nba_betting_money_line.csv', low_memory=False)
        games = pd.read_csv('combination/csv/entry_data/team_entry/entry_data_' + season + '.csv', low_memory=False)


        for model_pred, result, game_id in zip(pred, test, games['GAME_ID']):
            
            # Guardamos las cuotas de la casa de apuestas
            bookmaker = bets[bets['book_name'] == book]

            if(bookmaker[bookmaker['game_id'] == game_id].empty):
                continue

            bet_away = float(bookmaker[bookmaker['game_id'] == game_id]['price1'])
            
            bet_home = float(bookmaker[bookmaker['game_id'] == game_id]['price2'])
            
            # Calculamos las predicciones de la casa de apuestas
            if bet_home < 0:
                y_pred += [[1]]
            elif bet_home > 0:
                y_pred += [[0]]
            
            y_test += [[result]]
            m_pred += [[model_pred]]
            cont += 1
            
        # Convertimos las listas en arrays e imprimimos los resultados comparando nuestro modelo con las casas
        y_pred = np.concatenate(y_pred)
        y_test = np.concatenate(y_test)
        m_pred = np.concatenate(m_pred)
        print(book, season, accuracy_score(y_test, y_pred), cont)
        print(season, accuracy_score(y_test, m_pred))
