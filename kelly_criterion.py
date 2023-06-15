import numpy as np
import pandas as pd

seasons = ['201516', '201617', '201718']
model = 'Team_Model'
book_names = ['Pinnacle Sports', '5Dimes', 'Bookmaker', 'BetOnline', 'Bovada', 'Heritage', 'Intertops', 'YouWager', 'JustBet']

for book in book_names:
    cont = 0
    start = 100
    for season in seasons:

        # Cargamos los datos
        test = np.load('combination/results/' + model +'/test_' + season + '.npy')
        pred = np.load('combination/results/' + model +'/pred_' + season + '.npy')
        proba = np.load('combination/results/' + model +'/proba_' + season + '.npy')
        bets = pd.read_csv('bets/nba_betting_money_line.csv', low_memory=False)
        games = pd.read_csv('combination/csv/entry_data/team_entry/entry_data_' + season + '.csv', low_memory=False)


        # Para cada partido
        for odds, result, game_id in zip(proba, test, games['GAME_ID']):

            # Guardamos las cuotas de la casa de apuestas
            pinnacle = bets[bets['book_name'] == book]

            # Si no hay cuotas de la casa de apuestas, pasamos al siguiente partido
            if(pinnacle[pinnacle['game_id'] == game_id].empty):
                continue

            bet_away = float(pinnacle[pinnacle['game_id'] == game_id]['price1'])
            
            bet_home = float(pinnacle[pinnacle['game_id'] == game_id]['price2'])
            
            # Calculamos las cuotas decimales mediante las cuotas americanas
            if bet_home < 0:
                h_decimal_odds = 100/(bet_home*-1)
            elif bet_home > 0:
                h_decimal_odds = bet_home/100

            if bet_away < 0:
                a_decimal_odds = 100/(bet_away*-1)
            elif bet_away > 0:
                a_decimal_odds = bet_away/100


            # Calculamos el valor de kelly            
            kc = odds[1] - odds[0]/(h_decimal_odds*0.6)
            # Filtramos las apuestas que no nos interesan (cuotas bajas)
            if kc<0 or h_decimal_odds < 1.7:
                continue

            # Limitamos el valor de kelly
            if kc > 0.2:
                kc = 0.2
            
            # Actualizamos el estado de la cartera
            if kc > 0:
                if result == 1:
                    start += start*abs(kc)*h_decimal_odds
                else:
                    start -= start*abs(kc)

            elif kc < 0:
                if result == 0:
                    start += start*abs(kc)*a_decimal_odds
                else:
                    start -= start*abs(kc)

            # Si nos quedamos sin dinero, paramos
            if start < 0.1:
                break

            cont+=1
            
    # Imprimimos los resultados 
    print(start, book, cont)