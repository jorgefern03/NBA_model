import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np

def get_previous_day(game_date):
    """
    Devuelve la fecha del dia anterior a la fecha del partido
    :param game_date: fecha del partido
    :return: fecha del dia anterior
    """
    game_date = game_date.split('-')
    game_date = game_date[1] + '/' + game_date[2].split('T')[0] + '/' + game_date[0]
    game_date = game_date.split('/')
    game_date = pd.to_datetime(game_date[2] + '-' + game_date[0] + '-' + game_date[1])
    game_date = game_date - pd.Timedelta(days=1)
    game_date = game_date.strftime('%m/%d/%Y')
    return game_date

def get_day(game_date):
    """
    Devuelve la fecha en formato mm/dd/yyyy
    :param game_date: fecha del partido
    :return: fecha en formato mm/dd/yyyy
    """
    game_date = game_date.split('-')
    game_date = game_date[1] + '/' + game_date[2].split('T')[0] + '/' + game_date[0]
    game_date = game_date.split('/')
    game_date = pd.to_datetime(game_date[2] + '-' + game_date[0] + '-' + game_date[1])
    game_date = game_date.strftime('%m/%d/%Y')
    return game_date

def add_elo_rating(df, k):
    """
    Calcula el elo rating de cada equipo en cada partido
    :param df: dataframe con los partidos
    :param k: constante de elo
    :return: dataframe con los partidos y el elo rating de cada equipo
    """

    df['A_ELO_RATING'] = df['H_ELO_RATING'] = 1500
    n_game = 0
    team_ratings = {}
    last_game_result = {}
    for index, game in df.iterrows():
        # Calculamos el elo del equipo
        if game['H_TEAM_NAME'] not in team_ratings.keys():
            team_ratings[game['H_TEAM_NAME']] = 1500
        if game['A_TEAM_NAME'] not in team_ratings.keys():
            team_ratings[game['A_TEAM_NAME']] = 1500

        if game['H_TEAM_NAME'] not in last_game_result.keys():
            last_game_result[game['H_TEAM_NAME']] = game['H_WL']
        else:
            r_0 = team_ratings[game['H_TEAM_NAME']]
            r_opp = team_ratings[game['A_TEAM_NAME']]
            w = last_game_result[game['H_TEAM_NAME']]
            w_e = 1/(1+pow(10, (r_opp-r_0)/400))
            df.loc[index, 'H_ELO_RATING'] = r_0 + k*(float(w)-float(w_e))
            last_game_result[game['H_TEAM_NAME']] = game['H_WL']
        
        if game['A_TEAM_NAME'] not in last_game_result.keys():
            last_game_result[game['A_TEAM_NAME']] = game['A_WL']
        else:
            r_0 = team_ratings[game['A_TEAM_NAME']]
            r_opp = team_ratings[game['H_TEAM_NAME']]
            w = last_game_result[game['A_TEAM_NAME']]
            w_e = 1/(1+pow(10, (r_opp-r_0)/400))
            df.loc[index, 'A_ELO_RATING'] = r_0 + k*(float(w)-float(w_e))
            last_game_result[game['A_TEAM_NAME']] = game['A_WL']

        team_ratings[game['H_TEAM_NAME']] = df.loc[index, 'H_ELO_RATING']
        team_ratings[game['A_TEAM_NAME']] = df.loc[index, 'A_ELO_RATING']
        n_game += 1
        
    return df

def add_back_to_back(df):
    """
    Calcula si un equipo juega un partido back to back
    :param df: dataframe con los partidos
    :return: dataframe con los partidos y si cada equipo juega un partido back to back
    """
    df['H_BACK_TO_BACK'] = df['A_BACK_TO_BACK'] = np.nan
    n_game = 0
    for team in df['H_TEAM_ID'].unique():
        games_by_team = df.query('H_TEAM_ID == @team or A_TEAM_ID == @team')

        n_game = 0
        prev_index = 0
        for i, game in games_by_team.iterrows():

            if game['H_TEAM_ID'] == team:
                if n_game != 0:
                    if get_day(games_by_team.loc[prev_index, 'GAME_DATE']) == get_previous_day(game['GAME_DATE']):
                        games_by_team.loc[i, 'H_BACK_TO_BACK'] = 1
                    else:
                        games_by_team.loc[i, 'H_BACK_TO_BACK'] = 0
                else:
                    games_by_team.loc[i, 'H_BACK_TO_BACK'] = 0
            else:
                if n_game != 0:
                    if get_day(games_by_team.loc[prev_index, 'GAME_DATE']) == get_previous_day(game['GAME_DATE']):
                        games_by_team.loc[i, 'A_BACK_TO_BACK'] = 1
                    else:
                        games_by_team.loc[i, 'A_BACK_TO_BACK'] = 0
                else:
                    games_by_team.loc[i, 'A_BACK_TO_BACK'] = 0

            n_game += 1
            prev_index = i
        df.fillna(games_by_team, inplace=True)

    return df

def add_win_rate(df, last_n):
    """
    Calcula el win rate de los últimos n partidos de cada equipo
    :param df: dataframe con los partidos
    :param last_n: número de partidos a tener en cuenta
    :return: dataframe con los partidos y el win rate de los últimos n partidos de cada equipo
    """

    df['H_WIN_RATE_L10'] = df['A_WIN_RATE_L10'] = np.nan
    n_game = 0
    for team in df['H_TEAM_ID'].unique():
        games_by_team = df.query('H_TEAM_ID == @team or A_TEAM_ID == @team')

        n_game = 0
        wins = []
        for i, game in games_by_team.iterrows():
            
            
            if game['H_TEAM_ID'] == team:
                
                #Calculamos el win rate de los últimos 10 partidos
                if n_game >= last_n:
                    games_by_team.loc[i, 'H_WIN_RATE_L10'] = wins[n_game-last_n:n_game].count(1)/last_n

                elif n_game < last_n and n_game != 0:   
                    games_by_team.loc[i, 'H_WIN_RATE_L10'] = wins[0:n_game].count(1)/n_game

                else:
                    games_by_team.loc[i, 'H_WIN_RATE_L10'] = 0.5
                
                wins += [game['H_WL']]
            else:
                
                #Calculamos el win rate de los últimos 10 partidos
                if n_game >= last_n:
                    games_by_team.loc[i, 'A_WIN_RATE_L10'] = wins[n_game-last_n:n_game].count(1)/last_n
                elif n_game < last_n and n_game != 0:   
                    games_by_team.loc[i, 'A_WIN_RATE_L10'] = wins[0:n_game].count(1)/n_game
                else:
                    games_by_team.loc[i, 'A_WIN_RATE_L10'] = 0.5
                
                wins += [game['A_WL']]

            n_game += 1
        df.fillna(games_by_team, inplace=True)

    return df

def add_bookmaker_odds(df):
    """
    Añade las cuotas de las casas de apuestas a cada partido
    :param df: dataframe con los partidos
    :return: dataframe con los partidos y las cuotas de las casas de apuestas
    """
    
    bets = pd.read_csv('csv/bets/nba_betting_money_line.csv', low_memory=False)
    df['H_BOOKMAKER_WIN'] = 0
    book = '5Dimes'
    for game_id in df['GAME_ID']:

            pinnacle = bets[bets['book_name'] == book]

            if(pinnacle[pinnacle['game_id'] == game_id].empty):
                continue

            bet_home = float(pinnacle[pinnacle['game_id'] == game_id]['price2'])
            
            if bet_home < 0:
                df.loc[df['GAME_ID'] == game_id, 'H_BOOKMAKER_WIN'] = bet_home
            elif bet_home > 0:
                df.loc[df['GAME_ID'] == game_id, 'H_BOOKMAKER_WIN'] = bet_home

    return df


if __name__ == '__main__':

    seasons = ['201314', '201415', '201516', '201617', '201718', '201819']
    infolder = 'csv/matchups/matchups_'
    outfolder = 'csv/entry_data/entry_data_'

    for season in seasons:
        matchups = pd.read_csv(infolder + season + '.csv', low_memory=False)
        matchups = matchups.reindex(index=matchups.index[::-1])
        matchups.replace(to_replace=['W', 'L'], value=[1, 0], inplace=True)
        matchups = add_win_rate(matchups, 10)
        matchups = add_elo_rating(matchups, 20)
        matchups = add_back_to_back(matchups)
        matchups = add_bookmaker_odds(matchups)

        matchups.to_csv(outfolder + season + '.csv', index=False)
     



