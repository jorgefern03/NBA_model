# getStats.py - Obtains a grouping of stats for any team in the NBA

from nba_api.stats.endpoints import teamdashboardbygeneralsplits, leaguedashteamstats
import time
import pandas as pd
import numpy as np
from customHeaders import customHeaders
from json.decoder import JSONDecodeError


def getStatsForTeam(team_id, startDate, endDate, season):
    """
    Usa la API de NBA para obtener el diccionario que contiene las estadísticas básicas del equipo indicado en un rango de fechas.
    :param team_id: identificador del equipo
    :param startDate: fecha de inicio del rango, en formato 'mm/dd/yyyy'
    :param endDate: fecha de fin del rango, en formato 'mm/dd/yyyy'
    :param season: temporada
    :return: diccionario con las estadísticas básicas del equipo
    """
    # Añadimos esperas para no saturar la web de la NBA
    time.sleep(0.5)

    print(team_id, startDate, endDate, season)

    # Usamos la API de NBA para acceder al diccionario que contiene las estadísticas avanzadas de cada equipo
    try:
        advancedTeamInfo = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=team_id, measure_type_detailed_defense='Advanced', date_from_nullable=startDate, date_to_nullable=endDate, season=season, headers=customHeaders, timeout=100)
        advancedTeamDict  = advancedTeamInfo.get_normalized_dict()
        advancedTeamDashboard = advancedTeamDict['OverallTeamDashboard'][0]

    except JSONDecodeError as e:
        # Si no se puede acceder a las estadísticas avanzadas, devolvemos None
        print("mal")
        return None

    return advancedTeamDashboard

def get_previous_day(game_date):
    """
    Devuelve la fecha del día anterior a la fecha del partido
    :param game_date: fecha del partido
    :return: fecha del día anterior
    """
    game_date = game_date.split('-')
    game_date = game_date[1] + '/' + game_date[2].split('T')[0] + '/' + game_date[0]
    game_date = game_date.split('/')
    game_date = pd.to_datetime(game_date[2] + '-' + game_date[0] + '-' + game_date[1])
    game_date = game_date - pd.Timedelta(days=1)
    game_date = game_date.strftime('%m/%d/%Y')
    return game_date

def complete_first_games(games_df, season, not_stats):
    """
    Completa los datos de los primeros partidos de la temporada con los datos de la temporada anterior
    :param games_df: dataframe con los datos de los partidos
    :param season: temporada
    :param not_stats: columnas que no son estadísticas
    :return: dataframe con los datos completados
    """
    last_seasons = {'2013-14': '2012-13', '2014-15': '2013-14', '2015-16': '2014-15', '2016-17': '2015-16', '2017-18': '2016-17', '2018-19': '2017-18'}

    last_year_response = leaguedashteamstats.LeagueDashTeamStats(measure_type_detailed_defense='Advanced', per_mode_detailed='PerGame', plus_minus='N', pace_adjust='N', rank='N', league_id_nullable='00', season=last_seasons[season], season_type_all_star='Regular Season', po_round_nullable=0, month=0, opponent_team_id=0, team_id_nullable=0, period=0, last_n_games=0, two_way_nullable=0, headers=customHeaders, timeout=100)
    last_year_dict = {d['TEAM_ID']: d for d in last_year_response.get_normalized_dict()['LeagueDashTeamStats']}
    
    for index, game in games_df[games_df['CONTADOR'] == -1].iterrows():
        if game['CONTADOR'] != -1:
            continue
        for column in games_df.columns:
            if column not in not_stats:
                games_df.loc[index, column] = last_year_dict[game['TEAM_ID']][column]
        games_df.loc[index, 'CONTADOR'] = 0

    return games_df

not_stats = ['GAME_ID', 'TEAM_ID', 'SEASON_YEAR', 'GAME_DATE', 'MATCHUP', 'TEAM_NAME', 'WL', 'MIN', 'CONTADOR']
years = ['201314', '201415', '201516', '201617', '201718', '201819']
#years = ['201314']
#years = ['201415']
#years = ['201516']
#years = ['201617']
#years = ['201718']
#years = ['201819']



for year in years:
    contador = 0
    # Leemos el archivo que contiene todos los partidos de la temporada
    try:
        games_df = pd.read_csv('csv/daybyday/' + year + '_stats.csv', low_memory=False)
    except:
        games_df = pd.read_csv('csv/seasons/season_'+ year + '.csv', low_memory=False)
        games_df['CONTADOR'] = -1
        games_df.drop(['TEAM_ABBREVIATION', 'MIN', 'E_OFF_RATING', 'E_DEF_RATING', 'E_NET_RATING', 'E_PACE', 'PACE_PER40', 'W_RANK', 'L_RANK'], axis=1, inplace=True)

    season = year[0:4] + '-' + year[4:6]
    start_date = '09/01/'+ year[0:4] 
    
    # Para cada partido
    for index, game in games_df.iterrows():

        # Obtenemos la fecha del día anterior al partido
        game_date = get_previous_day(game['GAME_DATE'])

        # Si 'CONTADOR' es distinto de -1, significa que ya se han obtenido los datos de ese partido
        if game['CONTADOR'] != -1:
            continue
        try:
            # Hacemos la petición
            previous_stast = getStatsForTeam(game['TEAM_ID'], start_date, game_date, game['SEASON_YEAR']) 
        except IndexError:
            continue
        contador +=1
        # Cambiamos el valor de 'CONTADOR' para indicar que ya se han obtenido los datos de ese partido
        games_df.loc[index, 'CONTADOR'] = contador

        # Escribimos los datos en el dataframe
        for column in games_df.columns:
            if column not in not_stats:
                if previous_stast is not None:
                    games_df.loc[index, column] = previous_stast[column]
                else:
                    games_df.loc[index, 'CONTADOR'] = -1

        # Guardamos el dataframe en un archivo csv
        games_df.to_csv('csv/daybyday/' + year + '_stats.csv', index=False)
    
    # Completamos los datos de los primeros partidos de la temporada con los datos de la temporada anterior
    games_df = complete_first_games(games_df, season, not_stats)

    # Guardamos el dataframe en un archivo csv
    games_df.to_csv('csv/daybyday/' + year + '_stats.csv', index=False)

    # Eliminamos las columnas que no son estadísticas referentes a los partidos
    games_df.drop(list(games_df.filter(regex = 'RANK')), axis = 1, inplace = True) 
    games_df.drop(['CONTADOR'], axis=1, inplace=True)

    # Creamos dos dataframes, uno con los datos de los partidos de local y otro con los de visitante y 
    # añadimos los prefijos H_ y A_ respectivamente
    games_df_home = games_df[games_df.MATCHUP.str.contains('vs')].add_prefix('H_')
    games_df_away = games_df[games_df.MATCHUP.str.contains('@')].add_prefix('A_')
    
    # Renombramos las columnas para que coincidan con el nombre de las columnas del otro dataframe
    games_df_home.rename(columns={'H_GAME_ID':'GAME_ID', 'H_SEASON_YEAR': 'SEASON_YEAR','H_GAME_DATE': 'GAME_DATE', 'H_MATCHUP' : 'MATCHUP'}, inplace=True)
    games_df_away.rename(columns={'A_GAME_ID':'GAME_ID'}, inplace=True)
    # Eliminamos las columnas redundantes
    games_df_away.drop(['A_SEASON_YEAR', 'A_GAME_DATE', 'A_MATCHUP'], axis=1, inplace=True)

    # Unimos los dos dataframes en uno solo y lo guardamos en un csv
    entry_matrix = pd.merge(games_df_home, games_df_away,on='GAME_ID')
    entry_matrix.to_csv('csv/matchups/matchups_'+ year +'.csv', index=False)
    