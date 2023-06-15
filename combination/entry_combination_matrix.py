import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == '__main__':

    seasons = ['201314', '201415', '201516', '201617', '201718', '201819']
    team_folder = 'csv/entry_data/team_entry/entry_data_'
    player_folder = 'csv/entry_data/player_entry/entry_data_'
    outfolder = 'csv/entry_data/combination_entry/entry_data_'

    for season in seasons:
        players = pd.read_csv(player_folder + season + '.csv', low_memory=False)
        teams = pd.read_csv(team_folder + season + '.csv', low_memory=False)
        players.drop(['GAME_DATE', 'H_WL'], axis=1, inplace=True)
        # Combinamos los dos dataframes
        matriz_entrada = pd.merge(players, teams, on='GAME_ID', how='inner')
        matriz_entrada.to_csv(outfolder + season + '.csv', index=False)



