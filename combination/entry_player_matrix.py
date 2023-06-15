import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

matrices_og = []
seasons = ['201314', '201415', '201516', '201617', '201718', '201819']
folder_teams = 'csv/entry_data/team_entry/'
folder_players = 'csv/player_data/'

total_players = []
for season in seasons:
    total_players += [pd.read_csv(folder_players + 'player_data_' + season + '.csv', low_memory=False)]

total_players = pd.concat(total_players, ignore_index=True)

for season in seasons:

    matriz_teams = pd.read_csv(folder_teams + 'entry_data_' +
                                season + '.csv', low_memory=False)
    matriz_teams.replace(to_replace=['W', 'L'], value=[1, 0], inplace=True)
    
    matriz_players = pd.read_csv(folder_players + 'player_data_' +
                                season + '.csv', low_memory=False)

    players_id = pd.unique(total_players['PLAYER_ID'])
    games_id = pd.unique(matriz_teams['GAME_ID'])

    identity_home_df = pd.DataFrame(0, columns=players_id, index=matriz_teams.index)
    identity_away_df = pd.DataFrame(0, columns=players_id, index=matriz_teams.index)

    for index, game in matriz_teams.iterrows():
        players_in_game = matriz_players[matriz_players['GAME_ID'] == game['GAME_ID']]
        for i, player in players_in_game.iterrows():
            if player['TEAM_ID'] == game['H_TEAM_ID']:
                identity_home_df.at[index, player['PLAYER_ID']] = 1
            else:
                identity_away_df.at[index, player['PLAYER_ID']] = 1
    
    identity_df = pd.concat([identity_home_df.add_prefix('H_'), identity_away_df.add_prefix('A_')], axis=1)
    identity_df['H_WL'] = matriz_teams['H_WL']
    identity_df['GAME_ID'] = matriz_teams['GAME_ID']
    identity_df['GAME_DATE'] = matriz_teams['GAME_DATE']
    identity_df.to_csv('csv/entry_data/player_entry/entry_data_'+ season +'.csv', index=False)