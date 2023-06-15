import requests
import csv
import json
import pandas as pd

months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september']
years = ['201617', '201718', '201819', '201920', '202021', '202122']
years = ['201314', '201415', '201516', '201617', '201718', '201819', '201920', '202021', '202122']
infolder = ['raw_data/per_month/', 'raw_data/acumulativo_per_month/']
outfolder = ['csv/per_month/', 'csv/acumulativo_per_month/']
'''
for folder in range(len(infolder)):
    for year in years:
        for month in months:
            try:
                f = open(infolder[folder] + year + '/'+ year + '_' + month + '.json', 'r')
            except FileNotFoundError:
                continue
            data = json.load(f)

            season_2022 = pd.DataFrame(0, columns=data['resultSets'][0]['headers'], index=range(0, len(data['resultSets'][0]['rowSet'])))


            for index, game in enumerate(data['resultSets'][0]['rowSet']):
                season_2022.loc[index] = game

            season_2022.to_csv(outfolder[folder] + year + '/'+ year + '_' + month + '.csv', index=False)


'''    
'''
f = open('201516_global.json', 'r')
data = json.load(f)

season_2022 = pd.DataFrame(0, columns=data['resultSets'][0]['headers'], index=range(0, len(data['resultSets'][0]['rowSet'])))


for index, game in enumerate(data['resultSets'][0]['rowSet']):
    season_2022.loc[index] = game

season_2022.to_csv('csv/201516_global.csv', index=False)
'''

for year in years:
    try:
        f = open('raw_data/player_data/player_data_'+ year + '.json', 'r')
    except FileNotFoundError:
        continue
    data = json.load(f)

    season_2022 = pd.DataFrame(0, columns=data['resultSets'][0]['headers'], index=range(0, len(data['resultSets'][0]['rowSet'])))


    for index, game in enumerate(data['resultSets'][0]['rowSet']):
        season_2022.loc[index] = game

    season_2022.to_csv('csv/player_data/player_data_'+ year +'.csv', index=False)