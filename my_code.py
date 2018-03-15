# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# win/loss data
data_dir = 'input/'
df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults2.csv')

# change directory to get player data
data_dir = 'input/Players/'
df_players_2010 = pd.read_csv(data_dir + 'Players_2010.csv')
df_players_2011 = pd.read_csv(data_dir + 'Players_2011.csv')
df_players_2012 = pd.read_csv(data_dir + 'Players_2012.csv')
df_players_2013 = pd.read_csv(data_dir + 'Players_2013.csv')
df_players_2014 = pd.read_csv(data_dir + 'Players_2014.csv')
df_players_2015 = pd.read_csv(data_dir + 'Players_2015.csv')
df_players_2016 = pd.read_csv(data_dir + 'Players_2016.csv')
df_players_2017 = pd.read_csv(data_dir + 'Players_2017.csv')
#df_players_2018 = pd.read_csv(data_dir + 'Players_2018.csv')

df_players = pd.concat([df_players_2010, df_players_2011, df_players_2012, df_players_2013, df_players_2014, df_players_2015, df_players_2016, df_players_2017])

def get_avg_name_length(some_df):
    sum = 0
    size = 0
    current_TeamID = 0
    year = 2010
    temp = []
    for row in some_df.itertuples():
        if row[3] == current_TeamID:
            if row[4] != "TEAM":
                sum += len(row[4])
                size += 1
                year = row[2]
        else:
            try:
                temp_dict = {"Season":year, "TeamID": current_TeamID, "Length":(float(sum/size))}
                temp.append(temp_dict)
                sum = len(row[4])
                size = 1
                current_TeamID = row[3]
            except:
                sum = len(row[4])
                size = 1
                current_TeamID = row[3]
    avg_name_lengths = pd.DataFrame(temp)
    return avg_name_lengths

df_all_players = get_avg_name_length(df_players)

def get_length_difference(key_df, some_df):
    temp = []
    for row in key_df.itertuples():
        season = row[1]
        winning_id = row[3]
        losing_id = row[5]
        winning_team = some_df[(some_df["Season"] == season) & (some_df["TeamID"] == winning_id)]
        winning_team_length = float(winning_team["Length"])
        losing_team = some_df[(some_df["Season"] == season) & (some_df["TeamID"] == losing_id)]
        losing_team_length = float(losing_team["Length"])
        temp_dict = {"Difference": winning_team_length - losing_team_length}
        temp.append(temp_dict)
    name_differences = pd.DataFrame(temp)
    return name_differences

df_differences = get_length_difference(df_tour, df_all_players)

# remove extra info from win/loss data
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)

df_final = df_tour.join(df_differences)

df_wins = pd.DataFrame()
df_wins["LengthDifference"] = df_final["Difference"]
df_wins["Result"] = 1

df_losses = pd.DataFrame()
df_losses["LengthDifference"] = -df_final["Difference"]
df_losses["Result"] = 0

df_predictions = pd.concat((df_wins, df_losses))
print(df_predictions)

X_train = df_predictions.LengthDifference.values.reshape(-1,1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)

logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))

X = np.arange(-10, 10).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]

plt.plot(X, preds)
plt.xlabel('Team1 length - Team2 length')
plt.ylabel('P(Team1 will win)')

data_dir = 'input/'
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

'''
X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed

preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.head()
'''