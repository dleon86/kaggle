# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import random
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV

from kaggle_titanic.get_param_dist import get_param_dist

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
cwd = os.getcwd()

for dirname, _, filenames in os.walk(cwd):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session]

# use glob to get all the files in the directory
files = glob.glob(cwd + "\\**\\**\\*.csv")  # for debugging

# set the random seed for numpy and sklearn
seed = 42

# load the files into a dataframe
Xtrain = pd.read_csv(files[2])
Xtest = pd.read_csv(files[1])

# split ytrain from Xtrain['Survived']
ytrain = Xtrain['Survived']
Xtrain.drop('Survived', axis=1, inplace=True)

# Check for missing values
print(Xtrain.isnull().sum())
print(Xtest.isnull().sum())

# Drop Name, Ticket, Cabin, and Embarked
Xtrain.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
Xtest.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Fill missing values in Age with the median
Xtrain['Age'].fillna(Xtrain['Age'].median(), inplace=True)
Xtest['Age'].fillna(Xtest['Age'].median(), inplace=True)

# Fill missing values in Fare with the mean
Xtrain['Fare'].fillna(Xtrain['Fare'].mean(), inplace=True)
Xtest['Fare'].fillna(Xtest['Fare'].mean(), inplace=True)


#%%

# Create a new feature FamilySize
Xtrain['FamilySize'] = Xtrain['SibSp'] + Xtrain['Parch'] + 1
Xtest['FamilySize'] = Xtest['SibSp'] + Xtest['Parch'] + 1

# Create a new feature IsAlone
Xtrain['IsAlone'] = 1
Xtrain.loc[Xtrain['FamilySize'] > 1, 'IsAlone'] = 0
# Xtrain['IsAlone'].loc[Xtrain['FamilySize'] > 1] = 0
Xtest['IsAlone'] = 1
Xtest.loc[Xtest['FamilySize'] > 1, 'IsAlone'] = 0
# Xtest['IsAlone'].loc[Xtest['FamilySize'] > 1] = 0

# Drop SibSp, Parch, and PassengerId
Xtrain.drop(['SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True)
Xtest.drop(['SibSp', 'Parch', 'PassengerId'], axis=1, inplace=True)

# convert categorical columns to ordinal columns using the one hot encoder
from sklearn.preprocessing import OneHotEncoder

# Select the columns for one hot encoding
ohe_cols = ['Pclass', 'Sex', 'IsAlone']

# Create the one hot encoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

# fit the encoder to the data
encoder.fit(Xtrain[ohe_cols])

# transform the data
Xtrain_ohe = encoder.transform(Xtrain[ohe_cols])
Xtest_ohe  = encoder.transform(Xtest[ohe_cols])

# merge the dataframes
Xtrain = pd.concat([Xtrain, pd.DataFrame(Xtrain_ohe, columns=encoder.get_feature_names(ohe_cols))], axis=1)
Xtest  = pd.concat([Xtest,  pd.DataFrame(Xtest_ohe,  columns=encoder.get_feature_names(ohe_cols))], axis=1)

# drop the original columns
Xtrain = Xtrain.drop(ohe_cols, axis=1)
Xtest  =  Xtest.drop(ohe_cols, axis=1)



# #%%
#
# # check the data
# print(Xtrain.head())
# print(Xtest.head())
# print(ytrain.head())
#
# #%%
#
# # check the shape of the data
# print(Xtrain.shape)
# print(Xtest.shape)
# print(ytrain.shape)
#
# #%%
# # check the data types
# print(Xtrain.dtypes)
# print(Xtest.dtypes)
# print(ytrain.dtypes)
#

#%%
# set up a pipeline to test different models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# #%%
# set up the pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(svd_solver='full')),  ('model', None)])

n_iter_search = 3

full_model_list = ['LR', 'RFC', 'SVC', 'KNC', 'GNB', 'DTC', 'GBC', 'ABC', 'BC', 'ETC', 'MLPC', 'LDA', 'QDA']

model_list = random.sample(full_model_list, k=4)  #[full_model_list[i] for i in [3,2,7,9]]
print(model_list)


param_dists = get_param_dist(model_list, n_iter_search, pca=True)

grid = RandomizedSearchCV(pipe, param_dists, cv=3, scoring='accuracy', verbose=2, n_jobs=8, n_iter=n_iter_search,
                          random_state=seed, return_train_score=True, refit=True, error_score=np.nan)

grid.fit(Xtrain, ytrain)
y_pred = grid.predict(Xtest)

# generate the results
from kaggle_titanic.report import get_results_df, plot_results, extract_model_abbreviations, get_filename



#%%
# print the best parameters
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
# print(grid.best_index_)
# print(grid.cv_results_)


results_df = get_results_df(grid)

# print the results


# plot the results
# plot_results(results_df, model_list, n_iter_search)
#%%

# save the results
print(model_list)

filename = get_filename(model_list, n_iter_search)
pickle.dump(grid, open(filename, 'wb'))


# #%%
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # set up scoring functions
# scoring = {'roc_auc': 'roc_auc', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}
#
# # def plot_results(results_df, model_list, n_iter_search, scoring=scoring):
# # get the number of scoring metrics
# n_scoring = len(scoring)
# # get the number of models
# n_models = len(model_list)
# # get the number of parameters
# n_params = len(results_df.columns) - n_scoring
# # set up the figure
# fig, axes = plt.subplots(nrows=n_scoring, ncols=n_params, figsize=(n_params*5, n_scoring*5))
# # plot the results
# for i, metric in enumerate(scoring):
#     print(metric)
#     for j, param in enumerate(results_df.columns[:-n_scoring]):
#         print(param)
#         sns.barplot(x=results_df.index, y=results_df[param], ax=axes[i, j])
#         axes[i, j].set_title(f'{param} vs {metric} ({model_list[j]}, {n_iter_search} iterations)')
#         axes[i, j].set_xlabel(param)
#         axes[i, j].set_ylabel(metric)
# plt.show()
# # save the figure in the figures folder
# fig.savefig('./figures/results.png')
# # fig.savefig('./TabularPlaygroundSeptember2022//figures/results.png')