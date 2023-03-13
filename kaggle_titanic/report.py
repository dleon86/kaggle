# This is the code for scoring the models and assmembling the results:
# want roc_auc, accuracy, precision, recall, f1, confusion matrix
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#%%
# set up scoring functions
scoring = {'roc_auc': 'roc_auc', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 'f1': 'f1'}

def extract_model_abbreviations(grid):
    model_names = []
    for param_dist in grid.param_distributions:
        model_name = param_dist['model'].__class__.__name__
        abbreviated_name = "".join([char for char in model_name if char.isupper()])
        if abbreviated_name not in model_names:
            model_names.append(abbreviated_name)
    return model_names

# def extract_model_abbreviations(grid):
#     model_abbreviations = set()
#     for param_dict in grid.param_distributions:
#         if 'model_abbreviation' in param_dict:
#             model_abbreviations.add(param_dict['model_abbreviation'])
#     return sorted(list(model_abbreviations))
#
# # def extract_model_abbreviations(grid):
# #     model_names = []
# #     for param_dist in grid.param_distributions:
# #         model_name = param_dist['model'][0].__class__.__name__
# #         abbreviated_name = ""
# #         for char in model_name:
# #             if char.isupper():
# #                 abbreviated_name += char
# #         model_names.append(abbreviated_name)
# #     return model_names


def get_filename(model_names, n_iter_search):
    initials = ''.join([name.upper()+'_' for name in model_names])
    filename = f'input/models/titanic_model_{initials}{n_iter_search}.pkl'
    return filename

def get_model_list():
    return ['LR', 'RFC', 'SVC', 'KNC', 'GNB', 'DTC', 'GBC', 'ABC', 'BC', ' ETC', 'MLPC', 'LDA', 'QDA']


# make a function that returns the abbreviations for the models given a list of models and a list of abbreviations
def get_abbreviations(model_list):
    model_dict = {
        'LogisticRegression': 'LR',
        'RandomForestClassifier': 'RFC',
        'SVC': 'SVC',
        'KNeighborsClassifier': 'KNC',
        'GaussianNB': 'GNB',
        'DecisionTreeClassifier': 'DTC',
        'GradientBoostingClassifier': 'GBC',
        'AdaBoostClassifier': 'ABC',
        'BaggingClassifier': 'BC',
        'ExtraTreesClassifier': 'ETC',
        'MLPClassifier': 'MLPC',
        'LinearDiscriminantAnalysis': 'LDA',
        'QuadraticDiscriminantAnalysis': 'QDA'
    }
    return [model_dict[model] for model in model_list]


def rand_model_selector(n):
    model_list = get_model_list()
    return np.random.choice(model_list, size=n, replace=False)


# make a function that takes in the random search object and returns a dataframe with the results for each model in
# the random search object
def get_results_df(grid):
    param_dicts = grid.cv_results_['params']
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    model_list = [param_dict['model'].__class__.__name__ for param_dict in param_dicts]
    model_abbreviations = get_abbreviations(model_list)
    scoring = grid.scorer_
    # results_df = pd.DataFrame(index=model_abbreviations, columns=list(param_dicts[0].keys())[:-1] + list(scoring.keys()))
    results_df = pd.DataFrame(index=model_abbreviations, columns=list(param_dicts[0].keys())[:-1] + list(grid.scorer_.keys()))

    for param_dict, mean, std in zip(param_dicts, means, stds):
        model_abbreviation = get_abbreviations([param_dict['model'].__class__.__name__])[0]
        result = {k: v for k, v in param_dict.items() if k != 'model'}
        result.update({'mean': mean, 'std': std})
        results_df.loc[model_abbreviation] = pd.Series(result)
    return results_df


# def get_results_df(grid, scoring='accuracy'):
#     # get the model abbreviations
#     model_abbreviations = extract_model_abbreviations(grid)
#     # get the number of models
#     n_models = len(model_abbreviations)
#     # get the number of parameters
#     n_params = len(grid.param_distributions[0].keys()) - 1
#
#     # check if scoring is a string
#     if isinstance(scoring, str):
#         # convert it to a dictionary using get_scorer
#         scoring = {scoring: get_scorer(scoring)}
#
#     # get the number of scoring metrics
#     n_scoring = len(scoring)
#     # set up the results dataframe
#     results_df = pd.DataFrame(index=model_abbreviations,
#                               columns=list(grid.param_distributions[0].keys())[:-1] + list(scoring.keys()))
#     # fill in the results dataframe
#     for i, param_dict_idx in enumerate(range(len(grid.param_distributions))):
#         param_dict = grid.param_distributions[param_dict_idx]
#         model = param_dict['model_abbreviation']
#         for j, param in enumerate(param_dict.keys()[:-1]):
#             results_df.loc[model, param] = param_dict[param]
#         for k, metric in enumerate(scoring.keys()):
#             results_df.loc[model, metric] = grid.cv_results_[f'mean_test_{metric}'][param_dict_idx]
#     return results_df
#
# # def get_results_df(grid, scoring=scoring):
# #     # get the model abbreviations
# #     model_abbreviations = extract_model_abbreviations(grid)
# #     # get the number of models
# #     n_models = len(model_abbreviations)
# #     # get the number of parameters
# #     n_params = len(grid.param_distributions[0].keys()) - 1
# #     # get the number of scoring metrics
# #     n_scoring = len(scoring)
# #     # set up the results dataframe
# #     results_df = pd.DataFrame(index=model_abbreviations, columns=list(grid.param_distributions[0].keys())[:-1] + list(scoring.keys()))
# #     # fill in the results dataframe
# #     for i, model in enumerate(model_abbreviations):
# #         param_dict_idx_list = [idx for idx, d in enumerate(grid.param_distributions) if d.get('model_abbreviation') == model]
# #         if len(param_dict_idx_list) == 0:
# #             continue
# #         param_dict_idx = param_dict_idx_list[0]
# #         param_dict = grid.param_distributions[param_dict_idx]
# #         for j, param in enumerate(param_dict.keys()[:-1]):
# #             results_df.iloc[i, j] = param_dict[param]
# #         for k, metric in enumerate(scoring.keys()):
# #             results_df.iloc[i, n_params+k] = grid.cv_results_[f'mean_test_{metric}'][param_dict_idx]
# #     return results_df
# #
# #
# # # def get_results_df(grid):
# # #     # get the model abbreviations
# # #     model_abbreviations = extract_model_abbreviations(grid)
# # #     # get the number of models
# # #     n_models = len(model_abbreviations)
# # #     # get the number of parameters
# # #     n_params = len(grid.param_distributions[0].keys()) - 1
# # #     # get the number of scoring metrics
# # #     n_scoring = len(scoring)
# # #     # set up the results dataframe
# # #     results_df = pd.DataFrame(index=model_abbreviations, columns=list(grid.param_distributions[0].keys())[:-1] + list(scoring.keys()))
# # #     # fill in the results dataframe
# # #     for i, model in enumerate(model_abbreviations):
# # #         for j, param in enumerate(grid.param_distributions[i].keys()[:-1]):
# # #             results_df.iloc[i, j] = grid.param_distributions[i][param]
# # #         for k, metric in enumerate(scoring.keys()):
# # #             results_df.iloc[i, n_params+k] = grid.cv_results_[f'mean_test_{metric}'][i]
# # #     return results_df


def print_results(grid, scoring):
    # get the results dataframe
    results_df = get_results_df(grid, scoring)
    # print the results dataframe
    print(results_df)


# make a function that plots all of the results for each model in the random search object
def plot_results(results_df, model_list, n_iter_search, scoring=scoring):
    # get the number of scoring metrics
    n_scoring = len(scoring)
    # get the number of models
    n_models = len(model_list)
    # get the number of parameters
    n_params = len(results_df.columns) - n_scoring
    # set up the figure
    fig, axes = plt.subplots(nrows=n_scoring, ncols=n_params, figsize=(n_params*5, n_scoring*5))
    # plot the results
    for i, metric in enumerate(scoring):
        for j, param in enumerate(results_df.columns[:-n_scoring]):
            sns.barplot(x=results_df.index, y=results_df[param], ax=axes[i, j])
            axes[i, j].set_title(f'{param} vs {metric} ({model_list[j]}, {n_iter_search} iterations)')
            axes[i, j].set_xlabel(param)
            axes[i, j].set_ylabel(metric)
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/results.png')


# make a function that plots the confusion matrix for each model in the random search object
def plot_confusion_matrices(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the confusion matrices
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # make predictions
        yhat = grid.predict(xval)
        # get the confusion matrix
        cm = confusion_matrix(yval, yhat)
        # plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i])
        axes[i].set_title(f'{model} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/confusion_matrices.png')

# make a function that plots the roc_auc curve for each model in the random search object
def plot_roc_auc_curves(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the roc_auc curves
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # make predictions
        yhat = grid.predict(xval)
        # get the roc_auc score
        roc_auc = roc_auc_score(yval, yhat)
        # plot the roc_auc curve
        sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[i])
        sns.lineplot(x=[0, 1], y=[0, 1], ax=axes[i])
        axes[i].set_title(f'{model} ROC AUC Curve')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/roc_auc_curves.png')

# make a function that plots the precision recall curve for each model in the random search object
def plot_precision_recall_curves(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the precision recall curves
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # make predictions
        yhat = grid.predict(xval)
        # get the precision recall score
        precision = precision_score(yval, yhat)
        recall = recall_score(yval, yhat)
        # plot the precision recall curve
        sns.lineplot(x=[0, 1], y=[precision, precision], ax=axes[i])
        sns.lineplot(x=[recall, recall], y=[0, 1], ax=axes[i])
        axes[i].set_title(f'{model} Precision Recall Curve')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/precision_recall_curves.png')

# make a function that plots the learning curve for each model in the random search object
def plot_learning_curves(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the learning curves
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # get the learning curve
        train_sizes, train_scores, test_scores = learning_curve(grid, xtrain, ytrain, cv=5)
        # plot the learning curve
        sns.lineplot(x=train_sizes, y=train_scores.mean(axis=1), ax=axes[i])
        sns.lineplot(x=train_sizes, y=test_scores.mean(axis=1), ax=axes[i])
        axes[i].set_title(f'{model} Learning Curve')
        axes[i].set_xlabel('Training Set Size')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/learning_curves.png')


def learning_curve(grid, xtrain, ytrain, cv=5):
    # get the training set sizes
    train_sizes = np.linspace(0.1, 1.0, 10)
    # get the training and validation scores
    train_sizes, train_scores, validation_scores = learning_curve(grid, xtrain, ytrain, train_sizes=train_sizes, cv=cv)
    # return the training set sizes, training scores, and validation scores
    return train_sizes, train_scores, validation_scores


# make a function that plots the validation curve for each model in the random search object
def plot_validation_curves(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the validation curves
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # get the validation curve
        train_scores, test_scores = validation_curve(grid, xtrain, ytrain, param_name='max_depth', param_range=range(1, 21), cv=5)
        # plot the validation curve
        sns.lineplot(x=range(1, 21), y=train_scores.mean(axis=1), ax=axes[i])
        sns.lineplot(x=range(1, 21), y=test_scores.mean(axis=1), ax=axes[i])
        axes[i].set_title(f'{model} Validation Curve')
        axes[i].set_xlabel('Max Depth')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/validation_curves.png')


def validation_curve(grid, xtrain, ytrain, param_name='max_depth', param_range=range(1, 21), cv=5):
    # get the training and validation scores
    train_scores, validation_scores = validation_curve(grid, xtrain, ytrain, param_name=param_name, param_range=param_range, cv=cv)
    # return the training scores and validation scores
    return train_scores, validation_scores


def plot_calibration_curves(grid, scoring, xtrain, ytrain, xval, yval):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the calibration curves
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # make predictions
        yhat = grid.predict(xval)
        # get the calibration curve
        prob_true, prob_pred = calibration_curve(yval, yhat, n_bins=10)
        # plot the calibration curve
        sns.lineplot(x=prob_pred, y=prob_true, ax=axes[i])
        axes[i].set_title(f'{model} Calibration Curve')
        axes[i].set_xlabel('Predicted Probability')
        axes[i].set_ylabel('True Probability')
        axes[i].set_ylim(0, 1)
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/calibration_curves.png')


def calibration_curve(yval, yhat, n_bins=10):
    # get the true probabilities
    prob_true = np.array([np.mean(yval == i) for i in range(n_bins)])
    # get the predicted probabilities
    prob_pred = np.array([np.mean(yhat == i) for i in range(n_bins)])
    # return the true probabilities and predicted probabilities
    return prob_true, prob_pred


def plot_feature_importance(grid, xtrain, ytrain):
    # get the model abbreviations
    model_abbreviations = extract_model_abbreviations(grid)
    # get the number of models
    n_models = len(model_abbreviations)
    # set up the figure
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(5, n_models*5))
    # plot the feature importance
    for i, model in enumerate(model_abbreviations):
        # fit the model
        grid.fit(xtrain, ytrain)
        # get the feature importance
        feature_importance = grid.best_estimator_.feature_importances_
        # plot the feature importance
        sns.barplot(x=feature_importance, y=xtrain.columns, ax=axes[i])
        axes[i].set_title(f'{model} Feature Importance')
        axes[i].set_xlabel('Importance')
        axes[i].set_ylabel('Feature')
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/feature_importance.png')


def feature_importance(grid, xtrain, ytrain):
    # fit the model
    grid.fit(xtrain, ytrain)
    # get the feature importance
    feature_importance = grid.best_estimator_.feature_importances_
    # return the feature importance
    return feature_importance


def plot_feature_correlation(xtrain, ytrain):
    # get the correlation matrix
    correlation_matrix = xtrain.corr()
    # set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, ax=ax)
    plt.show()
    # save the figure in the figures folder
    fig.savefig('./TabularPlaygroundSeptember2022//figures/feature_correlation.png')


def feature_correlation(xtrain, ytrain):
    # get the correlation matrix
    correlation_matrix = xtrain.corr()
    # return the correlation matrix
    return correlation_matrix

def plot_results(results_df, model_list, n_iter_search):
    plot_results(results_df, model_list, n_iter_search)
    plot_roc_auc_curves(results_df, model_list, n_iter_search)
    plot_confusion_matrices(results_df, model_list, n_iter_search)
    plot_learning_curves(results_df, model_list, n_iter_search)
    plot_validation_curves(results_df, model_list, n_iter_search)
    plot_precision_recall_curves(results_df, model_list, n_iter_search)
    plot_feature_importance(results_df, model_list, n_iter_search)
    plot_feature_correlation(results_df, model_list, n_iter_search)
