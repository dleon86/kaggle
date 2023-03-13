from scipy.stats import uniform, randint
from sklearn.utils.fixes import loguniform

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_param_dist(models, n_iter_search=100, pca=False):
    param_dists = []
    for model in models:
        if 'LR' in model:
            param_dists.append( {
            'model': [LogisticRegression(max_iter=5000)],
            'model__penalty': ['l1', 'l2'],
            'model__C': list(loguniform(1e-4, 1e4).rvs(n_iter_search)),
            'model__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        })
        elif 'RFC' in model:
            param_dists.append( {
            'model': [RandomForestClassifier()],
            'model__n_estimators': randint(10, 1000),
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': randint(2, 11),
            'model__min_samples_split': randint(2, 11),
            'model__min_samples_leaf': randint(1, 11),
            'model__max_features': ['sqrt', 'log2']
        })
        elif 'SVC' in model:
            param_dists.append( {
            'model': [SVC()],
            'model__C': loguniform(1e-4, 1e4),
            'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__gamma': ['scale', 'auto'] + list(loguniform(1e-4, 1e-1).rvs(n_iter_search)),
            'model__degree': randint(2, 11),
            'model__coef0': uniform(0, 10)
            })
        elif 'KNC' in model:
            param_dists.append( {
            'model': [KNeighborsClassifier()],
            'model__n_neighbors': randint(1, 20),
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'model__leaf_size': randint(10, 50)
            })
        elif 'GNB' in model:
            param_dists.append( {
            'model': [GaussianNB()],
            })
        elif 'DTC' in model:
            param_dists.append( {
            'model': [DecisionTreeClassifier()],
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': randint(2, 11),
            'model__min_samples_split': randint(2, 11),
            'model__min_samples_leaf': randint(1, 11),
            'model__max_features': ['sqrt', 'log2']
            })
        elif 'GBC' in model:
            param_dists.append( {
            'model': [GradientBoostingClassifier()],
            'model__learning_rate': loguniform(1e-4, 1e0),
            'model__n_estimators': randint(10, 1000),
            'model__max_depth': randint(2, 11),
            'model__min_samples_split': randint(2, 11),
            'model__min_samples_leaf': randint(1, 11),
            'model__max_features': ['sqrt', 'log2']
            })
        elif 'ABC' in model:
            param_dists.append( {
            'model': [AdaBoostClassifier()],
            'model__n_estimators': randint(50, 1000),
            'model__learning_rate': loguniform(1e-4, 1),
            'model__algorithm': ['SAMME', 'SAMME.R']
            })
        elif 'BC' in model:
            param_dists.append( {
            'model': [BaggingClassifier()],
            'model__n_estimators': randint(10, 1000),
            'model__max_samples': uniform(0, 1),
            'model__max_features': uniform(0, 1),
            'model__bootstrap': [True, False],
            'model__bootstrap_features': [True, False],
            })
        elif 'ETC' in model:
            param_dists.append( {
            'model': [ExtraTreesClassifier()],
            'model__n_estimators': randint(10, 1000),
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': randint(2, 11),
            'model__min_samples_split': randint(2, 11),
            'model__min_samples_leaf': randint(1, 11),
            'model__max_features': ['sqrt', 'log2'],
            'model__bootstrap': [True, False],
            'model__class_weight': ['balanced', 'balanced_subsample'],
            })
        elif 'MLP' in model:
            param_dists.append( {
            'model': [MLPClassifier(max_iter=5000)],
            'model__hidden_layer_sizes': [(10,), (50,), (100,), (10,10), (50,50), (100,100)],
            'model__activation': ['relu', 'logistic', 'tanh'],
            'model__alpha': loguniform(1e-5, 1),
            'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
            'model__solver': ['adam', 'lbfgs', 'sgd'],
            })
        elif 'LDA' in model:
            param_dists.append( {
            'model': [LinearDiscriminantAnalysis()],
            'model__solver': ['lsqr', 'eigen'],
            'model__shrinkage': [None, 'auto', 0.5, 0.9],
            'model__n_components': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            })
        elif 'QDA' in model:
            param_dists.append( {
            'model': [QuadraticDiscriminantAnalysis(reg_param=0.0)],
            'model__reg_param': loguniform(1e-5, 1),
            'model__tol': loguniform(1e-5, 1e-1),
        })

    if pca:
        for param in param_dists:
            param['pca__n_components'] = randint(1, 11)

    return param_dists
