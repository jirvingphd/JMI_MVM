from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from scipy.stats import randint
import pandas as pd
import time
import re
from JMI_MVM import list2df

config_dict = {
    sklearn.linear_model.LogisticRegressionCV:[{

        }],
        sklearn.linear_model.LogisticRegression:[{
            'clf__penalty':['l1'],
            'clf__C':[0.1, 1, 10, 15 ],
            'clf__tol':[1e-5, 1e-4, 1e-3],
            'clf__solver':['liblinear', 'newton-cg'],
            'clf__n_jobs':[-1]
            }, {
            'clf__penalty':['l2'],
            'clf__C':[0.1, 1, 10, 15 ],
            'clf__tol':[1e-5, 1e-4, 1e-3],
            'clf__solver':['lbfgs', 'sag'],
            'clf__n_jobs':[-1]
            }], 
            sklearn.ensemble.RandomForestClassifier:[{
                'clf__n_estimators':[10, 50, 100], 
                'clf__criterion':['gini', 'entropy'],
                'clf__max_depth':[4, 6, 10], 
                'clf__min_samples_leaf':[0.1, 1, 5, 15],
                'clf__min_samples_split':[0.05 ,0.1, 0.2],
                'clf__n_jobs':[-1]
                }],
                sklearn.svm.SVC:[{
                    'clf__C': [0.1, 1, 10], 
                    'clf__kernel': ['linear']
                    },{
                    'clf__C': [1, 10], 
                    'clf__gamma': [0.001, 0.01], 
                    'clf__kernel': ['rbf']
                    }],
                    sklearn.ensemble.GradientBoostingClassifier:[{
                        'clf__loss':['deviance'], 
                        'clf__learning_rate': [0.1, 0.5, 1.0],
                        'clf__n_estimators': [50, 100, 150]
                        }], 
                        xgboost.sklearn.XGBClassifier:[{
                            'clf__learning_rate':[.001, .01],
                            'clf__n_estimators': [1000,  100],
                            'clf__max_depth': [3, 5]
                            }]
    }
    
random_config_dict = {
    sklearn.ensemble.RandomForestClassifier:{
        'clf__n_estimators': [100 ,500, 1000],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': randint(1,100),
        'max_features': randint(1,100),
        'clf__min_samples_leaf': randint(1, 100),
        'clf__min_samples_split': randint(2, 10),
        'clf__n_jobs':[-1]
        }, 
        xgboost.sklearn.XGBClassifier:{
            'silent': [False],
            'max_depth': [6, 10, 15, 20],
            'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            'gamma': [0, 0.25, 0.5, 1.0],
            'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
            'n_estimators': [100]
            }
    }


def pipe_search(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=StandardScaler(), random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a grid search with cross validation using with given estimator
    and parameters.
    
    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC
    params: dict, list of dicts,
            Dictionary with parameters names (string) as keys and lists of parameter 
            settings to try as values, or a list of such dictionaries, in which case
            the grids spanned by each dictionary in the list are explored.This enables
            searching over any sequence of parameter settings.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.

    Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'
    """
    # create dictioinary to store results.
    results = {}
    # Instantiate pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])
    # start timer and fit pipeline.                        
    start = time.time()
    pipe.fit(X_train, y_train)
    # Instantiate and fit gridsearch object.
    grid = GridSearchCV(estimator = pipe,
        param_grid = params,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)

    grid.fit(X_train, y_train)
    # Store results in dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_
    # End timer and print results if verbosity higher than 0.
    end = time.time()
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'\nest Estimator: {grid.best_estimator_}')
        print(f'\nTime Elapsed: {((end - start))/60} minutes')
    
    return results


def random_pipe(estimator, params, X_train, y_train, X_test, y_test, n_components='mle',
                scaler=StandardScaler(),n_iter=10, random_state=42, cv=3, verbose=2, n_jobs=-1):

    """
    Fits pipeline and performs a randomized grid search with cross validation.
    
    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC 
    params: dict, 
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for 
             sampling (such as those from scipy.stats.distributions). 
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off 
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.
    
     Returns:
    ------------
        dictionary:
                keys are: 'test_score' , 'best_accuracy' (training validation score),
                'best_params', 'best_estimator', 'results'
    
    """
    # Start timer
    start = time.time()
    # Create dictioinary for storing results.
    results = {}
    # Instantiate Pipeline object.
    pipe = Pipeline([('scaler', scaler),
                        ('pca', PCA(n_components=n_components,random_state=random_state)),
                        ('clf', estimator(random_state=random_state))])
    # Fit pipeline to training data.                    
    pipe.fit(X_train, y_train)
    # Instantiate RandomizedSearchCV object.
    grid = RandomizedSearchCV(estimator = pipe,
        param_distributions = params,
        n_iter = n_iter,
        scoring = 'accuracy',
        cv = cv, verbose = verbose, n_jobs=n_jobs, return_train_score = True)
    # Fit gridsearch object to training data.
    grid.fit(X_train, y_train)
    # Store Test scores in results dictionary.
    results['test_score'] = grid.score(X_test, y_test)
    results['best_accuracy'] = grid.best_score_
    results['best_params'] = grid.best_params_
    results['best_estimator'] = grid.best_estimator_
    results['results'] = grid.cv_results_
    # End timer
    end = time.time()
    # print concise results if verbosity greater than 0.
    if verbose > 0:
        name = str(estimator).split(".")[-1].split("'")[0]
        print(f'{name} \nBest Score: {grid.best_score_} \nBest Params: {grid.best_params_} ')
        print(f'Best Estimator: {grid.best_estimator_}')
        print(f'Time Elapsed: {((end - start))/60} minutes')
    
    return results

def compare_pipes(config_dict, X_train, y_train, X_test, y_test, n_components='mle',
                 search='random',scaler=StandardScaler(), n_iter=10, random_state=42,
                  cv=3, verbose=2, n_jobs=-1):
    """
    Runs any number of estimators through pipeline and gridsearch(exhaustive or radomized) with cross validations, 
    can print dataframe with scores, returns dictionary of all results.

    Parameters:
    --------------
    estimator: estimator object,
            This is assumed to implement the scikit-learn estimator interface. 
            Ex. sklearn.svm.SVC 
    params: dict, or list of dictionaries if using GridSearchcv, cannot pass lists if search='random
            Dictionary with parameters names (string) as keys and distributions or
             lists of parameters to try. Distributions must provide a rvs method for 
             sampling (such as those from scipy.stats.distributions). 
             If a list is given, it is sampled uniformly.
            MUST BE IN FORM: 'clf__param_'. ex. 'clf__C':[1, 10, 100]
    X_train, y_train, X_test, y_test: 
            training and testing data to fit, test to model
    n_components: int, float, None or str. default='mle'
            Number of components to keep. if n_components is not set all components are kept.
            If n_components == 'mle'  Minka’s MLE is used to guess the dimension. For PCA.
    search: str, 'random' or 'grid',
            Type of gridsearch to execute, 'random' = RandomizedSearchCV,
            'grid' = GridSearchCV.
    scaler: sklearn.preprocessing class instance,
            MUST BE IN FORM: StandardScaler(), (default=StandardScaler())
    n_iter: int,
            Number of parameter settings that are sampled. n_iter trades off 
            runtime vs quality of the solution.
    random_state: int, RandomState instance or None, optional, default=42
            Pseudo random number generator state used for random uniform sampling from lists of 
            possible values instead of scipy.stats distributions. If int, random_state is the 
            seed used by the random number generator; If RandomState instance, random_state 
            is the random number generator; If None, the random number generator is the 
            RandomState instance used by np.random.
    cv:  int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy. Possible inputs for cv are:
                None, to use the default 3-fold cross validation,
                integer, to specify the number of folds in a (Stratified)KFold,
                CV splitter,
                An iterable yielding (train, test) splits as arrays of indices.
    verbose : int,
            Controls the verbosity: the higher, the more messages.
    n_jobs : int or None, optional (default = -1)
            Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend 
            context. -1 means using all processors. See Glossary for more details.

    """

    #Start timer
    begin = time.time()
    # CreateDictionary to store results from each grid search. Create list for displaying results.
    compare_dict = {}
    df_list = [['estimator', 'Test Score', 'Best Accuracy Score']]
    # Loop through dictionary instantiate pipeline and grid search on each estimator.
    for k, v in config_dict.items():

        # perform RandomizedSearchCV.
        if search == 'random':

            # Assert params are in correct form, as to not raise error after running search.
            if type (v) == list:
                raise ValueError(f"'For random search, params must be dictionary, not list ")
            else:
                results = random_pipe(k, v, X_train, y_train, X_test, y_test, n_components, 
                                    scaler, n_iter, random_state, cv, verbose, n_jobs)
        # Perform GridSearchCV.
        elif search == 'grid':
            results = pipe_search(k, v, X_train, y_train, X_test, y_test, n_components, 
                                        scaler, random_state, cv, verbose, n_jobs )

        # Raise error if grid parameter not specified.
        else:
            raise ValueError(f"search expected 'random' or 'grid' instead got{search}")

        # append results to display list and dictionary.
        name = str(k).split(".")[-1].split("'")[0]
        compare_dict[name] = results
        df_list.append([name, results['test_score'], results['best_accuracy']])
        
    # Display results if verbosity greater than 0.
    finish = time.time()
    if verbose > 0:
        print(f'\nTotal runtime: {((finish - begin)/60)}')
        display(list2df(df_list))
    
    return compare_dict
    



