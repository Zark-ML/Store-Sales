import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, r_regression, SequentialFeatureSelector
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_log_error
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

class Feature_selection_custom:
    """
    This class provides custom feature selection methods based on your requirements.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Initialize the class with features (X) and target (y).

        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.DataFrame or pd.Series): Target variable.
        """
        self.X = X
        self.y = y

    def recursive_feature_elimination(self, estimator: BaseEstimator, n_features_to_select: int = 5):
        """
        Perform Recursive Feature Elimination (RFE).

        Args:
            estimator (BaseEstimator): A model with a `fit` method (e.g., LogisticRegression, RandomForest).
            n_features_to_select (int): Number of features to select.

        Returns:
            pd.DataFrame: Reduced feature set with selected features.
        """
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(self.X, self.y)
        selected_columns = self.X.columns[rfe.support_]
        return pd.DataFrame(self.X[selected_columns])

    def sequential_feature_selector(self, estimator: BaseEstimator, direction: str = 'forward', n_features_to_select: int = 5):
        """
        Perform Sequential Feature Selection (SFS).

        Args:
            estimator (BaseEstimator): A model with a `fit` method.
            direction (str): 'forward' for forward selection, 'backward' for backward elimination.
            n_features_to_select (int): Number of features to select.

        Returns:
            pd.DataFrame: Reduced feature set with selected features.
        """
        if direction not in ['forward', 'backward']:
            raise ValueError("Invalid direction. Use 'forward' or 'backward'.")
        
        sfs = SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features_to_select, direction=direction)
        sfs.fit(self.X, self.y)
        selected_columns = self.X.columns[sfs.get_support()]
        return pd.DataFrame(self.X[selected_columns])
    
    
    def boruta_select(self, estimator: BaseEstimator, **kwargs):
        """
        Perform Boruta feature selection.
        
        Arguments:
        estimator (BaseEstimator): A model with a `fit` method.
        kwargs (dict): Additional keyword arguments to be passed to BorutaPy.
        
        Keyword Arguments:
        n_estimators (int): Number of estimators for BorutaPy.
        perc (float): Percentage of features to select.
        alpha (float): Significance level for BorutaPy.
        two_step (bool): If True, perform two-step Boruta.
        max_iter (int): Maximum number of iterations for BorutaPy.
        random_state (int): Random state for BorutaPy.
        verbose (int): Verbosity level for BorutaPy.
        early_stopping (bool): If True, stop BorutaPy early if the score doesn't improve for n_iter_no_change iterations.
        n_iter_no_change (int): Number of iterations without improvement after which BorutaPy stops.
        """
        boruta_args = {
            'n_estimators': 1000,
            'perc': 100,
            'alpha': 0.05,
            'two_step': True,
            'max_iter': 100,
            'random_state': None,
            'verbose': 0,
            'early_stopping': False,
            'n_iter_no_change': 20
        }
        boruta_args.update(kwargs)
        boruta = BorutaPy(estimator=estimator, **boruta_args)
        boruta.fit(self.X, self.y)
        selected_columns = np.array(self.X.columns)[boruta.support_]
        return pd.DataFrame(self.X[selected_columns])

if __name__=='__main__':
    data = pd.read_csv("data.csv").iloc[:200000,:]
    
    data = data.drop(columns=["date"])  # Drop unnecessary column
    data_target = data["sales"]
    data_features = data.drop(columns=["sales"])

    model = RandomForestRegressor(n_jobs=10)
    feature_selector = Feature_selection_custom(data_features.values, data_target)
    print(feature_selector.boruta_select(model, random_state=42,verbose=1))

    

    