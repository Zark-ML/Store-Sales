import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE,SequentialFeatureSelector
from sklearn.base import BaseEstimator,TransformerMixin
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.feature_selection import mutual_info_regression



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
        print(boruta.support_)
        return pd.DataFrame(self.X[selected_columns])
    


    def xgboost_select(self, estimator: xgb.XGBRegressor = None, threshold: float = 0.0, **kwargs):
        """
        Perform XGBoost feature selection based on feature importances.

        Arguments:
        estimator (xgb.XGBRegressor): An XGBoost model instance. If None, defaults to XGBRegressor.
        threshold (float): Minimum importance threshold to select features. Features with importance > threshold will be selected.
        kwargs (dict): Additional keyword arguments to be passed to the XGBoost model.

        Keyword Arguments:
        n_estimators (int): Number of estimators for XGBoost.
        max_depth (int): Maximum depth of trees.
        learning_rate (float): Learning rate.
        subsample (float): Subsample ratio of the training instances.
        colsample_bytree (float): Subsample ratio of columns when constructing each tree.
        random_state (int): Random state for reproducibility.
        verbose (int): Verbosity level for XGBoost.
        early_stopping_rounds (int): Early stopping rounds for XGBoost.
        """
        # Default XGBoost arguments
        xgb_args = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': None,
            'verbose': 0,
            'early_stopping_rounds': 10
        }
        
        # Update default arguments with any passed in kwargs
        xgb_args.update(kwargs)
        
        # If no estimator is provided, use XGBRegressor with the specified arguments
        if estimator is None:
            estimator = xgb.XGBRegressor(**xgb_args)
        
        # Fit the model to the data
        estimator.fit(self.X, self.y, eval_set=[(self.X, self.y)])
        
        # Get feature importances and filter based on the threshold
        feature_importances = estimator.feature_importances_
        selected_columns = np.array(self.X.columns)[feature_importances > threshold]
        
        # Print the feature importances
        print("Feature Importances: ", feature_importances)
        
        # Return the names of selected features based on importance threshold
        return selected_columns


class RMRMFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select):
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        # Calculate mutual information for each feature with the target
        self.mi_scores_ = mutual_info_regression(X, y)
        return self

    def transform(self, X):
        # Select the top features based on mutual information
        selected_features = np.argsort(self.mi_scores_)[-self.n_features_to_select:]
        return X.iloc[:, selected_features]




if __name__=='__main__':
    data = pd.read_csv(r"C:\Users\user\Desktop\data\data.csv")
    
    # data = data.drop(columns=["date"])  # Drop unnecessary column
    # data_target = data["sales"]
    # data_features = data.drop(columns=["sales"])

    # rmrm_selector = RMRMFeatureSelector(n_features_to_select=10)  # Adjust the number of features to select
    # X_selected = rmrm_selector.fit_transform(data_features, data_target)

    # print(X_selected.columns)
    # Boruta
    # model = RandomForestRegressor(n_jobs=-1)
    # feature_selector = Feature_selection_custom(data_features, data_target)
    # print(feature_selector.boruta_select(model, random_state=42,verbose=1,max_iter=40))
    # xgbselector
    # model = Feature_selection_custom(data_features, data_target)
    # selected_features = model.xgboost_select(estimator=None, threshold=0.03, n_estimators=400, max_depth=8,verbose=1,n_jobs=10)
    # print("Selected features:", selected_features)

    
    """
    Boruta
    ['family', 'city', 'state', 'store_type', 'cluster', 'transactions',
       'day', 'day_of_week']
    
    xgb
    Selected features: ['family' 'city' 'state' 'store_type',
      'cluster' 'transactions''day_of_week' 'onpromotion' 'year']

    RMRM
    Selected features: ['day of week' 'store_type' 'year' 'state' 'oil' 'city' 'cluster' 'transaction' 'onpromotion' 'family']
    """

    

    