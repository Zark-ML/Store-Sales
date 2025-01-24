import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectKBest, r_regression
from sklearn.base import BaseEstimator

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

    def select_k_best(self, k: int = 10):
        """
        Select the top k features based on univariate statistical tests.

        Args:
            k (int): Number of top features to select.

        Returns:
            pd.DataFrame: Reduced feature set with k selected features.
        """
        selector = SelectKBest(score_func=r_regression, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_columns = self.X.columns[selector.get_support()]
        return pd.DataFrame(X_selected, columns=selected_columns)

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

    def r_regression_model(self, threshold: float = 0.1):
        """
        Select features using r_regression based on absolute correlation threshold.

        Args:
            threshold (float): Minimum absolute correlation required to select a feature.

        Returns:
            pd.DataFrame: Reduced feature set with selected features.
            pd.Series: Correlation values for all features.
        """
        correlations = r_regression(self.X, self.y)
        selected_mask = abs(correlations) >= threshold
        selected_columns = self.X.columns[selected_mask]
        return pd.DataFrame(self.X[selected_columns]), pd.Series(correlations, index=self.X.columns)

    def get_feature_support(self, method: str, *args, **kwargs):
        """
        Get a boolean mask of selected features for a given method.

        Args:
            method (str): The feature selection method ('k_best', 'rfe', or 'r_regression').
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            pd.Series: Boolean mask of selected features.
        """

        match(method):
            case 'k_best':
                selector = SelectKBest(score_func=r_regression, *args, **kwargs)
                selector.fit(self.X, self.y)
                return pd.Series(selector.get_support(), index=self.X.columns)
            case 'rfe':
                estimator = kwargs.get('estimator')
                if estimator is None:
                    raise ValueError("An 'estimator' argument is required for RFE.")
                rfe = RFE(estimator=estimator, *args, **kwargs)
                rfe.fit(self.X, self.y)
                return pd.Series(rfe.support_, index=self.X.columns)
            case 'r_regression':
                threshold = kwargs.get('threshold', 0.1)
                correlations = r_regression(self.X, self.y)
                selected_mask = abs(correlations) >= threshold
                return pd.Series(selected_mask, index=self.X.columns)
            case _:
                raise ValueError(f"Invalid feature selection method: {method}")