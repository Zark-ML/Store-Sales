import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, SelectKBest, r_regression
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_log_error

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
    
    def select_k_best(self,k: int = 10):
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
                return self.select_k_best(args,kwargs)
            case 'rfe':
                return self.recursive_feature_elimination(args,kwargs)
            case _:
                raise ValueError(f"Invalid feature selection method: {method}")


    def feature_selection_score_plot(self,method,estimator: BaseEstimator, max_k=None):
        """
        Plot model performance (e.g., accuracy) for different values of k (number of features selected) using feature selection.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target vector.
            estimator (BaseEstimator): A model with a `fit` method, such as LogisticRegression or RandomForestClassifier.
            max_k (int): The maximum number of top features to consider. If None, it uses the total number of features.
        """
        if max_k is None:
            max_k = self.X.shape[1]

        # List to store performance scores for each value of k
        scores = []

        # Loop through different values of k (number of features selected)
        for k in range(1, max_k + 1):
            # Initialize RFE with the given estimator and number of features to select
            if method == 'RFE':
                selector = RFE(estimator, n_features_to_select=k)
            elif method == 'k_best':
                selector = SelectKBest(score_func=r_regression, k=k)
            selector.fit(self.X, self.y)
            
            # Get the selected features and reduce X to the selected features
            X_selected = self.X.iloc[:, selector.support_]

            # Train the estimator on the selected features and evaluate performance
            estimator.fit(X_selected, self.y)
            y_pred = estimator.predict(X_selected)
            
            # Calculate accuracy or other metric (you can customize this)
            score = mean_squared_log_error(self.y, y_pred)
            scores.append(score)
        
        # Plot the performance (e.g., accuracy) vs number of features selected (k)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), scores, marker='o')
        plt.title('Performance vs Number of Features Selected (RFE)')
        plt.xlabel('Number of Features Selected')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(range(1, max_k + 1))  # Ensure ticks for all values of k
        plt.show()