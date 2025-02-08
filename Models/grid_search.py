import optuna
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from typing import Dict, Any, Callable


class OptunaOptimizer:
    """
    A custom wrapper for Optuna hyperparameter optimization.

    Attributes:
        objective (Callable): The objective function to optimize.
        n_trials (int): Number of trials for optimization.
        direction (str): Direction of optimization ('minimize' or 'maximize').
        study (optuna.Study): Optuna study object.
    """

    def __init__(self, objective: Callable[[optuna.Trial], float], 
                 n_trials: int = 100, direction: str = "minimize", verbose: bool = True) -> None:
        """
        Initializes the Optuna optimizer.

        Args:
            objective (Callable): The objective function to optimize.
            n_trials (int, optional): Number of trials. Defaults to 100.
            direction (str, optional): 'minimize' or 'maximize'. Defaults to 'minimize'.
            verbose (bool, optional): Enable verbose logging. Defaults to True.
        """
        self.objective = objective
        self.n_trials = n_trials
        self.direction = direction
        self.study = optuna.create_study(direction=self.direction)

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)  # Enable verbose logging

    def optimize(self) -> None:
        """Runs the optimization process with a progress bar."""
        self.study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

    def get_best_params(self) -> Dict[str, Any]:
        """Retrieves the best parameters from the study."""
        return self.study.best_trial.params

    def get_best_value(self) -> float:
        """Retrieves the best objective value."""
        return self.study.best_trial.value

    def print_results(self) -> None:
        """Prints the best parameters and their corresponding objective value."""
        print("\n🔹 Best Hyperparameters:")
        print(self.get_best_params())
        print("\n🔹 Best Objective Value (Neg MSE):", self.get_best_value())

        # Save best parameters to CSV
        best_params_df = pd.DataFrame([self.get_best_params()])
        best_params_df.to_csv("best_hyperparameters.csv", index=False)
        print("\n✅ Best parameters saved to 'best_hyperparameters.csv'.")


# Load dataset
data = pd.read_csv(r'C:\Users\Student\Desktop\data.csv')

# Define features and target
target_column = "sales"
feature_columns = ['day_of_week', 'store_type', 'year', 'state', 
                   'dcoilwtico', 'city', 'cluster', 'transactions', 
                   'onpromotion', 'family']

# Encode categorical variables
for col in ["store_type", "state", "city", "family"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Feature Scaling (StandardScaler on numerical columns)
scaler = StandardScaler()
data[["dcoilwtico", "transactions"]] = scaler.fit_transform(data[["dcoilwtico", "transactions"]])

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    data[feature_columns], data[target_column], test_size=0.2, random_state=42
)


def rf_objective(trial: optuna.Trial) -> float:
    """Objective function to optimize RandomForestRegressor hyperparameters."""
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    # Use cross-validation score as the evaluation metric
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    return np.mean(score)  # Maximize negative MSE (equivalent to minimizing MSE)


if __name__ == "__main__":
    # Optimize hyperparameters
    optimizer = OptunaOptimizer(objective=rf_objective, n_trials=50, direction="maximize")
    optimizer.optimize()
    optimizer.print_results()

    # Train final model with best parameters
    best_params = optimizer.get_best_params()
    final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"\n📊 Final Test RMSE: {rmse:.4f}")

