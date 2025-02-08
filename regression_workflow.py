import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib


# Base abstract preprocessor class
class DataPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, df, target_col):
        if df.isnull().values.any():
            print("Warning: Missing values found. Dropping missing values.")
            df = df.dropna()
        return df.drop(columns=[target_col]), df[target_col]


# A basic preprocessor for regression models
class BasicPreprocessor(DataPreprocessor):
    def preprocess(self, df, target_col, selected_columns=None, test_size=0.2, random_state=18):
        print(f"Initial data shape: {df.shape}")

        # Remove duplicate rows
        df = df.drop_duplicates()
        print(f"After duplicate removal: {df.shape}")

        # Select only required columns (ensure target is included)
        if selected_columns:
            selected_columns = list(set(selected_columns + [target_col]))
            df = df[selected_columns]
        print(f"After column selection: {df.shape}")

        # Convert object columns to numeric (coerce errors to NaN)
        for column in df.select_dtypes(include=["object"]).columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Drop any rows with NaN values after conversion
        df = df.dropna()
        print(f"After dropping NaNs: {df.shape}")

        if df.empty:
            raise ValueError("Data is empty after preprocessing. Please check your data and preprocessing steps.")

        X = df.drop(columns=[target_col])
        y = df[target_col]
        print(f"Final feature shape: {X.shape}, Target shape: {y.shape}")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        # Scale features using MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


# Workflow class to manage training and testing of regression models.
class RegressionWorkflow:
    def __init__(self, models_dir='models', model_path=None):
        self.models_dir = models_dir
        self.model_path = model_path
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}. Please check the path.")
        except pd.errors.EmptyDataError:
            raise ValueError("The provided CSV file is empty.")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def get_next_model_name(self, model_name):
        model_identifiers = {
            "Gradient Boosting": "gb_reg",
            "Decision Tree": "dt_reg",
            "Random Forest": "rf_reg"
        }
        # Get the simple model code (default to 'model' if not found)
        model_id = model_identifiers.get(model_name, "model")
        existing_models = os.listdir(self.models_dir)
        model_files = [f for f in existing_models if f.startswith(f'{model_id}_model') and f.endswith('.pkl')]

        # Extract model version numbers and determine the next number
        model_numbers = [int(f.split('_')[2].split('.')[0]) for f in model_files if
                         f.split('_')[2].split('.')[0].isdigit()]
        next_model_number = max(model_numbers, default=0) + 1

        return f'{model_id}_model_{next_model_number}.pkl'

    def train_model(self, train_path, target_col, selected_columns=None, test_size=0.2,
                    random_state=18, model_name="Gradient Boosting", model_params=None):
        # Load and preprocess data
        df = self.load_data(train_path)
        preprocessor = BasicPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess(df, target_col,
                                                                   selected_columns,
                                                                   test_size,
                                                                   random_state)

        if model_params is None:
            model_params = {}

        # Choose the regressor based on model name
        if model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=random_state, **model_params)
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(random_state=random_state, **model_params)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=random_state, **model_params)
        else:
            raise ValueError(f"Model '{model_name}' not supported. Please choose a supported model.")

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        msle = mean_squared_log_error(y_test, y_pred)
        print(f"Training completed. MSLE: {msle}")

        # Save the trained model
        model_file_name = self.get_next_model_name(model_name)
        self.model_path = os.path.join(self.models_dir, model_file_name)
        joblib.dump(model, self.model_path)
        print(f"Model saved successfully: {self.model_path}")

        return msle

    def test_model(self, test_path, target_col, selected_columns=None):
        # Load and preprocess testing data
        df = self.load_data(test_path)
        preprocessor = BasicPreprocessor()
        X_test, _, y_test, _ = preprocessor.preprocess(df, target_col, selected_columns)

        # Find the latest trained model file
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
        if not model_files:
            raise FileNotFoundError("No trained models found. Train a model first.")

        # Sort model files based on the version number and load the latest one
        model_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_model_path = os.path.join(self.models_dir, model_files[-1])
        model = joblib.load(latest_model_path)
        print(f"Loaded model from: {latest_model_path}")

        # Evaluate the model on the test set
        y_pred = model.predict(X_test)
        msle = mean_squared_log_error(y_test, y_pred)
        print(f"Test MSLE: {msle}")

        return msle


def main():
    # Update these paths and target column as needed
    train_path = "data/regression_data.csv"
    test_path = "data/regression_data.csv"
    target_column = "sales"

    workflow = RegressionWorkflow(models_dir='models')

    print("=== Training the Regression Model ===")
    train_msle = workflow.train_model(train_path=train_path,
                                      target_col=target_column,
                                      model_name="Gradient Boosting")
    print(f"Training MSLE: {train_msle}\n")

    print("=== Testing the Regression Model ===")
    test_msle = workflow.test_model(test_path=test_path,
                                    target_col=target_column)
    print(f"Testing MSLE: {test_msle}")


if __name__ == "__main__":
    main()
