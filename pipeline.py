import pickle
import os
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

# Sales predictor pipeline:

models_path = [rf"C:\Users\Student\Desktop\Store-Sales\family\{i}_family_random_forest_model_with_rmlse.pkl" for i in range(33)]

class SalesPredictor:
    def __init__(self, model_paths: list, data_path: str) -> None:
        """
        This class implements the Sales predictor pipeline.
        parameters:
        model_paths (list): list of paths to the pickled models
        data_path (str): path to the dataset
        """
        self.models: list = []
        for path in model_paths:
            with open(path, 'rb') as file:
                self.models.append(pickle.load(file, encoding='latin1'))
        self.data: pd.DataFrame = pd.read_csv(data_path)
        self.fitted: bool = False
        self.not_fitted_models = []  # To track models that are not fitted

    def check_if_fitted(self, model):
        """
        Check if the model (or pipeline step) has been fitted.
        """
        if hasattr(model, 'named_steps'):  # Check if the model is a pipeline
            for step_name, step_model in model.named_steps.items():
                if not hasattr(step_model, 'predict') or not hasattr(step_model, 'fit'):
                    print(f"Model step '{step_name}' is not fitted.")
                    return False
        else:
            # For non-pipeline models, just check if the model is fitted
            if not hasattr(model, 'predict') or not hasattr(model, 'fit'):
                print(f"Model is not fitted: {model}")
                return False
        return True

    def fit(self) -> None:
        """
        Loads the data and fits the models.
        """
        selected_columns = ["id", "onpromotion", "state", "store_type", "cluster", "year", "week_day", "transactions_15d_avg", "family"]

        self.data.drop(columns=["transactions"], inplace=True)
        self.data['date'] = pd.to_datetime(self.data['date'])

        # Extracting year, month, day, and day_of_week
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.strftime('%B')  # Replace numeric month with month name (e.g., January, February)
        self.data['day'] = self.data['date'].dt.day
        self.data['day_of_week'] = self.data['date'].dt.strftime('%A')  # Replace numeric day with day name (e.g., Sunday, Monday)
        weekday_map = {'Monday': 0, 'Tuesday': 0, 'Wednesday': 0, 'Thursday': 0, 'Friday': 0, 'Saturday': 1, 'Sunday': 1}
        self.data['week_day'] = self.data['day_of_week'].replace(weekday_map)

        # Replace the values in the 'family' column based on the mapping
        family_mapping = {family: idx for idx, family in enumerate(self.data['family'].unique())}
        self.data['family'] = self.data['family'].replace(family_mapping)

        # Replace the values in the 'city', 'state', and 'store_type' columns based on mappings
        city_mapping = {city: idx for idx, city in enumerate(self.data['city'].unique())}
        self.data['city'] = self.data['city'].replace(city_mapping)

        state_mapping = {state: idx for idx, state in enumerate(self.data['state'].unique())}
        self.data['state'] = self.data['state'].replace(state_mapping)

        store_type_mapping = {category: idx for idx, category in enumerate(self.data['store_type'].unique())}
        self.data['store_type'] = self.data['store_type'].replace(store_type_mapping)

        self.data = self.data[selected_columns]
        self.fitted = True

        # Check if all models are fitted correctly
        for i, model in enumerate(self.models):
            if not self.check_if_fitted(model):
                self.not_fitted_models.append(i)

        if self.not_fitted_models:
            print(f"The following models are not fitted: {self.not_fitted_models}")
        else:
            print("All models are fitted correctly.")

    def predict(self) -> None:
        """
        Predicts sales for the given data and creates a CSV file of the predictions.
        """
        if not self.fitted:
            raise Exception("Model not fitted. Please call the fit method first.")

        predictions = pd.DataFrame(columns=['id', 'sales'])
        specific_family = [11, 22, 25, 30]

        for family in range(33):
            # Correct data filtering for family
            test_data_for_family = self.data[self.data['family'] == family].copy()

            if test_data_for_family.empty:
                continue

            family_id = test_data_for_family['id']
            test_data = test_data_for_family.drop(columns=["id", "family"])

            model = self.models[family]

            try:
                # Check if the model is fitted before predicting
                if not self.check_if_fitted(model):
                    raise ValueError(f"Model for family {family} is not fitted properly.")
                test_predict = model.predict(test_data)
            except Exception as e:
                print(f"Error in model for family {family}: {e}")
                continue

            log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=True)

            if family in specific_family:
                # Apply inverse log transformation for specific families
                test_predict = log_transformer.inverse_transform(test_predict.reshape(-1, 1)).flatten()

            current_sales = pd.DataFrame({'id': family_id, 'sales': test_predict})

            predictions = pd.concat([predictions, current_sales], ignore_index=True)

        predictions = predictions.sort_values(by='id', ascending=True)

        # Save predictions
        predictions.to_csv("submit_data.csv", index=False)

        print("Predictions saved to 'submit_data.csv'")

        return predictions


# Define paths to your existing files
DATA_PATH = r"C:\Users\Student\Desktop\Store-Sales\test.csv"  # Ensure this file exists

def test_sales_predictor():
    """Runs a test on the SalesPredictor pipeline using real files."""

    # Ensure data file exists
    assert os.path.exists(DATA_PATH), f"Data file missing: {DATA_PATH}"

    # Initialize predictor
    predictor = SalesPredictor(models_path, DATA_PATH)

    # Fit the model (preprocess data)
    predictor.fit()
    assert predictor.fitted, "Model should be fitted after calling fit()"

    # Run predictions
    predictions = predictor.predict()

    # Validate predictions
    assert isinstance(predictions, pd.DataFrame), "Predictions should be a pandas DataFrame"
    assert len(predictions) > 0, "Predictions should not be empty"
    assert all(predictions['sales'] >= 0), "Predictions should be non-negative"

    print("✅ SalesPredictor test passed successfully!")

# Run the test
if __name__ == '__main__':
    test_sales_predictor()
