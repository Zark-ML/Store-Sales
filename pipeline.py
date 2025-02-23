import pickle
import os
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

# Sales predictor pipeline:

models_path = [rf"C:\Users\user\Desktop\Store-Sales\family_models\{i}_adaboost_model_with_rmlse.pkl" for i in range(33)]

class SalesPredictor:
    def __init__(self, model_paths: list, data_path: str) -> None:
        """
        This class implements the Sales predictor pipeline.
        parameters:
        model_paths (list): list of paths to the pickled models
        data_path (str): path to the dataset
        """
        self.models: list = []
        self.models_path = model_paths
        self.data: pd.DataFrame = pd.read_csv(data_path)
        self.fitted: bool = False


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

        store_type_mapping = {state: idx for idx, state in enumerate(self.data['store_type'].unique())}
        self.data['store_type'] = self.data['store_type'].replace(store_type_mapping)

        self.data = self.data[selected_columns]
        self.fitted = True


    def predict(self) -> None:
        """
        Predicts sales for the given data and creates a CSV file of the predictions.
        """
        if not self.fitted:
            raise Exception("Model not fitted. Please call the fit method first.")

        predictions = pd.DataFrame(columns=['id', 'sales'])

        for family in range(33):
            # Correct data filtering for family
            test_data_for_family = self.data[self.data['family'] == family].copy()

            if test_data_for_family.empty:
                continue

            family_id = test_data_for_family['id']
            test_data = test_data_for_family.drop(columns=["id", "family"])



            with open(self.models_path[family],"rb") as f:            
                model = pickle.load(f)
                test_predict = model.predict(test_data)

            with open(rf"C:\Users\user\Desktop\Store-Sales\family_models\{family}_target_transformer.pkl","rb") as f:
                transformer = pickle.load(f)
                test_predict = transformer.inverse_transform(test_predict.reshape(-1, 1)).flatten()

            current_sales = pd.DataFrame({'id': family_id, 'sales': test_predict})

            predictions = pd.concat([predictions, current_sales], ignore_index=True)

        predictions = predictions.sort_values(by='id', ascending=True)

        predictions = predictions.set_index("id")
        # Save predictions
        predictions.to_csv("submit_data.csv", index=True)

        print("Predictions saved to 'submit_data.csv'")

        return predictions


# Define paths to your existing files
DATA_PATH = r"C:\Users\user\Desktop\Store-Sales\test.csv"  # Ensure this file exists


if __name__ == '__main__':
    model = SalesPredictor(models_path,DATA_PATH)
    model.fit()
    print(model.predict())
