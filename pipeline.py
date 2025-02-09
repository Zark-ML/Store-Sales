import numpy as np
import pandas as pd
import pickle

# sales predictor pipeline:

models_path = [f"{i}_family_model.pkl" for i in range(33)]

class SalesPredictor:
    def __init__(self, model_paths:list,data_path: str)-> None:
        """
        this class implements the Sales predictor pipeline
        parameters:
        model_paths (list): list of paths to the pickled models
        data_path (str): path to the dataset
        """
        self.models : list = []
        for path in model_paths:
            with open(path, 'rb') as file:
                self.models.append(pickle.load(file))
        self.data : pd.DataFrame = pd.read_csv(data_path)
        self.fitted : bool = False
    def fit(self)->None:
        """
        loads the data and fits the models
        """
        selected_columns = ['family', 'onpromotion', 'city', 'store_type', 'cluster', 'year', 'transactions_7d_avg', 'sales_7d_avg', 'week_day']


        if "transactions_7d_avg" in self.data.columns:
            self.data["transactions_7d_avg"] = self.data["transactions"].rolling(window = 7).mean().shift(1).fillna(0)
        self.data.drop(columns=["transactions"], inplace = True)

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
        
        
        # Replace the values in the 'family' column based on the mapping
        city_mapping = {city: idx for idx, city in enumerate(self.data['city'].unique())}
        self.data['city'] = self.data['city'].replace(city_mapping)


        # Replace the values in the'state' column based on the mapping
        state_mapping = {state: idx for idx, state in enumerate(self.data['state'].unique())}
        self.data['state'] = self.data['state'].replace(state_mapping)


        # Replace the values in the'store_type' column based on the mapping
        store_type_mapping = {category: idx for idx, category in enumerate(self.data['store_type'].unique())}
        self.data['store_type'] = self.data['store_type'].replace(store_type_mapping)

        self.data = self.data[selected_columns]
        self.fitted = True

        def predict(self) -> None:
            """
            Predicts sales for the given data and creates a JSON file of the predictions.
            """


            if not self.fitted:
                raise Exception("Model not fitted. Please call the fit method first.")
            

            predictions = []


            for index, row in self.data.iterrows():
                # Convert the row (which is a Series) back to a DataFrame
                row_df = pd.DataFrame([row], columns=self.data.columns)
                
                family = row_df['family'].iloc[0]  # Get the family value for this row
                
                if family not in range(len(self.models)):
                    raise Exception(f"Model for family '{family}' not found.")
                
                # Make predictions using the appropriate family model
                model = self.models[family]
                predictions.append(model.predict(row_df.drop(columns=['family'])))
            
            # Save the predictions to the DataFrame and then to a CSV file
            self.data['predictions'] = predictions
            self.data.to_csv('predictions.csv', index=False)

            print("Predictions saved to 'predictions.csv'")

            return self.data['predictions']



#TODO:test the class
if __name__ == '__main__':
    pass