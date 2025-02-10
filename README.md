
# Store-Sales

## Project Purpose
The purpose of this project is to create a system that can predict the sales of different types of products across store networks. The main goal is to determine daily sales for specific product types.

## Key Features
- **family**: Identifies the type of product sold.
- **sales**: Represents the total sales for a product family at a particular store on a given date. Fractional values are possible, as products can be sold in fractional units (e.g., 1.5 kg of cheese vs. 1 bag of chips).
- **onpromotion**: Represents the total number of items in a product family that were being promoted at a store on a given date.

## Problem Solved and Usefulness
Managing inventory in retail stores is challenging due to fluctuations in customer demand. Without accurate sales predictions, stores risk either stockouts or overstocking, both of which result in financial losses.  
This project solves this problem by providing accurate sales forecasts for different product types. It helps store managers optimize inventory, plan promotions, and reduce operational costs, ultimately leading to improved customer satisfaction and increased profitability.

## Models Functionality
- **Accurate Sales Predictions**:  
  The system uses advanced predictive models to forecast sales for different product types, helping businesses make data-driven decisions.
  
- **Customizable Input**:  
  Users can input sales data manually or upload CSV files, offering flexibility based on data availability.

- **Real-time Forecasting**:  
  Provides daily sales predictions in real-time, helping businesses plan inventory and promotions efficiently.

- **Multi-Product Support**:  
  Capable of forecasting sales for multiple product types at once, streamlining inventory management across different categories.

- **CSV Export**:  
  Predicted sales can be exported to a CSV file for further analysis or reporting purposes.

## Technologies Used
- **Programming Languages**:  
  Python 3.x

- **Libraries/Frameworks**:  
  - Pandas (for data manipulation)  
  - Scikit-learn (for machine learning models)  
  - NumPy (for numerical operations)  
  - Matplotlib (for plotting graphs)  
  - Seaborn (for statistical data visualization)  
  - Pickle (for serializing models)

- **Tools**:  
  - PyCharm (for development)  
  - Jupyter Notebook (for development and testing)
  - OS (for interacting with the operating system)
