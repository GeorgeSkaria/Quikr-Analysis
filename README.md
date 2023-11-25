# Quikr-Analysis
A car price prediction model using Linear Regression


## Overview

This repository contains a linear regression model for predicting car prices based on various features such as car name, company, manufacturing year, kilometers driven, and fuel type.

## Data Cleaning

The dataset was initially loaded from the 'quikr_car.csv' file, and several cleaning steps were performed:

- Removed inconsistent and spam-like data in the 'name' and 'company' columns.
- Filtered out rows with non-numeric values in the 'year' column.
- Removed rows with 'Ask For Price' in the 'Price' column.
- Removed commas from the 'Price' column and converted it to integer.
- Extracted numeric values from the 'kms_driven' column and converted it to integer.
- Removed rows with non-numeric values in the 'kms_driven' column.
- Removed rows with NaN values in the 'fuel_type' column.

The final cleaned dataset contains 816 entries.

## Exploratory Data Analysis (EDA)

### Descriptive Statistics


- The dataset has 816 entries with 6 columns.
- The 'year', 'Price', and 'kms_driven' columns were converted to numeric types.
- The 'Price' column was filtered to exclude values greater than 6,000,000.

### Relationship Analysis

- Explored the relationship between car companies and prices using boxplots.
- Analyzed the relationship between manufacturing year and prices using swarmplots.
- Investigated the relationship between kilometers driven and prices using a scatter plot.
- Explored the impact of fuel type, manufacturing year, and company on prices using a relational plot.

## Model Development

- Applied one-hot encoding to categorical features ('name', 'company', 'fuel_type') using `OneHotEncoder`.
- Developed a linear regression model using the scikit-learn library.
- Utilized a pipeline to streamline data preprocessing and model training.

## Model Evaluation

- Split the dataset into training and testing sets.
- Evaluated the model using the R-squared score, achieving a score of approximately 0.56.
- Conducted a search for the best model by varying the random state in the train-test split, achieving a maximum R-squared score of approximately 0.899.

## Model Deployment

- Exported the final model using `pickle`.
- Example prediction: Predicted price for a 'Maruti Suzuki Swift' from 'Maruti' with a manufacturing year of 2019, 100 km driven, and fueled by 'Petrol' is approximately 456,549.33 INR.

## Files

- `LinearRegressionModel.pkl`: Pickled file containing the trained linear regression model.
- `Cleaned_Car_data.csv`: CSV file containing the cleaned dataset.



```python
import pickle

# Load the model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Make a prediction
prediction = model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'], data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))

print(prediction)
```

