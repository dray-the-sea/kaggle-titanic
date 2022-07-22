import pandas as pd

def numerify_categorical_columns(data, columns=None):
    """
    converts text columns to numeric categories 
    data: DataFrame with the data to be converted
    columns=None: array of column names. If None, will go through all the columns
    """
    
    for label, content in data.items():
        # if column needs processing, go ahead
        if not columns or label in columns:
            if pd.api.types.is_string_dtype(content):
                data[label] = content.astype("category").cat.as_ordered()
                print (f"converting {label} to category type")

            data[label] = pd.Categorical(content).codes + 1
    return data
