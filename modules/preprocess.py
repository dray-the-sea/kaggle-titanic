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
            else: 
                print (f"skipping conversion for {label}, not a string col")

            data[label] = pd.Categorical(content).codes + 1
        else: 
            print (f"{label}, not in col list")

    return data

def infer_cabin_features(data):
    """
    takes titanic data and infers a Deck and CabinNo features from Cabin. 
    Deck is the first letter of the Cabin value
    CabinNo is the numbers
    """

    MISSING_CABIN_VAL = "unknown"
    data.Cabin.fillna(MISSING_CABIN_VAL, inplace=True)

    data["Deck"] = data.Cabin.str.replace(pat='\d+', repl='', regex=True)
    data["CabinNo"] = data.Cabin.str.replace(pat='[ABCDEFG]', repl="", regex=True)
    data['MultiCabin'] = data.apply (lambda row: label_multi_cabin(row), axis=1)    


    return data


def label_multi_cabin(row):
    """
    Utility function that tells you whether the person had a multi-room reservation 
    """
    deck = row.Deck

    if deck == "unknown":
        #print(f"{deck} - unknown")
        return "unknown"
    elif len(deck) == 1:
        #print(f"{deck} - single")
        return "single"
    else:
        #print(f"{deck} - single or multi")
        return "multi"

