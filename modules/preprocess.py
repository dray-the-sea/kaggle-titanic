import pandas as pd


def aggregated_preprocess1(df):
    """
    preprocesses data with:
    1. mark all columns where data will be filled in (imputed?)
    2. "unknown" for NA cabin, creates MultiCabin field
    3. infers Age from Sex, Parch, SibSp
    4. converts Sex, Embarked, Deck, MultiCabin, Ticket to category columns
    5. converts all to numbers 
    6. drops Name, Cabin
    """
    df = mark_missing_labels(df)
    df = infer_cabin_features(df)
    df.Age = df.apply(lambda row: fill_with_median_of_pss(row, df), axis=1)    
    df.Fare = df.apply(lambda row: fill_fare_with_pclass_median(row, df), axis=1)   
    df = numerify_categorical_columns(df, columns=["Sex", "Embarked", "Deck", "MultiCabin", "Ticket"])
    df = df.drop("Name", axis=1).drop("Cabin", axis = 1)
    return df



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
    #data["CabinNo"] = data.Cabin.str.replace(pat='[ABCDEFG]', repl="", regex=True)
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

def mark_missing_labels(data):
    for label, content in data.items():
        if pd.isnull(content).sum():
                data[label+"_is_missing"] = pd.isnull(content)
    return data

def fill_age_neg_1(data):
    """
    Fills age values with -1; this is just one option for pre-processing age
    """

    data.Age.fillna(-1, inplace=True)
    #fill in non-numeric missing data
            

    return data

def fill_with_median_of_pss(row, data):
    # if age is known, leave it alone
    if row.Age > 0:
        return row.Age
    else:
        # if sex, Parch, SibSp are known, take the median of these
        pred_age = data.loc[((data.Parch == row.Parch) & (data.Sex == row.Sex)) & (data.SibSp == row.SibSp)].Age.median()

        if pred_age > 0:
            print(f"keys: {row.Parch}  {row.Sex}  {row.SibSp} predicts age: {pred_age} ")
            return pred_age
        else: 
            # Sex, Parch, or SibSp must have had a unique value. most likely it's SibSp b/c there's more options, try median without it
            pred_age = data.loc[((data.Parch == row.Parch) & (data.Sex == row.Sex))].Age.median()

        if pred_age > 0:
            print(f"keys: {row.Parch}  {row.Sex}  (excluding sibsp {row.SibSp}) predicts age: {pred_age} ")
            return pred_age
        else: 
            # Maybe Parch had a unique value. let's try without that one.
            pred_age = data.loc[((data.SibSp == row.SibSp) & (data.Sex == row.Sex))].Age.median()
        
        if pred_age > 0:
            print(f"keys: {row.Parch}  {row.Sex}  (excluding Parch {row.Parch}) predicts age: {pred_age} ")
            return pred_age  
        else:  
            pred_age = data.Age[data.Sex == row.Sex].median()

        print(f"keys: {row.Sex}  (excluding parch {row.Parch}, sibsp {row.SibSp}) predicts age: {pred_age} ")
        return pred_age

def fill_fare_with_pclass_median(row, data):
    if row.Fare > 0:
        return row.Fare
    else:
        # if sex, Parch, SibSp are known, take the median of these
        return data.Fare[data.Pclass == row.Pclass].median()

