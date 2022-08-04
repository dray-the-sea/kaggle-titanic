import pandas as pd
from sklearn.preprocessing import MinMaxScaler



def aggregated_preprocess2(df):
    df["Family"] = df.SibSp+df.Parch
    df[['LastName','TitleFirstName']] = df.Name.str.split(',', expand=True)
    df['Title'] = df.TitleFirstName.apply(lambda x: x.split('.') [0])
    df.Embarked.fillna("X", inplace=True)

    df["Age"] = df.apply(lambda row: fill_with_median_of_fam_size(row, df), axis=1) 

    """ 
    df["TicketSurvivalRate"] = .5
    for _, travel_group in df.groupby('Ticket'):
        if (len(travel_group) > 1):
            df["TicketSurvivalRate"] = travel_group.Survived.mean() 
    """

    df = set_age_groups(df)
    df = set_fare_groups(df)

    df = df.drop(['Name', 'SibSp', "Parch", 'LastName'], axis=1)

    return df

def set_age_groups(df):
    cut_ages = [1, 2, 3, 4, 5, 6]
    cut_bins = [ 0, 18, 23, 28.815, 35, 44.92, 100]
    df['AgeGroup'] = pd.cut(df['Age'], bins=cut_bins, labels = cut_ages)

    return df

def set_fare_groups(df):
    cut_fares = [1, 2, 3, 4, 5, 6]
    cut_fare_bins = [ 0, 7.775, 8.6625, 14.5, 26.3075,  53.1, 600]
    df['FareGroup'] = pd.cut(df['Fare'], bins=cut_fare_bins, labels = cut_fares)

    return df


def aggregated_preprocess1(df, report=False):
    """
    df - dataframe to pre-proess 
    report - if true, print report of the different operations 
    preprocesses data with:
    1. mark all columns where data will be filled in (imputed?)
    2. "unknown" for NA cabin, creates MultiCabin field
    3. infers Age from Sex, Parch, SibSp
    4. converts Sex, Embarked, Deck, MultiCabin, Ticket to category columns
    5. converts all to numbers 
    6. drops Name, Cabin, PassengerId
    """
    #df = mark_missing_labels(df)
    df = infer_cabin_features(df)

    df.Age = df.Age.fillna(-1)
    df["Age_is_missing"] = df.apply( lambda row: mark_missing(row, "Age", -1), axis=1)
    df["Age"] = df.apply(lambda row: fill_with_median_of_pss(row, df), axis=1)  

    df.Fare = df.Fare.fillna(-1)
    df["Fare_is_missing"] = df.apply( lambda row: mark_missing(row, "Fare", -1), axis=1)   
    df["Fare"] = df.apply(lambda row: fill_fare_with_pclass_median(row, df), axis=1)  


    df = numerify_categorical_columns(df, columns=["Sex", "Embarked", "Deck", "MultiCabin", "Ticket"], report=report)
    df = df.drop("Name", axis=1).drop("Cabin", axis = 1).drop("PassengerId", axis = 1)
    return df



def scale_aggregated1(data):
    """
    MinMax scale preprocessed dataframe
    """
    scaler = MinMaxScaler()

    copy = data.copy()
    scale_arr = scaler.fit_transform(copy)
    scale_df = pd.DataFrame(copy)
    scale_df.columns = data.columns
    
    return scale_df


def numerify_categorical_columns_0(data, columns=None, report=False):
    """
    converts text columns to numeric categories 
    data: DataFrame with the data to be converted
    columns=None: array of column names. If None, will go through all the columns
    report=False: will print process notes if set to true
    """
    process_report = {}

    for label, content in data.items():

        # if column needs processing, go ahead
        if not columns or label in columns:
            if pd.api.types.is_string_dtype(content):
                data[label] = content.astype("category").cat.as_ordered()
                process_report[label] = "converted to category type"
            else: 
                process_report[label] = "skipping conversion, not a string type"

            data[label] = pd.Categorical(content).codes
        else: 
            process_report[label] = "skipping conversion, not in conversion list"

    if report:
        for col, stat in process_report:
            print(f"{col}: {stat}")

    return data

def numerify_categorical_columns(data, columns=None, report=False):
    """
    converts text columns to numeric categories 
    data: DataFrame with the data to be converted
    columns=None: array of column names. If None, will go through all the columns
    report=False: will print process notes if set to true
    """
    process_report = {}

    for label, content in data.items():

        # if column needs processing, go ahead
        if not columns or label in columns:
            if pd.api.types.is_string_dtype(content):
                data[label] = content.astype("category").cat.as_ordered()
                process_report[label] = "converted to category type"
            else: 
                process_report[label] = "skipping conversion, not a string type"

            data[label] = pd.Categorical(content).codes + 1
        else: 
            process_report[label] = "skipping conversion, not in conversion list"

    if report:
        for col, stat in process_report:
            print(f"{col}: {stat}")

    return data

def infer_cabin_features(data, mark_missing=True):
    """
    takes titanic data and infers a Deck features from Cabin. 
    Deck is the first letter of the Cabin value
    Deck_is_missing tells you if the data has been filled
    """

    MISSING_CABIN_VAL = "unknown"
    data.Cabin.fillna(MISSING_CABIN_VAL, inplace=True)

    data["Deck"] = data.Cabin.str.replace(pat='\d+', repl='', regex=True)

    if mark_missing:
        data["Deck_is_missing"] = data.apply( lambda row: mark_missing(row, "Cabin", MISSING_CABIN_VAL), axis=1)

    return data

def mark_missing(row, col_name, missing_label):
    """
    check if row's col_name value is equal to missing_label
    """
    if row[col_name] == missing_label:
        return 1
    else:
        return 0


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

def fill_with_median_of_pss(row, data):
    # if age is known, leave it alone
 
    if row.Age > -1:
        return row.Age
    else:
        try:
            # if sex, Parch, SibSp are known, take the median of these
            pred_age = data.loc[((data.Parch == row.Parch) & (data.Sex == row.Sex)) & (data.SibSp == row.SibSp)].Age.median()

            if pred_age == -1:
                # Sex, Parch, or SibSp must have had a unique value. most likely it's SibSp b/c there's more options, try median without it
                pred_age = data.loc[((data.Parch == row.Parch) & (data.Sex == row.Sex))].Age.median()

            if pred_age == -1:
                # Maybe Parch had a unique value. let's try without that one.
                pred_age = data.loc[((data.SibSp == row.SibSp) & (data.Sex == row.Sex))].Age.median()
            
            if pred_age == -1:  
                pred_age = data.Age[data.Sex == row.Sex].median()

            #print(f"keys: {row.Sex}  (excluding parch {row.Parch}, sibsp {row.SibSp}) predicts age: {pred_age} ")
            return pred_age

        except:
                print(f"keys: {row.Parch}  {row.Sex}  {row.SibSp} caused an exception ")



def fill_with_median_of_fam_size(row, data):
    if row.Age > 0:
        return row.Age
    else:
        return data.loc[((data.Family == row.Family) & (data.Sex == row.Sex))].Age.mean()



def fill_fare_with_pclass_median(row, data):
    if row.Fare > 0:
        return row.Fare
    else:
        # if sex, Parch, SibSp are known, take the median of these
        fare = data.Fare[data.Pclass == row.Pclass].median()
        return fare
