from sklearn.model_selection import train_test_split
import pandas as pd


def generate_train_test_split(df, vars_to_drop=[]):
    """
    This function generates a (seeded) 80/20 train test split for the spotify data with the categorical "key" and "mode" are
    transformed to dummy variables, with no interaction terms included.

    Params:
    -------
    df : pandas.DataFrame
        The labels should match the original data frame of song data.
    vars_to_drop : list
        A list of labels containing variables not to be included in the model.

    Returns:
    --------
    (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series):
        A sequence of four data frames/series with the first two being the training and testing predictors
        and the last two being the labels of the training data and testing data.
    """

    all_predictors = df.drop(vars_to_drop + ["Label"], axis=1)
    response = df["Label"]

    # We drop_first because one of the variables in each of key,
    # mode is determined by the values of the others
    # (for example mode != 0 implies mode = 1)
    encoded_data = pd.get_dummies(
        all_predictors, columns=["key", "mode"], drop_first=True
    )
    return train_test_split(encoded_data, response, test_size=0.2, random_state=0)
