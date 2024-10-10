from sklearn.model_selection import train_test_split
import pandas as pd


def generate_train_test_split(df, vars_to_drop=[]):

    all_predictors = df.drop(vars_to_drop + ["Label"], axis=1)
    response = df["Label"]

    encoded_data = pd.get_dummies(all_predictors, columns=["key", "mode"])
    return train_test_split(encoded_data, response, test_size=0.2, random_state=0)
