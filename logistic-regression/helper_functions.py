from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

from ISLP import confusion_table


def generate_train_test_split(df, vars_to_drop=[]):
    """
    This function generates a (seeded) 80/20 train test split for the spotify data with the categorical "key" and "mode" are
    transformed to dummy variables, with no interaction terms included.

    Params:
    -------
    df : pandas.DataFrame
        The columns should include each of the columns of the original data frame of song data.
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


def iterate_forward_selection(predictor_df, response, initial_vars, other_vars):
    selected_vars = initial_vars.copy()
    new_other_vars = other_vars.copy()
    best_variable = None
    best_AIC = np.inf
    scores = None
    for var in other_vars:
        model = LogisticRegression(penalty=None, max_iter=1000)
        predictors = predictor_df[initial_vars + [var]]
        # print(new_other_vars)
        # print(var)
        model.fit(predictors, response)

        predict_proba = model.predict_proba(predictors)
        AIC = (
            2 * len(selected_vars)
            + 2
            + 2 * log_loss(response, predict_proba, normalize=False)
        )  # AIC, to be minimized
        if AIC <= best_AIC:
            best_AIC = AIC
            best_variable = var
            scores = cross_val_score(
                model,
                predictor_df[initial_vars + [var]],
                response,
                cv=KFold(shuffle=True, random_state=0),
            )

    new_other_vars.remove(best_variable)
    return selected_vars + [best_variable], new_other_vars, best_AIC, scores


def forward_selection_minimize_AIC(predictor_df, response):
    max_AIC = np.inf
    best_variable_subset = None
    max_n_variables = 40
    best_n_variables = 0
    selected_vars = []
    metric_dict = {"AIC": [], "Cross Validation Scores": [], "Selected Variables": []}
    other_vars = predictor_df.columns.to_list()
    for i in range(1, max_n_variables + 1):
        selected_vars, other_vars, AIC, scores = iterate_forward_selection(
            predictor_df, response, selected_vars, other_vars
        )
        metric_dict["AIC"].append(AIC)
        metric_dict["Cross Validation Scores"].append(scores)
        metric_dict["Selected Variables"].append(selected_vars)

    return pd.DataFrame(metric_dict, index=range(1, max_n_variables + 1))


def evaluate_sequentially_selected_models(predictor_df):

    # best_variables = []
    max_n_variables = 40
    metric_dict = {"Cross Validation Scores": [], "AIC": []}

    predictors_train, response_train = (
        predictor_df.drop(["Label"], axis=1),
        predictor_df["Label"],
    )
    for i in range(2, max_n_variables + 1):

        model_no_penalty = LogisticRegression(penalty=None, max_iter=1000)

        sfs_no_penalty = SequentialFeatureSelector(
            model_no_penalty, direction="forward", n_features_to_select=i
        )

        sfs_no_penalty.fit(predictors_train, response_train)

        selected_columns_no_penalty = sfs_no_penalty.get_feature_names_out()

        selected_predictors_train = predictors_train[selected_columns_no_penalty]

        scaled_interaction_model = LogisticRegression(penalty=None, max_iter=1000)

        scores = cross_val_score(
            scaled_interaction_model,
            selected_predictors_train,
            response_train,
            cv=KFold(shuffle=True, random_state=0),
        )

        scaled_interaction_model.fit(selected_predictors_train, response_train)

        predict_proba = scaled_interaction_model.predict_proba(
            selected_predictors_train
        )

        AIC = 2 * i + 2 * log_loss(response_train, predict_proba, normalize=False)
        metric_dict["AIC"].append(AIC)
        metric_dict["Cross Validation Scores"].append(scores)

    return pd.DataFrame(metric_dict, index=range(2, max_n_variables + 1))


def plot_metrics(metrics):
    min_n_variables = min(metrics.index)
    max_n_variables = max(metrics.index)
    fig, ax = plt.subplots(2)
    ax[0].set_xticks([])
    ax[0].set_ylim([0.5, 1])
    ax[0].set_xlim([min_n_variables, max_n_variables])

    ax[0].set_ylabel("Accuracy")
    min_score = [min(s) for s in metrics["Cross Validation Scores"]]
    max_score = [max(s) for s in metrics["Cross Validation Scores"]]
    mean_score = [np.mean(s) for s in metrics["Cross Validation Scores"]]
    ax[0].fill_between(
        metrics.index,
        [min(s) for s in metrics["Cross Validation Scores"]],
        [max(s) for s in metrics["Cross Validation Scores"]],
        color="orange",
        alpha=0.8,
        label="Accuracy Range",
    )
    ax[0].plot(
        metrics.index, mean_score, color="black", linestyle="--", label="Mean Accuracy"
    )
    ax[0].legend()
    ax[1].set_xlim([min_n_variables, max_n_variables])
    ax[1].set_xticks(range(min_n_variables, max_n_variables + 1, 2))
    ax[1].plot(metrics.index, metrics["AIC"])
    ax[1].set_xlabel("Number of Predictors in Model")
    ax[1].set_ylabel("AIC")
    plt.tight_layout()
    plt.show()
