import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import logit
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import RFE, SequentialFeatureSelector  # RFE only works for linear SVM
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, cohen_kappa_score

from collections import Counter
from cleanup import removeRows
from SVM.model import svm_model


train_file_path = '../project_train.csv'
test_file_path = '../project_test.csv'

data_train = pd.read_csv(train_file_path)
data_test = pd.read_csv(test_file_path)

features_needed = ['danceability','speechiness', 'liveness', 'loudness']  # Based on the feature selection process described in the report

training_set = data_train[features_needed + ['Label']]
rows_to_remove = [68, 94]  # 69: speechiness outlier, 95: loudness false data
print(f'Rows removed: \n {training_set.iloc[rows_to_remove]}')
training_set = training_set.drop(rows_to_remove)

testing_set = data_test[features_needed]

# Transformation and scaling
vars_to_transform = [
    "danceability",
    "speechiness",
    "liveness"
]

training_set_transformed = training_set.copy()
training_set_transformed[vars_to_transform] = training_set_transformed[vars_to_transform].transform(logit)
training_set_transformed[vars_to_transform + ['loudness']] = preprocessing.scale(training_set_transformed[vars_to_transform + ['loudness']])

testing_set_transformed = testing_set.copy()
testing_set_transformed[vars_to_transform] = testing_set_transformed[vars_to_transform].transform(logit)
testing_set_transformed[vars_to_transform + ['loudness']] = preprocessing.scale(testing_set_transformed[vars_to_transform + ['loudness']])

#########################################################
###                     PLOTS                         ###
#########################################################

# Pair plot of labelled data
display_train = training_set.copy()
display_train['Label'] = display_train['Label'].map({0: 'Dislike', 1: 'Like'})

sns.pairplot(display_train, hue="Label", diag_kws={'fill': False}, palette={'Dislike': 'red', 'Like': 'green'})
plt.show()

# Pair plot of unlabelled data
sns.pairplot(testing_set, hue=None)
plt.show()


# Distribution of transformed and scaled data compared to original
fig1, axes1 = plt.subplots(2, 2, figsize=(6, 6))
axes1 = axes1.flatten()

for i, col in enumerate(training_set[features_needed].columns):
    sns.histplot(training_set_transformed[col], ax=axes1[i],alpha=1, color='palegreen', label='Transformed, scaled')
    sns.histplot(training_set[col], ax=axes1[i], alpha=0.5, color='bisque', label='Original')
    axes1[i].set_xlabel(col)
    axes1[i].set_ylabel('Count')
fig1.subplots_adjust(top=0.8)
handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, ['Transformed, scaled', 'Original'], loc='upper center')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


fig2, axes2 = plt.subplots(2, 2, figsize=(6, 6))
axes2 = axes2.flatten()

for i, col in enumerate(testing_set.columns):
    sns.histplot(testing_set_transformed[col], ax=axes2[i],alpha=1, color='palegreen', label='Transformed, scaled')
    sns.histplot(testing_set[col], ax=axes2[i], alpha=0.5, color='bisque', label='Original')
    axes2[i].set_xlabel(col)
    axes2[i].set_ylabel('Count')
fig2.subplots_adjust(top=0.8)
handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, ['Transformed, scaled', 'Original'], loc='upper center')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


fig3, axes3 = plt.subplots(2, 2, figsize=(6, 6))
axes3 = axes3.flatten()

for i, col in enumerate(testing_set.columns):
    sns.histplot(testing_set_transformed[col], ax=axes3[i],alpha=1, color='palegreen', label='Test')
    sns.histplot(training_set_transformed[col], ax=axes3[i], alpha=0.5, color='bisque', label='Train')
    axes3[i].set_xlabel(col)
    axes3[i].set_ylabel('Count')
fig3.subplots_adjust(top=0.8)
handles, labels = axes2[0].get_legend_handles_labels()
fig3.legend(handles, ['Test', 'Train'], loc='upper center')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()


#########################################################
###                    TRAINING                       ###
#########################################################

predictors_train = training_set[features_needed]
response_train = training_set['Label']

model = SVC(kernel='rbf', random_state=0)
model.fit(predictors_train, response_train)

response_pred = model.predict(testing_set_transformed)

testing_set_transformed['Label'] = response_pred

print(testing_set_transformed)
