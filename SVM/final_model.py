import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.pyplot import xlabel
from scipy.special import logit
import seaborn as sns

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay


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
###                   DATA PLOTS                      ###
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
    sns.histplot(training_set_transformed[col], ax=axes1[i], alpha=1, color='palegreen', label='Transformed, scaled')
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

predictors_train = training_set_transformed[features_needed]
response_train = training_set_transformed['Label']

model = SVC(kernel='rbf', random_state=0)
model.fit(predictors_train, response_train)

response_pred = model.predict(testing_set_transformed)

testing_set_transformed['Label'] = response_pred

#print(testing_set_transformed)

#########################################################
###             TRAINING VISUALIZATION                ###
#########################################################

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(12, 12), sharex=False, sharey=False)

# Define custom colors for the classes
cmap_light = ListedColormap(['#ffcccc', '#ccffcc'])  # Light red and light green for decision boundaries
cmap_bold = ['red', 'green']  # Red and green for the points
formatter = FuncFormatter(lambda x, _: f'{abs(x):.1f}')

for i in range(len(predictors_train.columns)):
    for j in range(len(predictors_train.columns)):
        feature1 = j
        feature2 = i
        ax = axes[i, j]

        if i != j:
            X = predictors_train.to_numpy()[:, [feature1, feature2]]
            classifier = SVC(kernel='rbf', random_state=0).fit(X, response_train)

            disp = DecisionBoundaryDisplay.from_estimator(
                classifier, X, response_method="predict",
                xlabel=predictors_train.columns[feature1],
                ylabel=predictors_train.columns[feature2],
                alpha=0.5, ax=ax, cmap=cmap_light
            )

            ax.scatter(X[:, 0], X[:, 1], c=response_train, cmap=ListedColormap(cmap_bold), edgecolor="k", alpha=0.7)

            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.plot([0, 1], [1, 0], transform=ax.transAxes, color='black', linewidth=2)
            ax.set_xlabel(predictors_train.columns[feature1])
            ax.set_ylabel(predictors_train.columns[feature2])
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

        if j > 0:
            ax.set_ylabel('')

        if i < len(predictors_train.columns) - 1:
            ax.set_xlabel('')

plt.tight_layout()
plt.savefig('SVM_DecisionBoundaries.eps', format='eps')
