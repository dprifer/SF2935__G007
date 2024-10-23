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
rows_to_remove = [68, 84, 94]  # 68: speechiness outlier, 84: energy false data, 94: loudness false data
print(f'Rows removed: \n {training_set.iloc[rows_to_remove]}')
training_set = training_set.drop(rows_to_remove)
training_set = training_set.reset_index(drop=True)

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
###                    TRAINING                       ###
#########################################################

predictors_train = training_set_transformed[features_needed]
response_train = training_set_transformed['Label']

model = SVC(kernel='rbf', random_state=0)
model.fit(predictors_train, response_train)

response_pred = model.predict(testing_set_transformed)

testing_set_transformed['Label'] = response_pred

#print(testing_set_transformed)