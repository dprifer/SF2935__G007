from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score


def svm_model(kernel, predictor_train, response_train, predictors_test, response_test, c = 1, gamma = 'scale', metrics = 'single', i = 0):
    model = SVC(kernel=kernel, C=c, gamma=gamma, random_state=0)
    model.fit(predictor_train, response_train)

    # Evaluate SVM
    response_pred_linear = model.predict(predictors_test)
    accuracy = accuracy_score(response_test, response_pred_linear)
    precision = precision_score(response_test, response_pred_linear)
    recall = recall_score(response_test, response_pred_linear)
    kappa = cohen_kappa_score(response_test, response_pred_linear)
    if metrics == 'single':
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"Cohen's Kappa Score: {kappa:.2f}")
    elif metrics == 'sfs':
        print(f"With {i} features: Accuracy -- {accuracy * 100:.2f}%, precision: {precision * 100:.2f}%, recall: {recall * 100:.2f}%, kappa: {kappa:.2f}")

    return model, accuracy