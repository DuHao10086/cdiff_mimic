import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
# from cm_utils import plot_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


def auc_ci(y_pred, y_true, n_bootstraps=2000, ci_level=0.95):
    li = (1. - ci_level) / 2
    ui = 1 - li

    rng = np.random.RandomState(seed=20)
    bootstrapped_auc = []

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_auc.append(auc)

    sorted_scores = np.array(bootstrapped_auc)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]

    return confidence_lower, confidence_upper

def gen_ci(value_list, ci_level=0.95):
    li = (1. - ci_level) / 2
    ui = 1 - li
    sorted_scores = np.array(value_list)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(li * len(sorted_scores))]
    confidence_upper = sorted_scores[int(ui * len(sorted_scores))]
    return confidence_lower, confidence_upper

def gen_auc_plot(models, names, title, X, y, ci_level=None, save_name=None):
    plt.figure(figsize=(10, 10))
    for i, model in enumerate(models):
        if type(model) == list:
            f_hat = np.repeat(np.array(model)[..., np.newaxis], 2, axis=1)
        else:
            f_hat = model.predict_proba(X)
        roc = roc_curve(y, f_hat[:, 1])
        auc = roc_auc_score(y, f_hat[:, 1])
        if (ci_level != None):
            ci = auc_ci(y_pred=f_hat[:, 1], y_true=y, ci_level=ci_level)
            sns.lineplot(x=roc[0], y=roc[1], label='{0}\n(AUC = {1:.3f} [{2:.3f}, {3:.3f}])'.format(names[i], auc, *ci))
        else:
            sns.lineplot(x=roc[0], y=roc[1], label='{0}\n(AUC = {1:.3f})'.format(names[i], auc))
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('1 - Specificty')
    plt.ylabel('Sensitivity')
    plt.title(title)
    if save_name != None:
        plt.savefig(save_name + '.svg', bbox_inches='tight')
        plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()


def save_model(model, file_name):
    pickle.dump(model, open('./{0}'.format(file_name), 'wb'))


def perf_metrics_2X2(y_true, y_pred):
    """
    Returns the specificity, sensitivity, positive predictive value, and
    negative predictive value
    of a 2X2 table.

    where:
    0 = negative case
    1 = positive case

    Parameters
    ----------
    yobs :  array of positive and negative ``observed`` cases
    yhat : array of positive and negative ``predicted`` cases

    Returns
    -------
    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    pos_pred_val = TP/ (TP+FP)
    neg_pred_val = TN/ (TN+FN)

    """
    y_true = y_true.flatten()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    #     print(cnf_matrix)

    #     TN = cnf_matrix[0][0]
    #     FN = cnf_matrix[1][0]
    #     TP = cnf_matrix[1][1]
    #     FP = cnf_matrix[0][1]
    TP = np.sum(y_true[y_true == 1] == y_pred[y_true == 1])
    TN = np.sum(y_true[y_true == 0] == y_pred[y_true == 0])
    FP = np.sum(y_true[y_true == 0] != y_pred[y_true == 0])
    FN = np.sum(y_true[y_true == 1] != y_pred[y_true == 1])
    #     print(TP, TN, FP, FN)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    pos_pred_val = TP / (TP + FP)
    neg_pred_val = TN / (TN + FN)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = TP / (TP + FN)
    # Specificity or true negative rate
    specificity = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)

    return sensitivity, specificity, PPV, NPV


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)
    plt.grid(b=None)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=18)

    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)


def evaluate_model(model, X_test, y_test, model_name, cmap=plt.cm.Blues):
    y_prob = model.predict_proba(X_test)[:, 1]
    thres = np.percentile(y_prob, 95)
    decision = (y_prob >= thres).astype(int)
    acc = accuracy_score(y_pred=decision, y_true=y_test)
    auc = roc_auc_score(y_score=y_prob, y_true=y_test)
    sensitivity, specificity, pos_pred_val, neg_pred_val = perf_metrics_2X2(y_true=y_test, y_pred=decision)
    cnf_matrix = confusion_matrix(y_test, decision)
    np.set_printoptions(precision=2)

    plt.figure(figsize=(10, 10))
    plot_confusion_matrix(cnf_matrix, classes=['survived', 'dead'],
                          title='{} Confusion Matrix'.format(model_name),
                          cmap=cmap)
    ax = plt.gca()
    ax.grid(False)

    print("Thresholds: {:.2f}".format(thres))
    print("Accuracy: {:.2f}%".format(acc * 100))
    print("AUROC: {:.2f}".format(auc))
    print("Sensitivity: {:.2f}%".format(sensitivity * 100))
    print("Specificity: {:.2f}%".format(specificity * 100))
    print("PPV: {:.2f}%".format(pos_pred_val * 100))
    print("NPV: {:.2f}%".format(neg_pred_val * 100))


# Function for generating reliability curves
def gen_calib_plot(models, names, title, X, y, save_name=None):
    plt.figure(figsize=(10, 10))
    for i, model in enumerate(models):
        f_hat = model.predict_proba(X)
        fraction_of_positives, mean_predicted_value = calibration_curve(y, f_hat[:, 1], n_bins=10)
        brier_score = brier_score_loss(y, f_hat[:, 1], pos_label=y.max())
        sns.lineplot(x=mean_predicted_value, y=fraction_of_positives,
                     label='{0}\n(Brier score: {1:.3f})'.format(names[i], brier_score))
    plt.plot([0, 1], [0, 1], 'k:')
    plt.xlabel('Mean Predicted Value')
    plt.ylabel('Actual Probability of Outcome')
    plt.title(title)
    if save_name != None:
        plt.savefig(save_name + '.svg', bbox_inches='tight')
        plt.savefig(save_name + '.png', bbox_inches='tight')
    plt.show()

def train_model(model, X_train, y_train, X_test, y_test):
    fitted_model = model.fit(X_train, y_train)
    y_prob = fitted_model.predict_proba(X_test)[:, 1]
    thres = np.percentile(y_prob, 95)
    decision = (y_prob >= thres).astype(int)
    acc = accuracy_score(y_pred=decision, y_true=y_test)
    auc = roc_auc_score(y_score=y_prob, y_true=y_test)
    sensitivity, specificity, pos_pred_val, neg_pred_val = perf_metrics_2X2(y_true=y_test, y_pred=decision)
    print("Thresholds: {:.2f}".format(thres))
    print("Accuracy: {:.2f}%".format(acc*100))
    print("AUROC: {:.2f}".format(auc))
    print("Sensitivity: {:.2f}%".format(sensitivity*100))
    print("Specificity: {:.2f}%".format(specificity*100))
    print("PPV: {:.2f}%".format(pos_pred_val*100))
    print("NPV: {:.2f}%".format(neg_pred_val*100))
    return fitted_model

def model_fit(model, X_train, y_train, X_test, y_test):
    fitted_model = model.fit(X_train, y_train)
    calibrated_model = CalibratedClassifierCV(fitted_model, method='sigmoid', cv=5)
    calibrated_model = calibrated_model.fit(X_train, y_train)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    thres = np.percentile(y_prob, 95)
    decision = (y_prob >= thres).astype(int)
    acc = accuracy_score(y_pred=decision, y_true=y_test)
    auc = roc_auc_score(y_score=y_prob, y_true=y_test)
    sensitivity, specificity, pos_pred_val, neg_pred_val = perf_metrics_2X2(y_true=y_test, y_pred=decision)
    return fitted_model, calibrated_model, y_test, y_prob, decision, acc, auc, sensitivity, specificity, pos_pred_val, neg_pred_val, thres