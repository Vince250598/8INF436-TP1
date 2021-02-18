import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def anomaly_scores(original_DF, reduced_DF):
    loss = np.sum((np.array(original_DF) - np.array(reduced_DF)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=original_DF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))

    return loss


def plot_results(true_labels, anomaly_scores, return_preds=False):
    preds = pd.concat([true_labels, anomaly_scores], axis=1)
    preds.columns = ['trueLabel', 'anomalyScore']

    precision, recall, thresholds = metrics.precision_recall_curve(preds['trueLabel'], preds['anomalyScore'])

    average_precision = metrics.average_precision_score(preds['trueLabel'], preds['anomalyScore'])

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(average_precision))

    fpr, tpr, thresholds = metrics.roc_curve(preds['trueLabel'], preds['anomalyScore'])
    area_under_ROC = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics: Area under the curve = {0:0.2f}'.format(area_under_ROC))
    plt.legend(loc="lower right")
    plt.show()

    if return_preds:
        return preds


def scatter_plot(x_DF, y_DF, algo_name):
    temp_DF = pd.DataFrame(data=x_DF.loc[:, 0:1], index=x_DF.index)
    temp_DF = pd.concat((temp_DF, y_DF), axis=1, join="inner")
    temp_DF.columns = ['First Vector', 'Second Vector', 'Label']
    sns.lmplot(x="First Vector", y="Second Vector", hue="Label", data=temp_DF, fit_reg=False)
    ax = plt.gca()
    ax.set_title("Separation of Observations using " + algo_name)
    plt.show()


def compute_metrics(original_DF, reduced_DF, classes):
    scores = anomaly_scores(original_DF, reduced_DF)
    preds = plot_results(classes, scores, True)

    preds.sort_values(by="anomalyScore", ascending=False, inplace=True)
    cutoff = 43
    preds_Top = preds[:cutoff]
    print("Precision: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              cutoff, 2))
    print("Recall: ",
          np.round(
              preds_Top.anomalyScore[preds_Top.trueLabel == 1].count() /
              classes.sum(), 2))
    print("Solar flares Caught out of 43 Cases:", preds_Top.trueLabel.sum())
