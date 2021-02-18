from sklearn.decomposition import FastICA
import pandas as pd
from Tools import scatter_plot, plot_results, anomaly_scores
import numpy as np


def apply_ICA(n_components, random_state, features, classes):
    fast_ICA = FastICA(n_components=n_components, algorithm='parallel', whiten=True, max_iter=200,
                       random_state=random_state)

    features_fast_ICA = fast_ICA.fit_transform(features)
    features_fast_ICA = pd.DataFrame(data=features_fast_ICA, index=features.index)

    scatter_plot(features_fast_ICA, classes, "Independant Component Analysis")

    features_fast_ICA_inverse = fast_ICA.inverse_transform(features_fast_ICA)
    features_fast_ICA_inverse = pd.DataFrame(data=features_fast_ICA_inverse, index=features.index)

    anomaly_scores_ICA = anomaly_scores(features, features_fast_ICA_inverse)
    preds = plot_results(classes, anomaly_scores_ICA, True)

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
