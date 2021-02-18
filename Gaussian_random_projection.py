from sklearn.random_projection import GaussianRandomProjection
import pandas as pd
from Tools import scatter_plot, plot_results, anomaly_scores
import numpy as np


def apply_gaussian_random_projection(n_components, random_state, features, classes):
    GRP = GaussianRandomProjection(n_components=n_components, eps=None, random_state=random_state)

    features_GRP = GRP.fit_transform(features)
    features_GRP = pd.DataFrame(data=features_GRP, index=features.index)

    scatter_plot(features_GRP, classes, "Gaussian Random Projection")

    features_GRP_inverse = np.array(features_GRP).dot(GRP.components_) + np.array(features.mean(axis=0))
    features_GRP_inverse = pd.DataFrame(data=features_GRP_inverse, index=features.index)

    anomaly_scores_GRP = anomaly_scores(features, features_GRP_inverse)
    preds = plot_results(classes, anomaly_scores_GRP, True)

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
