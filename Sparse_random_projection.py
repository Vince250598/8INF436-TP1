from sklearn.random_projection import SparseRandomProjection
import pandas as pd
from Tools import scatter_plot, plot_results, anomaly_scores
import numpy as np


def apply_sparse_random_projection(n_components, eps, random_state, features, classes):
    SRP = SparseRandomProjection(n_components=n_components, density='auto', eps=eps, dense_output=True,
                                 random_state=random_state)

    features_SRP = SRP.fit_transform(features)
    features_SRP = pd.DataFrame(data=features_SRP, index=features.index)

    scatter_plot(features_SRP, classes, "Sparse Random Projection")

    features_SRP_inverse = np.array(features_SRP).dot(SRP.components_.toarray()) + np.array(features.mean(axis=0))
    features_SRP_inverse = pd.DataFrame(data=features_SRP_inverse, index=features.index)

    anomaly_scores_SRP = anomaly_scores(features, features_SRP_inverse)
    preds = plot_results(classes, anomaly_scores_SRP, True)

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
