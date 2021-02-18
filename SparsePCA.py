from sklearn.decomposition import SparsePCA
import pandas as pd
from Tools import scatter_plot, anomaly_scores, plot_results
import numpy as np


def apply_sparse_pca(n_components, alpha, random_state, features, classes):
    sparse_PCA = SparsePCA(n_components=n_components, alpha=alpha, random_state=random_state, n_jobs=-1)

    sparse_PCA.fit(features.loc[:, :])
    features_sparse_PCA = sparse_PCA.transform(features)
    features_sparse_PCA = pd.DataFrame(data=features_sparse_PCA, index=features.index)

    scatter_plot(features_sparse_PCA, classes, "Sparse PCA")

    features_sparse_PCA_inverse = np.array(features_sparse_PCA).dot(sparse_PCA.components_) + np.array(
        features.mean(axis=0))
    features_sparse_PCA_inverse = pd.DataFrame(data=features_sparse_PCA_inverse, index=features.index)

    anomaly_scores_sparse_PCA = anomaly_scores(features, features_sparse_PCA_inverse)
    preds = plot_results(classes, anomaly_scores_sparse_PCA, True)

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
