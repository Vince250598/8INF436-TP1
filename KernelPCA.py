from sklearn.decomposition import KernelPCA
import pandas as pd
from Tools import scatter_plot, anomaly_scores, plot_results
import numpy as np


def apply_kernel_pca(n_components, random_state, features, classes):
    kernel_pca = KernelPCA(n_components=n_components, kernel='rbf', gamma=None, fit_inverse_transform=True,
                           n_jobs=1, random_state=random_state)

    kernel_pca.fit(features)
    features_kernel_PCA = kernel_pca.transform(features)
    features_kernel_PCA = pd.DataFrame(data=features_kernel_PCA, index=features.index)

    scatter_plot(features_kernel_PCA, classes, "Kernel PCA")

    features_kernel_PCA_inverse = kernel_pca.inverse_transform(features_kernel_PCA)
    features_kernel_PCA_inverse = pd.DataFrame(data=features_kernel_PCA_inverse, index=features.index)

    anomaly_scores_kernel_PCA = anomaly_scores(features, features_kernel_PCA_inverse)
    preds = plot_results(classes, anomaly_scores_kernel_PCA, True)

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
