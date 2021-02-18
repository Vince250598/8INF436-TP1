from sklearn.decomposition import MiniBatchDictionaryLearning
import pandas as pd
from Tools import scatter_plot, anomaly_scores, plot_results
import numpy as np


def apply_dictionay_learning(n_components, batch_size, n_iter, random_state, features, classes):
    miniBatchDictionayLearning = MiniBatchDictionaryLearning(n_components=n_components, alpha=1, batch_size=batch_size,
                                                             n_iter=n_iter, random_state=random_state)

    miniBatchDictionayLearning.fit(features)
    features_miniBatchDictLearning = miniBatchDictionayLearning.fit_transform(features)
    features_miniBatchDictLearning = pd.DataFrame(data=features_miniBatchDictLearning, index=features.index)

    scatter_plot(features_miniBatchDictLearning, classes, "Mini-batch Dictionary Learning")

    features_miniBatchDictLearning_inverse = np.array(features_miniBatchDictLearning).dot(
        miniBatchDictionayLearning.components_) + np.array(features.mean(axis=0))
    features_miniBatchDictLearning_inverse = pd.DataFrame(data=features_miniBatchDictLearning_inverse,
                                                          index=features.index)

    anomaly_scores_miniBatchDictionaryLearning = anomaly_scores(features, features_miniBatchDictLearning_inverse)
    preds = plot_results(classes, anomaly_scores_miniBatchDictionaryLearning, True)

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
