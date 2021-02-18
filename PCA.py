from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from Tools import scatter_plot, compute_metrics


def explained_variance_pca(pca, n_components):
    importance_principal_components = pd.DataFrame(data=pca.explained_variance_ratio_)
    importance_principal_components = importance_principal_components.T

    for x in range(1, n_components):
        print('Variance Captured by First ' + str(x + 1) + ' Principal Components: ',
              importance_principal_components.loc[:, 0:x].sum(axis=1).values)

    sns.set(rc={'figure.figsize': (10, 10)})
    sns.barplot(data=importance_principal_components.loc[:, 0:9], color='k').set_title(
        'Variance explained by component')
    plt.show()


def apply_pca(n_components, whiten, random_state, features, classes):
    pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

    features_PCA = pca.fit_transform(features)
    features_PCA = pd.DataFrame(data=features_PCA, index=features.index)

    scatter_plot(features_PCA, classes, "PCA")

    features_PCA_inverse = pca.inverse_transform(features_PCA)
    features_PCA_inverse = pd.DataFrame(data=features_PCA_inverse, index=features.index)

    compute_metrics(features, features_PCA_inverse, classes)

    explained_variance_pca(pca, n_components)
