import pandas as pd
import sklearn.model_selection as ms
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PCA import apply_pca
from SparsePCA import apply_sparse_pca
from KernelPCA import apply_kernel_pca
from Gaussian_random_projection import apply_gaussian_random_projection
from Sparse_random_projection import apply_sparse_random_projection
from Dictionary_learning import apply_dictionay_learning
from Fast_ICA import apply_ICA

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

random_state_algorithms = 2021
random_state_train_test_split = 2018

data = pd.read_csv('SolarFlare.csv')

# Impression des informations sur le dataset
print(data.head())
print(data.info())
print(data.describe())

# Compte du nombre de valeurs pour chacune des classes
print(data['class'].value_counts())

# Separation du dataset en 2 ensembles, un pour les colonnes des attributs et l'autres pour les classes
data_X = data.copy().drop(['class'], axis=1)
data_Y = data['class'].copy()

# Encodage des attributs catégoriels et normalisation des autres attributs
features_to_scale = data_X.drop(['largest_spot_size', 'spot_distribution'], axis=1).columns

e_X = LabelEncoder()
s_X = StandardScaler(copy=True)

data_X = data_X.apply(e_X.fit_transform)
data_X.loc[:, features_to_scale] = s_X.fit_transform(data_X[features_to_scale])


# 30% de données de test
X_train, X_test, y_train, y_test = ms.train_test_split(data_X, data_Y, test_size=0.3, random_state=random_state_train_test_split
                                                       , stratify=data_Y)

# 40% de données de test
'''
X_train, X_test, y_train, y_test = ms.train_test_split(data_X, data_Y, test_size=0.4, random_state=random_state_train_test_split
                                                    , stratify=data_Y)
'''

# RÉSULTATS AVEC LES DONNÉES D'ENTRAINEMENT

# PCA avec 6 composants à garder
apply_pca(n_components=6, whiten=False, random_state=random_state_algorithms, features=X_train, classes=y_train)

# PCA avec 7 composants à garder
apply_pca(n_components=7, whiten=False, random_state=random_state_algorithms, features=X_train, classes=y_train)

# PCA avec 5 composants à garder
apply_pca(n_components=5, whiten=False, random_state=random_state_algorithms, features=X_train, classes=y_train)

# Sparse PCA avec alpha=0.0001 et 6 composants à garder
apply_sparse_pca(n_components=6, alpha=0.0001, random_state=random_state_algorithms, features=X_train, classes=y_train)

# Kernel PCA avec 6 composants à garder
apply_kernel_pca(n_components=6, random_state=random_state_algorithms, features=X_train, classes=y_train, )

# Gaussian Random Projection 6 composants à garder
apply_gaussian_random_projection(n_components=6, random_state=random_state_algorithms, features=X_train, classes=y_train)

# Gaussian Random Projection avec 5 composants à garder
apply_gaussian_random_projection(n_components=5, random_state=random_state_algorithms, features=X_train, classes=y_train)

# Sparse Random Projection avec 6 composants à garder
apply_sparse_random_projection(n_components=6, eps=.01, random_state=random_state_algorithms, features=X_train, classes=y_train)

# Dictionnary Learning avec 6 composants principaux
apply_dictionay_learning(n_components=6, batch_size=15, n_iter=1000, random_state=random_state_algorithms, features=X_train,
                         classes=y_train)

# ICA avec 6 composants principaux
apply_ICA(n_components=6, random_state=random_state_algorithms, features=X_train, classes=y_train)


# RÉSULTATS AVEC LES DONNÉES DE TEST

# PCA avec les données de test
apply_pca(n_components=6, whiten=False, random_state=random_state_algorithms, features=X_test, classes=y_test)

# Gaussian Random Projection avec les données de test
apply_gaussian_random_projection(n_components=6, random_state=random_state_algorithms, features=X_test, classes=y_test)

# ICA avec les données de test
apply_ICA(n_components=6, random_state=random_state_algorithms, features=X_test, classes=y_test)
