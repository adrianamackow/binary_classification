import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
from pprint import pprint

# Redukcja cech w rzadkich danych z wykorzystaniem funkcji TSVD

# wczytanie danych
data = np.load('data_3class_bagofwords.npy')
features = data
print(data.shape)
# np.savetxt('data_test.csv', data, delimiter=',')


# digits = datasets.load_digits()
features = StandardScaler().fit_transform(data)
features_sparse = csr_matrix(features)

# zdefiniowanie operacji TSVD
tsvd = TruncatedSVD(n_components=4873)
features_tsvd = tsvd.fit(features)
tsvd_var_ratios = tsvd.explained_variance_ratio_


def select_n_components(var_ratio, goal_var):
    total_variance = 0.0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components


n_components = select_n_components(tsvd_var_ratios, 0.95)
print(n_components)

features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
print("przed:", features_sparse.shape[1])
print("po:", features_sparse_tsvd.shape[1])
np.save('data_3class_bag_of_words_tsvd.npy', features_sparse_tsvd)
np.savetxt('data_3class_bag_of_words_tscd.csv', features_sparse_tsvd, delimiter=',')

# Zdefiniowanie i przeprowadzenie operacji TSVD z wartością o jeden mniejszą niż liczba cech
'''
data = np.load('data_bag_of.npy')
features = data
print(data.shape)


features = StandardScaler().fit_transform(data)
features_sparse = csr_matrix(features)
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)
tsvd_var_ratios = tsvd.explained_variance_ratio_


def select_n_components(var_ratio, goal_var):
    total_variance = 0.0
    n_components = 0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
    return n_components


n_components = select_n_components(tsvd_var_ratios, 0.95)
print(n_components)'''
