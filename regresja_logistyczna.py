import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#klasyfikator z użyciem regresji logistycznej
def get_vector(what):
    with open(what, "r") as f:
        v = f.read()
        v = eval(v)
        return np.asarray(v)

kf = KFold(n_splits=5, shuffle=True)
features = np.load('/Users/adrianamackow/PycharmProjects/cross2/data_ready_3class.npy')
target = get_vector("/Users/adrianamackow/PycharmProjects/cross2/target_file_3class")

results = []

for train, test in kf.split(features):
    logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")
    model = logistic_regression.fit(features[train], target[train])
    target_predict = model.predict(features[test])
    result = accuracy_score(target[test],target_predict)
    results.append(result)
print(results)
print("Średnia wartość accuracy: ", np.mean(results))
