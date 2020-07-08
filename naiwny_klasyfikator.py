import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#Klasyfikator z uzyciem naiwnego klasyfikatora Bayesa
def get_vector(what):
    with open(what, "r") as f:
        v = f.read()
        v = eval(v)
        return np.asarray(v)

kf = KFold(n_splits=5, shuffle=True)
features = np.load('data_3class_bagofwords.npy')
target = get_vector("target_file_3class")

results = []

for train, test in kf.split(features):
    classifer = MultinomialNB()
    model = classifer.fit(features[train], target[train])
    target_predict = model.predict(features[test])
    result = accuracy_score(target[test],target_predict)
    results.append(result)
print(results)
print("Średnia wartość accuracy: ", np.mean(results))



