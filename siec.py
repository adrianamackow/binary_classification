import numpy as np
from keras import models
from keras import layers
from sklearn.model_selection import KFold


def get_vector(what):
    with open(what, "r") as f:
        v = f.read()
        v = eval(v)
        return np.asarray(v)

# 5-krotna walidacja krzyzowa
kf = KFold(n_splits=5, shuffle=True)
data = np.load('data.npy')
target = get_vector("target_file")

#okreslenie żądanej liczby cech
number_of_features = 2977

#uruchomienie sieci neuronowej i dodanie warstw
network = models.Sequential()
network.add(layers.Dense(units=1024, activation="relu", input_shape=(number_of_features,)))
network.add(layers.Dense(units=512, activation="relu"))
network.add(layers.Dense(units=1, activation="sigmoid"))

#Kompilacja sieci neuronowej
network.compile(loss="binary_crossentropy",
                optimizer="rmsprop",
                metrics=["accuracy"])

#ustalenie wag
wages = network.get_weights()
results = []

#wytrenowanie sieci neuronowej z wykorzystaniem walidacji krzyzowej
for train, test in kf.split(data):
    network.set_weights(wages)
    network.fit(data[train], target[train], nb_epoch=3, verbose=1, batch_size=1000)
    score = network.evaluate(data[test], target[test], verbose=1)
    results.append(score)

print(results)
print("Walidacja krzyzowa: ", np.mean(results))



# wytrenowanie sieci neuronowej z wykorzystaniem odłożonego zbioru treningowego i testowego
# x_train, x_valid, y_train, y_valid = train_test_split(data, target, test_size=0.33, shuffle=True)
# history = network.fit(data,
#                       target,
#                       epochs=2,
#                       verbose=1,
#                       batch_size=100
#                       )

#
# print("score: ", score[1])
# results.append(score[1])
